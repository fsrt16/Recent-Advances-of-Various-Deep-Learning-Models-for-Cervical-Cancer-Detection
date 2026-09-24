"""
eval_student.py
───────────────
Standalone evaluation of the saved student checkpoint on the EXACT same
stratified test split used during training (random_state=42, test_size=0.10).

Runs:
  1) Standard inference   → fast sanity check
  2) 4-view TTA           → averaged softmax predictions

Uses cytoplasm masking + dynamically-computed mean/std to match the
training pipeline precisely.

Usage:
    python eval_student.py
    python eval_student.py --ckpt sipakmed_student_dual_densenet_best.pth
    python eval_student.py --ckpt sipakmed_student_dual_densenet_best.pth --data DATA_DIR
"""

import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, cohen_kappa_score, roc_auc_score
)
from tqdm import tqdm

# ── Import model + helpers from nonattnet.py ─────────────────────────
from nonattnet import (
    BioHCLSSSM, SIPaKMeDDataset, _Subset,
    apply_cytoplasm_mask, get_dataset_stats, get_transforms,
    CLASS_NAMES, NUM_CLASSES, DEVICE
)

# ── Reproducibility ───────────────────────────────────────────────────
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ═════════════════════════════════════════════════════════════════════
# 1. RECREATE THE EXACT SAME TEST SPLIT
# ═════════════════════════════════════════════════════════════════════

def get_test_samples(data_dir: str,
                     test_split: float = 0.10,
                     val_split: float = 0.111):
    """
    Mirrors create_dataloaders() exactly:
      All data  ──[stratified, seed=42]──► pool (90%) + test (10%)
    Returns (test_samples, d_mean, d_std, class_names).
    """
    all_samples = []
    ds = None
    for subfolder in ('Training', 'Testing'):
        root = os.path.join(data_dir, subfolder)
        if os.path.isdir(root):
            ds = SIPaKMeDDataset(root)
            all_samples.extend(ds.samples)

    if not all_samples:
        ds = SIPaKMeDDataset(data_dir)
        all_samples = ds.samples

    if not all_samples:
        raise RuntimeError(f"No images found under '{data_dir}'.")

    class_names = ds.class_names
    print(f"  Classes found : {class_names}")
    print(f"  Total images  : {len(all_samples)}")

    all_labels = [s[1] for s in all_samples]
    n_total    = len(all_samples)

    # Same stratified split as training (random_state=42)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_split,
                                      random_state=42)
    pool_idx, test_idx = next(sss_test.split(range(n_total), all_labels))
    pool_idx  = list(pool_idx)
    test_idx  = list(test_idx)

    # Reproduce train/val split so we can compute mean/std from train_samples
    pool_labels = [all_labels[i] for i in pool_idx]
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_split,
                                     random_state=42)
    rel_tr, _ = next(sss_val.split(range(len(pool_idx)), pool_labels))
    train_samples = [all_samples[pool_idx[i]] for i in rel_tr]
    test_samples  = [all_samples[i] for i in test_idx]

    print(f"  Train pool    : {len(train_samples)}  (used for mean/std only)")
    print(f"  Test set      : {len(test_samples)}")

    # Dynamic mean/std (from training pool, not test — avoids leakage)
    d_mean, d_std = get_dataset_stats(train_samples, img_size=256)
    return test_samples, d_mean, d_std, class_names


# ═════════════════════════════════════════════════════════════════════
# 2. STANDARD EVALUATION (no TTA)
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_standard(model, test_samples, d_mean, d_std,
                      img_size=224, batch_size=16, class_names=None):
    """Single-pass evaluation using the same eval transform as training."""
    if class_names is None:
        class_names = CLASS_NAMES

    eval_tf = get_transforms(img_size, 'eval', mean=d_mean, std=d_std)
    ds      = _Subset(test_samples, eval_tf)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    model.eval()
    preds_all, labels_all, probs_all = [], [], []

    for images, labels in tqdm(loader, desc="Standard eval"):
        images = images.to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                            enabled=DEVICE.type == 'cuda'):
            logits, _, _ = model(images)
        probs = torch.softmax(logits, dim=1)
        preds_all.extend(probs.argmax(1).cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
        labels_all.extend(labels.tolist())

    return _compute_metrics(preds_all, labels_all, probs_all, class_names)


# ═════════════════════════════════════════════════════════════════════
# 3. TTA EVALUATION (4 views)
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_tta(model, test_samples, d_mean, d_std,
                 img_size=224, batch_size=16, class_names=None):
    """
    4-view Test-Time Augmentation.
    Each view uses cytoplasm-masked images (via _Subset) with the same
    normalization computed from the training pool.
    Views: clean | h-flip | rot+10 | rot-10
    """
    if class_names is None:
        class_names = CLASS_NAMES

    tta_transforms = [
        # 0 — clean
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std),
        ]),
        # 1 — horizontal flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std),
        ]),
        # 2 — rotation +10°
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std),
        ]),
        # 3 — rotation -10°
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std),
        ]),
    ]

    model.eval()
    all_probs  = None
    all_labels = None

    for t_idx, tf in enumerate(tta_transforms):
        ds     = _Subset(test_samples, tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        probs_list, labels_list = [], []
        for images, labels in tqdm(loader,
                                   desc=f"TTA view {t_idx+1}/{len(tta_transforms)}",
                                   leave=False):
            images = images.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                enabled=DEVICE.type == 'cuda'):
                logits, _, _ = model(images)
            probs_list.append(torch.softmax(logits, dim=1).cpu())
            labels_list.append(labels)

        probs  = torch.cat(probs_list,  dim=0)
        labels = torch.cat(labels_list, dim=0)

        if all_probs is None:
            all_probs  = probs
            all_labels = labels
        else:
            all_probs += probs

    all_probs  /= len(tta_transforms)
    preds_all  = all_probs.argmax(dim=1).tolist()
    labels_all = all_labels.tolist()
    probs_all  = all_probs.tolist()

    return _compute_metrics(preds_all, labels_all, probs_all, class_names)


# ═════════════════════════════════════════════════════════════════════
# 4. METRICS HELPER
# ═════════════════════════════════════════════════════════════════════

def _compute_metrics(preds, labels, probs, class_names):
    n_cls    = len(class_names)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro',    zero_division=0)
    w_f1     = f1_score(labels, preds, average='weighted', zero_division=0)
    kappa    = cohen_kappa_score(labels, preds)

    try:
        auc = roc_auc_score(np.eye(n_cls)[labels], probs,
                            average="macro", multi_class="ovr")
    except Exception:
        auc = float('nan')

    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    per_class_sens, per_class_spec = {}, {}
    for c in range(n_cls):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class_sens[class_names[c]] = float(tp / (tp + fn + 1e-8))
        per_class_spec[class_names[c]] = float(tn / (tn + fp + 1e-8))

    return {
        "acc": acc, "macro_f1": macro_f1, "weighted_f1": w_f1,
        "kappa": kappa, "auc": auc,
        "sensitivity": per_class_sens,
        "specificity": per_class_spec,
        "confusion_matrix": cm,
        "report": classification_report(labels, preds,
                                        target_names=class_names,
                                        zero_division=0),
    }


def print_metrics(title: str, m: dict):
    bar = "=" * 50
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(f"  Accuracy    : {m['acc']*100:.2f}%")
    print(f"  Macro F1    : {m['macro_f1']:.4f}")
    print(f"  Weighted F1 : {m['weighted_f1']:.4f}")
    print(f"  Cohen κ     : {m['kappa']:.4f}")
    if not math.isnan(m['auc']):
        print(f"  AUC (macro) : {m['auc']:.4f}")
    print(f"\n  Per-class Sensitivity:")
    for cls, val in m['sensitivity'].items():
        print(f"    {cls:<35s}: {val*100:.1f}%")
    print(f"\n  Per-class Specificity:")
    for cls, val in m['specificity'].items():
        print(f"    {cls:<35s}: {val*100:.1f}%")
    print(f"\n  Classification Report:\n{m['report']}")
    print(f"\n  Confusion Matrix:\n{m['confusion_matrix']}")


# ═════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved student checkpoint on the stratified test split."
    )
    parser.add_argument("--ckpt",       default="sipakmed_student_dual_densenet_best.pth",
                        help="Path to student .pth checkpoint.")
    parser.add_argument("--data",       default="DATA_DIR",
                        help="Root data directory.")
    parser.add_argument("--config",     default="dual_densenet",
                        help="Model config name used during training.")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--img_size",   default=224, type=int)
    parser.add_argument("--no_tta",     action="store_true",
                        help="Skip TTA and only run standard eval.")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Student Checkpoint Evaluator")
    print(f"{'='*50}")
    print(f"  Device   : {DEVICE}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Data dir  : {args.data}")

    # ── 1. Recreate the exact test split ─────────────────────────────
    print("\n[1/3] Recreating stratified test split (seed=42)...")
    test_samples, d_mean, d_std, class_names = get_test_samples(args.data)

    # ── 2. Load model ────────────────────────────────────────────────
    print(f"\n[2/3] Loading model ({args.config}) from '{args.ckpt}'...")
    from nonattnet import MODEL_CONFIGS, create_model_variant
    student = create_model_variant(args.config)
    state = torch.load(args.ckpt, map_location=DEVICE)
    student.load_state_dict(state)
    student.eval()
    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"  Loaded. Parameters: {n_params:.1f}M")

    # ── 3. Evaluate ──────────────────────────────────────────────────
    print("\n[3/3] Running evaluation...")

    print("\n--- Standard (no TTA) ---")
    std_metrics = evaluate_standard(
        student, test_samples, d_mean, d_std,
        img_size=args.img_size, batch_size=args.batch_size,
        class_names=class_names
    )
    print_metrics("STANDARD TEST RESULTS", std_metrics)

    if not args.no_tta:
        print("\n--- 4-View TTA ---")
        tta_metrics = evaluate_tta(
            student, test_samples, d_mean, d_std,
            img_size=args.img_size, batch_size=args.batch_size,
            class_names=class_names
        )
        print_metrics("TTA TEST RESULTS (4 views)", tta_metrics)

        # Summary comparison
        print("\n" + "="*50)
        print("  SUMMARY COMPARISON")
        print("="*50)
        print(f"  {'Metric':<20} {'Standard':>10} {'TTA (4v)':>10}")
        print(f"  {'-'*40}")
        print(f"  {'Accuracy':<20} {std_metrics['acc']*100:>9.2f}% {tta_metrics['acc']*100:>9.2f}%")
        print(f"  {'Macro F1':<20} {std_metrics['macro_f1']:>10.4f} {tta_metrics['macro_f1']:>10.4f}")
        print(f"  {'Weighted F1':<20} {std_metrics['weighted_f1']:>10.4f} {tta_metrics['weighted_f1']:>10.4f}")
        print(f"  {'Cohen κ':<20} {std_metrics['kappa']:>10.4f} {tta_metrics['kappa']:>10.4f}")


if __name__ == "__main__":
    main()
