# Recent-Advances-of-Various-Deep-Learning-Models-for-Cervical-Cancer-Detection
Recent Advances of Various Hybrid Deep Learning Models for Cervical Cancer Detection and Diagnosis: A Comprehensive Study

A research repository for the study **“Recent Advances of Various Hybrid Deep Learning Models for Cervical Cancer Detection and Diagnosis: A Comprehensive Study.”** The project investigates convolutional, transformer, ensemble, and hybrid state-space architectures for automated cervical-cell classification from Pap-smear imagery.

> **Research prototype:** This repository is intended for research and educational use. Model predictions must not be used as a substitute for cytopathologist review, clinical diagnosis, or treatment decisions.

## Overview

Cervical cancer screening remains challenging because cytological images contain overlapping cells, staining variability, cellular-scale morphology, and class imbalance. This project benchmarks multiple deep-learning families and introduces **BioFusion-MambaNet (BFM-Net)**, a hybrid architecture designed to combine:

- **Multi-Scale Attention Fusion (MSAF)** for adaptive modeling of nuclear, cellular, and cluster-level features.
- **Cytoplasm State-Space Dynamics (CSSD/Mamba-style blocks)** for efficient long-range sequence modeling.
- **Teacher–student knowledge distillation** for transferring representations from a larger teacher to an efficient student.
- **Physics-informed regularization** for smoother and more biologically plausible representations.
- **Explainable AI (XAI)** using Grad-CAM, SHAP, and LIME to inspect model reasoning.

## Main Results

Experiments use the WSI cluster subset of the SIPaKMeD dataset, containing 966 images across five classes. On the reported held-out test split, BFM-Net achieved:

| Metric | Result |
|---|---:|
| Accuracy | 97.44% |
| Macro-F1 | 98.04% |
| Precision | 98.41% |
| Recall | 97.44% |
| Parameters | Approximately 18.5–25.51M, depending on configuration |

The study also reports 10-fold stratified cross-validation results for model-family comparisons, with BFM-Net obtaining approximately 97.27% accuracy and 98.06% macro-F1.

> Reported metrics depend on the exact split, preprocessing, augmentation, test-time augmentation, checkpoint, and implementation configuration. Please reproduce results before making comparisons or clinical claims.

## Dataset

The experiments use the **SIPaKMeD WSI cluster subset**. The study focuses on multicellular cluster images rather than isolated single-cell crops.

| Class | Abbreviation | Images | Description |
|---|---:|---:|---|
| `abnormal_Dyskeratotic` | DC | 271 | Pyknotic nuclei, keratinization, and perinuclear halos |
| `abnormal_Koilocytotic` | KC | 238 | Enlarged nuclei and perinuclear clearing associated with koilocytotic morphology |
| `benign_Metaplastic` | MC | 223 | Transitional squamous morphology and dense chromatin |
| `normal_Parabasal` | PC | 108 | Basaloid cells with a high nucleus-to-cytoplasm ratio |
| `normal_Superficial-Intermediate` | SIC | 126 | Mature squamous cells with abundant cytoplasm and pyknotic nuclei |
| **Total** | — | **966** | — |

The dataset is not redistributed in this repository. Obtain it from an authorized source and comply with its license and usage conditions.

## Method

### BFM-Net pipeline

The general pipeline is:

1. Load RGB Pap-smear cluster images.
2. Apply resizing, normalization, geometric augmentation, photometric augmentation, blur, perspective perturbation, RandAugment, and random erasing during training.
3. Extract features using a pretrained image backbone.
4. Convert the spatial representation into patch tokens.
5. Refine the representation using MSAF and CSSD/Mamba blocks.
6. Pool the token sequence and predict one of five cytological classes.
7. Train using a composite objective combining classification, physics-informed, and distillation losses.
8. Evaluate using accuracy, macro-F1, precision, recall, confusion matrices, and cross-validation analyses.
9. Generate visual explanations using Grad-CAM, SHAP, and LIME.

### Composite objective

The reported training objective is:

\[
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{phys}\mathcal{L}_{phys} + \mathcal{L}_{KD}
\]

where:

- `LCE` is the supervised classification loss.
- `Lphys` encourages smoothness and biologically motivated feature behavior.
- `LKD` transfers softened teacher predictions to the student.
- The production configuration uses approximately `lambda_phys = 0.3`, `kd_temp = 6`, and `kd_alpha = 0.5`.

## Repository Structure

The exact structure may evolve with implementation updates. A recommended organization is:

```text
.
├── configs/              # Experiment and model configurations
├── data/                 # Local dataset links or metadata; do not commit raw data
├── datasets/             # Dataset loaders and preprocessing utilities
├── models/               # CNN, transformer, MSAF, CSSD, and BFM-Net modules
├── losses/               # CE, physics-informed, and distillation losses
├── training/             # Training, validation, checkpointing, and scheduling
├── evaluation/           # Metrics, cross-validation, statistical tests, and plots
├── xai/                  # Grad-CAM, SHAP, and LIME utilities
├── checkpoints/          # Local model weights; normally ignored by Git
├── results/              # Metrics, figures, and experiment summaries
├── requirements.txt      # Python dependencies
└── README.md
```

## Installation

Create a Python environment and install the project dependencies:

```bash
git clone https://github.com/fsrt16/Recent-Advances-of-Various-Deep-Learning-Models-for-Cervical-Cancer-Detection.git
cd Recent-Advances-of-Various-Deep-Learning-Models-for-Cervical-Cancer-Detection

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for training. Install the PyTorch build that matches your operating system and CUDA version before installing the remaining dependencies if required.

## Data Preparation

1. Download the SIPaKMeD dataset from an authorized source.
2. Select the WSI/multicellular cluster subset used in the study.
3. Arrange images under class-specific directories, for example:

```text
data/sipakmed_wsi/
├── abnormal_Dyskeratotic/
├── abnormal_Koilocytotic/
├── benign_Metaplastic/
├── normal_Parabasal/
└── normal_Superficial-Intermediate/
```

4. Update the dataset path in the relevant configuration or command-line argument.
5. Keep isolated single-cell crops separate from the WSI cluster subset to avoid mixing experimental modalities.

Do not commit patient-sensitive data, downloaded datasets, or private annotations to the repository.

## Training

The study describes the following general training settings:

- Stratified data partitioning.
- AdamW optimization.
- Learning-rate warm-up followed by cosine annealing.
- Gradient clipping with a maximum norm of 1.0.
- Class-balanced sampling for imbalanced classes.
- Early stopping based on validation macro-F1.
- Automatic mixed precision when CUDA is available.
- Fixed random seed, reported as 42 in the study.
- Optional 8-view test-time augmentation for inference.

Use the training entry point exposed by the repository, for example:

```bash
python train.py \
  --data-dir data/sipakmed_wsi \
  --model bfm_net \
  --image-size 224 \
  --epochs 30 \
  --batch-size 32 \
  --seed 42
```

If the implementation uses a different entry point or argument names, consult the available scripts and configuration files.

## Evaluation

Evaluate a trained checkpoint using the repository’s evaluation script:

```bash
python evaluate.py \
  --data-dir data/sipakmed_wsi \
  --checkpoint checkpoints/bfm_net_best.pth \
  --split test
```

Recommended outputs include:

- Accuracy.
- Macro-averaged precision, recall, and F1-score.
- Per-class precision, recall, and F1-score.
- Confusion matrix.
- One-vs-rest ROC-AUC where appropriate.
- Parameter count and inference configuration.
- Fold-wise metrics for cross-validation.

The manuscript describes three complementary evaluation settings: an 80/20 stratified held-out test split, 5-fold cross-validation for development and validation monitoring, and 10-fold stratified cross-validation for statistical comparison across model families.

## Explainability

The project evaluates model explanations using methods such as:

- **Grad-CAM:** Class-discriminative activation maps for identifying influential image regions.
- **SHAP:** Feature-attribution analysis using superpixel perturbations.
- **LIME:** Local surrogate explanations and explanation stability analysis.
- **Deletion/insertion:** Faithfulness-oriented attribution curves.
- **ROAR:** Performance changes after removing highly attributed regions.

Example command:

```bash
python explain.py \
  --checkpoint checkpoints/bfm_net_best.pth \
  --input path/to/image.bmp \
  --method gradcam
```

Explanations should be interpreted as model-behavior diagnostics rather than proof that the highlighted region is clinically causal.

## Reproducibility

For reproducible experiments:

- Use the same dataset modality, class mapping, and split protocol.
- Record the commit hash, configuration, random seed, PyTorch version, CUDA version, and GPU model.
- Save train/validation/test file lists when permitted.
- Do not tune hyperparameters on the held-out test set.
- Report both aggregate and per-fold results.
- Preserve the exact augmentation and test-time augmentation policies.
- Distinguish the final production configuration from component ablations.

The headline BFM-Net result combines the complete training and inference pipeline and should not be interpreted as the result of any single A1–A4 ablation row.

## Limitations

- SIPaKMeD is a benchmark dataset and may not represent the full variability of laboratories, scanners, staining protocols, populations, and clinical workflows.
- Held-out test performance does not establish clinical validity or prospective utility.
- Dataset-level or slide-level leakage can substantially inflate medical-image performance if splits are not carefully defined.
- Physics-informed terms in this project are computational priors and should not be interpreted as a complete biological model.
- XAI maps can be unstable and require quantitative and expert-based validation.
- External, multicenter, prospective validation is required before any clinical deployment.



## License

Add the repository’s intended license here. If no license has been selected, code and documentation should be treated as **all rights reserved** rather than assumed to be freely reusable. Dataset access and reuse remain subject to the original dataset license and terms.

