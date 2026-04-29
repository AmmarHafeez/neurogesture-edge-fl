# Current EMG Baseline Results

This document summarizes the current local EMG baseline pipeline for the UCI EMG Data for Gestures dataset. Generated metrics, figures, datasets, and model artifacts are intentionally kept out of Git and can be reproduced locally from the pipeline scripts.

## Dataset Parsing

The raw UCI EMG Data for Gestures files were parsed into a tabular sample dataset. Unmarked rows were dropped before downstream window generation.

- Total parsed rows after dropping unmarked rows: 1,512,750
- Subjects: 36
- Recordings: 72
- Files parsed: 72
- Files failed: 0

## Window Generation

Parsed samples were converted into fixed-length EMG windows grouped by subject, recording, and gesture.

- Window size: 200 samples
- Stride: 100 samples
- Total windows: 14,469
- Subjects represented: 36
- Recordings represented: 72
- Min windows per subject: 313
- Max windows per subject: 609
- Mean windows per subject: 401.92

## Window Label Distribution

| Gesture | Windows |
| --- | ---: |
| rest | 2,391 |
| fist | 2,322 |
| wrist_flexion | 2,387 |
| wrist_extension | 2,407 |
| radial_deviation | 2,411 |
| ulnar_deviation | 2,420 |
| extended_palm | 131 |

## Baseline Models

The current baseline evaluation uses classical machine-learning models:

- Logistic Regression
- Random Forest

The feature mode is flattened EMG windows. Each `200 x 8` EMG window becomes a `1,600`-feature vector for classical models.

## Evaluation Protocols

### Random Split

The random split uses a stratified train/test split.

- Test size: 20%
- Train windows: 11,575
- Test windows: 2,894

### Subject Split

The subject split holds out complete subjects so that no subject appears in both train and test sets.

- Train windows: 11,478
- Test windows: 2,991

## Baseline Results

### Random Split

- Best model: Random Forest
- Accuracy: 0.8082
- Macro F1: 0.6960
- Balanced accuracy: 0.6993

### Subject Split

- Best model: Random Forest
- Macro F1: 0.6838
- Balanced accuracy: 0.6843

## Deep-Learning Baseline Support

A lightweight PyTorch deep-learning baseline has been added for windowed EMG classification. The initial supported model is a compact 1D CNN that consumes windows as `(batch, channels, samples)` tensors and reports metrics using the same fixed label order as the classical baseline. A small TCN option is also available for later comparison.

No deep-learning performance values are listed here until the training command is run locally and the resulting metrics are reviewed.

## Data Note

The `extended_palm` class is highly underrepresented:

- 131 total windows
- 26 windows in the random test split
- 0 windows in the current subject-split test set

Accuracy alone is not sufficient for this dataset because of class imbalance. Macro F1 and balanced accuracy are more informative because they give more visibility into minority-class behavior and class-level recall.

## Current Limitations

- Baseline models use flattened windows and do not yet model temporal structure deeply.
- The `extended_palm` class is underrepresented.
- Subject-split results depend on which subjects are held out.
- Personalization, federated training, and edge deployment have not been added yet.

## Next Steps

- Train a lightweight deep model such as a 1D CNN or TCN.
- Add per-subject normalization experiments.
- Add personalization and calibration experiments.
- Add federated learning simulation.
- Add ONNX export and latency benchmarks.
