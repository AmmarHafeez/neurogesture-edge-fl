# EMG Pipeline Results

This document summarizes the completed local pipeline for the UCI EMG Data for Gestures dataset. Generated metrics, figures, datasets, checkpoints, and model artifacts are intentionally kept out of Git and can be reproduced locally with the commands below.

## Dataset Parsing Summary

- Parsed rows after dropping unmarked rows: 1,512,750
- Subjects: 36
- Recordings: 72
- Files parsed: 72
- Files failed: 0

## Window Generation Summary

- Window size: 200 samples
- Stride: 100 samples
- Total windows: 14,469
- Subjects represented: 36
- Recordings represented: 72

## Window Label Distribution

| Gesture | Windows |
|---|---:|
| rest | 2,391 |
| fist | 2,322 |
| wrist_flexion | 2,387 |
| wrist_extension | 2,407 |
| radial_deviation | 2,411 |
| ulnar_deviation | 2,420 |
| extended_palm | 131 |

## Model Comparison

| Model | Split | Normalization | Macro F1 | Balanced accuracy |
|---|---|---|---:|---:|
| Logistic Regression | Random split | flattened raw windows | 0.2648 | 0.2837 |
| Random Forest | Random split | flattened raw windows | 0.6960 | 0.6993 |
| Random Forest | Subject split | flattened raw windows | 0.6838 | 0.6843 |
| CNN-1D | Random split | global channel z-score | 0.8724 | 0.9088 |
| CNN-1D | Subject split | global channel z-score | 0.7377 | 0.7350 |
| CNN-1D | Random split | per-window channel z-score | 0.5041 | 0.5404 |
| CNN-1D | Subject split | per-window channel z-score | 0.4545 | 0.4520 |

## Key Observations

- CNN-1D with global channel z-score currently performs best.
- Global channel z-score substantially outperforms per-window z-score in the current CNN setup.
- Random split is useful for sanity checking, but subject split is more realistic for user generalization.
- The `extended_palm` class is highly underrepresented and should be handled carefully in later experiments.
- Accuracy alone is not sufficient because of class imbalance; macro F1 and balanced accuracy are more informative.

## Personalized Calibration

Personalized calibration adapts the CNN-1D subject-split checkpoint to held-out target subjects. The experiment uses `global_channel_zscore` normalization, samples up to 10 windows per class from each held-out subject, and evaluates on the remaining windows from the same subject. Calibration and evaluation windows do not overlap.

| Calibration mode | Subjects evaluated | Subjects skipped | Mean before macro F1 | Mean after macro F1 | Mean delta macro F1 | Mean before balanced accuracy | Mean after balanced accuracy | Mean delta balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Last layer | 8 | 0 | 0.7335 | 0.7704 | +0.0369 | 0.7326 | 0.7666 | +0.0340 |
| Full model | 8 | 0 | 0.7335 | 0.7764 | +0.0430 | 0.7326 | 0.7795 | +0.0469 |

Key observations:

- Personalized calibration improved mean macro F1 for both modes.
- Full-model fine-tuning produced the larger mean macro-F1 gain in the current run.
- Last-layer calibration is lighter because fewer parameters are updated.
- Full-model calibration can adapt more strongly but may require more care to avoid overfitting.
- `extended_palm` remains difficult because it has low support and is absent for some target-subject evaluation sets.

## Edge Deployment Benchmark

The trained CNN-1D checkpoint can be exported to ONNX FP32, quantized with ONNX Runtime dynamic INT8 quantization, and benchmarked on CPU. Generated ONNX models and benchmark JSON files are intentionally kept out of Git and can be reproduced locally.

| Model format | Size KB | Mean latency ms | Median latency ms | P95 latency ms | Throughput windows/sec | Runtime provider |
|---|---:|---:|---:|---:|---:|---|
| ONNX FP32 | 149.52 | 0.149 | 0.144 | 0.203 | 6706.50 | CPUExecutionProvider |
| ONNX INT8 | 45.32 | 2.087 | 1.842 | 3.362 | 479.16 | CPUExecutionProvider |

Key observations:

- The FP32 ONNX model is already very small at about 149.52 KB.
- Dynamic INT8 quantization reduced model size to about 45.32 KB.
- This is roughly a 70% model-size reduction.
- In this CPU benchmark, INT8 was slower than FP32.
- This can happen for small convolutional models because quantization overhead and CPU operator support can outweigh speed gains.
- Therefore, INT8 should be treated here as a size-optimization path, not automatically as a latency optimization.
- FP32 ONNX currently provides the best latency in this local benchmark.

## Federated Learning Simulation

Manual PyTorch FedAvg support has been added for subject-level federated simulation. Each subject is treated as one client, client training runs on that subject's local windows, and the server aggregates only model parameters with sample-weighted FedAvg. The simulation uses the deterministic subject split, computes `global_channel_zscore` normalization from federated training subjects only, and evaluates the global model on held-out subjects after each round.

No federated benchmark metrics are listed here until the simulation command is run locally.

## Reproducibility Commands

Parse the raw dataset:

```bash
python src/data/make_dataset.py --raw-dir data/raw/EMG_data_for_gestures-master --output data/processed/emg_samples.parquet --summary reports/metrics/dataset_summary.json
```

Generate fixed-length windows:

```bash
python src/data/make_windows.py --input data/processed/emg_samples.parquet --output data/processed/emg_windows.npz --summary reports/metrics/window_summary.json --window-size 200 --stride 100
```

Run classical baselines:

```bash
python src/training/train_baseline.py --windows data/processed/emg_windows.npz --results reports/metrics/baseline_results.json --figures-dir reports/figures --models-dir models
```

Run the normalized CNN baseline:

```bash
python src/training/train_deep.py --windows data/processed/emg_windows.npz --results reports/metrics/deep_results.json --models-dir models --model cnn1d --epochs 10 --batch-size 128 --normalization global_channel_zscore
```

Run personalized calibration after creating `models/cnn1d_subject_split_best.pt`:

```bash
python src/personalization/evaluate_calibration.py --windows data/processed/emg_windows.npz --base-model models/cnn1d_subject_split_best.pt --results reports/metrics/personalization_results.json --mode last_layer --calibration-per-class 10 --epochs 5 --batch-size 64
```

Run full-model personalized calibration:

```bash
python src/personalization/evaluate_calibration.py --windows data/processed/emg_windows.npz --base-model models/cnn1d_subject_split_best.pt --results reports/metrics/personalization_results_full_model.json --mode full_model --calibration-per-class 10 --epochs 5 --batch-size 64
```

Export the CNN checkpoint to ONNX FP32:

```bash
python src/edge/export_onnx.py --checkpoint models/cnn1d_subject_split_best.pt --output models/onnx/cnn1d_fp32.onnx --windows data/processed/emg_windows.npz
```

Quantize the ONNX model to INT8:

```bash
python src/edge/quantize_onnx.py --input models/onnx/cnn1d_fp32.onnx --output models/onnx/cnn1d_int8.onnx
```

Benchmark ONNX latency:

```bash
python src/edge/benchmark_latency.py --fp32-model models/onnx/cnn1d_fp32.onnx --int8-model models/onnx/cnn1d_int8.onnx --output reports/metrics/edge_benchmark.json --warmup 20 --runs 200
```

Run the FedAvg federated simulation:

```bash
python src/federated/simulate_fedavg.py --windows data/processed/emg_windows.npz --results reports/metrics/federated_results.json --rounds 5 --clients-per-round 8 --local-epochs 1 --batch-size 64
```

## Current Limitations

- Federated learning simulation is available, but local benchmark results have not been listed here yet.
- Subject-split results depend on which subjects are held out.
- The `extended_palm` class has low support.
- Current CNN results are a baseline, not final optimization.
