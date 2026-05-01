# Reproducibility

This document lists the command sequence used to reproduce the local pipeline. Generated data, reports, figures, checkpoints, and ONNX files are intentionally ignored by Git.

## Environment

Create and activate a Python 3.12 environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Tests

```bash
pytest
```

## Dataset Parsing

Place the raw UCI EMG Data for Gestures dataset under `data/raw/EMG_data_for_gestures-master/`, then run:

```bash
python src/data/make_dataset.py --raw-dir data/raw/EMG_data_for_gestures-master --output data/processed/emg_samples.parquet --summary reports/metrics/dataset_summary.json
```

## Window Generation

```bash
python src/data/make_windows.py --input data/processed/emg_samples.parquet --output data/processed/emg_windows.npz --summary reports/metrics/window_summary.json --window-size 200 --stride 100
```

## Classical Baseline

```bash
python src/training/train_baseline.py --windows data/processed/emg_windows.npz --results reports/metrics/baseline_results.json --figures-dir reports/figures --models-dir models
```

## CNN Training

```bash
python src/training/train_deep.py --windows data/processed/emg_windows.npz --results reports/metrics/deep_results.json --models-dir models --model cnn1d --epochs 10 --batch-size 128 --normalization global_channel_zscore
```

This command writes the subject-split checkpoint required by personalization and ONNX export.

## Personalization

Last-layer calibration:

```bash
python src/personalization/evaluate_calibration.py --windows data/processed/emg_windows.npz --base-model models/cnn1d_subject_split_best.pt --results reports/metrics/personalization_results.json --mode last_layer --calibration-per-class 10 --epochs 5 --batch-size 64
```

Full-model calibration:

```bash
python src/personalization/evaluate_calibration.py --windows data/processed/emg_windows.npz --base-model models/cnn1d_subject_split_best.pt --results reports/metrics/personalization_results_full_model.json --mode full_model --calibration-per-class 10 --epochs 5 --batch-size 64
```

## ONNX Export

```bash
python src/edge/export_onnx.py --checkpoint models/cnn1d_subject_split_best.pt --output models/onnx/cnn1d_fp32.onnx --windows data/processed/emg_windows.npz
```

## ONNX Quantization

```bash
python src/edge/quantize_onnx.py --input models/onnx/cnn1d_fp32.onnx --output models/onnx/cnn1d_int8.onnx
```

## Edge Benchmark

```bash
python src/edge/benchmark_latency.py --fp32-model models/onnx/cnn1d_fp32.onnx --int8-model models/onnx/cnn1d_int8.onnx --output reports/metrics/edge_benchmark.json --warmup 20 --runs 200
```

## FedAvg

```bash
python src/federated/simulate_fedavg.py --windows data/processed/emg_windows.npz --results reports/metrics/federated_results.json --rounds 5 --clients-per-round 8 --local-epochs 1 --batch-size 64
```

## FedProx

```bash
python src/federated/simulate_fedprox.py --windows data/processed/emg_windows.npz --results reports/metrics/fedprox_results.json --rounds 5 --clients-per-round 8 --local-epochs 1 --batch-size 64 --mu 0.01
```

## Troubleshooting

- Install dependencies with `pip install -r requirements.txt`.
- The raw dataset must exist under `data/raw/EMG_data_for_gestures-master/` before dataset parsing.
- `data/processed/emg_samples.parquet` must exist before window generation.
- `data/processed/emg_windows.npz` must exist before training, personalization, federated simulation, or ONNX export.
- `src/training/train_deep.py` must be run before ONNX export and personalization because `models/cnn1d_subject_split_best.pt` is required.
- Generated data, reports, figures, checkpoints, and ONNX files are not committed and should be regenerated locally when needed.
