# neurogesture-edge-fl

## Overview

`neurogesture-edge-fl` is a local-first Python project for EMG gesture recognition in wearable-control scenarios. The repository is structured for iterative development of subject-aware evaluation, personalization, privacy-preserving federated learning simulation, and edge-ready inference.

The first version contains only the project skeleton, configuration files, starter modules, and smoke tests. Full machine learning pipelines are intentionally left for later iterations.

## Background

Surface electromyography (EMG) captures muscle activation patterns through sensor channels placed on the skin. Gesture recognition systems can use EMG windows to infer intended hand or arm actions for wearable interfaces, assistive control, and embedded interaction.

This project is organized around practical constraints that matter for EMG systems: variation across subjects, limited calibration data, privacy-sensitive signals, and deployment on constrained local devices.

## Dataset

The project uses the UCI EMG Data for Gestures dataset:

Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., & Lobov, S. (2018). EMG Data for Gestures. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZP5C

Place the raw dataset under:

```text
data/raw/EMG_data_for_gestures-master/
```

The raw dataset contains subject folders and text files. Each raw text file is expected to contain 10 columns: time, 8 EMG sensor channels, and class label.

## Window generation

Parsed EMG samples can be converted into fixed-length model windows after creating `data/processed/emg_samples.parquet`. Windows are grouped by subject, recording, and gesture, sorted by time, and stored as compressed arrays under `data/processed/emg_windows.npz`.

```bash
python src/data/make_windows.py --input data/processed/emg_samples.parquet --output data/processed/emg_windows.npz --summary reports/metrics/window_summary.json --window-size 200 --stride 100
```

## Planned architecture

- Data ingestion for parsing the UCI text files into consistent tabular data.
- Preprocessing for windowing, normalization, and starter feature extraction.
- Baseline models using classical machine learning methods.
- Deep learning models for 1D CNN and TCN experiments.
- Subject-aware evaluation splits and reports.
- Personalization workflows for adapting models to individual users.
- Federated learning simulation for privacy-preserving local training experiments.
- Edge inference utilities and a FastAPI service for local control integration.

## Project structure

```text
configs/       Configuration files for data, models, training, and API settings.
data/          Local raw, interim, and processed datasets.
demo/          Small scripts for local demonstrations.
docs/          Project notes and design documentation.
reports/       Generated figures, metrics, and model cards.
src/           Source package for data, preprocessing, models, training, evaluation, API, and utilities.
tests/         Pytest smoke tests for starter preprocessing utilities.
```

## Quickstart

Create and activate a Python 3.12 environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Start the local API:

```bash
uvicorn src.api.main:app --reload
```

With Docker:

```bash
docker compose up --build
```

## Roadmap

- Add robust parsing for all UCI subject files.
- Build reproducible processed datasets with metadata.
- Add subject-aware train, validation, and test split utilities.
- Implement baseline feature models with scikit-learn.
- Add PyTorch 1D CNN and TCN training loops.
- Add personalization experiments with limited subject calibration data.
- Add federated learning simulation across subject partitions.
- Add edge export and local inference benchmarks.

## Limitations

- The current repository is a skeleton and does not include complete training logic.
- Raw data is not included and must be placed locally.
- API predictions are placeholders until trained model loading is implemented.
- Edge deployment targets and performance budgets are not defined yet.

## Citation

If you use the dataset through this project, cite:

Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., & Lobov, S. (2018). EMG Data for Gestures. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZP5C
