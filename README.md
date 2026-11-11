# DSFF for UAV Propeller Fault Diagnosis

Training and inference code for the proposed Dual-Scale Feature Fusion Network (DSFF).

## Install

Use Python 3.9 or 3.10 in a virtual environment:

```bash
python -m pip install -r requirements.txt
```

For CUDA 12.1, install `torch==2.5.1` from `https://download.pytorch.org/whl/cu121` before installing the remaining requirements. CPU is also supported.

## Download and verify data

The measured acceleration data are provided by [AeroLabPUT/UAV_measurement_data](https://github.com/AeroLabPUT/UAV_measurement_data), under `3DR_Solo/Range_data`. Cite the [acquisition paper](https://doi.org/10.1007/s10846-024-02101-7) and follow the upstream data terms.

```bash
python download_data.py --data-dir data/Range_data
python data_utils.py --data-dir data/Range_data
```

If downloading fails, obtain the nine CSVs from the upstream repository manually and pass their directory to `--data-dir`. Exact filenames, SHA-256 hashes and label order are stored in `data_manifest.json`. Changed or incomplete recordings fail verification.

## Train

```bash
python train.py --data-dir data/Range_data --output-dir runs/dsff_seed42 --seed 42 --device auto
```


## Infer

```bash
python infer.py --run-dir runs/dsff_seed42 --csv data/Range_data/Solo_ACCEL_16g_GYRO_2kdps_range_2000.csv --start-sample 96
```

