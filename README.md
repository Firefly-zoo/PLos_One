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

The implementation follows the manuscript's documented **264-input protocol**: twelve acceleration channels at 500 Hz; non-overlapping 16-sample windows; single-level db4 DWT with symmetric extension; 11 approximation and 11 detail coefficients per channel. Features are reshaped to 12 × 22. A fixed stratified split (seed 42) gives 33,868 training, 7,258 validation and 7,258 test windows. Standardization is fitted on training data only.

Training uses AdamW (lr 5e-4, weight decay 1e-4), cosine decay to 1e-6, batch size 256, label smoothing .05, gradient clipping 1.0 and at most 100 epochs. Early stopping counts 20 epochs without strictly higher validation accuracy. Validation-accuracy ties use lower loss for checkpoint selection but do not reset patience. Test evaluation uses the selected checkpoint. The architecture uses dropout .15, high-frequency channel dropout .20 and stochastic depth .10.

The output directory must be empty. Each run saves `checkpoint.pth`, its paired `scaler.npz`, settings, split indices/manifest, training history and test predictions/metrics. Use `--device cpu` for CPU or `--device cuda:0` for a specific GPU. For training seeds 43–46, invoke the same command with another `--seed` and a new output directory. Changing `--epochs` changes the cosine schedule and is a custom setting. Hardware/library differences can change trained results.

## Infer

```bash
python infer.py --run-dir runs/dsff_seed42 --csv data/Range_data/Solo_ACCEL_16g_GYRO_2kdps_range_2000.csv --start-sample 96
```

Inference reads sixteen samples beginning at the zero-based `--start-sample`, computes the same DWT and standardization, and outputs the predicted class and nine probabilities. CSV acceleration must be in g, with columns `A_aX,A_aY,A_aZ` through `D_aX,D_aY,D_aZ`. Use weights and scaler from the same training run.

## Scope

This source release contains only the proposed method and the data download/verification utilities. It follows the manuscript's 264-input experiment; the original reported 99.45% evaluation used a different feature set. Window splits within these recordings do not measure independent-flight generalization.

---

中文：本包仅提供我们方法的模型、训练、推理、原始数据下载与 SHA-256 校验。先下载并校验数据，再运行 `train.py`，随后将生成的运行目录传给 `infer.py`。默认参数对应论文描述的 264 维实验。训练权重和输出由使用者自行生成。
