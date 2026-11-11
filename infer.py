import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from data_utils import CHANNELS, dwt_features, metadata
from model import DSFF


def main():
    parser = argparse.ArgumentParser(description='Predict one 16-sample acceleration window using a trained DSFF run')
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True, help='Training output containing checkpoint.pth and scaler.npz')
    parser.add_argument('--start-sample', type=int, default=0)
    args = parser.parse_args()
    if args.start_sample < 0:
        parser.error('--start-sample must be nonnegative')
    frame = pd.read_csv(args.csv, usecols=CHANNELS)
    window = frame[CHANNELS].iloc[args.start_sample:args.start_sample + 16].to_numpy(dtype=np.float64)
    if window.shape != (16, 12):
        parser.error('Selected interval must contain sixteen complete acceleration samples')
    features = dwt_features(window.T[None, ...])
    with np.load(args.run_dir / 'scaler.npz') as saved:
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_ = saved['mean'], saved['scale']
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = 264
        values = scaler.transform(features).astype(np.float32)
    torch.set_num_threads(4)
    model = DSFF().eval()
    model.load_state_dict(torch.load(args.run_dir / 'checkpoint.pth', map_location='cpu', weights_only=True))
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(values)), dim=-1).numpy()[0]
    names = metadata()['class_names']
    print(json.dumps({'predicted_class': names[int(probabilities.argmax())],
                      'probabilities': dict(zip(names, probabilities.astype(float).tolist()))}, indent=2))


if __name__ == '__main__':
    main()
