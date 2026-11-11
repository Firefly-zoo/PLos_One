import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CHANNELS = [f'{rotor}_a{axis}' for rotor in 'ABCD' for axis in 'XYZ']


def metadata():
    return json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_source(directory):
    for record in metadata()['recordings']:
        path = Path(directory) / record['file']
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Run download_data.py or obtain the upstream CSVs.')
        if sha256(path) != record['sha256']:
            raise ValueError(f'SHA-256 mismatch: {path.name}')


def dwt_features(windows):
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (12, 16):
        raise ValueError('Expected measured acceleration in g with shape (batch, 12, 16)')
    if not np.isfinite(values).all():
        raise ValueError('Acceleration must contain only finite values')
    approximation, detail = pywt.dwt(values, 'db4', mode='symmetric', axis=-1)
    return np.concatenate([approximation, detail], axis=-1).reshape(len(values), 264).astype(np.float32)


def load_dataset(directory):
    validate_source(directory)
    features, targets, records, starts = [], [], [], []
    for label, record in enumerate(metadata()['recordings']):
        frame = pd.read_csv(Path(directory) / record['file'], usecols=CHANNELS)
        if len(frame) != record['rows']:
            raise ValueError(f'Unexpected row count: {record["file"]}')
        raw = frame[CHANNELS].to_numpy(dtype=np.float64)
        count = len(raw) // 16
        windows = raw[:count * 16].reshape(count, 16, 12).transpose(0, 2, 1)
        features.append(dwt_features(windows))
        targets.extend([label] * count)
        records.extend([record['file']] * count)
        starts.extend(range(0, count * 16, 16))
    manifest = pd.DataFrame({'record': records, 'start_sample': starts, 'label': targets})
    return np.concatenate(features), np.asarray(targets, dtype=np.int64), manifest


def split_indices(targets):
    remaining, test = train_test_split(np.arange(len(targets)), test_size=0.15,
                                       stratify=targets, random_state=42)
    train, validation = train_test_split(remaining, test_size=0.15 / 0.85,
                                         stratify=targets[remaining], random_state=42)
    return train, validation, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify the nine original recordings against SHA-256 hashes')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/Range_data')
    args = parser.parse_args()
    validate_source(args.data_dir)
    print('All nine source files verified.')
