import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import sklearn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, label_binarize
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import ROOT, load_dataset, metadata, sha256, split_indices
from model import DSFF


SETTINGS = {
    'input_features': 264, 'channels': 12, 'window_samples': 16, 'stride_samples': 16,
    'wavelet': 'db4', 'wavelet_mode': 'symmetric', 'split_seed': 42,
    'batch_size': 256, 'learning_rate': 5e-4, 'weight_decay': 1e-4,
    'cosine_min_lr': 1e-6, 'label_smoothing': 0.05,
    'gradient_clip': 1.0, 'patience': 20,
}


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=SETTINGS['label_smoothing'])
    loss_sum, count, correct, scores = 0.0, 0, 0, []
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss_sum += criterion(logits, targets).item() * len(targets)
            count += len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            scores.append(torch.softmax(logits, dim=1).cpu().numpy())
    return loss_sum / count, correct / count, np.concatenate(scores)


def fit(features, targets, members, output_dir, seed, epochs, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory must be empty; choose a new --output-dir for each run')
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(features[members[0]])
    generator = torch.Generator().manual_seed(seed)
    loaders = []
    for subset_index, indices in enumerate(members):
        values = scaler.transform(features[indices]).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(values), torch.from_numpy(targets[indices]))
        loaders.append(DataLoader(dataset, batch_size=SETTINGS['batch_size'],
                                  shuffle=subset_index == 0,
                                  generator=generator if subset_index == 0 else None,
                                  num_workers=0, pin_memory=device.type == 'cuda'))
    config = dict(SETTINGS, seed=seed, epochs_max=epochs, device=str(device),
                  class_names=metadata()['class_names'],
                  split_counts=dict(zip(['train', 'validation', 'test'], map(len, members))),
                  model_sha256=sha256(ROOT / 'model.py'), runner_sha256=sha256(Path(__file__)),
                  preprocessing_sha256=sha256(ROOT / 'data_utils.py'),
                  data_manifest_sha256=sha256(ROOT / 'data_manifest.json'),
                  runtime={'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                           'pandas': pd.__version__, 'pywavelets': pywt.__version__,
                           'sklearn': sklearn.__version__})
    write_json(output_dir / 'config.json', config)
    np.savez_compressed(output_dir / 'scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    np.savez_compressed(output_dir / 'split_indices.npz', train=members[0], validation=members[1], test=members[2])
    model = DSFF().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['learning_rate'],
                                  weight_decay=SETTINGS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                           eta_min=SETTINGS['cosine_min_lr'])
    criterion = nn.CrossEntropyLoss(label_smoothing=SETTINGS['label_smoothing'])
    best_accuracy, best_loss, best_epoch, bad_epochs = -1.0, float('inf'), 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for batch, labels in loaders[0]:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip'])
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
        validation_loss, validation_accuracy, _ = evaluate(model, loaders[1], device)
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': loss_sum / count, 'train_accuracy': correct / count,
                        'validation_loss': validation_loss, 'validation_accuracy': validation_accuracy})
        improved = validation_accuracy > best_accuracy + 1e-12
        if improved or (abs(validation_accuracy - best_accuracy) < 1e-12 and validation_loss < best_loss):
            best_accuracy, best_loss, best_epoch = validation_accuracy, validation_loss, epoch
            best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}
        bad_epochs = 0 if improved else bad_epochs + 1
        print(f'Epoch {epoch}: train_acc={correct / count:.6f} val_acc={validation_accuracy:.6f}', flush=True)
        if bad_epochs >= SETTINGS['patience']:
            break
    model.load_state_dict(best_state)
    _, _, scores = evaluate(model, loaders[2], device)
    test_labels = targets[members[2]]
    predictions = scores.argmax(axis=1)
    result = {'accuracy': float(accuracy_score(test_labels, predictions)),
              'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)),
              'ap_macro': float(average_precision_score(label_binarize(test_labels, classes=np.arange(9)),
                                                       scores, average='macro')),
              'confusion_matrix': confusion_matrix(test_labels, predictions, labels=np.arange(9)).tolist(),
              'best_epoch': best_epoch, 'epochs_completed': len(history),
              'validation_accuracy_selected': best_accuracy, 'test_samples': len(test_labels)}
    torch.save(best_state, output_dir / 'checkpoint.pth')
    np.savez_compressed(output_dir / 'predictions.npz', labels=test_labels, scores=scores, sample_indices=members[2])
    write_json(output_dir / 'history.json', history)
    write_json(output_dir / 'result.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description='Train DSFF with the documented 264-input protocol')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/Range_data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/dsff_seed42')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0, ...')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('--output-dir must be empty')
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch build')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    features, targets, manifest = load_dataset(args.data_dir)
    members = split_indices(targets)
    fit(features, targets, members, args.output_dir, args.seed, args.epochs, device)
    manifest['subset'] = ''
    for name, indices in zip(['train', 'validation', 'test'], members):
        manifest.loc[indices, 'subset'] = name
    manifest.to_csv(args.output_dir / 'sample_manifest.csv', index_label='sample_index')


if __name__ == '__main__':
    main()
