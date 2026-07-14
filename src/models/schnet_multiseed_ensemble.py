"""Generate OOF and hold-out predictions from a multi-seed SchNet ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from src.config import DATASET_PATH, MODELS_DIR, RESULTS_DIR
from src.models.schnet_common import NODE_FEATURE_COLUMNS, SchNetEmbeddingGlobalDescRegressor, compute_global_descriptors, prepare_graphs

DATA_SEED = 42
MODEL_SEEDS = [42, 43, 44]
MODEL_TAG = "schnet_selected9_nodeattr15_cutoff8_dropout10"
FOLDS, BATCH_SIZE = 5, 32


def build_model(device, n_global):
    return SchNetEmbeddingGlobalDescRegressor(
        num_global_features=n_global,
        num_node_features=len(NODE_FEATURE_COLUMNS),
    ).to(device)


def predict(model, loader, device):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch, batch.global_desc, batch.node_attr)
            preds.extend(pred.cpu().numpy()); targets.extend(batch.y.view(-1).cpu().numpy())
    return np.asarray(preds), np.asarray(targets)


def main() -> None:
    dataset = torch.load(DATASET_PATH, weights_only=False)
    descriptors = np.asarray([compute_global_descriptors(d.smiles) for d in dataset], dtype=np.float32)
    y = np.asarray([d.y.item() for d in dataset], dtype=np.float32)
    strata = np.asarray(pd.qcut(y, q=10, labels=False, duplicates="drop"))
    indices = np.arange(len(dataset))
    trainval_idx, test_idx, trainval_strata, _ = train_test_split(
        indices, strata, test_size=0.15, random_state=DATA_SEED, stratify=strata)
    final_train_idx, _ = train_test_split(
        trainval_idx, test_size=0.15, random_state=DATA_SEED, stratify=trainval_strata)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=DATA_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oof_rows = []
    for fold, (train_rel, val_rel) in enumerate(splitter.split(trainval_idx, trainval_strata), 1):
        train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
        scaler = StandardScaler().fit(descriptors[train_idx])
        val_data = prepare_graphs([dataset[i] for i in val_idx], val_idx, descriptors, scaler)
        loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        seed_preds, targets_ref = {}, None
        for model_seed in MODEL_SEEDS:
            path = MODELS_DIR / f"best_{MODEL_TAG}_cv_fold{fold}_seed{model_seed}_checkpoint.pt"
            if not path.exists():
                continue
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = build_model(device, descriptors.shape[1]); model.load_state_dict(checkpoint["model_state_dict"])
            preds, targets = predict(model, loader, device)
            seed_preds[model_seed] = preds
            targets_ref = targets if targets_ref is None else targets_ref
        if not seed_preds:
            raise RuntimeError(f"No checkpoints found for fold {fold}.")
        avg = np.mean(list(seed_preds.values()), axis=0)
        for row, idx in enumerate(val_idx):
            record = {"original_idx": int(idx), "fold": fold,
                      "target_logBB": float(targets_ref[row]),
                      "schnet_oof_pred_multiseed": float(avg[row])}
            for seed, preds in seed_preds.items():
                record[f"schnet_oof_pred_seed{seed}"] = float(preds[row])
            oof_rows.append(record)
    pd.DataFrame(oof_rows).sort_values("original_idx").to_csv(
        RESULTS_DIR / f"{MODEL_TAG}_schnet_multiseed_oof_predictions.csv", index=False)

    scaler = StandardScaler().fit(descriptors[final_train_idx])
    test_data = prepare_graphs([dataset[i] for i in test_idx], test_idx, descriptors, scaler)
    loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, labels, targets_ref = [], [], None
    for model_seed in MODEL_SEEDS:
        for fold in range(1, FOLDS+1):
            path = MODELS_DIR / f"best_{MODEL_TAG}_cv_fold{fold}_seed{model_seed}_checkpoint.pt"
            if not path.exists():
                continue
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = build_model(device, descriptors.shape[1]); model.load_state_dict(checkpoint["model_state_dict"])
            preds, targets = predict(model, loader, device)
            all_preds.append(preds); labels.append(f"seed{model_seed}_fold{fold}")
            targets_ref = targets if targets_ref is None else targets_ref
    pred_array = np.asarray(all_preds); ensemble = pred_array.mean(axis=0)
    print(f"Test R2: {r2_score(targets_ref, ensemble):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(targets_ref, ensemble)):.4f}")
    print(f"Test MAE: {mean_absolute_error(targets_ref, ensemble):.4f}")
    output = pd.DataFrame({"target_logBB": targets_ref, "schnet_multiseed_ensemble_pred": ensemble})
    for label, preds in zip(labels, pred_array):
        output[f"pred_{label}"] = preds
    output.to_csv(RESULTS_DIR / f"{MODEL_TAG}_schnet_multiseed_ensemble_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
