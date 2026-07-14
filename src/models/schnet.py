"""Train SchNet models across multiple seeds and stratified folds."""

from __future__ import annotations

import copy
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from src.config import DATASET_PATH, MODELS_DIR, RESULTS_DIR
from src.models.schnet_common import NODE_FEATURE_COLUMNS, SchNetEmbeddingGlobalDescRegressor, compute_global_descriptors, prepare_graphs

DATA_SEED = 42
MODEL_SEEDS = [42, 43, 44]
MODEL_TAG = "schnet_selected9_nodeattr15_cutoff8_dropout10"
EPOCHS, PATIENCE, BATCH_SIZE, FOLDS = 3000, 30, 32, 5
MIN_DELTA, LEARNING_RATE, WEIGHT_DECAY = 1e-4, 1e-3, 1e-4


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed); random.seed(worker_seed)


def build_model(device, n_global):
    return SchNetEmbeddingGlobalDescRegressor(
        num_global_features=n_global,
        num_node_features=len(NODE_FEATURE_COLUMNS),
    ).to(device)


def train_epoch(model, loader, optimizer, device):
    model.train(); total = 0.0
    for batch in loader:
        batch = batch.to(device); optimizer.zero_grad()
        pred = model(batch.z, batch.pos, batch.batch, batch.global_desc, batch.node_attr)
        loss = F.mse_loss(pred, batch.y.view(-1)); loss.backward(); optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch, batch.global_desc, batch.node_attr)
            preds.extend(pred.cpu().numpy()); targets.extend(batch.y.view(-1).cpu().numpy())
    preds, targets = np.asarray(preds), np.asarray(targets)
    mse = mean_squared_error(targets, preds)
    return mse, np.sqrt(mse), mean_absolute_error(targets, preds), r2_score(targets, preds), preds, targets


def fit(model, train_loader, val_loader, device, checkpoint_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_rmse, best_epoch, stale = float("inf"), 0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mse, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mse": val_mse,
                        "val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2})
        if val_rmse < best_rmse - MIN_DELTA:
            best_rmse, best_epoch, stale = val_rmse, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()),
                        "optimizer_state_dict": optimizer.state_dict(), "best_val_rmse": best_rmse}, checkpoint_path)
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, pd.DataFrame(history), best_epoch


def main() -> None:
    dataset = torch.load(DATASET_PATH, weights_only=False)
    descriptors = np.asarray([compute_global_descriptors(d.smiles) for d in dataset], dtype=np.float32)
    y = np.asarray([d.y.item() for d in dataset], dtype=np.float32)
    strata = np.asarray(pd.qcut(y, q=10, labels=False, duplicates="drop"))
    indices = np.arange(len(dataset))
    trainval_idx, test_idx, trainval_strata, _ = train_test_split(
        indices, strata, test_size=0.15, random_state=DATA_SEED, stratify=strata)
    final_train_idx, final_val_idx = train_test_split(
        trainval_idx, test_size=0.15, random_state=DATA_SEED, stratify=trainval_strata)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=DATA_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_seed in MODEL_SEEDS:
        seed_everything(model_seed); cv_rows = []
        for fold, (train_rel, val_rel) in enumerate(splitter.split(trainval_idx, trainval_strata), 1):
            train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
            scaler = StandardScaler().fit(descriptors[train_idx])
            train_data = prepare_graphs([dataset[i] for i in train_idx], train_idx, descriptors, scaler)
            val_data = prepare_graphs([dataset[i] for i in val_idx], val_idx, descriptors, scaler)
            generator = torch.Generator().manual_seed(model_seed + fold)
            train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                      generator=generator, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
            model = build_model(device, descriptors.shape[1])
            checkpoint = MODELS_DIR / f"best_{MODEL_TAG}_cv_fold{fold}_seed{model_seed}_checkpoint.pt"
            model, history, best_epoch = fit(model, train_loader, val_loader, device, checkpoint)
            val_mse, val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, device)
            history.to_csv(RESULTS_DIR / f"{MODEL_TAG}_cv_fold{fold}_seed{model_seed}_history.csv", index=False)
            cv_rows.append({"fold": fold, "best_epoch": best_epoch, "val_mse": val_mse,
                            "val_rmse": val_rmse, "val_mae": val_mae, "val_r2": val_r2})
        pd.DataFrame(cv_rows).to_csv(RESULTS_DIR / f"{MODEL_TAG}_5fold_cv_seed{model_seed}_results.csv", index=False)

        scaler = StandardScaler().fit(descriptors[final_train_idx])
        train_data = prepare_graphs([dataset[i] for i in final_train_idx], final_train_idx, descriptors, scaler)
        val_data = prepare_graphs([dataset[i] for i in final_val_idx], final_val_idx, descriptors, scaler)
        test_data = prepare_graphs([dataset[i] for i in test_idx], test_idx, descriptors, scaler)
        generator = torch.Generator().manual_seed(model_seed)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                  generator=generator, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        model = build_model(device, descriptors.shape[1])
        checkpoint = MODELS_DIR / f"best_{MODEL_TAG}_holdout_seed{model_seed}_checkpoint.pt"
        model, history, best_epoch = fit(model, train_loader, val_loader, device, checkpoint)
        test_mse, test_rmse, test_mae, test_r2, test_pred, test_target = evaluate(model, test_loader, device)
        pd.DataFrame([{"model_seed": model_seed, "best_epoch": best_epoch,
                       "test_mse": test_mse, "test_rmse": test_rmse,
                       "test_mae": test_mae, "test_r2": test_r2}]).to_csv(
            RESULTS_DIR / f"{MODEL_TAG}_seed{model_seed}_test_results.csv", index=False)
        pd.DataFrame({"target_logBB": test_target, "predicted_logBB": test_pred,
                      "residual": test_target-test_pred}).to_csv(
            RESULTS_DIR / f"{MODEL_TAG}_seed{model_seed}_predictions.csv", index=False)
        torch.save(model.state_dict(), MODELS_DIR / f"{MODEL_TAG}_seed{model_seed}_model.pt")


if __name__ == "__main__":
    main()
