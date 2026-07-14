"""Build a five-fold Extra Trees ensemble from selected descriptors."""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import DATASET_PATH, MODELS_DIR, RESULTS_DIR, SEED

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount",
    "NHOHCount", "NOCount", "NumAromaticRings", "NumAliphaticRings",
]


def compute_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)
    funcs = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
             Descriptors.NumHDonors, Descriptors.NumHAcceptors,
             Descriptors.NumRotatableBonds, Descriptors.RingCount,
             Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
             Descriptors.NHOHCount, Descriptors.NOCount,
             Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings]
    return np.asarray([f(mol) for f in funcs], dtype=np.float32)


def main() -> None:
    dataset = torch.load(DATASET_PATH, weights_only=False)
    x = np.asarray([compute_descriptors(d.smiles) for d in dataset], dtype=np.float32)
    y = np.asarray([d.y.item() for d in dataset], dtype=np.float32)
    strata = np.asarray(pd.qcut(y, q=10, labels=False, duplicates="drop"))
    indices = np.arange(len(dataset))
    trainval_idx, test_idx, trainval_strata, _ = train_test_split(
        indices, strata, test_size=0.15, random_state=SEED, stratify=strata)

    selected = pd.read_csv(RESULTS_DIR / "extratrees_sfs_selected_features.csv")["selected_feature"].tolist()
    selected_idx = [DESCRIPTOR_NAMES.index(name) for name in selected]
    with (MODELS_DIR / "extratrees_sfs_model.pkl").open("rb") as fh:
        best_pipeline = pickle.load(fh)
    model_params = best_pipeline.named_steps["model"].get_params()

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_test_preds, oof_rows = [], []
    for fold, (train_rel, val_rel) in enumerate(splitter.split(trainval_idx, trainval_strata), 1):
        train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(**model_params)),
        ])
        pipe.fit(x[train_idx][:, selected_idx], y[train_idx])
        val_pred = pipe.predict(x[val_idx][:, selected_idx])
        fold_test_preds.append(pipe.predict(x[test_idx][:, selected_idx]))
        for idx, pred, target in zip(val_idx, val_pred, y[val_idx]):
            oof_rows.append({"original_idx": int(idx), "fold": fold,
                             "target_logBB": float(target),
                             "extratrees_oof_pred": float(pred)})
        with (MODELS_DIR / f"extratrees_sfs_fold{fold}_model.pkl").open("wb") as fh:
            pickle.dump(pipe, fh)

    oof = pd.DataFrame(oof_rows).sort_values("original_idx")
    oof.to_csv(RESULTS_DIR / "extratrees_sfs_oof_predictions.csv", index=False)
    ensemble = np.asarray(fold_test_preds).mean(axis=0)
    final_pred = best_pipeline.predict(x[test_idx][:, selected_idx])
    print(f"OOF R2: {r2_score(oof.target_logBB, oof.extratrees_oof_pred):.4f}")
    print(f"Test R2: {r2_score(y[test_idx], ensemble):.4f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y[test_idx], ensemble)):.4f}")
    print(f"Test MAE: {mean_absolute_error(y[test_idx], ensemble):.4f}")
    pd.DataFrame({"target": y[test_idx], "extratrees_final_pred": final_pred,
                  "extratrees_ensemble_5fold_pred": ensemble}).to_csv(
        RESULTS_DIR / "extratrees_sfs_ensemble_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
