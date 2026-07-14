"""Select RDKit descriptors and tune an Extra Trees regressor."""

from __future__ import annotations

import os
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import torch
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import DATASET_PATH, MODELS_DIR, RESULTS_DIR, SEED

warnings.filterwarnings("ignore")

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount",
    "NHOHCount", "NOCount", "NumAromaticRings", "NumAliphaticRings",
]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(DESCRIPTOR_NAMES), dtype=np.float32)
    return np.asarray([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol),
        Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
        Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol),
    ], dtype=np.float32)


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse, np.sqrt(mse), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def main() -> None:
    set_seed(SEED)
    dataset = torch.load(DATASET_PATH, weights_only=False)
    x = np.asarray([compute_descriptors(d.smiles) for d in dataset], dtype=np.float32)
    y = np.asarray([d.y.item() for d in dataset], dtype=np.float32)
    strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
    indices = np.arange(len(dataset))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=SEED, stratify=strata
    )
    x_trainval, y_trainval = x[trainval_idx], y[trainval_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    selector = SFS(
        estimator=ExtraTreesRegressor(n_estimators=150, random_state=SEED, n_jobs=-1),
        k_features=(4, 10), forward=True, floating=False, scoring="r2",
        cv=3, n_jobs=-1, verbose=2,
    )
    selector.fit(x_trainval, y_trainval)
    selected_idx = list(selector.k_feature_idx_)
    selected_features = [DESCRIPTOR_NAMES[i] for i in selected_idx]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ExtraTreesRegressor(random_state=SEED, n_jobs=-1)),
    ])
    search = GridSearchCV(
        pipeline,
        {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 8, 12, 20],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7, 1.0],
        },
        scoring="r2", cv=5, n_jobs=-1, verbose=2,
    )
    search.fit(x_trainval[:, selected_idx], y_trainval)
    test_pred = search.best_estimator_.predict(x_test[:, selected_idx])
    mse, rmse, mae, r2 = metrics(y_test, test_pred)

    pd.DataFrame([{
        "model": "ExtraTrees_SFS", "selected_features": str(selected_features),
        "best_cv_r2": search.best_score_, "test_mse": mse, "test_rmse": rmse,
        "test_mae": mae, "test_r2": r2, "best_params": str(search.best_params_),
    }]).to_csv(RESULTS_DIR / "extratrees_sfs_results.csv", index=False)
    pd.DataFrame({"target": y_test, "prediction": test_pred,
                  "residual": test_pred - y_test}).to_csv(
        RESULTS_DIR / "extratrees_sfs_test_predictions.csv", index=False)
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        RESULTS_DIR / "extratrees_sfs_selected_features.csv", index=False)
    with (MODELS_DIR / "extratrees_sfs_model.pkl").open("wb") as fh:
        pickle.dump(search.best_estimator_, fh)


if __name__ == "__main__":
    main()
