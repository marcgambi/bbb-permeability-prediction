"""Combine SchNet and Extra Trees predictions through a weighted ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import RESULTS_DIR

SCHNET_TAG = "schnet_selected9_nodeattr15_cutoff8_dropout10"
ALPHA_GRID = np.arange(0.0, 1.01, 0.05)


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse, np.sqrt(mse), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def select_alpha(y_true, schnet_pred, extratrees_pred):
    scores = []
    for alpha in ALPHA_GRID:
        pred = alpha * schnet_pred + (1.0 - alpha) * extratrees_pred
        scores.append((r2_score(y_true, pred), float(alpha)))
    best_r2, best_alpha = max(scores)
    return best_alpha, best_r2


def main() -> None:
    schnet_oof = pd.read_csv(RESULTS_DIR / f"{SCHNET_TAG}_schnet_multiseed_oof_predictions.csv")
    et_oof = pd.read_csv(RESULTS_DIR / "extratrees_sfs_oof_predictions.csv")
    merged = schnet_oof.merge(et_oof, on="original_idx", suffixes=("_schnet", "_et"))

    if len(merged) != len(schnet_oof) or len(merged) != len(et_oof):
        raise ValueError("Incomplete OOF merge.")
    if not np.allclose(merged.target_logBB_schnet, merged.target_logBB_et, atol=1e-4):
        raise ValueError("OOF targets are not aligned.")
    if not (merged.fold_schnet == merged.fold_et).all():
        raise ValueError("OOF folds are not aligned.")

    merged = merged.rename(columns={"target_logBB_schnet": "target_logBB", "fold_schnet": "fold"})
    y = merged.target_logBB.to_numpy()
    schnet_pred = merged.schnet_oof_pred_multiseed.to_numpy()
    et_pred = merged.extratrees_oof_pred.to_numpy()

    curve = []
    for alpha in ALPHA_GRID:
        pred = alpha * schnet_pred + (1.0 - alpha) * et_pred
        mse, rmse, mae, r2 = evaluate(y, pred)
        curve.append({"alpha_schnet": alpha, "alpha_extratrees": 1-alpha,
                      "oof_mse": mse, "oof_rmse": rmse, "oof_mae": mae, "oof_r2": r2})
    pd.DataFrame(curve).to_csv(RESULTS_DIR / "hybrid_alpha_curve_oof.csv", index=False)

    best_alpha, best_oof_r2 = select_alpha(y, schnet_pred, et_pred)
    nested_rows = []
    for fold in sorted(merged.fold.unique()):
        inner = merged[merged.fold != fold]
        outer = merged[merged.fold == fold]
        alpha, _ = select_alpha(inner.target_logBB.to_numpy(),
                                inner.schnet_oof_pred_multiseed.to_numpy(),
                                inner.extratrees_oof_pred.to_numpy())
        pred = alpha * outer.schnet_oof_pred_multiseed.to_numpy() + (1-alpha) * outer.extratrees_oof_pred.to_numpy()
        mse, rmse, mae, r2 = evaluate(outer.target_logBB.to_numpy(), pred)
        nested_rows.append({"fold": fold, "best_alpha_schnet": alpha,
                            "hybrid_mse": mse, "hybrid_rmse": rmse,
                            "hybrid_mae": mae, "hybrid_r2": r2})
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(RESULTS_DIR / "hybrid_nested_cv_results.csv", index=False)

    schnet_test = pd.read_csv(RESULTS_DIR / f"{SCHNET_TAG}_schnet_multiseed_ensemble_test_predictions.csv")
    et_test = pd.read_csv(RESULTS_DIR / "extratrees_sfs_ensemble_test_predictions.csv")
    y_test = schnet_test.target_logBB.to_numpy()
    if not np.allclose(y_test, et_test.target.to_numpy(), atol=1e-4):
        raise ValueError("Test targets are not aligned.")

    schnet_test_pred = schnet_test.schnet_multiseed_ensemble_pred.to_numpy()
    et_test_pred = et_test.extratrees_ensemble_5fold_pred.to_numpy()
    hybrid_pred = best_alpha * schnet_test_pred + (1-best_alpha) * et_test_pred
    mse, rmse, mae, r2 = evaluate(y_test, hybrid_pred)

    pd.DataFrame([{"alpha_schnet": best_alpha, "alpha_extratrees": 1-best_alpha,
                   "hybrid_test_mse": mse, "hybrid_test_rmse": rmse,
                   "hybrid_test_mae": mae, "hybrid_test_r2": r2,
                   "oof_alpha_selection_r2": best_oof_r2,
                   "cv_hybrid_mean_r2": nested.hybrid_r2.mean(),
                   "cv_hybrid_std_r2": nested.hybrid_r2.std()}]).to_csv(
        RESULTS_DIR / "hybrid_test_results.csv", index=False)
    pd.DataFrame({"target_logBB": y_test, "schnet_ensemble_pred": schnet_test_pred,
                  "extratrees_pred": et_test_pred, "hybrid_pred": hybrid_pred,
                  "residual": hybrid_pred-y_test}).to_csv(
        RESULTS_DIR / "hybrid_test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
