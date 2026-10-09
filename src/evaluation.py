"""
evaluation.py

Functions for the evaluation of the models: engine-based cross-validation
folds, the NASA score and the cross-validation of an estimator with the
preprocessing refitted inside each fold.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

from config import CANDIDATE_FEATURES, N_SPLITS, RUL, SEED, UNIT
from preprocessing import (
    add_deviation,
    apply_cond_norm,
    fit_cond_stats,
    select_non_constant,
)

# A fold is a pair (train_units, val_units) of arrays of engine IDs
Fold = Tuple[np.ndarray, np.ndarray]


def make_folds(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> List[Fold]:
    """Generate the cross-validation folds on the engine IDs.

    The split is done on the engines and not on the rows: rows of the same
    engine are strongly correlated, so they must never be split between
    train and validation.

    Args:
        df: dataframe with the UNIT column.
        n_splits: number of folds. Default N_SPLITS.
        seed: random seed of the shuffling. Default SEED.

    Returns:
        A list of (train_units, val_units), one pair for each fold. It
        depends only on the engine IDs.
    """
    units = np.sort(df[UNIT].unique())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(units[tr], units[va]) for tr, va in kf.split(units)]


def get_fold(
    df: pd.DataFrame,
    train_units: Sequence[int],
    val_units: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fold to any dataframe (with any set of features).

    Args:
        df: dataframe with the UNIT column.
        train_units: engine IDs of the training portion.
        val_units: engine IDs of the validation portion.

    Returns:
        A tuple (train_df, val_df) with the rows of the given engines.
    """
    return df[df[UNIT].isin(train_units)], df[df[UNIT].isin(val_units)]


def nasa_score(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    reduce: str = "sum",
) -> float:
    """Compute the asymmetric NASA scoring function.

    With d = y_pred - y_true, the penalty of each prediction is
    exp(-d / 13) - 1 if d < 0 and exp(d / 10) - 1 otherwise, so that
    overestimating the RUL is penalised more than underestimating it.
    Lower is better.

    Args:
        y_true: true RUL values.
        y_pred: predicted RUL values.
        reduce: how to aggregate the penalties, "sum" or "mean".
            Default "sum".

    Returns:
        The aggregated score.

    Raises:
        ValueError: if reduce is neither "sum" nor "mean".
    """
    if reduce not in ("sum", "mean"):
        raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}.")

    d = np.asarray(y_pred) - np.asarray(y_true)
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum() if reduce == "sum" else s.mean())


def cv_evaluate(
    estimator: BaseEstimator,
    df: pd.DataFrame,
    folds: Sequence[Fold],
    candidate_features: Sequence[str] = CANDIDATE_FEATURES,
) -> pd.DataFrame:
    """Cross-validate an estimator on the engine-based folds.

    In each fold the whole preprocessing (per-condition normalisation,
    deviation features, removal of the constant features) is fitted on the
    training portion only and then applied to the validation portion.

    Args:
        estimator: unfitted scikit-learn regressor. It is cloned in each
            fold.
        df: training dataframe with the UNIT, RUL and COND columns.
        folds: list of (train_units, val_units), from make_folds.
        candidate_features: features available before the removal of the
            constant ones. Default CANDIDATE_FEATURES.

    Returns:
        A dataframe with one row per fold and the columns fold, rmse, mae,
        r2 and nasa_score. The NASA score is the mean over the rows of the
        validation portion, so it is not comparable with the NASA score
        summed over the last cycle of each engine of the test set.
    """
    fold_results = []
    for i, (tr_units, val_units) in enumerate(folds, 1):
        train_df, val_df = get_fold(df, tr_units, val_units)

        # Preprocessing fitted ONLY on the train set of the fold
        cond_mean, cond_std = fit_cond_stats(train_df)
        train_df = add_deviation(apply_cond_norm(train_df, cond_mean, cond_std))
        val_df = add_deviation(apply_cond_norm(val_df, cond_mean, cond_std))

        features = select_non_constant(train_df, candidate_features)

        m = clone(estimator).fit(train_df[features], train_df[RUL])
        pred = m.predict(val_df[features])

        fold_results.append({
            "fold": i,
            "rmse": root_mean_squared_error(val_df[RUL], pred),
            "mae": mean_absolute_error(val_df[RUL], pred),
            "r2": r2_score(val_df[RUL], pred),
            "nasa_score": nasa_score(val_df[RUL], pred, reduce="mean"),
        })
    return pd.DataFrame(fold_results)
