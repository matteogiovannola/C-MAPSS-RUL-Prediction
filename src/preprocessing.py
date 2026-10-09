"""
preprocessing.py

Preprocessing functions for the C-MAPSS data: operating-condition
clustering, per-condition sensor normalisation, deviation from the initial
state of each engine and removal of constant features.

All the statistics are learned with a "fit" function on the training data
only and then applied, never recomputed, to any other data.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    COND,
    CONSTANT_STD,
    CYCLE,
    DEV_SUFFIX,
    K0,
    N_COND,
    OP_COLS,
    SEED,
    SENSOR_COLS,
    UNIT,
)


def fit_cond_model(
    df: pd.DataFrame,
    op_cols: Sequence[str] = OP_COLS,
    k: int = N_COND,
    seed: int = SEED,
) -> Pipeline:
    """Learn the operating-condition clusters.

    Standardises the operational settings and clusters them with KMeans.

    Args:
        df: dataframe containing the operational settings.
        op_cols: names of the operational setting columns.
            Default OP_COLS.
        k: number of clusters (operating conditions). Default N_COND.
        seed: random seed of KMeans. Default SEED.

    Returns:
        The fitted pipeline (StandardScaler + KMeans).
    """
    return make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=k, n_init=10, random_state=seed),
    ).fit(df[list(op_cols)])


def add_cond(
    df: pd.DataFrame,
    cond_model: Pipeline,
    op_cols: Sequence[str] = OP_COLS,
) -> pd.DataFrame:
    """Add the operating condition (0..k-1) to a copy of the dataframe.

    Args:
        df: dataframe containing the operational settings.
        cond_model: pipeline returned by fit_cond_model.
        op_cols: names of the operational setting columns.
            Default OP_COLS.

    Returns:
        A copy of df with the additional COND column.
    """
    out = df.copy()
    out[COND] = cond_model.predict(out[list(op_cols)])
    return out


def fit_cond_stats(
    train_df: pd.DataFrame,
    sensor_cols: Sequence[str] = SENSOR_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean and std of each sensor for each operating condition.

    Args:
        train_df: training dataframe, with the COND column.
        sensor_cols: names of the sensor columns. Default SENSOR_COLS.

    Returns:
        A tuple (mean, std) of two dataframes with one row per operating
        condition and one column per sensor. The std is clipped at
        CONSTANT_STD to avoid divisions by zero.
    """
    g = train_df.groupby(COND)[list(sensor_cols)]
    mean = g.mean()
    std = g.std().clip(lower=CONSTANT_STD)
    return mean, std


def apply_cond_norm(
    df: pd.DataFrame,
    mean: pd.DataFrame,
    std: pd.DataFrame,
    sensor_cols: Sequence[str] = SENSOR_COLS,
) -> pd.DataFrame:
    """Apply the per-condition z-score to the sensors.

    Each sensor value is normalised with the mean and std of its own
    operating condition. The statistics are the given ones and are never
    recomputed on df.

    Args:
        df: dataframe to normalise, with the COND column.
        mean: per-condition sensor means, from fit_cond_stats.
        std: per-condition sensor stds, from fit_cond_stats.
        sensor_cols: names of the sensor columns. Default SENSOR_COLS.

    Returns:
        A copy of df with the normalised sensors.
    """
    sensor_cols = list(sensor_cols)
    out = df.copy()
    m = mean.loc[df[COND]].to_numpy()   # one row of means for each row of df
    s = std.loc[df[COND]].to_numpy()
    out[sensor_cols] = (df[sensor_cols].to_numpy() - m) / s
    return out


def add_deviation(
    df: pd.DataFrame,
    sensor_cols: Sequence[str] = SENSOR_COLS,
    k: int = K0,
) -> pd.DataFrame:
    """Add the deviation of each sensor from the initial state of the engine.

    The initial state of an engine is the mean of each sensor over its
    first k cycles. It is subtracted from every row of that engine. The new
    columns are named "<sensor>" + DEV_SUFFIX.

    Args:
        df: dataframe with the UNIT and CYCLE columns. It must contain the
            full history of each engine, not only its last row.
        sensor_cols: names of the sensor columns. Default SENSOR_COLS.
        k: number of initial cycles used as baseline. Default K0.

    Returns:
        A copy of df, sorted by (UNIT, CYCLE), with the deviation columns.
    """
    sensor_cols = list(sensor_cols)
    dev_cols = [f"{c}{DEV_SUFFIX}" for c in sensor_cols]

    out = df.sort_values([UNIT, CYCLE]).copy()

    # Baseline: mean of the first k cycles of each engine
    baseline = out.groupby(UNIT).head(k).groupby(UNIT)[sensor_cols].mean()

    # Subtract from every row the baseline of its own engine
    base = baseline.loc[out[UNIT]].to_numpy()
    out[dev_cols] = out[sensor_cols].to_numpy() - base
    return out


def select_non_constant(
    train_df: pd.DataFrame,
    cols: Sequence[str],
    thr: float = CONSTANT_STD,
) -> List[str]:
    """Select the columns that are not constant on the training set.

    Args:
        train_df: training dataframe.
        cols: candidate columns.
        thr: minimum std for a column to be kept. Default CONSTANT_STD.

    Returns:
        The columns whose std on train_df is >= thr.
    """
    stds = train_df[list(cols)].std()
    return stds[stds >= thr].index.tolist()
