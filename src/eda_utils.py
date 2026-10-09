"""
eda_utils.py

Utility functions for the exploratory data analysis (EDA) of the
C-MAPSS dataset (RUL prediction).

All plotting functions return the Figure object instead of only calling
plt.show() internally, so the caller can decide whether to show it, save
it, or compose it into a grid with other plots.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from utils import save_figure


# ---------------------------------------------------------------------------
# Summary / statistics functions
# ---------------------------------------------------------------------------

def dataset_summary(dataset: pd.DataFrame, verbose: bool = True) -> dict:
    """Compute a summary of the dataset (shape, dtypes, descriptive stats).

    Args:
        dataset: DataFrame to summarize.
        verbose: if True, print the summary (default True).

    Returns:
        A dictionary with the keys "shape", "dtypes" and "describe",
        so the result can be reused even without printing.

    Raises:
        ValueError: if the dataset is empty.
    """
    if dataset.empty:
        raise ValueError("The dataset is empty: cannot compute a summary.")

    summary = {
        "shape": dataset.shape,
        "dtypes": dataset.dtypes,
        "describe": dataset.describe(),
    }

    if verbose:
        print("Dataset shape:")
        print(summary["shape"])

        print("\nDataset dtypes:")
        print(summary["dtypes"])

        print("\nDataset statistics:")
        print(summary["describe"])

    return summary


def count_units(dataset: pd.DataFrame) -> int:
    """Return the number of distinct engines (unit numbers) in the dataset.

    Args:
        dataset: DataFrame containing the "unit number" column.

    Returns:
        Number of distinct unit numbers.

    Raises:
        KeyError: if the "unit number" column is missing.
    """
    if "unit number" not in dataset.columns:
        raise KeyError("The 'unit number' column is not present in the dataset.")

    return dataset["unit number"].nunique()


def cycles_per_unit(dataset: pd.DataFrame) -> pd.Series:
    """Compute the maximum number of cycles run by each engine.

    Args:
        dataset: DataFrame with "unit number" and "cycle" columns.

    Returns:
        Series indexed by unit number, holding the maximum number of
        cycles (lifetime) of each engine.

    Raises:
        KeyError: if the required columns are missing.
    """
    required = {"unit number", "cycle"}
    missing = required - set(dataset.columns)
    if missing:
        raise KeyError(f"Missing columns in the dataset: {missing}")

    return dataset.groupby("unit number")["cycle"].max()


def missing_values(dataset: pd.DataFrame) -> pd.Series:
    """Count the missing values in each column of the dataset.

    Args:
        dataset: DataFrame to check.

    Returns:
        Series with the NaN count per column.
    """
    return dataset.isnull().sum()


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_multiple_sensor_evolution(
    dataset: pd.DataFrame,
    engine: int,
    sensors: Sequence[str],
    save: bool = False,
    show: bool = True,
    figsize: tuple[int, int] = (12, 6),
) -> Figure:
    """Plot the evolution over time of one or more sensors for a given engine.

    Args:
        dataset: DataFrame containing the sensor data.
        engine: unit number of the engine to analyze.
        sensors: list of sensor column names to plot.
        save: if True, save the figure via save_figure().
        show: if True, call plt.show() before returning (default True,
            for compatibility with interactive notebook use; set to False
            if you only want to get the figure, e.g. to compose it into a
            grid with other plots).
        figsize: figure size.

    Returns:
        The created Figure object.

    Raises:
        ValueError: if the requested engine is not in the dataset, or if
            one of the requested sensors does not exist as a column.
    """
    engine_data = dataset[dataset["unit number"] == engine]

    if engine_data.empty:
        raise ValueError(f"No data found for unit number = {engine}.")

    missing_sensors = [s for s in sensors if s not in dataset.columns]
    if missing_sensors:
        raise ValueError(f"Sensors not present in the dataset: {missing_sensors}")

    fig, ax = plt.subplots(figsize=figsize)

    for sensor in sensors:
        ax.plot(engine_data["cycle"], engine_data[sensor], label=sensor)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Value")
    ax.set_title(f"Engine {engine} sensor evolution")
    ax.legend(loc="upper left")
    ax.grid(True)

    fig.tight_layout()

    if save:
        save_figure(fig, f"{engine}_evolution")

    if show:
        plt.show()

    return fig


def rolling_mean(
    dataset: pd.DataFrame,
    engine: int,
    sensor: str,
    window: int = 20,
    save: bool = False,
    show: bool = True,
    figsize: tuple[int, int] = (12, 5),
) -> tuple[Figure, pd.Series]:
    """Compute and plot the rolling mean of a sensor for a given engine.

    Args:
        dataset: DataFrame containing the sensor data.
        engine: unit number of the engine to analyze.
        sensor: name of the sensor column.
        window: rolling mean window size (default 20).
        save: if True, save the figure via save_figure().
        show: if True, call plt.show() before returning.
        figsize: figure size.

    Returns:
        A tuple (fig, rolling_series), where rolling_series is the Series
        with the computed rolling mean (indexed like engine_data), so the
        result can be reused without recomputing it.

    Raises:
        ValueError: if the requested engine or sensor is not present.
    """
    if sensor not in dataset.columns:
        raise ValueError(f"Sensor '{sensor}' not present in the dataset.")

    engine_data = dataset[dataset["unit number"] == engine].copy()

    if engine_data.empty:
        raise ValueError(f"No data found for unit number = {engine}.")

    rolling_series = engine_data[sensor].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        engine_data["cycle"],
        engine_data[sensor],
        alpha=0.3,
        label="Raw signal",
    )
    ax.plot(
        engine_data["cycle"],
        rolling_series,
        linewidth=2,
        label=f"Rolling mean (window={window})",
    )

    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor)
    ax.set_title(f"Unit {engine} - {sensor}")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save:
        save_figure(fig, f"{engine}_{sensor}_rolling_mean")

    if show:
        plt.show()

    return fig, rolling_series


def rolling_std(
    dataset: pd.DataFrame,
    engine: int,
    sensor: str,
    window: int = 20,
    save: bool = False,
    show: bool = True,
    figsize: tuple[int, int] = (12, 5),
) -> tuple[Figure, pd.Series]:
    """Compute and plot the rolling standard deviation of a sensor for a given engine.

    Complements rolling_mean(): a change in the rolling standard deviation
    over time can highlight changes in signal variability that may also be
    related to engine degradation.

    Args:
        dataset: DataFrame containing the sensor data.
        engine: unit number of the engine to analyze.
        sensor: name of the sensor column.
        window: rolling window size (default 20).
        save: if True, save the figure via save_figure().
        show: if True, call plt.show() before returning.
        figsize: figure size.

    Returns:
        A tuple (fig, rolling_series), where rolling_series is the Series
        with the computed rolling standard deviation (indexed like
        engine_data), so the result can be reused without recomputing it.

    Raises:
        ValueError: if the requested engine or sensor is not present.
    """
    if sensor not in dataset.columns:
        raise ValueError(f"Sensor '{sensor}' not present in the dataset.")

    engine_data = dataset[dataset["unit number"] == engine].copy()

    if engine_data.empty:
        raise ValueError(f"No data found for unit number = {engine}.")

    rolling_series = engine_data[sensor].rolling(window=window).std()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        engine_data["cycle"],
        engine_data[sensor],
        alpha=0.3,
        label="Raw signal",
    )
    ax.plot(
        engine_data["cycle"],
        rolling_series,
        linewidth=2,
        color="tab:red",
        label=f"Rolling std (window={window})",
    )

    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor)
    ax.set_title(f"Unit {engine} - {sensor} (rolling std)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save:
        save_figure(fig, f"{engine}_{sensor}_rolling_std")

    if show:
        plt.show()

    return fig, rolling_series


def plot_sensor_dendrogram(
    dataset: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    method: str = "average",
    figsize: tuple[int, int] = (12, 6),
    title: str = "Sensor clustering",
    save: bool = False,
    show: bool = True,
) -> tuple[Figure, np.ndarray]:
    """Plot a hierarchical clustering dendrogram of the sensors.

    The distance between two sensors is defined as 1 - |correlation|:
    strongly correlated sensors (positively or negatively) end up close
    to each other.

    Args:
        dataset: DataFrame containing the sensor data.
        features: list of columns to use; if None, all columns whose name
            contains "sensor" are selected automatically.
        method: linkage method to use (default "average").
        figsize: figure size.
        title: plot title.
        save: if True, save the figure via save_figure().
        show: if True, call plt.show() before returning.

    Returns:
        A tuple (fig, Z), where Z is the linkage matrix computed by
        scipy, reusable for example to cut the dendrogram with
        fcluster() without recomputing everything from scratch.

    Raises:
        ValueError: if there are fewer than two sensors with non-zero
            variance on which to compute the correlation.
    """
    if features is None:
        sensor_data = dataset.filter(like="sensor")
    else:
        missing = [f for f in features if f not in dataset.columns]
        if missing:
            raise ValueError(f"Columns not present in the dataset: {missing}")
        sensor_data = dataset[features]

    # drop features with no variability (correlation is undefined)
    sensor_data = sensor_data.loc[:, sensor_data.std() != 0]

    if sensor_data.shape[1] < 2:
        raise ValueError(
            "At least two sensors with non-zero variance are required "
            "to compute the dendrogram."
        )

    corr = sensor_data.corr()
    distance = 1 - corr.abs()

    Z = linkage(squareform(distance, checks=False), method=method)

    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(Z, labels=corr.columns.tolist(), leaf_rotation=90, ax=ax)

    ax.set_title(title)
    ax.set_ylabel("Distance")

    fig.tight_layout()

    if save:
        save_figure(fig, "sensor_clustering")

    if show:
        plt.show()

    return fig, Z


def plot_sensor_by_operating_condition(
    dataset: pd.DataFrame,
    sensor: str,
    x: str = "RUL",
    condition_col: str = "operating condition",
    figsize: tuple[int, int] = (10, 5),
    save: bool = False,
    show: bool = True,
) -> Figure:
    """Plot the evolution of a sensor against RUL/cycle, split by operating condition.

    For each operating condition, the sensor values are averaged over each
    value of x and plotted as a separate line.

    Args:
        dataset: DataFrame containing the sensor data.
        sensor: name of the sensor column.
        x: column to use on the x axis (default "RUL"; "cycle" also works).
        condition_col: name of the column holding the operating condition
            (default "operating condition").
        figsize: figure size.
        save: if True, save the figure via save_figure().
        show: if True, call plt.show() before returning.

    Returns:
        The created Figure object.

    Raises:
        ValueError: if the sensor, the x column or the operating condition
            column is not present in the dataset.
    """
    if sensor not in dataset.columns:
        raise ValueError(f"Sensor '{sensor}' not present in the dataset.")

    if x not in dataset.columns:
        raise ValueError(f"Column '{x}' not present in the dataset.")

    if condition_col not in dataset.columns:
        raise ValueError(f"Column '{condition_col}' not present in the dataset.")

    fig, ax = plt.subplots(figsize=figsize)

    conditions = sorted(dataset[condition_col].unique())

    for condition in conditions:
        subset = dataset[dataset[condition_col] == condition]

        # mean sensor value for each value of x
        trend = subset.groupby(x)[sensor].mean().sort_index()

        ax.plot(
            trend.index,
            trend.values,
            label=f"Operating condition {condition}",
        )

    ax.set_xlabel(x)
    ax.set_ylabel(sensor)
    ax.set_title(f"{sensor} trend by operating condition")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    if save:
        save_figure(fig, f"{sensor}_by_operating_condition")

    if show:
        plt.show()

    return fig