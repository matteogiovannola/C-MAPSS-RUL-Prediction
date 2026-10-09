"""
config.py

Global constants shared by the notebooks and by the other modules in src/:
experiment settings, column names and the lists of sensor and candidate
features used by the preprocessing.
"""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
SEED: int = 42       # random seed used everywhere (folds, KMeans, models)
N_SPLITS: int = 5    # number of CV folds (the split is done on engine IDs)
N_ITER: int = 20     # random-search iterations for the hyperparameter tuning
RUL_CAP: int = 125   # RUL values above this threshold are clipped

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
UNIT: str = "unit number"
CYCLE: str = "cycle"
RUL: str = "rul"
COND: str = "cond"   # operating condition (KMeans cluster), added by add_cond
DROP_COLS: List[str] = [UNIT, RUL]   # columns never used as features
OP_COLS: List[str] = [
    "operational setting 1",
    "operational setting 2",
    "operational setting 3",
]
SENSOR_COLS: List[str] = [f"sensor{i}" for i in range(1, 22)]   # sensor1 .. sensor21

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------
N_COND: int = 6               # number of operating conditions (KMeans clusters)
CONSTANT_STD: float = 1e-3    # std below this threshold = constant feature
K0: int = 10                  # initial cycles used as baseline for the deviation features
DEV_SUFFIX: str = "_dev"      # suffix of the deviation columns

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
DEV_COLS: List[str] = [f"{c}{DEV_SUFFIX}" for c in SENSOR_COLS]

# Features before the (per-fold) removal of the constant ones
CANDIDATE_FEATURES: List[str] = SENSOR_COLS + [CYCLE] + DEV_COLS
