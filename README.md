# NASA C-MAPSS FD004 - Remaining Useful Life (RUL) prediction

Prediction of the Remaining Useful Life of turbofan engines on the C-MAPSS **FD004** subset (6 operating conditions, 2 fault modes). The project has two notebooks: an exploratory data analysis and the modelling pipeline (feature engineering, model selection, hyperparameter tuning, evaluation on the official test set and model export).

## Results

The target is clipped at 125 cycles and the validation and test metrics are computed on the clipped RUL (see *Modelling choices*). The metrics on the original RUL are reported for reference and are not comparable.

| Evaluation | RMSE | NASA score |
|---|---|---|
| 5-fold CV on the training engines (tuned Extra Trees, clipped RUL) | ≈ 13.96 | per-row mean, not comparable with the test |
| Test set, last cycle of each engine, clipped RUL | ≈ 14.75 | ≈ 1,210 |
| Test set, last cycle of each engine, original RUL | ≈ 26.9 | ≈ 5,059 |

## Project structure

```
project/
├── data/
│   ├── raw/                  # original C-MAPSS .txt files
│   └── csv/                  # train.csv, test.csv, target_test.csv (generated)
├── figures/                  # figures saved by the EDA notebook (generated)
├── models/                   # exported model (generated)
├── notebooks/
│   ├── EDA.ipynb                              # exploratory data analysis
│   └── cmapss_fd004_rul_prediction.ipynb      # modelling pipeline
├── src/
│   ├── config.py             # constants and feature lists
│   ├── data_loader.py        # .txt -> .csv conversion
│   ├── eda_utility.py        # summary and plotting functions for the EDA
│   ├── evaluation.py         # engine-based folds, NASA score, cross-validation
│   ├── preprocessing.py      # condition clustering, normalisation, deviation features
│   └── utils.py              # figure saving
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the C-MAPSS dataset (NASA Prognostics Center of Excellence data repository) and put `train_FD004.txt`, `test_FD004.txt` and `RUL_FD004.txt` in `data/raw/`. The conversion to CSV (with the column names) is done in section 2 of the EDA notebook, with `data_loader.txt_to_csv`, and produces the files in `data/csv/`.

## Usage

1. Run `notebooks/EDA.ipynb` from top to bottom: it creates the CSV files in `data/csv/` and saves the figures in `figures/`.
2. Run `notebooks/cmapss_fd004_rul_prediction.ipynb` from top to bottom. It reads `train.csv` and `test.csv` from `data/csv/` and `RUL_FD004.txt` from `data/raw/`, and saves the model in `models/extra_trees_fd004.joblib`.

## Method

1. **EDA**: dataset description, sensor evolution, rolling statistics, sensor clustering, analysis by operating condition.
2. **Folds** on the engine IDs (never on the rows), reused in every experiment.
3. **Baseline**: Random Forest with default hyperparameters on the raw columns.
4. **Feature engineering**, each step judged with the same CV:
   - per-operating-condition normalisation of the sensors (KMeans on the operational settings): results essentially unchanged;
   - rolling mean/std: no improvement, discarded;
   - deviation from the initial state of each engine: clear improvement, kept;
   - removal of constant features, refitted in each fold.
5. **Model selection**: Random Forest, Extra Trees and HistGradientBoosting perform similarly in CV; Extra Trees is chosen as a compromise between RMSE and NASA score.
6. **Tuning**: randomised search on Extra Trees only (small gain over the defaults).
7. **Final evaluation**: refit on the whole training set, evaluated once on the last cycle of each test engine.

## Modelling choices

- **RUL clipped at 125.** The target is clipped at 125 cycles, and both the validation folds and the test set are evaluated against the clipped RUL, to focus the evaluation on the degradation phase rather than on the long healthy phase, where telling a RUL of 150 from 300 is hard and weighs heavily on the error.
- **Possible use with real data.** With real data the RUL is not clipped, but the model keeps predicting values capped at about 125. A predicted RUL below 125 indicates that the engine has entered the degradation phase and may need preventive maintenance, while a prediction close to 125 indicates that it is still healthy. The model can therefore be used almost as a classifier (*does this engine need maintenance?*).

## Inference

The preprocessing functions are not stored in the `.joblib` file: load them from `src/`.

```python
import joblib
from config import UNIT, CYCLE
from preprocessing import add_cond, add_deviation, apply_cond_norm

artifact = joblib.load("models/extra_trees_fd004.joblib")

# raw_df must contain the full history of each engine
df = add_cond(raw_df, artifact["cond_model"])
df = add_deviation(apply_cond_norm(df, artifact["cond_mean"], artifact["cond_std"]))
last = df.loc[df.groupby(UNIT)[CYCLE].idxmax()]
rul_pred = artifact["model"].predict(last[artifact["features"]])
```

## Limitations

- Differences between the three model families are within the fold-to-fold variability.
- The KMeans of the operating conditions is fitted once on the whole training set (unsupervised, operational settings only).
- Only FD004 has been evaluated.