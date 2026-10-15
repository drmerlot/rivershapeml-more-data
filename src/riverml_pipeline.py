"""RiverShapeML deployment pipeline (NOAA-OWP/predict-riverML), reproduced for reuse.

Mirrors their data_loader.py / inference.py: fill missing values with the saved training medians,
engineer features, select the model's feature list (in order), and predict log1p(target) with XGBoost.
Inputs: data/raw/predict-riverML/ (ml_inputs.parquet, model_space/, models/), see get_data.sh.

    X, ids = load_features("TW_bf")            # all ~2.65M reaches, or columns=..., rows filter
    model = load_model("TW_bf")
    width_m = predict(model, X)
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
RML = ROOT / "data/raw/predict-riverML"
ID = "FEATUREID"
# settings used for all models trained here: regularized and early-stopped versions of the deployed settings,
# chosen by src/tune_regularization.py (validation split; same held-out accuracy, ~4x fewer trees, less memorization)
MODEL_PARAMS = {"TW_bf": dict(n_estimators=2478, learning_rate=0.01, max_depth=6, min_child_weight=5, reg_lambda=1,
                              subsample=0.8, colsample_bytree=0.7, random_state=42),
                "Y_bf": dict(n_estimators=2416, learning_rate=0.01, max_depth=6, min_child_weight=5, reg_lambda=1,
                             subsample=0.8, colsample_bytree=0.7, random_state=42)}
# hyperparameters of their deployed "update" models (read from the pickled XGBRegressor); used only for reference
# and as the starting point of the regularization sweep
THEIR_PARAMS = {"TW_bf": dict(n_estimators=9765, learning_rate=0.01, max_depth=9, subsample=0.8,
                              colsample_bytree=0.7, random_state=42),
                "Y_bf": dict(n_estimators=11000, learning_rate=0.01, max_depth=9, subsample=0.8,
                             colsample_bytree=0.7, random_state=42)}


def feature_list(target):
    return json.loads((RML / f"model_space/final_model_features_{target}.json").read_text())


def engineer(df, center_x=-22894.21, center_y=-281.82):
    """Their engineer_features(), verbatim in effect."""
    eps = 1e-6
    d = df
    d["StreamPower_Index"] = d["nwm_max"] * d["slope"]
    d["unit_discharge"] = d["bf_ff"] / (d["TotDASqKM"] + eps)
    d["vegetation_stability_index"] = d["LAI"] / (d["slope"] + eps)
    d["rock_resistance_index"] = d["RckDepWs"] * d["slope"]
    d["log_discharge"] = np.log1p(d["bf_ff"])
    d["log_drainage_area"] = np.log1p(d["TotDASqKM"])
    d["slope_squared"] = d["slope"] ** 2
    d["Runoff_Stability_Index"] = (d["qsb_tavg"] / (d["bf_ff"] + eps)).clip(0, 1)
    d["BankStability_Index"] = d["LAI"] / (d["StreamPower_Index"] + eps)
    d["Erodibility_Index"] = d["StreamPower_Index"] * d["RckDepWs"]
    d["Sinuosity_Slope_Ratio"] = d["Sinuosity"] / (d["slope"] + eps)
    d = pd.get_dummies(d, columns=["flowline_type"], prefix="flowline")
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform(d["long"].values, d["lat"].values)
    d["projected_x"], d["projected_y"] = x, y
    d["dist_from_center"] = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    d["angle_from_center"] = np.arctan2(y - center_y, x - center_x)
    return d


RAW_COLS = ["FEATUREID", "nwm_max", "slope", "bf_ff", "TotDASqKM", "LAI", "RckDepWs", "qsb_tavg", "Sinuosity",
            "flowline_type", "lat", "long", "suggested_width_m", "BFIWs", "ArbolateSu", "StreamOrde", "WtDepWs",
            "roughness", "elevation_diff"]


def load_features(target, ids=None):
    """Model-ready features for `target` ('TW_bf' or 'Y_bf'); optionally only reaches in `ids`."""
    df = pd.read_parquet(RML / "data/ml_inputs.parquet", columns=RAW_COLS)
    if ids is not None:
        df = df[df[ID].isin(set(ids))]
    df = df.reset_index(drop=True)
    med = pd.read_parquet(RML / f"model_space/metrics/median_imput_{target}.parquet")["Median"].to_dict()
    df = df.fillna(value={k: v for k, v in med.items() if k in df})
    df = engineer(df)
    feats = feature_list(target)
    for c in feats:
        if c not in df:
            df[c] = 0  # as their clean_predictors() does for missing one-hot columns
    return df[feats].astype(float), df[ID].to_numpy()


def load_model(target):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pickle.load(open(RML / f"models/trained_xgboost_model_update_{target}_final.pickle.dat", "rb"))


def predict(model, X):
    """Their inference: expm1 of the log1p prediction, absolute value."""
    return np.abs(np.expm1(model.predict(X)))
