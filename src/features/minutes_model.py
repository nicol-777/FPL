"""
Rotation-risk sub-model: predicts the probability a player starts
(plays >=60 minutes) in a gameweek, using only pre-match information.

This matters because a huge share of FPL prediction error comes from
"will they even play" rather than "how well will they play" — a player
rotated to the bench scores ~0 regardless of how good their underlying
stats are. This model isolates that signal so it can be fed into the
points model as a feature (rotation_risk_prob).

Run:
    python src/features/minutes_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

TARGET = "started"  # derived: minutes >= 60

FEATURE_COLS = [
    "starts_avg_last3", "starts_avg_last5",
    "minutes_avg_last3", "minutes_avg_last5", "minutes_avg_season",
    "value", "position", "was_home",
]


def load_features() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    df = df[~df["is_first_gw_for_player"]].reset_index(drop=True)
    df[TARGET] = (df["minutes"] >= 60).astype(int)
    return df


def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    df["position"] = df["position"].astype("category")
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y, df


def chronological_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, val_fraction: float = 0.2):
    order_key = df[["season", "round"]].apply(tuple, axis=1)
    rank = order_key.rank(method="dense")
    cutoff_rank = rank.quantile(1 - val_fraction)
    train_mask = rank <= cutoff_rank
    val_mask = ~train_mask
    print(f"Train: {train_mask.sum()} rows, Val: {val_mask.sum()} rows")
    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


def train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        enable_categorical=True,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    return model


def evaluate(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    brier = brier_score_loss(y_val, probs)
    print(f"\nValidation AUC:   {auc:.3f}  (0.5 = no better than chance, 1.0 = perfect)")
    print(f"Validation Brier: {brier:.3f}  (lower is better, 0 = perfect)")

    # naive baseline: "they'll do what they did last week" (last game's start status)
    return auc, brier


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = load_features()

    print("Preparing X/y...")
    X, y, df = prepare_xy(df)
    print(f"Base rate (started): {y.mean():.1%}")

    print("Splitting chronologically...")
    X_train, X_val, y_train, y_val = chronological_split(df, X, y)

    print("Training rotation-risk model...")
    model = train_model(X_train, y_train, X_val, y_val)

    evaluate(model, X_val, y_val)

    model_path = MODEL_DIR / "minutes_model.json"
    model.save_model(model_path)
    print(f"\nSaved model -> {model_path}")


if __name__ == "__main__":
    main()
