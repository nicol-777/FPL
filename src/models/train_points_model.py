"""
Train an XGBoost model to predict a player's FPL points for a gameweek,
using the leakage-safe rolling features from build_features.py.

Splits chronologically (train on earlier gameweeks, validate on later ones)
rather than randomly — random splits leak future form into the past for
time-series data like this.

Run:
    python src/models/train_points_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

TARGET = "total_points"

# EXPLICIT ALLOW-LIST, not a drop-list. Only columns known BEFORE the
# gameweek kicks off may be used as features. Anything not listed here —
# including all raw same-gameweek match stats (bonus, bps, minutes,
# goals_scored, ict_index, etc.) — is excluded by default. This is the
# safe direction for a leakage-prone dataset: a forgotten drop leaks the
# target, a forgotten allow just under-uses a feature.
PRE_MATCH_BASE_FEATURES = [
    "value",           # player's price — known before the gameweek
    "was_home",        # fixture is known in advance
    "position",        # static attribute
    "rotation_risk_prob",  # predicted P(starts >= 60 mins), from minutes_model
]

TEAM_STRENGTH_FEATURES = [
    "strength_overall_home", "strength_overall_away",
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
    "home_strength", "opponent_defence_strength",
    "fixture_difficulty",
    # opp_/own_team_goals_* columns are NOT listed here — they already
    # end in _avg_last3/_avg_last5 and get picked up automatically by
    # the rolling_cols suffix match in prepare_xy(). Listing them here
    # too caused a duplicate-column crash.
]

CATEGORICAL_COLS = ["position"]


def load_features() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    return df


def prepare_xy(df: pd.DataFrame):
    df = df.copy()

    # drop rows with no prior form yet — nothing meaningful to learn from
    df = df[~df["is_first_gw_for_player"]].reset_index(drop=True)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Only rolling-form columns (suffixed _avg_last3 / _avg_last5 /
    # _avg_season) plus the explicit pre-match allow-list are used.
    rolling_cols = [c for c in df.columns if
                     c.endswith("_avg_last3") or c.endswith("_avg_last5") or c.endswith("_avg_season")]
    allowed_base = [c for c in PRE_MATCH_BASE_FEATURES + TEAM_STRENGTH_FEATURES if c in df.columns]

    feature_cols = list(dict.fromkeys(rolling_cols + allowed_base))  # de-duped, order preserved
    print(f"Using {len(feature_cols)} features (all pre-match / rolling-form only):")
    print(f"  {feature_cols}")

    X = df[feature_cols]
    y = df[TARGET]
    return X, y, df


def chronological_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, val_fraction: float = 0.2):
    """
    Split so validation is always LATER than training, across both
    stacked seasons — e.g. train on 2024-25 + most of 2025-26, validate
    on the most recent gameweeks of 2025-26.
    """
    order_key = df[["season", "round"]].apply(tuple, axis=1)
    rank = order_key.rank(method="dense")
    cutoff_rank = rank.quantile(1 - val_fraction)

    train_mask = rank <= cutoff_rank
    val_mask = ~train_mask

    print(f"Train: {train_mask.sum()} rows "
          f"(through {df.loc[train_mask, ['season', 'round']].iloc[-1].tolist()})")
    print(f"Val:   {val_mask.sum()} rows "
          f"(from {df.loc[val_mask, ['season', 'round']].iloc[0].tolist()} onward)")

    return X[train_mask], X[val_mask], y[train_mask], y[val_mask], df[val_mask]


def train_model(X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        enable_categorical=True,
        early_stopping_rounds=30,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def evaluate(model, X_val, y_val, df_val: pd.DataFrame):
    preds = model.predict(X_val)
    preds = np.clip(preds, 0, None)  # points can't be negative

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    print(f"\nValidation MAE (all rows):  {mae:.3f} points")
    print(f"Validation RMSE (all rows): {rmse:.3f} points")

    if "total_points_avg_last5" in X_val.columns:
        baseline_preds = X_val["total_points_avg_last5"].fillna(y_val.mean())
        baseline_mae = mean_absolute_error(y_val, baseline_preds)
        print(f"Baseline MAE (5-gw rolling avg, all rows): {baseline_mae:.3f} points")
        print(f"Model improvement over baseline (all rows): {baseline_mae - mae:+.3f} points")

    # The real test: how good is this for players who actually played?
    # Bench/unused-sub rows (0 minutes) are trivially easy for both the
    # model and the baseline to get right, and dominate the row count —
    # they inflate the "all rows" numbers without being useful for
    # transfer decisions, since you'd never transfer in a 0-minute player.
    played_mask = df_val["minutes"] >= 60
    print(f"\n--- Restricted to rows where the player actually started (>=60 mins), "
          f"{played_mask.sum()} of {len(df_val)} rows ---")

    y_played = y_val[played_mask]
    preds_played = preds[played_mask]
    mae_played = mean_absolute_error(y_played, preds_played)
    rmse_played = np.sqrt(mean_squared_error(y_played, preds_played))
    print(f"Validation MAE (started):  {mae_played:.3f} points")
    print(f"Validation RMSE (started): {rmse_played:.3f} points")

    if "total_points_avg_last5" in X_val.columns:
        baseline_played = X_val.loc[played_mask, "total_points_avg_last5"].fillna(y_played.mean())
        baseline_mae_played = mean_absolute_error(y_played, baseline_played)
        print(f"Baseline MAE (started): {baseline_mae_played:.3f} points")
        print(f"Model improvement over baseline (started): {baseline_mae_played - mae_played:+.3f} points")

    return mae, rmse


def print_feature_importance(model, X: pd.DataFrame, top_n: int = 15):
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\nTop {top_n} features:")
    print(importance.head(top_n).to_string())


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = load_features()

    print("Preparing X/y...")
    X, y, df = prepare_xy(df)

    print("Splitting chronologically...")
    X_train, X_val, y_train, y_val, df_val = chronological_split(df, X, y)

    print("Training model...")
    model = train_model(X_train, y_train, X_val, y_val)

    evaluate(model, X_val, y_val, df_val)
    print_feature_importance(model, X_train)

    model_path = MODEL_DIR / "points_model.json"
    model.save_model(model_path)
    print(f"\nSaved model -> {model_path}")


if __name__ == "__main__":
    main()
