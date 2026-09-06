"""
Generates predicted points for every current player, for the next
upcoming gameweek, using LIVE current-season data (from fetch_fpl_api.py)
and the trained points model.

Rebuilds the same feature set the model was trained on, but from this
season's actual data so far — early in the season, rolling windows will
be based on fewer games (down to 1), which is expected and handled the
same way training did (min_periods=1).

Run:
    python src/models/predict.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

ROLLING_WINDOWS = [3, 5]
STAT_COLS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "ict_index", "bps", "influence", "creativity", "threat", "starts",
    "expected_goals", "expected_assists", "expected_goal_involvements",
]

# Must match train_points_model.py's feature list exactly.
FEATURE_COLS = (
    [f"{c}_avg_last{w}" for w in ROLLING_WINDOWS for c in STAT_COLS]
    + ["minutes_avg_season", "value", "was_home", "position",
       "rotation_risk_prob",
       "strength_overall_home", "strength_overall_away",
       "strength_attack_home", "strength_attack_away",
       "strength_defence_home", "strength_defence_away",
       "home_strength", "opponent_defence_strength", "fixture_difficulty",
       "opp_team_goals_for_avg_last3", "opp_team_goals_against_avg_last3",
       "opp_team_goals_for_avg_last5", "opp_team_goals_against_avg_last5",
       "own_team_goals_for_avg_last3", "own_team_goals_against_avg_last3",
       "own_team_goals_for_avg_last5", "own_team_goals_against_avg_last5"]
)


def get_next_gameweek(gameweeks: pd.DataFrame) -> int:
    upcoming = gameweeks[gameweeks["finished"] == False]  # noqa: E712
    if upcoming.empty:
        raise RuntimeError("No upcoming gameweek found — season may be over, or gameweeks.csv is stale.")
    return int(upcoming.sort_values("id").iloc[0]["id"])


def build_rolling_form(history: pd.DataFrame) -> pd.DataFrame:
    """Same logic as build_features.py's add_rolling_form, applied to this season only."""
    history = history.sort_values(["element", "round"]).reset_index(drop=True)
    stat_cols = [c for c in STAT_COLS if c in history.columns]
    grouped = history.groupby("element")

    for window in ROLLING_WINDOWS:
        for col in stat_cols:
            history[f"{col}_avg_last{window}"] = (
                grouped[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
            )
    history["minutes_avg_season"] = grouped["minutes"].transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    # keep only each player's MOST RECENT row — that's their current form
    # going into the next (not-yet-played) gameweek
    latest = history.sort_values("round").groupby("element").tail(1)
    return latest


def get_next_fixture_per_team(fixtures: pd.DataFrame, next_gw: int) -> pd.DataFrame:
    gw_fixtures = fixtures[fixtures["event"] == next_gw]
    home = gw_fixtures.rename(columns={
        "team_h": "team", "team_a": "opponent_team", "team_h_difficulty": "fixture_difficulty",
    })[["team", "opponent_team", "fixture_difficulty"]]
    home["was_home"] = True

    away = gw_fixtures.rename(columns={
        "team_a": "team", "team_h": "opponent_team", "team_a_difficulty": "fixture_difficulty",
    })[["team", "opponent_team", "fixture_difficulty"]]
    away["was_home"] = False

    return pd.concat([home, away], ignore_index=True)


def build_team_form(fixtures: pd.DataFrame) -> pd.DataFrame:
    finished = fixtures[fixtures["finished"] == True]  # noqa: E712
    if finished.empty:
        return pd.DataFrame()

    home_rows = finished.rename(columns={
        "team_h": "team", "team_h_score": "goals_for", "team_a_score": "goals_against",
    })[["event", "team", "goals_for", "goals_against"]]
    away_rows = finished.rename(columns={
        "team_a": "team", "team_a_score": "goals_for", "team_h_score": "goals_against",
    })[["event", "team", "goals_for", "goals_against"]]

    team_matches = pd.concat([home_rows, away_rows], ignore_index=True)
    team_matches = team_matches.sort_values(["team", "event"]).reset_index(drop=True)

    grouped = team_matches.groupby("team")
    for window in ROLLING_WINDOWS:
        team_matches[f"team_goals_for_avg_last{window}"] = (
            grouped["goals_for"].transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        team_matches[f"team_goals_against_avg_last{window}"] = (
            grouped["goals_against"].transform(lambda s: s.rolling(window, min_periods=1).mean())
        )

    latest = team_matches.sort_values("event").groupby("team").tail(1)
    return latest[["team"] + [c for c in latest.columns if c.startswith("team_goals_")]]


def compute_rotation_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Uses the saved minutes_model if present; otherwise a simple heuristic fallback.
    Expects df to already have starts_avg_last3/5, minutes_avg_last3/5, minutes_avg_season,
    value, position, was_home from the earlier merges."""
    model_path = MODEL_DIR / "minutes_model.json"
    fill_cols = ["starts_avg_last3", "starts_avg_last5", "minutes_avg_last3",
                 "minutes_avg_last5", "minutes_avg_season"]
    for col in fill_cols:
        if col not in df.columns:
            df[col] = 0
    df[fill_cols] = df[fill_cols].fillna(0)

    if model_path.exists():
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        feats = df[fill_cols + ["value", "position", "was_home"]].copy()
        feats["position"] = feats["position"].astype("category")
        df["rotation_risk_prob"] = model.predict_proba(feats)[:, 1]
    else:
        print("  no saved minutes_model.json found — using a simple fallback (minutes_avg_last3 / 90)")
        df["rotation_risk_prob"] = (df["minutes_avg_last3"].fillna(0) / 90).clip(0, 1)

    return df


def main():
    print("Loading live data...")
    players = pd.read_csv(RAW_DIR / "players.csv")
    teams = pd.read_csv(RAW_DIR / "teams.csv")
    positions = pd.read_csv(RAW_DIR / "positions.csv")
    gameweeks = pd.read_csv(RAW_DIR / "gameweeks.csv")
    fixtures = pd.read_csv(RAW_DIR / "fixtures.csv")
    history = pd.read_csv(RAW_DIR / "player_gameweek_history.csv")

    next_gw = get_next_gameweek(gameweeks)
    print(f"Predicting for gameweek {next_gw}")

    print("Attaching player static info...")
    pos_map = positions.set_index("id")["singular_name_short"].to_dict()
    players["position"] = players["element_type"].map(pos_map)
    players["value"] = players["now_cost"] / 10.0

    print("Computing this season's rolling form...")
    history_latest = build_rolling_form(history)

    print("Getting next fixture for each team...")
    next_fixture = get_next_fixture_per_team(fixtures, next_gw)

    df = players.merge(next_fixture, left_on="team", right_on="team", how="left")
    # a player's fixture depends on THEIR team, but players.csv already has a
    # 'team' column with the team id — this merge attaches the right fixture
    df = df.merge(
        history_latest.drop(columns=["position"], errors="ignore"),
        left_on="id", right_on="element", how="left", suffixes=("", "_hist"),
    )

    print("Attaching team strength ratings...")
    team_strength = teams[[
        "id", "strength_overall_home", "strength_overall_away",
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]].rename(columns={"id": "opponent_team"})
    df = df.merge(team_strength, on="opponent_team", how="left")

    df["home_strength"] = np.where(df["was_home"], df["strength_attack_home"], df["strength_attack_away"])
    df["opponent_defence_strength"] = np.where(df["was_home"], df["strength_defence_away"], df["strength_defence_home"])

    print("Computing dynamic team form from this season's results so far...")
    team_form = build_team_form(fixtures)
    if not team_form.empty:
        team_form_cols = [c for c in team_form.columns if c.startswith("team_goals_")]
        opp_form = team_form.rename(columns={"team": "opponent_team", **{c: f"opp_{c}" for c in team_form_cols}})
        df = df.merge(opp_form, on="opponent_team", how="left")
        own_form = team_form.rename(columns={"team": "team", **{c: f"own_{c}" for c in team_form_cols}})
        df = df.merge(own_form, on="team", how="left", suffixes=("", "_own"))

    print("Computing rotation risk...")
    df = compute_rotation_risk(df)

    # fill any remaining gaps (new players, early-season sparsity) with 0 —
    # matches how the model saw sparse-history rows during training
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="ignore")
    df["position"] = df["position"].astype("category")
    numeric_feature_cols = [c for c in FEATURE_COLS if c != "position"]
    df[numeric_feature_cols] = df[numeric_feature_cols].fillna(0)

    print("Loading trained model and predicting...")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / "points_model.json")
    df["predicted_points"] = np.clip(model.predict(df[FEATURE_COLS]), 0, None)

    out = df[["id", "web_name", "team", "position", "value", "predicted_points"]].rename(
        columns={"id": "player_id", "web_name": "name", "value": "price"}
    )
    # map numeric team id to name for readability
    team_names = teams.set_index("id")["name"].to_dict()
    out["team"] = out["team"].map(team_names)

    out_path = PROCESSED_DIR / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} player predictions -> {out_path}")
    print(out.sort_values("predicted_points", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()