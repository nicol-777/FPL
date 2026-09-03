"""
Build a training-ready feature set from the historical merged_gw data
(fetched by fetch_historical.py from the vaastav archive).

For every (player, gameweek) row, builds features using ONLY information
available before that gameweek was played — rolling form windows are
shifted by one gameweek to avoid leaking the target into the features.

Rolling form is computed WITHIN each season only (not carried across the
2024-25 -> 2025-26 boundary), since squads and roles change over summer.

Output: data/processed/features.csv
    One row per player per gameweek played, with target column `total_points`.

Run:
    python src/features/build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
HIST_DIR = RAW_DIR / "historical"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

ROLLING_WINDOWS = [3, 5]

STAT_COLS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "ict_index", "bps", "influence", "creativity", "threat", "starts",
    "expected_goals", "expected_assists", "expected_goal_involvements",
]


def load_merged() -> pd.DataFrame:
    path = HIST_DIR / "merged_gw_all_seasons.csv"
    df = pd.read_csv(path)
    df["player_id"] = df["element"]
    return df


def load_team_strength() -> pd.DataFrame:
    """Loads each season's teams.csv and stacks them, keyed by (season, team id)."""
    frames = []
    for season_dir in sorted(HIST_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        teams_path = season_dir / "teams.csv"
        if not teams_path.exists():
            continue
        teams = pd.read_csv(teams_path)
        keep_cols = [c for c in [
            "id", "strength_overall_home", "strength_overall_away",
            "strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away",
        ] if c in teams.columns]
        teams = teams[keep_cols].copy()
        teams["season"] = season_dir.name
        frames.append(teams)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(columns={"id": "opponent_team"})


def load_fixture_difficulty() -> pd.DataFrame:
    """
    Loads each season's fixtures.csv for the official per-fixture FDR
    (1-5 difficulty rating), keyed by (season, fixture id).
    """
    frames = []
    for season_dir in sorted(HIST_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        fixtures_path = season_dir / "fixtures.csv"
        if not fixtures_path.exists():
            continue
        fixtures = pd.read_csv(fixtures_path)
        keep_cols = [c for c in ["id", "team_h_difficulty", "team_a_difficulty"] if c in fixtures.columns]
        fixtures = fixtures[keep_cols].copy()
        fixtures["season"] = season_dir.name
        frames.append(fixtures)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(columns={"id": "fixture"})


def load_fixtures_full() -> pd.DataFrame:
    """
    Loads each season's fixtures.csv with enough columns to compute
    per-team rolling goals for/against, plus FDR and team IDs.
    """
    frames = []
    for season_dir in sorted(HIST_DIR.iterdir()):
        if not season_dir.is_dir():
            continue
        fixtures_path = season_dir / "fixtures.csv"
        if not fixtures_path.exists():
            continue
        cols = ["id", "event", "team_h", "team_a", "team_h_score", "team_a_score",
                "team_h_difficulty", "team_a_difficulty"]
        fixtures = pd.read_csv(fixtures_path)
        fixtures = fixtures[[c for c in cols if c in fixtures.columns]].copy()
        fixtures["season"] = season_dir.name
        frames.append(fixtures)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(columns={"id": "fixture"})


def build_team_form(fixtures_full: pd.DataFrame) -> pd.DataFrame:
    """
    Turns fixture-level results into a per-team, per-gameweek table of
    goals scored/conceded, then computes rolling (shifted, leakage-safe)
    attacking/defensive form for each team. This captures CURRENT form
    (e.g. a defense on a bad run) rather than a static season rating.
    """
    if fixtures_full.empty or "team_h_score" not in fixtures_full.columns:
        return pd.DataFrame()

    home_rows = fixtures_full.rename(columns={
        "team_h": "team", "team_h_score": "goals_for", "team_a_score": "goals_against",
    })[["season", "event", "team", "goals_for", "goals_against"]]

    away_rows = fixtures_full.rename(columns={
        "team_a": "team", "team_a_score": "goals_for", "team_h_score": "goals_against",
    })[["season", "event", "team", "goals_for", "goals_against"]]

    team_matches = pd.concat([home_rows, away_rows], ignore_index=True)
    team_matches = team_matches.dropna(subset=["goals_for", "goals_against"])
    team_matches = team_matches.sort_values(["team", "season", "event"]).reset_index(drop=True)

    grouped = team_matches.groupby(["team", "season"])
    for window in ROLLING_WINDOWS:
        team_matches[f"team_goals_for_avg_last{window}"] = (
            grouped["goals_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        team_matches[f"team_goals_against_avg_last{window}"] = (
            grouped["goals_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    return team_matches.rename(columns={"event": "round"})


def add_dynamic_opponent_form(df: pd.DataFrame, fixtures_full: pd.DataFrame, team_form: pd.DataFrame) -> pd.DataFrame:
    """
    Attaches CURRENT-form goals for/against for both the player's own
    team (attacking momentum) and their opponent (defensive vulnerability
    right now, not a static preseason rating).
    """
    if team_form.empty or fixtures_full.empty:
        return df

    # figure out the player's own team id for this fixture
    fixture_teams = fixtures_full[["season", "fixture", "team_h", "team_a"]].drop_duplicates(
        subset=["season", "fixture"], keep="last"
    )
    df = df.merge(fixture_teams, on=["season", "fixture"], how="left")
    df["player_team_id"] = np.where(df["was_home"], df["team_h"], df["team_a"])

    team_form_cols = [c for c in team_form.columns if c.startswith("team_goals_")]

    # Guard against fan-out: if a team has more than one row for the same
    # (season, round) — e.g. rescheduled/double-gameweek data quirks in the
    # source — collapse to one row per (season, round, team) so the merge
    # below stays many-to-one, not many-to-many.
    team_form_dedup = (
        team_form[["season", "round", "team"] + team_form_cols]
        .drop_duplicates(subset=["season", "round", "team"], keep="last")
    )

    # opponent's dynamic form
    opp_form = team_form_dedup.rename(
        columns={"team": "opponent_team", **{c: f"opp_{c}" for c in team_form_cols}}
    )
    df = df.merge(opp_form, on=["season", "round", "opponent_team"], how="left")

    # own team's dynamic form
    own_form = team_form_dedup.rename(
        columns={"team": "player_team_id", **{c: f"own_{c}" for c in team_form_cols}}
    )
    df = df.merge(own_form, on=["season", "round", "player_team_id"], how="left")

    return df.drop(columns=["team_h", "team_a"], errors="ignore")


def add_team_strength(df: pd.DataFrame, team_strength: pd.DataFrame) -> pd.DataFrame:
    if team_strength.empty:
        return df
    return df.merge(team_strength, on=["season", "opponent_team"], how="left")


def add_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling averages computed per player, WITHIN season, shifted by 1 GW."""
    df = df.sort_values(["player_id", "season", "round"]).reset_index(drop=True)

    stat_cols = [c for c in STAT_COLS if c in df.columns]
    grouped = df.groupby(["player_id", "season"])

    for window in ROLLING_WINDOWS:
        for col in stat_cols:
            df[f"{col}_avg_last{window}"] = (
                grouped[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    df["minutes_avg_season"] = grouped["minutes"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    return df


def add_fixture_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines team-strength-based proxies with the official FPL FDR
    (1-5 rating), oriented to the player's own team (was_home decides
    whether team_h_difficulty or team_a_difficulty applies to them).
    """
    if "was_home" not in df.columns:
        return df
    if "strength_attack_home" in df.columns:
        df["home_strength"] = np.where(
            df["was_home"], df["strength_attack_home"], df["strength_attack_away"]
        )
        df["opponent_defence_strength"] = np.where(
            df["was_home"], df["strength_defence_away"], df["strength_defence_home"]
        )
    if "team_h_difficulty" in df.columns:
        df["fixture_difficulty"] = np.where(
            df["was_home"], df["team_h_difficulty"], df["team_a_difficulty"]
        )
    return df


def add_rotation_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains the minutes/rotation-risk model inline (on the same
    leakage-safe rolling features) and attaches its predicted
    P(starts >= 60 mins) as a feature for the points model.
    """
    import xgboost as xgb

    df = df.copy()
    df["started"] = (df["minutes"] >= 60).astype(int)

    feature_cols = ["starts_avg_last3", "starts_avg_last5",
                     "minutes_avg_last3", "minutes_avg_last5", "minutes_avg_season",
                     "value", "position", "was_home"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    fit_rows = df[~df["is_first_gw_for_player"]].copy()
    fit_rows["position"] = fit_rows["position"].astype("category")

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="binary:logistic", enable_categorical=True,
    )
    model.fit(fit_rows[feature_cols], fit_rows["started"])

    all_rows = df.copy()
    all_rows["position"] = all_rows["position"].astype("category")
    df["rotation_risk_prob"] = model.predict_proba(all_rows[feature_cols])[:, 1]

    return df.drop(columns=["started"])


def build():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading combined historical gameweek data...")
    df = load_merged()
    original_row_count = len(df)
    print(f"  {len(df)} rows across seasons: {sorted(df['season'].unique())}")

    print("Loading team strength ratings...")
    team_strength = load_team_strength()
    df = add_team_strength(df, team_strength)

    print("Loading official fixture difficulty (FDR) ratings...")
    fixture_diff = load_fixture_difficulty()
    if not fixture_diff.empty:
        df = df.merge(fixture_diff, on=["season", "fixture"], how="left")

    print("Computing dynamic team form (current goals for/against, not static ratings)...")
    fixtures_full = load_fixtures_full()
    team_form = build_team_form(fixtures_full)
    df = add_dynamic_opponent_form(df, fixtures_full, team_form)

    print("Computing rolling form features (leakage-safe, within-season)...")
    df = add_rolling_form(df)

    print("Adding fixture difficulty proxies...")
    df = add_fixture_difficulty(df)

    df["is_first_gw_for_player"] = (
        df.groupby(["player_id", "season"])["round"].transform("min") == df["round"]
    )

    print("Training rotation-risk sub-model and attaching predictions...")
    df = add_rotation_risk(df)

    # Sanity check: merges above must never change the row count. If this
    # fires, a merge fanned out (duplicate keys on one side) and silently
    # corrupted the dataset — investigate before trusting anything downstream.
    if len(df) != original_row_count:
        raise ValueError(
            f"Row count changed from {original_row_count} to {len(df)} during feature "
            f"building — a merge likely duplicated rows. Do not proceed with training on this output."
        )

    out_path = PROCESSED_DIR / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns -> {out_path}")


if __name__ == "__main__":
    build()
