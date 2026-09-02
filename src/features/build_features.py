"""
Build a training-ready feature set from the raw FPL data.

For every (player, gameweek) row, builds features using ONLY information
available before that gameweek was played — rolling form windows are
shifted by one gameweek to avoid leaking the target into the features.

Output: data/processed/features.csv
    One row per player per gameweek they played (or were eligible for),
    with the target column `total_points` (that gameweek's actual score).

Run:
    python src/features/build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

ROLLING_WINDOWS = [3, 5]  # gameweeks of trailing form to compute


def load_raw():
    history = pd.read_csv(RAW_DIR / "player_gameweek_history.csv")
    players = pd.read_csv(RAW_DIR / "players.csv")
    teams = pd.read_csv(RAW_DIR / "teams.csv")
    positions = pd.read_csv(RAW_DIR / "positions.csv")
    return history, players, teams, positions


def add_static_player_info(history: pd.DataFrame, players: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Attach position and current team to each history row."""
    player_info = players[["id", "element_type", "team", "web_name"]].rename(
        columns={"id": "player_id", "element_type": "position_id"}
    )
    history = history.merge(player_info, on="player_id", how="left")

    pos_info = positions[["id", "singular_name_short"]].rename(
        columns={"id": "position_id", "singular_name_short": "position"}
    )
    history = history.merge(pos_info, on="position_id", how="left")
    return history


def add_team_strength(history: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Attach the opponent's overall attack/defence strength ratings."""
    team_strength = teams[
        ["id", "strength_overall_home", "strength_overall_away",
         "strength_attack_home", "strength_attack_away",
         "strength_defence_home", "strength_defence_away"]
    ].rename(columns={"id": "opponent_team"})
    history = history.merge(team_strength, on="opponent_team", how="left")
    return history


def add_rolling_form(history: pd.DataFrame) -> pd.DataFrame:
    """
    Computes trailing rolling averages per player, shifted by 1 gameweek
    so the current gameweek's own result is never included in its features.
    """
    history = history.sort_values(["player_id", "round"]).reset_index(drop=True)

    stat_cols = ["total_points", "minutes", "goals_scored", "assists",
                 "clean_sheets", "ict_index", "bps", "influence",
                 "creativity", "threat"]
    stat_cols = [c for c in stat_cols if c in history.columns]

    grouped = history.groupby("player_id")

    for window in ROLLING_WINDOWS:
        for col in stat_cols:
            history[f"{col}_avg_last{window}"] = (
                grouped[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    # career-to-date average minutes as a simple "starter reliability" signal
    history["minutes_avg_season"] = grouped["minutes"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    return history


def add_fixture_difficulty(history: pd.DataFrame) -> pd.DataFrame:
    """
    Uses FPL's own difficulty rating for this fixture if present in the
    history export (element-summary rows include `was_home`); combine
    with opponent strength as a proxy for fixture difficulty.
    """
    if "was_home" in history.columns:
        history["home_strength"] = np.where(
            history["was_home"],
            history["strength_attack_home"],
            history["strength_attack_away"],
        )
        history["opponent_defence_strength"] = np.where(
            history["was_home"],
            history["strength_defence_away"],
            history["strength_defence_home"],
        )
    return history


def build():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    history, players, teams, positions = load_raw()

    print("Attaching player position/team info...")
    history = add_static_player_info(history, players, positions)

    print("Attaching opponent team strength...")
    history = add_team_strength(history, teams)

    print("Computing rolling form features (leakage-safe)...")
    history = add_rolling_form(history)

    print("Adding fixture difficulty proxies...")
    history = add_fixture_difficulty(history)

    # Drop rows with no minutes history yet (first gameweek of the season
    # per player has no prior form — leave a raw baseline row but flag it)
    history["is_first_gw_for_player"] = history.groupby("player_id")["round"].transform("min") == history["round"]

    out_path = PROCESSED_DIR / "features.csv"
    history.to_csv(out_path, index=False)
    print(f"Saved {len(history)} rows, {history.shape[1]} columns -> {out_path}")


if __name__ == "__main__":
    build()