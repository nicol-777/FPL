"""
Fetch live data from the official FPL API.

No auth required. Pulls:
  - bootstrap-static: all players, teams, and gameweek (event) metadata
  - fixtures: full season fixture list with difficulty ratings
  - element-summary/{id}: per-gameweek history for a single player

Run:
    python src/data/fetch_fpl_api.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_bootstrap_static() -> dict:
    """Pulls the main FPL dataset: players, teams, positions, gameweeks."""
    resp = requests.get(f"{BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    return resp.json()


def fetch_fixtures() -> list:
    """Pulls the full season fixture list, including FDR (fixture difficulty)."""
    resp = requests.get(f"{BASE_URL}/fixtures/")
    resp.raise_for_status()
    return resp.json()


def fetch_player_history(player_id: int) -> dict:
    """Pulls one player's full gameweek-by-gameweek history."""
    resp = requests.get(f"{BASE_URL}/element-summary/{player_id}/")
    resp.raise_for_status()
    return resp.json()


def save_bootstrap_static(data: dict, out_dir: Path) -> None:
    pd.DataFrame(data["elements"]).to_csv(out_dir / "players.csv", index=False)
    pd.DataFrame(data["teams"]).to_csv(out_dir / "teams.csv", index=False)
    pd.DataFrame(data["events"]).to_csv(out_dir / "gameweeks.csv", index=False)
    pd.DataFrame(data["element_types"]).to_csv(out_dir / "positions.csv", index=False)
    print(f"Saved players, teams, gameweeks, positions -> {out_dir}")


def save_fixtures(fixtures: list, out_dir: Path) -> None:
    pd.DataFrame(fixtures).to_csv(out_dir / "fixtures.csv", index=False)
    print(f"Saved fixtures -> {out_dir}")


def fetch_all_player_histories(player_ids: list[int], out_dir: Path, pause: float = 0.3) -> None:
    """
    Pulls gameweek history for every player. Rate-limited with a small
    pause between requests since this is one call per player.
    """
    all_gw_rows = []
    all_history_past = []

    for i, pid in enumerate(player_ids, start=1):
        try:
            data = fetch_player_history(pid)
        except requests.RequestException as e:
            print(f"  skipped player {pid}: {e}")
            continue

        for row in data.get("history", []):
            row["player_id"] = pid
            all_gw_rows.append(row)

        for row in data.get("history_past", []):
            row["player_id"] = pid
            all_history_past.append(row)

        if i % 50 == 0:
            print(f"  fetched {i}/{len(player_ids)} players")

        time.sleep(pause)

    pd.DataFrame(all_gw_rows).to_csv(out_dir / "player_gameweek_history.csv", index=False)
    pd.DataFrame(all_history_past).to_csv(out_dir / "player_season_history.csv", index=False)
    print(f"Saved per-player gameweek + past-season history -> {out_dir}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching bootstrap-static...")
    bootstrap = fetch_bootstrap_static()
    save_bootstrap_static(bootstrap, RAW_DIR)

    print("Fetching fixtures...")
    fixtures = fetch_fixtures()
    save_fixtures(fixtures, RAW_DIR)

    player_ids = [p["id"] for p in bootstrap["elements"]]
    print(f"Fetching gameweek history for {len(player_ids)} players (this takes a few minutes)...")
    fetch_all_player_histories(player_ids, RAW_DIR)

    print("Done.")


if __name__ == "__main__":
    main()