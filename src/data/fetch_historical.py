"""
Fetch full gameweek-by-gameweek history for past seasons from the
vaastav/Fantasy-Premier-League public archive on GitHub.

Unlike the live FPL API (which only has this season's gameweeks so far),
this gives full completed seasons to train on immediately.

Run:
    python src/data/fetch_historical.py
"""

from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "historical"

SEASONS = ["2024-25", "2025-26"]

FILES_PER_SEASON = {
    "merged_gw.csv": "gws/merged_gw.csv",   # gw-by-gw stats, every player
    "players_raw.csv": "players_raw.csv",   # player overview for that season
    "teams.csv": "teams.csv",               # team strength ratings for that season
    "fixtures.csv": "fixtures.csv",         # per-fixture FDR (difficulty) ratings
}


def fetch_csv(url: str) -> pd.DataFrame:
    resp = requests.get(url)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def fetch_season(season: str, out_dir: Path) -> None:
    season_dir = out_dir / season
    season_dir.mkdir(parents=True, exist_ok=True)

    for local_name, remote_path in FILES_PER_SEASON.items():
        url = f"{BASE_URL}/{season}/{remote_path}"
        print(f"  fetching {url}")
        try:
            df = fetch_csv(url)
        except requests.RequestException as e:
            print(f"    skipped {local_name} for {season}: {e}")
            continue
        df.to_csv(season_dir / local_name, index=False)
        print(f"    saved {local_name} ({len(df)} rows)")


def combine_seasons(out_dir: Path) -> None:
    """Stack every season's merged_gw.csv into one file, tagged by season."""
    frames = []
    for season in SEASONS:
        path = out_dir / season / "merged_gw.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["season"] = season
        frames.append(df)

    if not frames:
        print("No season data found to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined_path = out_dir / "merged_gw_all_seasons.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined {len(frames)} seasons -> {combined_path} ({len(combined)} total rows)")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        print(f"Fetching {season}...")
        fetch_season(season, RAW_DIR)

    combine_seasons(RAW_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()