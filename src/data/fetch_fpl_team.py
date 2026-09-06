"""
Pulls your current FPL squad automatically using just your team (entry) ID
— no login required. Uses the PUBLIC picks endpoint.

Find your team ID: log into FPL, go to "Points", the URL is
fantasy.premierleague.com/entry/<YOUR_ID>/event/<gw>

NOTE on selling prices: the public endpoint doesn't expose your exact
per-player purchase price (that requires a logged-in session). By default
this treats your selling price as the CURRENT market price for each
player, which is a reasonable approximation but can overstate what you'd
actually get back — FPL's real rule is you only keep HALF of any profit,
rounded down to the nearest £0.1m. If you want exact selling prices,
supply a CSV via --purchase-prices with columns: player_id, purchase_price
(check your FPL app/site for what you paid for each player).

Run:
    python src/data/fetch_fpl_team.py --team-id 1234567 --gw 3
    python src/data/fetch_fpl_team.py --team-id 1234567 --gw 3 --purchase-prices my_prices.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://fantasy.premierleague.com/api"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def fetch_picks(team_id: int, gw: int) -> dict:
    resp = requests.get(f"{BASE_URL}/entry/{team_id}/event/{gw}/picks/")
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap() -> dict:
    resp = requests.get(f"{BASE_URL}/bootstrap-static/")
    resp.raise_for_status()
    return resp.json()


def estimate_selling_price(current_price: float, purchase_price: float | None) -> float:
    """
    FPL's real rule: if the player has risen in value, you keep the
    purchase price PLUS half the profit, rounded down to the nearest 0.1.
    If they've fallen, you just get the current (lower) price back.
    """
    if purchase_price is None:
        return current_price  # approximation — no purchase data available
    if current_price <= purchase_price:
        return current_price
    profit = current_price - purchase_price
    kept_profit = np.floor((profit / 2) * 10) / 10
    return round(purchase_price + kept_profit, 1)


def build_squad_df(picks_data: dict, bootstrap: dict, purchase_prices: pd.DataFrame | None) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])[["id", "name"]].rename(columns={"id": "team_id", "name": "team_name"})
    positions = pd.DataFrame(bootstrap["element_types"])[["id", "singular_name_short"]].rename(
        columns={"id": "position_id", "singular_name_short": "position"}
    )

    players = players.merge(teams, left_on="team", right_on="team_id", how="left")
    players = players.merge(positions, left_on="element_type", right_on="position_id", how="left")
    players["current_price"] = players["now_cost"] / 10.0

    # The picks JSON has its OWN "position" field (squad slot 1-15), which
    # would silently collide with the player's actual position (GK/DEF/MID/FWD)
    # during the merge below — rename it out of the way first.
    picks = pd.DataFrame(picks_data["picks"]).rename(columns={"position": "squad_slot"})
    squad = picks.merge(
        players[["id", "web_name", "team_name", "position", "current_price"]],
        left_on="element", right_on="id", how="left",
    )
    squad = squad.rename(columns={"element": "player_id", "web_name": "name", "team_name": "team"})
    squad["is_starting"] = squad["multiplier"] > 0

    if purchase_prices is not None:
        squad = squad.merge(purchase_prices, on="player_id", how="left")
        squad["selling_price"] = squad.apply(
            lambda r: estimate_selling_price(r["current_price"], r.get("purchase_price")), axis=1
        )
    else:
        squad["selling_price"] = squad["current_price"]  # approximation

    bank = picks_data.get("entry_history", {}).get("bank", 0) / 10.0
    squad_value = picks_data.get("entry_history", {}).get("value", 0) / 10.0

    return squad[[
        "player_id", "name", "team", "position", "current_price", "selling_price",
        "is_captain", "is_vice_captain", "is_starting",
    ]], bank, squad_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-id", type=int, required=True)
    parser.add_argument("--gw", type=int, required=True, help="Gameweek to pull your picks from (usually the most recently completed one)")
    parser.add_argument("--purchase-prices", type=str, default=None,
                         help="Optional CSV with columns: player_id, purchase_price — for exact selling prices")
    args = parser.parse_args()

    print(f"Fetching picks for team {args.team_id}, gameweek {args.gw}...")
    picks_data = fetch_picks(args.team_id, args.gw)

    print("Fetching current player prices...")
    bootstrap = fetch_bootstrap()

    purchase_prices = None
    if args.purchase_prices:
        purchase_prices = pd.read_csv(args.purchase_prices)
        print(f"Loaded {len(purchase_prices)} purchase prices for accurate selling-price calculation.")
    else:
        print("No purchase-prices file given — selling prices will be APPROXIMATED as current market price.")

    squad_df, bank, squad_value = build_squad_df(picks_data, bootstrap, purchase_prices)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "my_squad.csv"
    squad_df.to_csv(out_path, index=False)

    print(f"\nSquad value: £{squad_value:.1f}m | Bank: £{bank:.1f}m | Total budget available: £{squad_value + bank:.1f}m")
    print(f"Saved squad -> {out_path}")
    print(squad_df.to_string(index=False))


if __name__ == "__main__":
    main()