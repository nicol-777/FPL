"""
Recommends transfers by comparing your current squad's expected points
against the best achievable alternatives, accounting for your bank
balance, selling prices, and the -4 point cost of any transfer beyond
your free transfer(s).

Searches 0, 1, and 2-transfer scenarios exhaustively (your squad is only
15 players, so this is fast) and recommends whichever nets the highest
predicted points after any hit is subtracted.

Requires:
    data/processed/my_squad.csv       (from fetch_fpl_team.py)
    data/processed/predictions.csv    (from predict.py)

Run:
    python src/optimization/transfer_planner.py --free-transfers 1
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from squad_selector import select_squad  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
HIT_COST = 4  # points lost per transfer beyond your free transfer(s)


def load_data():
    squad = pd.read_csv(PROCESSED_DIR / "my_squad.csv")
    predictions = pd.read_csv(PROCESSED_DIR / "predictions.csv")
    return squad, predictions


def best_xi_points(player_ids: list, predictions: pd.DataFrame) -> float:
    """Given a set of 15 player_ids, find the best-scoring valid starting XI among them."""
    pool = predictions[predictions["player_id"].isin(player_ids)].copy()
    if len(pool) != 15:
        return float("-inf")  # invalid squad (wrong size / missing predictions)
    try:
        result = select_squad(pool, budget=10_000, locked_player_ids=list(player_ids))
    except RuntimeError:
        return float("-inf")
    return result.predicted_points_starting_xi


def evaluate_transfer_set(
    current_ids: set, sell_ids: tuple, predictions: pd.DataFrame,
    squad_info: pd.DataFrame, budget_available: float,
):
    """
    Tries the best possible replacement(s) for the given sell_ids, same
    position, within budget. Returns (new_ids, points, cost_freed) or None
    if no valid replacement combination exists.
    """
    remaining_ids = current_ids - set(sell_ids)
    freed_cash = squad_info[squad_info["player_id"].isin(sell_ids)]["selling_price"].sum()
    available = budget_available + freed_cash

    sell_positions = squad_info[squad_info["player_id"].isin(sell_ids)]["position"].tolist()

    candidates = predictions[~predictions["player_id"].isin(current_ids)].copy()

    # Greedy per-slot replacement: for each position needed, take the
    # highest-predicted-points affordable player not already chosen.
    # (Simple and fast; a full joint ILP over replacements would be more
    # exhaustive but is unnecessary for 1-2 transfer searches.)
    chosen = []
    spent = 0.0
    for pos in sell_positions:
        pool = candidates[
            (candidates["position"] == pos) & (~candidates["player_id"].isin([c[0] for c in chosen]))
        ].sort_values("predicted_points", ascending=False)
        affordable = pool[pool["price"] <= (available - spent)]
        if affordable.empty:
            return None
        pick = affordable.iloc[0]
        chosen.append((pick["player_id"], pick["price"]))
        spent += pick["price"]

    new_ids = remaining_ids | {c[0] for c in chosen}
    points = best_xi_points(list(new_ids), predictions)
    return new_ids, points, spent


def plan_transfers(squad: pd.DataFrame, predictions: pd.DataFrame, free_transfers: int, max_transfers: int = 2):
    current_ids = set(squad["player_id"])
    bank = squad.attrs.get("bank", 0.0)

    baseline_points = best_xi_points(list(current_ids), predictions)
    print(f"Current squad's best predicted XI: {baseline_points:.2f} points (0 transfers)")

    results = [{"n_transfers": 0, "sell": [], "buy": [], "net_points": baseline_points, "raw_points": baseline_points}]

    for n in range(1, max_transfers + 1):
        hit = max(0, n - free_transfers) * HIT_COST
        for sell_ids in combinations(current_ids, n):
            outcome = evaluate_transfer_set(current_ids, sell_ids, predictions, squad, bank)
            if outcome is None:
                continue
            new_ids, points, spent = outcome
            if points == float("-inf"):
                continue
            net = points - hit
            bought_ids = new_ids - (current_ids - set(sell_ids))
            results.append({
                "n_transfers": n,
                "sell": [squad.set_index("player_id").loc[i, "name"] for i in sell_ids],
                "buy": [predictions.set_index("player_id").loc[i, "name"] for i in bought_ids],
                "net_points": net,
                "raw_points": points,
            })

    results_df = pd.DataFrame(results).sort_values("net_points", ascending=False)
    return results_df, baseline_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--free-transfers", type=int, default=1,
                         help="How many free transfers you currently have (check the FPL app)")
    parser.add_argument("--bank", type=float, default=None,
                         help="Override bank balance in £m (otherwise read from my_squad.csv if available)")
    args = parser.parse_args()

    print("Loading squad and predictions...")
    squad, predictions = load_data()

    if args.bank is not None:
        squad.attrs["bank"] = args.bank
    else:
        squad.attrs["bank"] = 0.0
        print("No --bank given — assuming £0.0m in the bank. Pass --bank X.X if you have money spare.")

    print(f"Planning with {args.free_transfers} free transfer(s), up to 2-transfer scenarios...")
    results_df, baseline = plan_transfers(squad, predictions, args.free_transfers)

    print("\n--- Top options ---")
    for _, row in results_df.head(5).iterrows():
        if row["n_transfers"] == 0:
            print(f"HOLD: {row['net_points']:.2f} predicted points (no changes)")
        else:
            hit_note = f" (includes -{max(0, row['n_transfers'] - args.free_transfers) * HIT_COST} hit)" \
                if row["n_transfers"] > args.free_transfers else ""
            print(f"{row['n_transfers']} transfer(s): OUT {row['sell']} -> IN {row['buy']} "
                  f"= {row['net_points']:.2f} net points{hit_note}")

    best = results_df.iloc[0]
    improvement = best["net_points"] - baseline
    if improvement > 0.5:
        print(f"\nRecommendation: make the top move above (+{improvement:.2f} points over holding).")
    else:
        print(f"\nRecommendation: HOLD — no transfer clearly beats your current squad this gameweek.")


if __name__ == "__main__":
    main()