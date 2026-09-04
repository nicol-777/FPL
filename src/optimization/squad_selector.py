"""
Picks the optimal FPL squad (15 players + best starting XI) using integer
linear programming: maximize total predicted points subject to FPL's real
constraints.

Constraints enforced:
    - Total spend <= budget (default £100.0m)
    - Exactly 2 GK, 5 DEF, 5 MID, 3 FWD (15-man squad)
    - Max 3 players from any single club
    - Starting XI: 1 GK, valid formation (>=3 DEF, >=2 MID, >=1 FWD, 11 total)

Requires a DataFrame with columns:
    player_id, name, team, position (GK/DEF/MID/FWD), price, predicted_points

Run standalone for a quick example:
    python src/optimization/squad_selector.py
"""

from dataclasses import dataclass, field

import pandas as pd
import pulp

POSITION_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
SQUAD_SIZE = 15
STARTING_XI_SIZE = 11
MIN_STARTING_DEF = 3
MIN_STARTING_MID = 2
MIN_STARTING_FWD = 1
MIN_STARTING_GK = 1


@dataclass
class SquadResult:
    squad: pd.DataFrame
    starting_xi: pd.DataFrame
    bench: pd.DataFrame
    total_cost: float
    predicted_points_starting_xi: float


def select_squad(
    players: pd.DataFrame,
    budget: float = 100.0,
    locked_player_ids: list | None = None,
    excluded_player_ids: list | None = None,
) -> SquadResult:
    """
    players: DataFrame with player_id, name, team, position, price, predicted_points
    budget: total budget in £m
    locked_player_ids: player_ids that MUST be included (e.g. players you already own
        and don't want touched)
    excluded_player_ids: player_ids that must NOT be included (e.g. injured, suspended)
    """
    locked_player_ids = locked_player_ids or []
    excluded_player_ids = excluded_player_ids or []

    df = players[~players["player_id"].isin(excluded_player_ids)].reset_index(drop=True)
    ids = df["player_id"].tolist()

    prob = pulp.LpProblem("fpl_squad_selection", pulp.LpMaximize)

    # squad membership + starting-XI membership are separate binary decisions
    in_squad = {i: pulp.LpVariable(f"squad_{i}", cat="Binary") for i in ids}
    in_xi = {i: pulp.LpVariable(f"xi_{i}", cat="Binary") for i in ids}

    row_by_id = df.set_index("player_id")

    # Objective: maximize predicted points of the STARTING XI only
    # (bench players contribute ~0 unless there's a forced substitution,
    # which this simple version doesn't model)
    prob += pulp.lpSum(
        in_xi[i] * row_by_id.loc[i, "predicted_points"] for i in ids
    )

    # a player can only start if they're in the squad
    for i in ids:
        prob += in_xi[i] <= in_squad[i]

    # squad size + budget
    prob += pulp.lpSum(in_squad[i] for i in ids) == SQUAD_SIZE
    prob += pulp.lpSum(in_squad[i] * row_by_id.loc[i, "price"] for i in ids) <= budget

    # position quotas for the 15-man squad
    for pos, quota in POSITION_QUOTAS.items():
        pos_ids = [i for i in ids if row_by_id.loc[i, "position"] == pos]
        prob += pulp.lpSum(in_squad[i] for i in pos_ids) == quota

    # max 3 per club
    for club in df["team"].unique():
        club_ids = [i for i in ids if row_by_id.loc[i, "team"] == club]
        prob += pulp.lpSum(in_squad[i] for i in club_ids) <= MAX_PER_CLUB

    # starting XI constraints
    prob += pulp.lpSum(in_xi[i] for i in ids) == STARTING_XI_SIZE
    gk_ids = [i for i in ids if row_by_id.loc[i, "position"] == "GK"]
    def_ids = [i for i in ids if row_by_id.loc[i, "position"] == "DEF"]
    mid_ids = [i for i in ids if row_by_id.loc[i, "position"] == "MID"]
    fwd_ids = [i for i in ids if row_by_id.loc[i, "position"] == "FWD"]

    prob += pulp.lpSum(in_xi[i] for i in gk_ids) == MIN_STARTING_GK
    prob += pulp.lpSum(in_xi[i] for i in def_ids) >= MIN_STARTING_DEF
    prob += pulp.lpSum(in_xi[i] for i in mid_ids) >= MIN_STARTING_MID
    prob += pulp.lpSum(in_xi[i] for i in fwd_ids) >= MIN_STARTING_FWD

    # locked players must be in the squad
    for i in locked_player_ids:
        if i in in_squad:
            prob += in_squad[i] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver did not find an optimal solution: {pulp.LpStatus[prob.status]}")

    squad_ids = [i for i in ids if in_squad[i].value() == 1]
    xi_ids = [i for i in ids if in_xi[i].value() == 1]

    squad_df = df[df["player_id"].isin(squad_ids)].copy()
    xi_df = df[df["player_id"].isin(xi_ids)].copy()
    bench_df = squad_df[~squad_df["player_id"].isin(xi_ids)].copy()

    return SquadResult(
        squad=squad_df.sort_values(["position", "predicted_points"], ascending=[True, False]),
        starting_xi=xi_df.sort_values(["position", "predicted_points"], ascending=[True, False]),
        bench=bench_df.sort_values("predicted_points", ascending=False),
        total_cost=squad_df["price"].sum(),
        predicted_points_starting_xi=xi_df["predicted_points"].sum(),
    )


def print_result(result: SquadResult):
    print(f"\nTotal cost: £{result.total_cost:.1f}m")
    print(f"Predicted starting XI points: {result.predicted_points_starting_xi:.1f}")
    print("\n--- Starting XI ---")
    print(result.starting_xi[["name", "team", "position", "price", "predicted_points"]].to_string(index=False))
    print("\n--- Bench ---")
    print(result.bench[["name", "team", "position", "price", "predicted_points"]].to_string(index=False))


if __name__ == "__main__":
    # Minimal smoke-test example with fabricated data — real usage loads
    # predictions from the points model instead.
    example = pd.DataFrame([
        {"player_id": 1, "name": "Example GK1", "team": "A", "position": "GK", "price": 5.0, "predicted_points": 4.0},
        {"player_id": 2, "name": "Example GK2", "team": "B", "position": "GK", "price": 4.0, "predicted_points": 3.0},
    ])
    print("This module is meant to be imported with real prediction data.")
    print("Run predict_and_select.py (once built) for a full end-to-end example.")