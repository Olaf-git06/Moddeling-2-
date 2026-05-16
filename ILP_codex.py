"""ILP implementation for the ambulance maximum coverage problem.

This file implements the Chapter 2 ILP from the thesis:

    max  sum_i d_i y_i
    s.t. sum_j x_j = p
         y_i <= sum_j a_ji x_j      for all demand locations i
         x_j, y_i in {0, 1}

where a_ji = 1 when base location j can reach demand location i within
15 minutes, and 0 otherwise.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pulp


SECONDS_IN_15_MINUTES = 15 * 60

DEFAULT_DATASETS = [
    "Data assignment ambulance 2 Small(1).xlsx",
    "Data assignment ambulance 2 Large.xlsx",
]


@dataclass(frozen=True)
class AmbulanceData:
    """Data needed by the ILP."""

    file_path: Path
    region: str
    p: int
    demand_locations: List[int]
    candidate_bases: List[int]
    demand: Dict[int, float]
    travel_time_seconds: Dict[int, Dict[int, float]]


@dataclass(frozen=True)
class ILPResult:
    """Clean result object returned after solving the ILP."""

    file_path: Path
    region: str
    status: str
    solve_time_seconds: float
    p: int
    selected_bases: List[int]
    covered_locations: List[int]
    total_demand: float
    covered_demand: float
    coverage_rate: float
    average_response_seconds: Optional[float]
    demand_weighted_response_seconds: Optional[float]
    objective_value: float


def _clean_postal_code(value: object) -> int:
    """Convert Excel values like 1117.0 or '1117' into integer postal codes."""

    if pd.isna(value):
        raise ValueError("Missing postal code in Excel input.")
    return int(float(value))


def read_ambulance_data(file_path: str | Path) -> AmbulanceData:
    """Read p, travel times, and demand from the assignment Excel workbook."""

    path = Path(file_path)

    info = pd.read_excel(path, sheet_name="General Information")
    first_column = info.iloc[:, 0].astype(str).str.strip().str.lower()
    p_rows = info.loc[first_column == "p"]
    if p_rows.empty:
        raise ValueError(f"Could not find p in the General Information sheet of {path}.")

    region = str(info.columns[1]).strip()
    p = int(p_rows.iloc[0, 1])

    demand_df = pd.read_excel(path, sheet_name="Demand")
    demand = {
        _clean_postal_code(row.iloc[0]): float(row.iloc[1])
        for _, row in demand_df.iterrows()
    }

    raw_times = pd.read_excel(path, sheet_name="Traveltimes (seconds)", header=None)
    demand_locations = [_clean_postal_code(v) for v in raw_times.iloc[1, 1:].tolist()]
    candidate_bases = [_clean_postal_code(v) for v in raw_times.iloc[2:, 0].tolist()]

    travel_time_seconds: Dict[int, Dict[int, float]] = {}
    for row_offset, base in enumerate(candidate_bases, start=2):
        travel_time_seconds[base] = {}
        for col_offset, location in enumerate(demand_locations, start=1):
            value = raw_times.iat[row_offset, col_offset]
            if pd.isna(value):
                time_value = 0.0 if base == location else math.inf
            else:
                time_value = float(value)
            travel_time_seconds[base][location] = time_value

    missing_demands = sorted(set(demand_locations) - set(demand))
    if missing_demands:
        raise ValueError(
            "Demand sheet is missing demand values for these locations: "
            f"{missing_demands[:10]}"
        )

    return AmbulanceData(
        file_path=path,
        region=region,
        p=p,
        demand_locations=demand_locations,
        candidate_bases=candidate_bases,
        demand=demand,
        travel_time_seconds=travel_time_seconds,
    )


def build_coverage_matrix(
    data: AmbulanceData,
    max_response_seconds: int = SECONDS_IN_15_MINUTES,
) -> Dict[int, Dict[int, int]]:
    """Build a_ji: 1 if base j reaches demand location i in time, else 0."""

    return {
        base: {
            location: int(data.travel_time_seconds[base][location] <= max_response_seconds)
            for location in data.demand_locations
        }
        for base in data.candidate_bases
    }


def solve_maximum_coverage_ilp(
    data: AmbulanceData,
    max_response_seconds: int = SECONDS_IN_15_MINUTES,
    p: Optional[int] = None,
    solver_msg: bool = False,
    time_limit_seconds: Optional[int] = None,
) -> ILPResult:
    """Solve the thesis ILP using PuLP."""

    p_to_place = data.p if p is None else p
    if p_to_place < 0:
        raise ValueError("p must be nonnegative.")
    if p_to_place > len(data.candidate_bases):
        raise ValueError(
            f"p={p_to_place} is larger than the number of candidate bases "
            f"({len(data.candidate_bases)})."
        )

    I = data.demand_locations
    J = data.candidate_bases
    d = data.demand
    a = build_coverage_matrix(data, max_response_seconds=max_response_seconds)

    model = pulp.LpProblem("Ambulance_Maximum_Coverage", pulp.LpMaximize)

    # x_j = 1 when an ambulance/base is selected at candidate location j.
    x = pulp.LpVariable.dicts("x", J, lowBound=0, upBound=1, cat=pulp.LpBinary)

    # y_i = 1 when demand location i is covered by at least one selected base.
    y = pulp.LpVariable.dicts("y", I, lowBound=0, upBound=1, cat=pulp.LpBinary)

    model += pulp.lpSum(d[i] * y[i] for i in I), "Total_covered_demand"
    model += pulp.lpSum(x[j] for j in J) == p_to_place, "Choose_exactly_p_bases"

    for i in I:
        model += (
            y[i] <= pulp.lpSum(a[j][i] * x[j] for j in J),
            f"Cover_location_{i}",
        )

    solver = pulp.PULP_CBC_CMD(msg=solver_msg, timeLimit=time_limit_seconds)
    start_time = time.perf_counter()
    model.solve(solver)
    solve_time = time.perf_counter() - start_time

    selected_bases = sorted(j for j in J if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5)

    # Use the selected bases to report actual coverage, including zero-demand locations.
    covered_locations = sorted(
        i
        for i in I
        if any(a[j][i] == 1 for j in selected_bases)
    )

    total_demand = sum(d[i] for i in I)
    covered_demand = sum(d[i] for i in covered_locations)
    coverage_rate = covered_demand / total_demand if total_demand else 0.0
    objective_value = float(pulp.value(model.objective) or 0.0)

    response_times = _nearest_response_times(
        covered_locations=covered_locations,
        selected_bases=selected_bases,
        travel_time_seconds=data.travel_time_seconds,
    )

    average_response = (
        sum(response_times.values()) / len(response_times) if response_times else None
    )
    weighted_denominator = sum(d[i] for i in response_times)
    demand_weighted_response = (
        sum(d[i] * response_times[i] for i in response_times) / weighted_denominator
        if weighted_denominator
        else None
    )

    return ILPResult(
        file_path=data.file_path,
        region=data.region,
        status=pulp.LpStatus[model.status],
        solve_time_seconds=solve_time,
        p=p_to_place,
        selected_bases=selected_bases,
        covered_locations=covered_locations,
        total_demand=total_demand,
        covered_demand=covered_demand,
        coverage_rate=coverage_rate,
        average_response_seconds=average_response,
        demand_weighted_response_seconds=demand_weighted_response,
        objective_value=objective_value,
    )


def _nearest_response_times(
    covered_locations: Iterable[int],
    selected_bases: Iterable[int],
    travel_time_seconds: Dict[int, Dict[int, float]],
) -> Dict[int, float]:
    """For each covered location, find the fastest selected base."""

    selected = list(selected_bases)
    response_times: Dict[int, float] = {}
    for location in covered_locations:
        best_time = min(travel_time_seconds[base][location] for base in selected)
        if math.isfinite(best_time):
            response_times[location] = best_time
    return response_times


def print_result(result: ILPResult) -> None:
    """Pretty console output."""

    avg_minutes = _seconds_to_minutes(result.average_response_seconds)
    weighted_minutes = _seconds_to_minutes(result.demand_weighted_response_seconds)

    print("=" * 72)
    print(f"Dataset: {result.file_path.name}")
    print(f"Region: {result.region}")
    print(f"Solver status: {result.status}")
    print(f"p: {result.p}")
    print(f"Selected base locations: {result.selected_bases}")
    print(f"Covered demand: {result.covered_demand:,.0f} / {result.total_demand:,.0f}")
    print(f"Coverage rate: {100 * result.coverage_rate:.2f}%")
    print(f"Covered demand locations: {len(result.covered_locations)}")
    print(f"Objective value: {result.objective_value:,.0f}")
    print(f"Average response time for covered locations: {avg_minutes}")
    print(f"Demand-weighted response time for covered locations: {weighted_minutes}")
    print(f"Solve time: {result.solve_time_seconds:.3f} seconds")


def _seconds_to_minutes(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value / 60:.2f} minutes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the Chapter 2 ambulance maximum coverage ILP with PuLP."
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help=(
            "Excel workbook to solve. Can be passed multiple times. "
            "Defaults to the small and large assignment workbooks if present."
        ),
    )
    parser.add_argument(
        "--p",
        type=int,
        default=None,
        help="Override the number of bases/ambulances from the General Information sheet.",
    )
    parser.add_argument(
        "--max-response-seconds",
        type=int,
        default=SECONDS_IN_15_MINUTES,
        help="Coverage threshold in seconds. Default: 900 seconds = 15 minutes.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Optional CBC solver time limit in seconds.",
    )
    parser.add_argument(
        "--msg",
        action="store_true",
        help="Show solver output from CBC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.files or [name for name in DEFAULT_DATASETS if Path(name).exists()]
    if not files:
        raise FileNotFoundError(
            "No input files found. Pass an Excel workbook with --file."
        )

    for file_name in files:
        data = read_ambulance_data(file_name)
        result = solve_maximum_coverage_ilp(
            data,
            max_response_seconds=args.max_response_seconds,
            p=args.p,
            solver_msg=args.msg,
            time_limit_seconds=args.time_limit,
        )
        print_result(result)


if __name__ == "__main__":
    main()
