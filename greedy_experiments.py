"""
Run and save greedy experiment scenarios.

Outputs:
  - experiment_results/greedy_experiments_<timestamp>.csv
  - experiment_results/greedy_experiments_<timestamp>.json
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

import greedy as greedy_mod
from data import (
    TimetableDataset,
    _build_class_demand_map,
    _build_consecutive_slots,
    _build_conflict_matrix,
    _build_demand_compatibility,
    _build_rooms_by_type,
    _build_slots_by_day,
    _build_subject_demand_map,
    _build_teacher_demand_map,
    _compute_priorities,
    load_dataset,
)
from ga import _build_candidate_set, _build_room_index, fitness_function, hard_penalty, soft_penalty
from greedy import convert_to_chromosome, greedy_best_of_n, greedy_solve


@dataclass
class RunResult:
    scenario: str
    variant: str
    run_id: int
    seed: int | None
    hard: int
    soft: float
    fitness: float
    assigned: int
    total_demands: int
    unscheduled: int
    runtime_sec: float


def _rebuild_dataset_indexes(ds: TimetableDataset) -> TimetableDataset:
    ds.rooms_by_type = _build_rooms_by_type(ds.rooms)
    ds.slots_by_day = _build_slots_by_day(ds.timeslots)
    ds.consecutive_slots = _build_consecutive_slots(ds.slots_by_day)
    ds.teacher_demand_map = _build_teacher_demand_map(ds.demands)
    ds.class_demand_map = _build_class_demand_map(ds.demands)
    ds.subject_demand_map = _build_subject_demand_map(ds.demands)
    ds.demand_by_id = {d.id: d for d in ds.demands}
    compat_rooms, compat_slots = _build_demand_compatibility(
        ds.demands,
        ds.rooms_by_type,
        ds.consecutive_slots,
        ds.slots_by_day,
    )
    ds.demand_compatible_rooms = compat_rooms
    ds.demand_compatible_slots = compat_slots
    ds.conflict_matrix = _build_conflict_matrix(ds.demands, ds.class_demand_map)
    _compute_priorities(
        ds.demands,
        ds.demand_compatible_rooms,
        ds.conflict_matrix,
        ds.demand_compatible_slots,
    )
    return ds


def _clone_with_room_ratio(ds: TimetableDataset, ratio: float) -> TimetableDataset:
    cloned = deepcopy(ds)
    keep = max(1, int(len(cloned.rooms) * ratio))
    cloned.rooms = sorted(cloned.rooms, key=lambda r: r.id)[:keep]
    return _rebuild_dataset_indexes(cloned)


def _clone_with_slot_ratio(ds: TimetableDataset, ratio: float) -> TimetableDataset:
    cloned = deepcopy(ds)
    keep = max(1, int(len(cloned.timeslots) * ratio))
    cloned.timeslots = sorted(cloned.timeslots, key=lambda s: (s.day, s.period, s.id))[:keep]
    return _rebuild_dataset_indexes(cloned)


def _clone_with_teacher_ratio(ds: TimetableDataset, ratio: float) -> TimetableDataset:
    cloned = deepcopy(ds)
    for demand in cloned.demands:
        if not demand.candidate_teachers:
            continue
        keep = max(1, int(math.ceil(len(demand.candidate_teachers) * ratio)))
        demand.candidate_teachers = sorted(demand.candidate_teachers)[:keep]
    return _rebuild_dataset_indexes(cloned)


@contextmanager
def _ablation_mode(mode: str):
    original_slot = greedy_mod._sort_slot_groups
    original_teacher = greedy_mod._sort_teachers
    original_room = greedy_mod._sort_rooms
    try:
        if mode in {"no_slot_sort", "all_off"}:
            greedy_mod._sort_slot_groups = lambda slot_groups, demand, state, conflict_matrix, schedule: list(slot_groups)
        if mode in {"no_teacher_sort", "all_off"}:
            greedy_mod._sort_teachers = lambda candidate_ids, teachers, state, slot_group: list(candidate_ids)
        if mode in {"no_room_sort", "all_off"}:
            greedy_mod._sort_rooms = lambda rooms, demand: list(rooms)
        yield
    finally:
        greedy_mod._sort_slot_groups = original_slot
        greedy_mod._sort_teachers = original_teacher
        greedy_mod._sort_rooms = original_room


def _evaluate_schedule(schedule: dict, ds: TimetableDataset, runtime_sec: float) -> tuple[int, float, float, int, int]:
    chrom = convert_to_chromosome(schedule, ds)
    room_index = _build_room_index(ds)
    candidate_set = _build_candidate_set(ds)
    hard = hard_penalty(chrom, ds, room_index=room_index, candidate_set=candidate_set)
    soft = soft_penalty(chrom, ds)
    fit, _, _ = fitness_function(chrom, ds, room_index=room_index, candidate_set=candidate_set)
    assigned = len(schedule)
    total = len(ds.demands)
    return hard, soft, fit, assigned, total


def _run_once(
    scenario: str,
    variant: str,
    run_id: int,
    ds: TimetableDataset,
    seed: int | None,
    runner: Callable[[TimetableDataset], tuple[dict, list[str]]],
) -> RunResult:
    if seed is not None:
        random.seed(seed)
    t0 = time.perf_counter()
    schedule, unscheduled = runner(ds)
    elapsed = time.perf_counter() - t0
    hard, soft, fit, assigned, total = _evaluate_schedule(schedule, ds, elapsed)
    return RunResult(
        scenario=scenario,
        variant=variant,
        run_id=run_id,
        seed=seed,
        hard=hard,
        soft=soft,
        fitness=fit,
        assigned=assigned,
        total_demands=total,
        unscheduled=len(unscheduled),
        runtime_sec=elapsed,
    )


def _aggregate(results: list[RunResult]) -> list[dict]:
    groups: dict[tuple[str, str], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.scenario, r.variant), []).append(r)

    rows: list[dict] = []
    for (scenario, variant), items in sorted(groups.items()):
        hard_vals = [x.hard for x in items]
        uns_vals = [x.unscheduled for x in items]
        soft_vals = [x.soft for x in items]
        fit_vals = [x.fitness for x in items]
        rt_vals = [x.runtime_sec for x in items]
        row = {
            "scenario": scenario,
            "variant": variant,
            "runs": len(items),
            "hard_mean": mean(hard_vals),
            "hard_std": pstdev(hard_vals) if len(hard_vals) > 1 else 0.0,
            "hard_best": min(hard_vals),
            "hard_worst": max(hard_vals),
            "unscheduled_mean": mean(uns_vals),
            "soft_mean": mean(soft_vals),
            "fitness_mean": mean(fit_vals),
            "runtime_mean_sec": mean(rt_vals),
        }
        rows.append(row)
    return rows


def run_all() -> tuple[list[RunResult], list[dict], Path, Path]:
    base_ds = load_dataset()
    results: list[RunResult] = []
    repeats = 5

    # S1 baseline
    results.append(
        _run_once(
            scenario="S1_baseline",
            variant="greedy_single_pass",
            run_id=1,
            ds=base_ds,
            seed=42,
            runner=lambda ds: greedy_solve(ds, verbose=False),
        )
    )

    # S2 multi-restart
    for n in [1, 3, 5, 10, 20]:
        for i in range(1, repeats + 1):
            seed = 1000 + n * 100 + i
            results.append(
                _run_once(
                    scenario="S2_multi_restart",
                    variant=f"restarts_{n}",
                    run_id=i,
                    ds=base_ds,
                    seed=seed,
                    runner=lambda ds, n=n: greedy_best_of_n(
                        ds, n=n, verbose=False, verbose_best=False
                    ),
                )
            )

    # S3 seed sensitivity (fixed restart=10)
    for i, seed in enumerate(range(1, 21), start=1):
        results.append(
            _run_once(
                scenario="S3_seed_effect",
                variant="restart_10",
                run_id=i,
                ds=base_ds,
                seed=seed,
                runner=lambda ds: greedy_best_of_n(
                    ds, n=10, verbose=False, verbose_best=False
                ),
            )
        )

    # S4 ablation heuristics
    for mode in ["full", "no_slot_sort", "no_teacher_sort", "no_room_sort", "all_off"]:
        with _ablation_mode(mode):
            results.append(
                _run_once(
                    scenario="S4_ablation",
                    variant=mode,
                    run_id=1,
                    ds=base_ds,
                    seed=42,
                    runner=lambda ds: greedy_solve(ds, verbose=False),
                )
            )

    # S5 demand-order noise
    ordered = base_ds.get_demands_sorted_by_priority()
    for noise in [0.0, 0.1, 0.2, 0.3, 0.5]:
        for i in range(1, repeats + 1):
            seed = 2000 + int(noise * 1000) + i

            def _runner(ds: TimetableDataset, noise=noise):
                demand_order = greedy_mod._partial_shuffle(ordered, noise_ratio=noise)
                return greedy_solve(ds, verbose=False, demand_order=demand_order)

            results.append(
                _run_once(
                    scenario="S5_order_noise",
                    variant=f"noise_{noise:.1f}",
                    run_id=i,
                    ds=base_ds,
                    seed=seed,
                    runner=_runner,
                )
            )

    # S6 room pressure
    for ratio in [0.9, 0.8, 0.7, 0.6]:
        ds_ratio = _clone_with_room_ratio(base_ds, ratio)
        results.append(
            _run_once(
                scenario="S6_room_pressure",
                variant=f"rooms_{int(ratio * 100)}pct",
                run_id=1,
                ds=ds_ratio,
                seed=42,
                runner=lambda ds: greedy_solve(ds, verbose=False),
            )
        )

    # S7 slot pressure
    for ratio in [1.0, 0.9, 0.8, 0.7]:
        ds_ratio = _clone_with_slot_ratio(base_ds, ratio)
        results.append(
            _run_once(
                scenario="S7_slot_pressure",
                variant=f"slots_{int(ratio * 100)}pct",
                run_id=1,
                ds=ds_ratio,
                seed=42,
                runner=lambda ds: greedy_solve(ds, verbose=False),
            )
        )

    # S8 candidate teacher pressure
    for ratio in [1.0, 0.8, 0.6, 0.4]:
        ds_ratio = _clone_with_teacher_ratio(base_ds, ratio)
        results.append(
            _run_once(
                scenario="S8_teacher_candidate_pressure",
                variant=f"cand_{int(ratio * 100)}pct",
                run_id=1,
                ds=ds_ratio,
                seed=42,
                runner=lambda ds: greedy_solve(ds, verbose=False),
            )
        )

    aggregate = _aggregate(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "experiment_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"greedy_experiments_{ts}.csv"
    json_path = out_dir / f"greedy_experiments_{ts}.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "variant",
                "run_id",
                "seed",
                "hard",
                "soft",
                "fitness",
                "assigned",
                "total_demands",
                "unscheduled",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(asdict(item))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "runs": [asdict(x) for x in results],
                "aggregate": aggregate,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return results, aggregate, csv_path, json_path


def main() -> None:
    results, aggregate, csv_path, json_path = run_all()
    print(f"Saved run-level CSV: {csv_path}")
    print(f"Saved aggregate JSON: {json_path}")
    print(f"Total runs: {len(results)}")
    print("Top variants by hard_mean:")
    for row in sorted(aggregate, key=lambda x: (x["hard_mean"], x["unscheduled_mean"]))[:10]:
        print(
            f"- {row['scenario']} | {row['variant']}: "
            f"hard_mean={row['hard_mean']:.2f}, unscheduled_mean={row['unscheduled_mean']:.2f}, "
            f"runtime_mean={row['runtime_mean_sec']:.3f}s"
        )


if __name__ == "__main__":
    main()
