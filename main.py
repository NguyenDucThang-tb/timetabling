"""
Full timetabling pipeline:
    Greedy -> GA -> Local Search -> Backtracking Repair
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import config
from backtracking_repair import repair_with_backtracking
from data import TimetableDataset, load_dataset
from ga import (
    Chromosome,
    _build_candidate_set,
    _build_room_index,
    fitness_function,
    plot_fitness,
    run_ga,
)
from greedy import convert_to_chromosome, greedy_solve
import local_search as ls


def _extract_ga_outputs(ga_out: tuple) -> tuple[Chromosome, list[float], list[float]]:
    """
    Normalize run_ga outputs across GA versions.

    Expected minimum structure:
        index 0: best chromosome
        index 2: best fitness history
        index 3: mean fitness history
    """
    if len(ga_out) < 4:
        raise ValueError(f"run_ga returned {len(ga_out)} values, expected at least 4")

    ga_best = ga_out[0]
    best_history = ga_out[2]
    mean_history = ga_out[3]
    return ga_best, best_history, mean_history


def _evaluate(chrom: Chromosome, ds: TimetableDataset) -> tuple[float, int, float]:
    room_index = _build_room_index(ds)
    candidate_set = _build_candidate_set(ds)
    return fitness_function(chrom, ds, room_index=room_index, candidate_set=candidate_set)


def _chromosome_to_local_schedule(chrom: Chromosome, ds: TimetableDataset) -> ls.Schedule:
    room_by_id = {r.id: r for r in ds.rooms}
    schedule = ls.Schedule()
    for did, asgn in chrom.items():
        if not asgn.is_assigned():
            continue
        room_obj = room_by_id.get(asgn.room_id)
        if room_obj is None:
            continue
        schedule.assignment[did] = {
            "teacher": asgn.teacher_id,
            "room": room_obj,
            "slots": list(asgn.slot_group),
        }
    return schedule


def _local_schedule_to_chromosome(schedule: ls.Schedule, ds: TimetableDataset) -> Chromosome:
    from ga import Assignment

    chrom: Chromosome = {}
    for d in ds.demands:
        did = d.id
        asgn = schedule.assignment.get(did)
        if asgn is None:
            chrom[did] = Assignment(None, None, None)
            continue

        room = asgn["room"]
        room_id = room.id if hasattr(room, "id") else str(room)
        slot_group = list(asgn["slots"]) if asgn.get("slots") else None
        chrom[did] = Assignment(
            teacher_id=asgn.get("teacher"),
            room_id=room_id,
            slot_group=slot_group,
        )
    return chrom


def _run_local_search_stage(ds: TimetableDataset, ga_best: Chromosome) -> tuple[Chromosome, float, int, float]:
    local_initial = _chromosome_to_local_schedule(ga_best, ds)
    local_best = ls.local_search(local_initial, ds)
    chrom_best = _local_schedule_to_chromosome(local_best, ds)
    score, hard, soft = _evaluate(chrom_best, ds)
    return chrom_best, score, hard, soft


def _serialize(chrom: Chromosome) -> dict[str, dict]:
    data: dict[str, dict] = {}
    for did, asgn in sorted(chrom.items(), key=lambda x: x[0]):
        if not asgn.is_assigned():
            data[did] = {
                "assigned": False,
                "teacher_id": None,
                "room_id": None,
                "slot_ids": [],
            }
            continue

        data[did] = {
            "assigned": True,
            "teacher_id": asgn.teacher_id,
            "room_id": asgn.room_id,
            "slot_ids": [s.id for s in asgn.slot_group],
        }
    return data


def _stage_log(name: str, score: float, hard: int, soft: float, chrom: Chromosome) -> None:
    assigned = sum(1 for a in chrom.values() if a.is_assigned())
    print(f"[{name}] fitness={score:.2f} | hard={hard} | soft={soft:.2f} | assigned={assigned}/{len(chrom)}")


def run_pipeline(
    data_dir: Optional[str] = None,
    save_output: bool = True,
    output_file: Optional[str] = None,
    plot: bool = False,
) -> dict:
    print("=" * 70)
    print("TIMETABLING FULL PIPELINE")
    print("=" * 70)

    ds = load_dataset(data_dir)
    print(ds.summary())

    print("\n[1/4] Greedy initialization")
    greedy_schedule, unscheduled = greedy_solve(ds, verbose=config.VERBOSE)
    greedy_chrom = convert_to_chromosome(greedy_schedule, ds)
    g_fit, g_hard, g_soft = _evaluate(greedy_chrom, ds)
    _stage_log("GREEDY", g_fit, g_hard, g_soft, greedy_chrom)
    if unscheduled:
        print(f"[GREEDY] unscheduled_demands={len(unscheduled)}")

    print("\n[2/4] Genetic Algorithm")
    ga_out = run_ga(ds, greedy_chrom)
    ga_best, best_history, mean_history = _extract_ga_outputs(ga_out)
    if ga_best is None:
        ga_best = greedy_chrom
    ga_fit_chk, ga_hard, ga_soft = _evaluate(ga_best, ds)
    _stage_log("GA", ga_fit_chk, ga_hard, ga_soft, ga_best)

    if plot and config.PLOT_FITNESS and best_history:
        plot_fitness(best_history, mean_history)

    print("\n[3/4] Local Search")
    ls_best, ls_fit, ls_hard, ls_soft = _run_local_search_stage(ds, ga_best)
    _stage_log("LOCAL_SEARCH", ls_fit, ls_hard, ls_soft, ls_best)

    print("\n[4/4] Backtracking Repair")
    repair_result = repair_with_backtracking(ds, ls_best)
    repaired = repair_result.schedule
    r_fit, r_hard, r_soft = _evaluate(repaired, ds)
    _stage_log("REPAIR", r_fit, r_hard, r_soft, repaired)
    print(
        f"[REPAIR] success={repair_result.success} | nodes={repair_result.nodes_visited} "
        f"| backtracks={repair_result.backtracks} | repaired_demands={repair_result.repaired_demands} "
        f"| time={repair_result.elapsed_seconds:.2f}s"
    )

    result = {
        "metrics": {
            "greedy": {"fitness": g_fit, "hard": g_hard, "soft": g_soft},
            "ga": {"fitness": ga_fit_chk, "hard": ga_hard, "soft": ga_soft},
            "local_search": {"fitness": ls_fit, "hard": ls_hard, "soft": ls_soft},
            "repair": {"fitness": r_fit, "hard": r_hard, "soft": r_soft},
            "repair_details": {
                "success": repair_result.success,
                "nodes_visited": repair_result.nodes_visited,
                "backtracks": repair_result.backtracks,
                "hard_before": repair_result.hard_before,
                "hard_after": repair_result.hard_after,
                "soft_before": repair_result.soft_before,
                "soft_after": repair_result.soft_after,
                "elapsed_seconds": repair_result.elapsed_seconds,
                "repaired_demands": repair_result.repaired_demands,
            },
        },
        "schedule": _serialize(repaired),
    }

    if save_output and getattr(config, "SAVE_RESULT", True):
        out = output_file or getattr(config, "OUTPUT_FILE", "final_schedule.json")
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parent / out_path
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[SAVE] Wrote final schedule to: {out_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full timetabling pipeline")
    parser.add_argument("--data", default=None, help="Path to data directory (CSV files)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbose logs")
    parser.add_argument("--no-save", action="store_true", help="Do not write output JSON file")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--plot", action="store_true", help="Plot GA fitness chart")
    args = parser.parse_args()

    config.VERBOSE = not args.quiet
    run_pipeline(
        data_dir=args.data,
        save_output=not args.no_save,
        output_file=args.output,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
