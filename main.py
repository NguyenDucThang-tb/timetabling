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
from local_search import Schedule as LocalSearchSchedule
from local_search import evaluate as evaluate_local_search
from local_search import solve as solve_local_search


def _evaluate(chrom: Chromosome, ds: TimetableDataset) -> tuple[float, int, float]:
    room_index = _build_room_index(ds)
    candidate_set = _build_candidate_set(ds)
    return fitness_function(chrom, ds, room_index=room_index, candidate_set=candidate_set)


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
    print(f"[{name}] fitness={score:.2f} | ga_hard={hard} | ga_soft={soft:.2f} | assigned={assigned}/{len(chrom)}")


def _stage_log_local(
    name: str,
    score: float,
    hard: int,
    soft: float,
    schedule: LocalSearchSchedule,
    ds: TimetableDataset,
) -> None:
    assigned = len(schedule.assignment)
    print(f"[{name}] fitness={score:.2f} | ga_hard={hard} | ga_soft={soft:.2f} | assigned={assigned}/{len(ds.demands)}")


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
    print("[metrics] All pipeline stage metrics below use GA fitness/hard/soft.")

    print("\n[1/4] Greedy initialization")
    greedy_schedule, unscheduled = greedy_solve(ds, verbose=config.VERBOSE)
    greedy_chrom = convert_to_chromosome(greedy_schedule, ds)
    g_fit, g_hard, g_soft = _evaluate(greedy_chrom, ds)
    _stage_log("GREEDY", g_fit, g_hard, g_soft, greedy_chrom)
    if unscheduled:
        print(f"[GREEDY] unscheduled_demands={len(unscheduled)}")

    print("\n[2/4] Genetic Algorithm")
    ga_best, ga_fit, best_history, mean_history = run_ga(ds, greedy_chrom)
    if ga_best is None:
        ga_best = greedy_chrom
    ga_fit_chk, ga_hard, ga_soft = _evaluate(ga_best, ds)
    _stage_log("GA", ga_fit_chk, ga_hard, ga_soft, ga_best)

    if plot and config.PLOT_FITNESS and best_history:
        plot_fitness(best_history, mean_history)

    print("\n[3/4] Local Search")
    ls_best = solve_local_search(ds, ga_schedule=ga_best)
    ls_fit, ls_hard, ls_soft = evaluate_local_search(ls_best, ds)
    _stage_log_local("LOCAL_SEARCH", ls_fit, ls_hard, ls_soft, ls_best, ds)

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
    print(
        f"[GA-HARD TREND] greedy={g_hard} -> ga={ga_hard} -> "
        f"local_search={ls_hard} -> repair={r_hard}"
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
