"""
Full pipeline evaluation runner + report asset generator.

It executes scenario matrix, saves run-level/aggregate results, and renders
tables/charts for report usage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["axes.unicode_minus"] = False

import config
import local_search as ls
from backtracking_repair import repair_with_backtracking
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
from ga import (
    Assignment,
    Chromosome,
    _build_candidate_set,
    _build_room_index,
    fitness_function,
    run_ga,
)
from greedy import convert_to_chromosome, greedy_solve


@dataclass
class RunRecord:
    scenario: str
    variant: str
    run_id: int
    seed: int
    profile: str
    use_ga: bool
    use_ls: bool
    use_repair: bool
    hard_greedy: int
    hard_ga: int
    hard_ls: int
    hard_repair: int
    soft_greedy: float
    soft_ga: float
    soft_ls: float
    soft_repair: float
    fit_greedy: float
    fit_ga: float
    fit_ls: float
    fit_repair: float
    unscheduled_greedy: int
    assigned_final: int
    total_demands: int
    runtime_greedy_sec: float
    runtime_ga_sec: float
    runtime_ls_sec: float
    runtime_repair_sec: float
    runtime_total_sec: float
    repair_success: bool
    repair_nodes: int
    repair_backtracks: int
    repair_repaired_demands: int


def evaluate_chromosome(chrom: Chromosome, ds: TimetableDataset) -> tuple[float, int, float]:
    room_index = _build_room_index(ds)
    candidate_set = _build_candidate_set(ds)
    return fitness_function(chrom, ds, room_index=room_index, candidate_set=candidate_set)


def chromosome_to_local_schedule(chrom: Chromosome, ds: TimetableDataset) -> ls.Schedule:
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


def local_schedule_to_chromosome(schedule: ls.Schedule, ds: TimetableDataset) -> Chromosome:
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


def rebuild_dataset_indexes(ds: TimetableDataset) -> TimetableDataset:
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


def clone_with_room_ratio(ds: TimetableDataset, ratio: float) -> TimetableDataset:
    cloned = deepcopy(ds)
    keep = max(1, int(len(cloned.rooms) * ratio))
    cloned.rooms = sorted(cloned.rooms, key=lambda r: r.id)[:keep]
    return rebuild_dataset_indexes(cloned)


def clone_with_slot_ratio(ds: TimetableDataset, ratio: float) -> TimetableDataset:
    cloned = deepcopy(ds)
    keep = max(1, int(len(cloned.timeslots) * ratio))
    cloned.timeslots = sorted(cloned.timeslots, key=lambda s: (s.day, s.period, s.id))[:keep]
    return rebuild_dataset_indexes(cloned)


def clone_with_teacher_ratio(ds: TimetableDataset, ratio: float) -> TimetableDataset:
    cloned = deepcopy(ds)
    for demand in cloned.demands:
        if not demand.candidate_teachers:
            continue
        keep = max(1, int(math.ceil(len(demand.candidate_teachers) * ratio)))
        demand.candidate_teachers = sorted(demand.candidate_teachers)[:keep]
    return rebuild_dataset_indexes(cloned)


FAST_PROFILE_OVERRIDES = {
    "POP_SIZE": 24,
    "GENERATIONS": 18,
    "ELITISM_COUNT": 3,
    "NO_IMPROVE_LIMIT": 10,
    "RESTART_THRESHOLD": 8,
    "LOCAL_SEARCH_ITERATIONS": 20,
    "NEIGHBOR_SAMPLE_SIZE": 10,
    "TABU_TENURE": 6,
    "MAX_REPAIR_STEPS": 120,
    "MAX_REPAIR_DEPTH": 10,
    "REPAIR_TIMEOUT_SECONDS": 5,
    "REPAIR_NO_IMPROVE_LIMIT": 60,
    "VERBOSE": False,
    "PLOT_FITNESS": False,
}


def stable_seed(namespace: str, scenario: str, variant: str, run_id: int, base: int) -> int:
    """
    Build deterministic seed independent of Python hash randomization.
    """
    raw = f"{namespace}|{scenario}|{variant}|{run_id}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return base + int(digest[:8], 16) % 900000


def parse_stress_levels(raw: str) -> list[int]:
    levels = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not levels:
        raise ValueError("stress-levels cannot be empty")
    for level in levels:
        if level <= 0 or level > 100:
            raise ValueError(f"Invalid stress level {level}; expected 1..100")
    return levels


@contextmanager
def config_overrides(overrides: dict[str, Any]):
    original: dict[str, Any] = {}
    for key, value in overrides.items():
        original[key] = getattr(config, key)
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(config, key, value)


def run_one_pipeline(
    ds: TimetableDataset,
    scenario: str,
    variant: str,
    run_id: int,
    seed: int,
    profile: str,
    use_ga: bool,
    use_ls: bool,
    use_repair: bool,
) -> RunRecord:
    random.seed(seed)
    np.random.seed(seed)
    config.RANDOM_SEED = seed

    total_t0 = time.perf_counter()

    # Stage 1: Greedy
    t0 = time.perf_counter()
    greedy_schedule, unscheduled = greedy_solve(ds, verbose=False)
    runtime_greedy = time.perf_counter() - t0
    greedy_chrom = convert_to_chromosome(greedy_schedule, ds)
    g_fit, g_hard, g_soft = evaluate_chromosome(greedy_chrom, ds)

    # Stage 2: GA
    ga_chrom = greedy_chrom
    ga_fit, ga_hard, ga_soft = g_fit, g_hard, g_soft
    runtime_ga = 0.0
    if use_ga:
        t0 = time.perf_counter()
        ga_out = run_ga(ds, greedy_chrom)
        runtime_ga = time.perf_counter() - t0
        ga_chrom = ga_out[0]
        ga_fit, ga_hard, ga_soft = evaluate_chromosome(ga_chrom, ds)

    # Stage 3: Local Search
    ls_chrom = ga_chrom
    ls_fit, ls_hard, ls_soft = ga_fit, ga_hard, ga_soft
    runtime_ls = 0.0
    if use_ls:
        t0 = time.perf_counter()
        local_initial = chromosome_to_local_schedule(ga_chrom, ds)
        local_best = ls.local_search(local_initial, ds)
        ls_chrom = local_schedule_to_chromosome(local_best, ds)
        ls_fit, ls_hard, ls_soft = evaluate_chromosome(ls_chrom, ds)
        runtime_ls = time.perf_counter() - t0

    # Stage 4: Repair
    final_chrom = ls_chrom
    r_fit, r_hard, r_soft = ls_fit, ls_hard, ls_soft
    runtime_repair = 0.0
    repair_success = False
    repair_nodes = 0
    repair_backtracks = 0
    repair_repaired_demands = 0
    if use_repair:
        t0 = time.perf_counter()
        repair_result = repair_with_backtracking(ds, ls_chrom)
        runtime_repair = time.perf_counter() - t0
        final_chrom = repair_result.schedule
        r_fit, r_hard, r_soft = evaluate_chromosome(final_chrom, ds)
        repair_success = bool(repair_result.success)
        repair_nodes = int(repair_result.nodes_visited)
        repair_backtracks = int(repair_result.backtracks)
        repair_repaired_demands = int(repair_result.repaired_demands)

    total_runtime = time.perf_counter() - total_t0
    assigned_final = sum(1 for a in final_chrom.values() if a.is_assigned())

    return RunRecord(
        scenario=scenario,
        variant=variant,
        run_id=run_id,
        seed=seed,
        profile=profile,
        use_ga=use_ga,
        use_ls=use_ls,
        use_repair=use_repair,
        hard_greedy=g_hard,
        hard_ga=ga_hard,
        hard_ls=ls_hard,
        hard_repair=r_hard,
        soft_greedy=g_soft,
        soft_ga=ga_soft,
        soft_ls=ls_soft,
        soft_repair=r_soft,
        fit_greedy=g_fit,
        fit_ga=ga_fit,
        fit_ls=ls_fit,
        fit_repair=r_fit,
        unscheduled_greedy=len(unscheduled),
        assigned_final=assigned_final,
        total_demands=len(ds.demands),
        runtime_greedy_sec=runtime_greedy,
        runtime_ga_sec=runtime_ga,
        runtime_ls_sec=runtime_ls,
        runtime_repair_sec=runtime_repair,
        runtime_total_sec=total_runtime,
        repair_success=repair_success,
        repair_nodes=repair_nodes,
        repair_backtracks=repair_backtracks,
        repair_repaired_demands=repair_repaired_demands,
    )


def aggregate_runs(records: list[RunRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.scenario, r.variant), []).append(r)

    out: list[dict[str, Any]] = []
    for (scenario, variant), items in sorted(groups.items()):
        row = {
            "scenario": scenario,
            "variant": variant,
            "runs": len(items),
            "hard_final_mean": mean([x.hard_repair for x in items]),
            "hard_final_std": pstdev([x.hard_repair for x in items]) if len(items) > 1 else 0.0,
            "hard_final_best": min(x.hard_repair for x in items),
            "hard_final_worst": max(x.hard_repair for x in items),
            "hard_greedy_mean": mean([x.hard_greedy for x in items]),
            "hard_ga_mean": mean([x.hard_ga for x in items]),
            "hard_ls_mean": mean([x.hard_ls for x in items]),
            "soft_final_mean": mean([x.soft_repair for x in items]),
            "runtime_total_mean_sec": mean([x.runtime_total_sec for x in items]),
            "runtime_ga_mean_sec": mean([x.runtime_ga_sec for x in items]),
            "runtime_ls_mean_sec": mean([x.runtime_ls_sec for x in items]),
            "runtime_repair_mean_sec": mean([x.runtime_repair_sec for x in items]),
            "repair_success_rate": mean([1.0 if x.repair_success else 0.0 for x in items]),
            "repair_repaired_demands_mean": mean([x.repair_repaired_demands for x in items]),
        }
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_charts(aggregate: list[dict[str, Any]], records: list[RunRecord], out_dir: Path, tag: str) -> list[Path]:
    outputs: list[Path] = []

    # Chart 1: hard by stage for core scenarios
    core_order = ["P1", "P2", "P3", "P4", "A1", "A2"]
    core = [r for r in aggregate if r["scenario"] in core_order]
    core.sort(key=lambda x: core_order.index(x["scenario"]))
    labels = [f"{x['scenario']}:{x['variant']}" for x in core]
    hg = [x["hard_greedy_mean"] for x in core]
    hga = [x["hard_ga_mean"] for x in core]
    hls = [x["hard_ls_mean"] for x in core]
    hf = [x["hard_final_mean"] for x in core]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.plot(x, hg, marker="o", label="Greedy")
    plt.plot(x, hga, marker="o", label="GA")
    plt.plot(x, hls, marker="o", label="Local Search")
    plt.plot(x, hf, marker="o", label="Final")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Hard penalty")
    plt.title("Hard Penalty Across Pipeline Stages")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    p = out_dir / f"pipeline_hard_by_stage_{tag}.png"
    plt.savefig(p, dpi=180)
    plt.close()
    outputs.append(p)

    # Chart 2: runtime breakdown for core scenarios
    plt.figure(figsize=(10, 5))
    rg = [mean([r.runtime_greedy_sec for r in records if r.scenario == c["scenario"] and r.variant == c["variant"]]) for c in core]
    rga = [mean([r.runtime_ga_sec for r in records if r.scenario == c["scenario"] and r.variant == c["variant"]]) for c in core]
    rls = [mean([r.runtime_ls_sec for r in records if r.scenario == c["scenario"] and r.variant == c["variant"]]) for c in core]
    rr = [mean([r.runtime_repair_sec for r in records if r.scenario == c["scenario"] and r.variant == c["variant"]]) for c in core]
    x = np.arange(len(labels))
    plt.bar(x, rg, label="Greedy")
    plt.bar(x, rga, bottom=rg, label="GA")
    bottom2 = np.array(rg) + np.array(rga)
    plt.bar(x, rls, bottom=bottom2, label="LS")
    bottom3 = bottom2 + np.array(rls)
    plt.bar(x, rr, bottom=bottom3, label="Repair")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Seconds")
    plt.title("Runtime Breakdown by Scenario")
    plt.legend()
    plt.tight_layout()
    p = out_dir / f"pipeline_runtime_breakdown_{tag}.png"
    plt.savefig(p, dpi=180)
    plt.close()
    outputs.append(p)

    # Chart 3: stress scenarios curve
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    stress_cfg = [
        ("S1_room_pressure", "room_", "Room Availability (%)", axes[0]),
        ("S2_slot_pressure", "slot_", "Slot Availability (%)", axes[1]),
        ("S3_teacher_pressure", "teacher_", "Teacher Candidate (%)", axes[2]),
    ]
    for scenario, prefix, xlabel, ax in stress_cfg:
        subset = [x for x in aggregate if x["scenario"] == scenario]
        subset.sort(key=lambda r: float(r["variant"].split("_")[-1].replace("pct", "")))
        x = [float(r["variant"].split("_")[-1].replace("pct", "")) for r in subset]
        h = [r["hard_final_mean"] for r in subset]
        rt = [r["runtime_total_mean_sec"] for r in subset]
        ax.plot(x, h, marker="o", label="Hard final")
        ax2 = ax.twinx()
        ax2.plot(x, rt, marker="s", color="#C1121F", label="Runtime")
        ax.set_xlabel(xlabel)
        ax.set_title(scenario)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Hard final")
    fig.suptitle("Stress Tests: Quality and Runtime")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / f"pipeline_stress_curves_{tag}.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    outputs.append(p)

    return outputs


def render_markdown(
    aggregate: list[dict[str, Any]],
    records: list[RunRecord],
    chart_paths: list[Path],
    out_md: Path,
    generated_at: str,
    profile: str,
) -> None:
    lines: list[str] = []
    lines.append("# Full Pipeline Evaluation")
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Profile: `{profile}`")
    lines.append("")
    lines.append("## Table B - Stage-wise Summary")
    lines.append("")
    lines.append("| Scenario | Variant | Runs | Hard(Greedy) | Hard(GA) | Hard(LS) | Hard(Final) | Soft(Final) | Runtime Total (s) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in aggregate:
        lines.append(
            f"| {r['scenario']} | {r['variant']} | {r['runs']} | "
            f"{r['hard_greedy_mean']:.2f} | {r['hard_ga_mean']:.2f} | {r['hard_ls_mean']:.2f} | "
            f"{r['hard_final_mean']:.2f} | {r['soft_final_mean']:.2f} | {r['runtime_total_mean_sec']:.2f} |"
        )
    lines.append("")
    lines.append("## Table C - Repair Effect")
    lines.append("")
    lines.append("| Scenario | Variant | Hard before Repair | Hard after Repair | Repaired Demands (mean) | Repair Time (s) | Success Rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for rec in records:
        if not rec.use_repair:
            continue
        grouped.setdefault((rec.scenario, rec.variant), []).append(rec)
    for (scenario, variant), items in sorted(grouped.items()):
        hb = mean([x.hard_ls for x in items])
        ha = mean([x.hard_repair for x in items])
        rd = mean([x.repair_repaired_demands for x in items])
        rt = mean([x.runtime_repair_sec for x in items])
        sr = mean([1.0 if x.repair_success else 0.0 for x in items])
        lines.append(f"| {scenario} | {variant} | {hb:.2f} | {ha:.2f} | {rd:.2f} | {rt:.2f} | {sr:.2f} |")
    lines.append("")
    lines.append("## Charts")
    for p in chart_paths:
        lines.append(f"- `{p.name}`")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("```bash")
    lines.append("python full_pipeline_evaluation.py --profile fast")
    lines.append("```")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full timetabling pipeline")
    parser.add_argument("--profile", choices=["fast", "default"], default="fast")
    parser.add_argument("--runs-core", type=int, default=1)
    parser.add_argument("--runs-stress", type=int, default=1)
    parser.add_argument(
        "--stress-levels",
        default="90,70",
        help="Comma-separated percentages for stress scenarios, e.g. 90,80,70",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "experiment_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_ds = load_dataset()

    overrides = FAST_PROFILE_OVERRIDES if args.profile == "fast" else {}
    records: list[RunRecord] = []

    scenarios_core = [
        ("P1", "greedy_only", False, False, False),
        ("P2", "greedy_ga", True, False, False),
        ("P3", "greedy_ga_ls", True, True, False),
        ("P4", "full_pipeline", True, True, True),
        ("A1", "without_ls_keep_repair", True, False, True),
        ("A2", "without_repair_keep_ls", True, True, False),
    ]

    stress_levels = parse_stress_levels(args.stress_levels)

    with config_overrides(overrides):
        for scenario, variant, use_ga, use_ls, use_repair in scenarios_core:
            for run_id in range(1, args.runs_core + 1):
                seed = stable_seed("core", scenario, variant, run_id, base=1000)
                rec = run_one_pipeline(
                    ds=deepcopy(base_ds),
                    scenario=scenario,
                    variant=variant,
                    run_id=run_id,
                    seed=seed,
                    profile=args.profile,
                    use_ga=use_ga,
                    use_ls=use_ls,
                    use_repair=use_repair,
                )
                records.append(rec)
                print(
                    f"[{scenario}:{variant}] run={run_id} seed={seed} "
                    f"hard {rec.hard_greedy}->{rec.hard_ga}->{rec.hard_ls}->{rec.hard_repair} "
                    f"time={rec.runtime_total_sec:.2f}s"
                )

        stress_variants: list[tuple[str, str, TimetableDataset]] = []
        for pct in stress_levels:
            stress_variants.append(("S1_room_pressure", f"room_{pct}pct", clone_with_room_ratio(base_ds, pct / 100.0)))
            stress_variants.append(("S2_slot_pressure", f"slot_{pct}pct", clone_with_slot_ratio(base_ds, pct / 100.0)))
            stress_variants.append(("S3_teacher_pressure", f"teacher_{pct}pct", clone_with_teacher_ratio(base_ds, pct / 100.0)))

        for scenario, variant, ds_stress in stress_variants:
            for run_id in range(1, args.runs_stress + 1):
                seed = stable_seed("stress", scenario, variant, run_id, base=2000)
                rec = run_one_pipeline(
                    ds=deepcopy(ds_stress),
                    scenario=scenario,
                    variant=variant,
                    run_id=run_id,
                    seed=seed,
                    profile=args.profile,
                    use_ga=True,
                    use_ls=True,
                    use_repair=True,
                )
                records.append(rec)
                print(
                    f"[{scenario}:{variant}] run={run_id} seed={seed} "
                    f"hard {rec.hard_greedy}->{rec.hard_ga}->{rec.hard_ls}->{rec.hard_repair} "
                    f"time={rec.runtime_total_sec:.2f}s"
                )

    aggregate = aggregate_runs(records)

    run_csv = out_dir / f"full_pipeline_runs_{ts}.csv"
    agg_csv = out_dir / f"full_pipeline_aggregate_{ts}.csv"
    json_path = out_dir / f"full_pipeline_report_{ts}.json"
    md_path = out_dir / f"README_experiment_{ts}.md"

    run_fields = list(asdict(records[0]).keys()) if records else []
    write_csv(run_csv, [asdict(r) for r in records], run_fields)

    agg_fields = list(aggregate[0].keys()) if aggregate else []
    write_csv(agg_csv, aggregate, agg_fields)

    chart_paths = render_charts(aggregate, records, out_dir, ts)
    render_markdown(
        aggregate=aggregate,
        records=records,
        chart_paths=chart_paths,
        out_md=md_path,
        generated_at=datetime.now().isoformat(),
        profile=args.profile,
    )

    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "profile": args.profile,
                "runs_core": args.runs_core,
                "runs_stress": args.runs_stress,
                "run_count": len(records),
                "aggregate_count": len(aggregate),
                "run_csv": str(run_csv),
                "aggregate_csv": str(agg_csv),
                "readme_md": str(md_path),
                "charts": [str(p) for p in chart_paths],
                "runs": [asdict(r) for r in records],
                "aggregate": aggregate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved run CSV: {run_csv}")
    print(f"Saved aggregate CSV: {agg_csv}")
    print(f"Saved report JSON: {json_path}")
    print(f"Saved README: {md_path}")
    print("Charts:")
    for p in chart_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
