"""
Re-render full pipeline charts from existing CSV outputs with Vietnamese-safe font.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["axes.unicode_minus"] = False


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def plot_core_quality_runtime(agg_rows: list[dict], out_path: Path) -> None:
    core_order = ["P1", "P2", "P3", "P4", "A1", "A2"]
    core = [r for r in agg_rows if r["scenario"] in core_order]
    core.sort(key=lambda r: core_order.index(r["scenario"]))

    labels = [f"{r['scenario']}\n{r['variant']}" for r in core]
    hard = [float(r["hard_final_mean"]) for r in core]
    runtime = [float(r["runtime_total_mean_sec"]) for r in core]

    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(core))
    bars = ax1.bar(x, hard, color="#1f77b4", alpha=0.85)
    ax1.set_ylabel("Hard penalty cuối", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("So sánh chất lượng và thời gian của các phương án pipeline")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, runtime, color="#d62728", marker="o", linewidth=2)
    ax2.set_ylabel("Thời gian tổng (giây)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    for rect, h in zip(bars, hard):
        ax1.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.7, f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    for i, t in enumerate(runtime):
        ax2.text(i, t + 0.2, f"{t:.1f}s", color="#d62728", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hard_by_stage(agg_rows: list[dict], out_path: Path) -> None:
    core_order = ["P1", "P2", "P3", "P4", "A1", "A2"]
    core = [r for r in agg_rows if r["scenario"] in core_order]
    core.sort(key=lambda x: core_order.index(x["scenario"]))
    labels = [f"{x['scenario']}:{x['variant']}" for x in core]
    hg = [float(x["hard_greedy_mean"]) for x in core]
    hga = [float(x["hard_ga_mean"]) for x in core]
    hls = [float(x["hard_ls_mean"]) for x in core]
    hf = [float(x["hard_final_mean"]) for x in core]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.plot(x, hg, marker="o", label="Greedy")
    plt.plot(x, hga, marker="o", label="GA")
    plt.plot(x, hls, marker="o", label="Local Search")
    plt.plot(x, hf, marker="o", label="Final")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Hard penalty")
    plt.title("Hard penalty theo từng stage")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_runtime_breakdown(run_rows: list[dict], out_path: Path) -> None:
    core_order = ["P1", "P2", "P3", "P4", "A1", "A2"]
    labels = []
    rg = []
    rga = []
    rls = []
    rr = []

    for sc in core_order:
        sub = [r for r in run_rows if r["scenario"] == sc]
        if not sub:
            continue
        labels.append(f"{sc}:{sub[0]['variant']}")
        rg.append(np.mean([float(r["runtime_greedy_sec"]) for r in sub]))
        rga.append(np.mean([float(r["runtime_ga_sec"]) for r in sub]))
        rls.append(np.mean([float(r["runtime_ls_sec"]) for r in sub]))
        rr.append(np.mean([float(r["runtime_repair_sec"]) for r in sub]))

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, rg, label="Greedy")
    plt.bar(x, rga, bottom=rg, label="GA")
    b2 = np.array(rg) + np.array(rga)
    plt.bar(x, rls, bottom=b2, label="LS")
    b3 = b2 + np.array(rls)
    plt.bar(x, rr, bottom=b3, label="Repair")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Giây")
    plt.title("Phân rã thời gian theo stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_stress(agg_rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    stress_cfg = [
        ("S1_room_pressure", "Phòng khả dụng (%)", axes[0]),
        ("S2_slot_pressure", "Slot khả dụng (%)", axes[1]),
        ("S3_teacher_pressure", "Teacher candidate (%)", axes[2]),
    ]
    for scenario, xlabel, ax in stress_cfg:
        subset = [x for x in agg_rows if x["scenario"] == scenario]
        subset.sort(key=lambda r: float(r["variant"].split("_")[-1].replace("pct", "")))
        x = [float(r["variant"].split("_")[-1].replace("pct", "")) for r in subset]
        h = [float(r["hard_final_mean"]) for r in subset]
        rt = [float(r["runtime_total_mean_sec"]) for r in subset]
        ax.plot(x, h, marker="o", label="Hard cuối")
        ax2 = ax.twinx()
        ax2.plot(x, rt, marker="s", color="#C1121F", label="Thời gian")
        ax.set_xlabel(xlabel)
        ax.set_title(scenario)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Hard cuối")
    fig.suptitle("Stress test: chất lượng và thời gian")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full pipeline visuals")
    parser.add_argument("--tag", required=True, help="Timestamp tag, e.g. 20260521_054346")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent / "experiment_results"
    agg_csv = base / f"full_pipeline_aggregate_{args.tag}.csv"
    run_csv = base / f"full_pipeline_runs_{args.tag}.csv"
    if not agg_csv.exists() or not run_csv.exists():
        raise FileNotFoundError("Cannot find aggregate/run CSV for the specified tag.")

    agg_rows = load_rows(agg_csv)
    run_rows = load_rows(run_csv)

    p1 = base / f"pipeline_core_quality_runtime_{args.tag}.png"
    p2 = base / f"pipeline_hard_by_stage_{args.tag}.png"
    p3 = base / f"pipeline_runtime_breakdown_{args.tag}.png"
    p4 = base / f"pipeline_stress_curves_{args.tag}.png"

    plot_core_quality_runtime(agg_rows, p1)
    plot_hard_by_stage(agg_rows, p2)
    plot_runtime_breakdown(run_rows, p3)
    plot_stress(agg_rows, p4)

    print(f"Rendered: {p1}")
    print(f"Rendered: {p2}")
    print(f"Rendered: {p3}")
    print(f"Rendered: {p4}")


if __name__ == "__main__":
    main()
