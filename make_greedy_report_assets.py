"""
Generate summary tables and charts from greedy experiment JSON output.

Usage:
    python make_greedy_report_assets.py
    python make_greedy_report_assets.py --input experiment_results/greedy_experiments_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["axes.unicode_minus"] = False


def _latest_json(results_dir: Path) -> Path:
    files = sorted(results_dir.glob("greedy_experiments_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No greedy_experiments_*.json found in {results_dir}")
    return files[0]


def _extract_tag(path: Path) -> str:
    m = re.search(r"greedy_experiments_(\d{8}_\d{6})", path.stem)
    return m.group(1) if m else "latest"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(v: float) -> str:
    return f"{v:.2f}"


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Greedy Experiment Summary",
        "",
        "| Scenario | Variant | Runs | Hard Mean | Hard Std | Hard Best | Hard Worst | Unscheduled Mean | Soft Mean | Runtime Mean (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | {r['variant']} | {r['runs']} | "
            f"{_format_float(float(r['hard_mean']))} | {_format_float(float(r['hard_std']))} | "
            f"{_format_float(float(r['hard_best']))} | {_format_float(float(r['hard_worst']))} | "
            f"{_format_float(float(r['unscheduled_mean']))} | {_format_float(float(r['soft_mean']))} | "
            f"{_format_float(float(r['runtime_mean_sec']))} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _num_from_variant(variant: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", variant)
    return float(m.group(1)) if m else 0.0


def _plot_hard_all(rows: list[dict], out: Path) -> None:
    ordered = sorted(rows, key=lambda x: (x["hard_mean"], x["unscheduled_mean"], x["runtime_mean_sec"]))
    labels = [f"{r['scenario']}:{r['variant']}" for r in ordered]
    values = [float(r["hard_mean"]) for r in ordered]

    plt.figure(figsize=(12, 10))
    plt.barh(labels, values, color="#2D6A9F")
    plt.xlabel("Hard Mean")
    plt.title("Greedy Variants - Hard Mean (Lower is better)")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_multi_restart(rows: list[dict], out: Path) -> None:
    s2 = [r for r in rows if r["scenario"] == "S2_multi_restart"]
    s2.sort(key=lambda r: _num_from_variant(r["variant"]))
    x = [_num_from_variant(r["variant"]) for r in s2]
    hard = [float(r["hard_mean"]) for r in s2]
    runtime = [float(r["runtime_mean_sec"]) for r in s2]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, hard, marker="o", color="#1B4332", label="Hard Mean")
    ax1.set_xlabel("Number of Restarts")
    ax1.set_ylabel("Hard Mean", color="#1B4332")
    ax1.tick_params(axis="y", labelcolor="#1B4332")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, runtime, marker="s", color="#D00000", label="Runtime Mean (s)")
    ax2.set_ylabel("Runtime Mean (s)", color="#D00000")
    ax2.tick_params(axis="y", labelcolor="#D00000")

    fig.suptitle("S2 Multi-Restart: Hard vs Runtime")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_order_noise(rows: list[dict], out: Path) -> None:
    s5 = [r for r in rows if r["scenario"] == "S5_order_noise"]
    s5.sort(key=lambda r: _num_from_variant(r["variant"]))
    x = [_num_from_variant(r["variant"]) for r in s5]
    hard = [float(r["hard_mean"]) for r in s5]
    soft = [float(r["soft_mean"]) for r in s5]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, hard, marker="o", color="#2A6F97", label="Hard Mean")
    ax1.set_xlabel("Noise Ratio")
    ax1.set_ylabel("Hard Mean", color="#2A6F97")
    ax1.tick_params(axis="y", labelcolor="#2A6F97")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, soft, marker="^", color="#E85D04", label="Soft Mean")
    ax2.set_ylabel("Soft Mean", color="#E85D04")
    ax2.tick_params(axis="y", labelcolor="#E85D04")

    fig.suptitle("S5 Demand Order Noise: Hard vs Soft")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_pressure(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    configs = [
        ("S6_room_pressure", "rooms_", "Room Availability (%)", axes[0]),
        ("S7_slot_pressure", "slots_", "Slot Availability (%)", axes[1]),
        ("S8_teacher_candidate_pressure", "cand_", "Candidate Teacher Ratio (%)", axes[2]),
    ]

    for scenario, prefix, xlabel, ax in configs:
        subset = [r for r in rows if r["scenario"] == scenario]
        subset.sort(key=lambda r: _num_from_variant(r["variant"]))
        x = [_num_from_variant(r["variant"]) for r in subset]
        hard = [float(r["hard_mean"]) for r in subset]
        uns = [float(r["unscheduled_mean"]) for r in subset]

        ax.plot(x, hard, marker="o", color="#264653", label="Hard Mean")
        ax.plot(x, uns, marker="s", color="#E76F51", label="Unscheduled Mean")
        ax.set_xlabel(xlabel)
        ax.set_title(scenario)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Resource Pressure Scenarios")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tables/charts for greedy experiments")
    parser.add_argument("--input", default=None, help="Path to greedy_experiments_*.json")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "experiment_results"
    input_path = Path(args.input) if args.input else _latest_json(results_dir)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"]
    aggregate_sorted = sorted(aggregate, key=lambda r: (r["scenario"], r["variant"]))
    aggregate_hard_rank = sorted(aggregate, key=lambda r: (r["hard_mean"], r["unscheduled_mean"], r["runtime_mean_sec"]))

    tag = _extract_tag(input_path)
    summary_csv = results_dir / f"greedy_summary_{tag}.csv"
    ranking_csv = results_dir / f"greedy_ranking_by_hard_{tag}.csv"
    summary_md = results_dir / f"greedy_summary_{tag}.md"

    fields = [
        "scenario",
        "variant",
        "runs",
        "hard_mean",
        "hard_std",
        "hard_best",
        "hard_worst",
        "unscheduled_mean",
        "soft_mean",
        "fitness_mean",
        "runtime_mean_sec",
    ]
    _write_csv(summary_csv, aggregate_sorted, fields)
    _write_csv(ranking_csv, aggregate_hard_rank, fields)
    _write_markdown(summary_md, aggregate_sorted)

    chart_hard_all = results_dir / f"chart_hard_all_variants_{tag}.png"
    chart_s2 = results_dir / f"chart_s2_multi_restart_{tag}.png"
    chart_s5 = results_dir / f"chart_s5_order_noise_{tag}.png"
    chart_pressure = results_dir / f"chart_pressure_s6_s7_s8_{tag}.png"

    _plot_hard_all(aggregate, chart_hard_all)
    _plot_multi_restart(aggregate, chart_s2)
    _plot_order_noise(aggregate, chart_s5)
    _plot_pressure(aggregate, chart_pressure)

    print(f"Input JSON: {input_path}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Ranking CSV: {ranking_csv}")
    print(f"Summary MD: {summary_md}")
    print("Charts:")
    print(f"- {chart_hard_all}")
    print(f"- {chart_s2}")
    print(f"- {chart_s5}")
    print(f"- {chart_pressure}")


if __name__ == "__main__":
    main()
