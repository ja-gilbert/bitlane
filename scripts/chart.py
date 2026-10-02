# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Draw results/throughput.png from the results/bench_<design>.json files.

Run it with `uv run scripts/chart.py` after `bitlane bench` has written the results.
uv reads the block above and supplies matplotlib, which bitlane itself does not need.
One panel per design, test count across, tests per second up, both on log scales.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"
DESIGNS = ["mux", "alu", "fsm"]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#52514e", "#898781", "#e1e0d9"
# (colour, marker, line, filled marker) for each column of the results. The two Verilator
# lines share a colour: one thread is the dashed, hollow one.
STYLES = {
    "C reference": ("#eb6834", "s", "-", True),
    "Icarus": (MUTED, "D", "-", True),
    "Verilator, 1 thread": ("#1baf7a", "^", "--", False),
    "GPU": ("#2a78d6", "o", "-", True),
}
ALL_CORES = ("#1baf7a", "^", "-", True)  # "Verilator, N threads", whatever N is


def short(value: float, _position: int) -> str:
    """Tick label: 1000 -> 1k, 1e6 -> 1M, 1e9 -> 1B. Matplotlib also passes the tick's
    position, which a label doesn't need."""
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if value >= size:
            return f"{value / size:g}{unit}"
    return f"{value:g}"


def main() -> None:
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK})
    figure, axes = plt.subplots(1, len(DESIGNS), figsize=(11, 4.4), sharey=True)
    figure.set_facecolor(SURFACE)
    reports = [
        json.loads((RESULTS / f"bench_{design}.json").read_text()) for design in DESIGNS
    ]
    for axis, report in zip(axes, reports):
        rows = report["rows"]
        columns = [key for key in rows[0] if key not in ("tests", "cycles", "cold")]
        tests = [row["tests"] for row in rows]
        for column in columns:
            colour, marker, line, filled = STYLES.get(column, ALL_CORES)
            axis.plot(
                tests,
                [row["tests"] / row[column] for row in rows],
                label=column,
                color=colour,
                linestyle=line,
                linewidth=2,
                marker=marker,
                markersize=7,
                markerfacecolor=colour if filled else SURFACE,
                markeredgecolor=SURFACE if filled else colour,
                markeredgewidth=1.5,
            )
        cycles = rows[0]["cycles"]
        per_test = "1 cycle" if cycles == 1 else f"{cycles} cycles"
        axis.set_title(f"{report['design']}: {per_test} per test", loc="left")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(tests)
        axis.minorticks_off()
        axis.xaxis.set_major_formatter(short)
        axis.yaxis.set_major_formatter(short)
        axis.set_xlabel("tests in one run")
        axis.set_facecolor(SURFACE)
        axis.grid(axis="y", color=GRID, linewidth=1)
        axis.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("tests per second")
    axes[0].set_ylim(1e4, 1e10)  # whole decades, so the lowest line has a tick below it
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        handlelength=3.5,  # long enough to show which line is dashed
    )
    figure.text(
        0.01,
        0.01,
        f"Warm runs, median of {reports[0]['repeats']}; GPU times include the copies "
        f"across PCIe. {reports[0]['machine']}",
        color=MUTED,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.93))
    figure.savefig(RESULTS / "throughput.png", dpi=200)


if __name__ == "__main__":
    main()
