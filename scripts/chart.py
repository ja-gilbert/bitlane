# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
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
# Matplotlib's default colours, fixed per simulator so that a line keeps its colour in
# every panel and matches the one legend. "Verilator, N threads" takes C3 whatever N is.
COLOURS = {
    "C reference": "C0",
    "Icarus": "C1",
    "Verilator, 1 thread": "C2",
    "GPU": "C4",
}


def main() -> None:
    reports = [
        json.loads((RESULTS / f"bench_{design}.json").read_text()) for design in DESIGNS
    ]
    figure, axes = plt.subplots(1, len(reports), figsize=(11, 4.4), sharey=True)
    for axis, report in zip(axes, reports):
        rows = report["rows"]
        tests = [row["tests"] for row in rows]
        for column in rows[0]:
            if column not in ("tests", "cycles", "cold"):
                rates = [row["tests"] / row[column] for row in rows]
                colour = COLOURS.get(column, "C3")
                axis.plot(tests, rates, marker="o", color=colour, label=column)
        cycles = rows[0]["cycles"]
        per_test = "1 cycle" if cycles == 1 else f"{cycles} cycles"
        axis.set_title(f"{report['design']}, {per_test} per test")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("tests in one run")
        axis.grid(True)
    axes[0].set_ylabel("tests per second")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels))
    # All three runs share one machine and one repeat count, so the first says it all.
    figure.text(
        0.01,
        0.01,
        f"Warm runs, median of {reports[0]['repeats']}; GPU times include the copies "
        f"across PCIe. {reports[0]['machine']}",
        fontsize=8,
    )
    # Keep the bottom 4% for the footer and the top 8% for the legend.
    figure.tight_layout(rect=(0, 0.04, 1, 0.92))
    figure.savefig(RESULTS / "throughput.png", dpi=200)


if __name__ == "__main__":
    main()
