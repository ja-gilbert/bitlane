"""Run Yosys on a Verilog design and write the gate-level JSON netlist."""

import json
import subprocess
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "synth.ys"


def synth(verilog: Path, top: str, out: Path) -> None:
    """Flatten module `top` of `verilog` to simple gates and $_DFF_P_ flops.

    Writes the Yosys JSON netlist to `out`. Yosys prints its own warnings and
    errors; if it fails, exit with a one-line message.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    commands = (
        f"read_verilog {verilog}; hierarchy -check -top {top}; "
        f"script {SCRIPT}; write_json {out}"
    )
    result = subprocess.run(["yosys", "-q", "-p", commands], check=False)
    if result.returncode != 0:
        raise SystemExit(f"bitlane: yosys failed on {verilog} (see the error above)")


def cell_counts(netlist: Path) -> Counter[str]:
    """Number of cells of each type in a Yosys JSON netlist."""
    modules = json.loads(netlist.read_text())["modules"]
    (module,) = modules.values()  # flattened, so exactly one module
    return Counter(cell["type"] for cell in module["cells"].values())
