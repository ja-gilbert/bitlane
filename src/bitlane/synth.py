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
    # The whole run as one script, with the paths quoted because Yosys splits commands
    # at spaces, and the flow pasted in because its script command keeps the quotes.
    run = out.with_suffix(".ys")
    run.write_text(
        f'read_verilog "{verilog}"\n'
        f"hierarchy -check -top {top}\n"
        f"{SCRIPT.read_text()}"
        f'write_json "{out}"\n'
    )
    result = subprocess.run(["yosys", "-q", "-s", run], check=False)
    if result.returncode != 0:
        raise SystemExit(f"bitlane: yosys failed on {verilog} (see the error above)")


def cell_counts(netlist: Path) -> Counter[str]:
    """Number of cells of each type in a Yosys JSON netlist."""
    modules = json.loads(netlist.read_text())["modules"]
    (module,) = modules.values()  # flattened, so exactly one module
    return Counter(cell["type"] for cell in module["cells"].values())
