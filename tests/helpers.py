"""Shared by the test modules: the real designs, synthesized once per run."""

from functools import cache
from pathlib import Path

from bitlane.netlist import Netlist, read_netlist
from bitlane.synth import synth

ROOT = Path(__file__).parents[1]


@cache
def load(name: str) -> tuple[Netlist, Path]:
    """Synthesize designs/<name>.v with Yosys (once per run) and read it back."""
    path = ROOT / "build" / f"{name}.json"
    synth(ROOT / "designs" / f"{name}.v", name, path)
    return read_netlist(path), path
