"""Verilator against the NumPy simulator on every design, one and four threads."""

import numpy as np
import pytest
from helpers import DESIGNS, RESET, ROOT, load

from bitlane.icarus import first_mismatch
from bitlane.levels import pack
from bitlane.refsim import simulate
from bitlane.stimulus import random_inputs
from bitlane.verilator import run_verilator


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("name", DESIGNS)
def test_matches_verilator(name, threads):
    netlist, _ = load(name)
    reset = RESET.get(name)
    inputs = random_inputs(netlist, n_cycles=20, n_tests=200, reset=reset)
    workdir = ROOT / "build" / "verilator" / name  # kept, so later runs skip the build
    verilog = ROOT / "designs" / f"{name}.v"
    theirs, seconds = run_verilator(verilog, name, netlist, inputs, workdir, threads)
    ours = simulate(pack(netlist), inputs)
    # Verilator is two-state, so none of its outputs is ever unknown (x).
    unknown = {port: np.zeros_like(values, dtype=bool) for port, values in ours.items()}
    # As with Icarus, tests run back to back, so a clocked design's reset cycle is skipped.
    assert first_mismatch(ours, theirs, unknown, first_cycle=1 if reset else 0) is None
    assert seconds > 0
