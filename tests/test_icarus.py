"""Each simulator against Icarus Verilog on every design, same stimulus."""

import pytest
from helpers import DESIGNS, ROOT, load

from bitlane.icarus import first_mismatch, run_icarus
from bitlane.levels import pack
from bitlane.stimulus import random_inputs

RESET = {"counter": "rst", "fsm": "rst"}  # the clocked designs and their reset port


@pytest.mark.parametrize("name", DESIGNS)
def test_matches_icarus(name, tmp_path, simulate):
    netlist, _ = load(name)
    reset = RESET.get(name)
    inputs = random_inputs(netlist, n_cycles=20, n_tests=200, reset=reset)
    ours = simulate(pack(netlist), inputs)
    theirs, unknown = run_icarus(
        ROOT / "designs" / f"{name}.v", name, netlist, inputs, tmp_path
    )
    # A clocked design starts each test in the previous test's state (x for the first),
    # so its reset cycle is not compared; a combinational one is compared everywhere.
    assert first_mismatch(ours, theirs, unknown, first_cycle=1 if reset else 0) is None
