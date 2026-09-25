"""Level sorting on the real designs and hand-made netlists with and without a loop."""

import pytest
from helpers import load

from bitlane.levels import CombLoopError, gate_levels
from bitlane.netlist import Flop, Gate, Netlist


@pytest.mark.parametrize("name", ["counter", "adder"])
def test_every_input_comes_from_a_lower_level(name):
    netlist, _ = load(name)
    level = gate_levels(netlist)
    driver = {gate.output: i for i, gate in enumerate(netlist.gates)}
    for i, gate in enumerate(netlist.gates):
        for net in gate.inputs:
            if net in driver:
                assert level[driver[net]] < level[i]
    assert min(level) == 1


def test_two_gate_loop_is_an_error():
    # a = NOT b; b = NOT a; nets 2 and 3, nothing else.
    loop = Netlist(
        n_nets=4,
        inputs={},
        outputs={"a": [2]},
        clock=None,
        gates=[Gate("NOT", [3], 2), Gate("NOT", [2], 3)],
        flops=[],
        names={2: "a", 3: "b"},
    )
    with pytest.raises(CombLoopError, match="driving a, b"):
        gate_levels(loop)


def test_flop_feedback_is_not_a_loop():
    # A toggle flop: q = flop(d); d = NOT q. Nets: 2 clk, 3 q, 4 d.
    toggle = Netlist(
        n_nets=5,
        inputs={"clk": [2]},
        outputs={"q": [3]},
        clock="clk",
        gates=[Gate("NOT", [3], 4)],
        flops=[Flop(d=4, q=3)],
    )
    assert gate_levels(toggle) == [1]
