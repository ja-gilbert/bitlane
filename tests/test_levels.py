"""Level sorting and packing on the real designs and on hand-made netlists."""

from collections import Counter

import numpy as np
import pytest
from helpers import DESIGNS, load

from bitlane.levels import KINDS, CombLoopError, gate_levels, pack
from bitlane.netlist import Flop, Gate, Netlist


@pytest.mark.parametrize("name", DESIGNS)
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


# A toggle flop: q = flop(d); d = NOT q. Nets: 2 clk, 3 q, 4 d.
TOGGLE = Netlist(
    n_nets=5,
    inputs={"clk": [2]},
    outputs={"q": [3]},
    clock="clk",
    gates=[Gate("NOT", [3], 4)],
    flops=[Flop(d=4, q=3)],
)


def test_flop_feedback_is_not_a_loop():
    assert gate_levels(TOGGLE) == [1]


@pytest.mark.parametrize("name", DESIGNS)
def test_packed_levels_only_read_what_is_known(name):
    netlist, _ = load(name)
    packed = pack(netlist)
    assert packed.level_start[0] == 0 and packed.level_start[-1] == len(netlist.gates)
    known = {0, 1} | {n for nets in netlist.inputs.values() for n in nets}
    known |= {flop.q for flop in netlist.flops}
    known = set(packed.row[list(known)])
    for start, end in zip(packed.level_start, packed.level_start[1:]):
        ins, outs = packed.in_nets[start:end], packed.out_net[start:end]
        assert all(r in known for r in ins.flat)
        # A gate never writes a row that another gate of its level still reads.
        assert not set(outs) & set(ins.flat)
        known |= set(outs)
    assert Counter(KINDS[k] for k in packed.kind) == Counter(
        g.kind for g in netlist.gates
    )
    assert len(packed.flop_d) == len(packed.flop_q) == len(netlist.flops)


def test_rows_are_reused_and_the_constants_keep_theirs():
    netlist, _ = load("alu")
    packed = pack(netlist)
    assert packed.row[0] == 0 and packed.row[1] == 1
    assert packed.n_rows < netlist.n_nets  # some output took a dead net's row
    assert packed.n_rows == max(packed.out_net) + 1


def test_packed_toggle_layout():
    packed = pack(TOGGLE)
    assert packed.row.tolist() == [0, 1, 2, 3, 4]  # nothing to reuse
    assert packed.n_rows == 5
    assert packed.kind.tolist() == [KINDS.index("NOT")]
    assert packed.in_nets.tolist() == [[3, 0, 0]]
    assert packed.out_net.tolist() == [4]
    assert packed.level_start.tolist() == [0, 1]
    assert packed.flop_d.tolist() == [4] and packed.flop_q.tolist() == [3]
    assert packed.in_nets.dtype == np.int32 and packed.kind.dtype == np.uint8
