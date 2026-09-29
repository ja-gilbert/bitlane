"""The three benchmark designs against integer models of what the Verilog says."""

import numpy as np
from helpers import load

from bitlane.levels import pack
from bitlane.stimulus import random_inputs


def test_mux_selects_one_input(simulate):
    netlist, _ = load("mux")
    inputs = random_inputs(netlist, n_cycles=1, n_tests=4096)
    out = simulate(pack(netlist), inputs)
    sel = inputs["sel"].astype(np.intp)  # choose wants a signed index
    expected = np.choose(sel, [inputs["a"], inputs["b"], inputs["c"], inputs["d"]])
    assert np.array_equal(out["y"], expected)


def test_alu_matches_integer_model(simulate):
    netlist, _ = load("alu")
    inputs = random_inputs(netlist, n_cycles=1, n_tests=4096)
    inputs["b"][:, :100] = inputs["a"][:, :100]  # some ties, so < and <= differ
    out = simulate(pack(netlist), inputs)
    a, b, op = inputs["a"], inputs["b"], inputs["op"]
    shift = b & 7
    results = [
        (a + b) & 0xFF,
        (a - b) & 0xFF,
        a & b,
        a | b,
        a ^ b,
        (a << shift) & 0xFF,
        a >> shift,
        a < b,
    ]
    y = np.choose(op.astype(np.intp), results)
    assert np.array_equal(out["y"], y)
    assert np.array_equal(out["zero"], y == 0)


def test_fsm_matches_state_table(simulate):
    netlist, _ = load("fsm")
    n_cycles, n_tests = 300, 1024
    inputs = random_inputs(netlist, n_cycles, n_tests, reset="rst")
    out = simulate(pack(netlist), inputs)
    # The state table from fsm.v, indexed next_state[state, in].
    next_state = np.array([[0, 1], [0, 2], [3, 2], [0, 1]], dtype=np.uint64)
    state = np.zeros(n_tests, dtype=np.uint64)
    for cycle in range(n_cycles):
        bit, rst = inputs["in"][cycle], inputs["rst"][cycle]
        assert np.array_equal(out["state"][cycle], state)
        assert np.array_equal(out["hit"][cycle], (state == 3) & (bit == 1))
        state = np.where(rst == 1, 0, next_state[state, bit])
    assert out["hit"].any()  # the sequence 1101 did occur
