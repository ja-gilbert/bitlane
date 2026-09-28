"""Packed-bit reference simulator: 32 tests per 32-bit word.

Every net holds one uint32 word per group of 32 tests: bit t of word w is the
net's value in test 32*w + t. One bitwise operation on two words therefore runs
a gate for 32 tests at once, and a MUX is (b & s) | (a & ~s): take b's bit where
s is 1 and a's bit where s is 0. Values live in vals[net, word], net-major, so
one gate's words sit together: the layout the CUDA kernel will use with one
thread per (gate, word).

A cycle applies the inputs, evaluates the gates, samples the outputs, and then
lets every flop capture its D: the clock edge. Flops start at 0, so a design
must be reset by its own stimulus before its outputs mean anything.
"""

from itertools import pairwise

import numpy as np

from bitlane.levels import Packed

LANES = np.arange(32, dtype=np.uint32)  # bit position of each test within a word
OPS = [  # in KINDS order (NOT, AND, OR, XOR, MUX); each takes the input words a, b, s
    lambda a, b, s: ~a,
    lambda a, b, s: a & b,
    lambda a, b, s: a | b,
    lambda a, b, s: a ^ b,
    lambda a, b, s: (b & s) | (a & ~s),
]


def new_vals(packed: Packed, n_tests: int) -> np.ndarray:
    """vals[net, word], all zero except the constant-1 net; flops start at 0 too."""
    n_words = (n_tests + 31) // 32  # the last word's spare lanes are padding
    vals = np.zeros((packed.netlist.n_nets, n_words), dtype=np.uint32)
    vals[1] = 0xFFFFFFFF  # net 1 is constant 1
    return vals


def pack_inputs(
    packed: Packed, inputs: dict[str, np.ndarray], vals: np.ndarray
) -> None:
    """Write the driven input ports into vals; inputs not given are left alone."""
    n_words = vals.shape[1]
    for name, values in inputs.items():
        padded = np.zeros(n_words * 32, dtype=np.uint64)
        padded[: len(values)] = values
        by_word = padded.reshape(n_words, 32)  # test 32*w + t sits at [w, t]
        for bit, net in enumerate(packed.netlist.inputs[name]):
            lane_bits = ((by_word >> bit) & 1).astype(np.uint32)
            vals[net] = np.bitwise_or.reduce(lane_bits << LANES, axis=1)


def eval_gates(packed: Packed, vals: np.ndarray) -> None:
    """Evaluate every gate in place, one level at a time.

    Gather the input words of all gates in the level, apply each kind's op to
    its gates in one array operation, scatter the results. Every (gate, word)
    element of those array operations is what one GPU thread will do, and one
    level is one kernel launch.
    """
    for start, end in pairwise(packed.level_start):
        kind = packed.kind[start:end]
        ins = packed.in_nets[start:end]
        a, b, s = vals[ins[:, 0]], vals[ins[:, 1]], vals[ins[:, 2]]
        out = packed.out_net[start:end]
        for code, op in enumerate(OPS):
            mask = kind == code
            vals[out[mask]] = op(a[mask], b[mask], s[mask])


def unpack_outputs(packed: Packed, vals: np.ndarray, n_tests: int) -> dict:
    """Each output port's value in every test, as an integer array."""
    outputs = {}
    for name, nets in packed.netlist.outputs.items():
        values = np.zeros(n_tests, dtype=np.uint64)
        for bit, net in enumerate(nets):
            lane_bits = (vals[net][:, None] >> LANES) & 1  # (n_words, 32)
            values |= lane_bits.reshape(-1)[:n_tests].astype(np.uint64) << bit
        outputs[name] = values
    return outputs


def simulate(packed: Packed, inputs: dict[str, np.ndarray]) -> dict:
    """Outputs of a clocked design, cycle by cycle.

    inputs[port] and the returned outputs[port] are (n_cycles, n_tests) arrays.
    """
    n_cycles, n_tests = next(iter(inputs.values())).shape
    vals = new_vals(packed, n_tests)
    outputs = {
        name: np.zeros((n_cycles, n_tests), dtype=np.uint64)
        for name in packed.netlist.outputs
    }
    for cycle in range(n_cycles):
        pack_inputs(packed, {name: v[cycle] for name, v in inputs.items()}, vals)
        eval_gates(packed, vals)
        for name, values in unpack_outputs(packed, vals, n_tests).items():
            outputs[name][cycle] = values
        # The clock edge: every flop captures its D at once.
        vals[packed.flop_q] = vals[packed.flop_d]
    return outputs


def evaluate(packed: Packed, inputs: dict[str, np.ndarray]) -> dict:
    """Outputs of a combinational design for every test: one cycle, no flops."""
    outputs = simulate(packed, {name: v[np.newaxis] for name, v in inputs.items()})
    return {name: v[0] for name, v in outputs.items()}
