"""Packed-bit reference simulator: 32 tests per 32-bit word.

Every net holds one uint32 word per group of 32 tests: bit t of word w is the
net's value in test 32*w + t. One bitwise operation on two words therefore runs
a gate for 32 tests at once, and a MUX is (b & s) | (a & ~s): take b's bit where
s is 1 and a's bit where s is 0. Values live in vals[net, word], net-major, so
one gate's words sit together: the layout the CUDA kernel will use with one
thread per (gate, word).
"""

from itertools import pairwise

import numpy as np

from bitlane.levels import Packed

OPS = [  # in KINDS order (NOT, AND, OR, XOR, MUX); each takes the input words a, b, s
    lambda a, b, s: ~a,
    lambda a, b, s: a & b,
    lambda a, b, s: a | b,
    lambda a, b, s: a ^ b,
    lambda a, b, s: (b & s) | (a & ~s),
]


def pack_inputs(packed: Packed, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """vals[net, word] with the driven input ports filled in; other inputs stay 0."""
    n_tests = len(next(iter(inputs.values())))
    n_words = (n_tests + 31) // 32  # the last word's spare lanes are padding
    vals = np.zeros((packed.netlist.n_nets, n_words), dtype=np.uint32)
    vals[1] = 0xFFFFFFFF  # net 1 is constant 1
    for name, values in inputs.items():
        for bit, net in enumerate(packed.netlist.inputs[name]):
            for t in range(n_tests):
                if (int(values[t]) >> bit) & 1:
                    vals[net, t // 32] |= 1 << (t % 32)
    return vals


def eval_gates(packed: Packed, vals: np.ndarray) -> None:
    """Evaluate every gate in place, one level at a time.

    The innermost statement, one gate on one word, is exactly what one GPU
    thread will do, and one level is one kernel launch.
    """
    for start, end in pairwise(packed.level_start):
        for gate in range(start, end):
            a, b, s = packed.in_nets[gate]
            y = packed.out_net[gate]
            op = OPS[packed.kind[gate]]
            for word in range(vals.shape[1]):
                vals[y, word] = op(vals[a, word], vals[b, word], vals[s, word])


def unpack_outputs(packed: Packed, vals: np.ndarray, n_tests: int) -> dict:
    """Each output port's value in every test, as an integer array."""
    outputs = {}
    for name, nets in packed.netlist.outputs.items():
        values = np.zeros(n_tests, dtype=np.uint64)
        for bit, net in enumerate(nets):
            for t in range(n_tests):
                if (int(vals[net, t // 32]) >> (t % 32)) & 1:
                    values[t] |= 1 << bit
        outputs[name] = values
    return outputs


def evaluate(packed: Packed, inputs: dict[str, np.ndarray]) -> dict:
    """Outputs of a combinational design for every test."""
    vals = pack_inputs(packed, inputs)
    eval_gates(packed, vals)
    return unpack_outputs(packed, vals, len(next(iter(inputs.values()))))
