"""The C reference simulator, called through ctypes.

cuda/ builds one shared library with CMake. Python passes it the Packed arrays
as pointers, the stimulus already packed for every cycle, and a buffer for the
packed outputs, so one call runs the whole simulation with no Python in the loop.
The CUDA kernel joins the same library in week 3.
"""

import ctypes
import subprocess
from functools import cache
from pathlib import Path

import numpy as np

from bitlane.levels import Packed
from bitlane.refsim import new_vals, pack_inputs, unpack_outputs

ROOT = Path(__file__).resolve().parents[2]
INT = ctypes.c_int
I32 = np.ctypeslib.ndpointer(np.int32, flags="C")
U32 = np.ctypeslib.ndpointer(np.uint32, flags="C")
U8 = np.ctypeslib.ndpointer(np.uint8, flags="C")


@cache
def library() -> ctypes.CDLL:
    """Build the shared library with CMake if it is out of date, then load it."""
    build = ROOT / "build" / "cuda"
    for command in (
        ["cmake", "-S", ROOT / "cuda", "-B", build],
        ["cmake", "--build", build],
    ):
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    lib = ctypes.CDLL(build / "libbitlane.so")
    lib.bitlane_simulate.restype = None
    # fmt: off
    lib.bitlane_simulate.argtypes = [
        INT, INT, INT,  # n_nets, n_words, n_cycles
        INT, I32,       # n_levels, level_start
        U8, I32, I32,   # kind, in_nets, out_net
        INT, I32, I32,  # n_flops, flop_d, flop_q
        INT, I32, U32,  # n_stim, stim_net, stim_words
        INT, I32, U32,  # n_probe, probe_net, probe_words
    ]
    # fmt: on
    return lib


def run(
    packed: Packed, stim_net: np.ndarray, stim_words: np.ndarray, probe_net: np.ndarray
) -> np.ndarray:
    """The packed outputs, (n_cycles, n_probe, n_words), for packed stimulus of
    shape (n_cycles, n_stim, n_words). This call is what a benchmark times."""
    n_cycles, _, n_words = stim_words.shape
    probe_words = np.zeros((n_cycles, len(probe_net), n_words), dtype=np.uint32)
    # fmt: off
    library().bitlane_simulate(
        packed.netlist.n_nets, n_words, n_cycles,
        len(packed.level_start) - 1, packed.level_start,
        packed.kind, packed.in_nets, packed.out_net,
        len(packed.flop_d), packed.flop_d, packed.flop_q,
        len(stim_net), stim_net, stim_words,
        len(probe_net), probe_net, probe_words,
    )
    # fmt: on
    return probe_words


def simulate(packed: Packed, inputs: dict[str, np.ndarray]) -> dict:
    """The same contract as refsim.simulate, computed by the C reference."""
    netlist = packed.netlist
    n_cycles, n_tests = next(iter(inputs.values())).shape
    stim_net = np.array(
        [n for name in inputs for n in netlist.inputs[name]], dtype=np.int32
    )
    probe_net = np.array(
        [n for nets in netlist.outputs.values() for n in nets], dtype=np.int32
    )

    vals = new_vals(packed, n_tests)  # scratch rows, for packing and unpacking only
    stim_words = np.zeros((n_cycles, len(stim_net), vals.shape[1]), dtype=np.uint32)
    for cycle in range(n_cycles):
        pack_inputs(packed, {name: v[cycle] for name, v in inputs.items()}, vals)
        stim_words[cycle] = vals[stim_net]

    probe_words = run(packed, stim_net, stim_words, probe_net)

    outputs = {
        name: np.zeros((n_cycles, n_tests), dtype=np.uint64) for name in netlist.outputs
    }
    for cycle in range(n_cycles):
        vals[probe_net] = probe_words[cycle]
        for name, values in unpack_outputs(packed, vals, n_tests).items():
            outputs[name][cycle] = values
    return outputs
