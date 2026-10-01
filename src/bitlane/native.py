"""The C reference and the CUDA kernel, called through ctypes.

cuda/ builds one shared library with CMake. Python passes it the Packed arrays
as pointers, the stimulus already packed for every cycle, and a buffer for the
packed outputs, so one call runs the whole simulation with no Python in the loop.
The two entry points, bitlane_simulate and bitlane_simulate_gpu, take the same
arguments (see cuda/bitlane.h); `gpu=True` picks the second.
"""

import ctypes
import subprocess
from functools import cache
from pathlib import Path

import numpy as np

from bitlane.levels import Packed
from bitlane.netlist import Netlist
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
    # fmt: off
    argtypes = [
        INT, INT, INT,  # n_nets, n_words, n_cycles
        INT, I32,       # n_levels, level_start
        U8, I32, I32,   # kind, in_nets, out_net
        INT, I32, I32,  # n_flops, flop_d, flop_q
        INT, I32, U32,  # n_stim, stim_net, stim_words
        INT, I32, U32,  # n_probe, probe_net, probe_words
    ]
    # fmt: on
    for name in ("bitlane_simulate", "bitlane_simulate_gpu"):
        if hasattr(lib, name):  # the GPU entry point exists once kernel.cu is built
            getattr(lib, name).restype = None
            getattr(lib, name).argtypes = argtypes
    return lib


def run(
    packed: Packed,
    stim_net: np.ndarray,
    stim_words: np.ndarray,
    probe_net: np.ndarray,
    gpu: bool = False,
) -> np.ndarray:
    """The packed outputs, (n_cycles, n_probe, n_words), for packed stimulus of
    shape (n_cycles, n_stim, n_words). This call is what a benchmark times."""
    n_cycles, _, n_words = stim_words.shape
    probe_words = np.zeros((n_cycles, len(probe_net), n_words), dtype=np.uint32)
    entry = "bitlane_simulate_gpu" if gpu else "bitlane_simulate"
    # fmt: off
    getattr(library(), entry)(
        packed.netlist.n_nets, n_words, n_cycles,
        len(packed.level_start) - 1, packed.level_start,
        packed.kind, packed.in_nets, packed.out_net,
        len(packed.flop_d), packed.flop_d, packed.flop_q,
        len(stim_net), stim_net, stim_words,
        len(probe_net), probe_net, probe_words,
    )
    # fmt: on
    return probe_words


def probe_nets(netlist: Netlist) -> np.ndarray:
    """Every output port's nets, in port order: the rows the C code copies out."""
    return np.array(
        [n for nets in netlist.outputs.values() for n in nets], dtype=np.int32
    )


def pack_stimulus(packed: Packed, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """The driven input nets and their packed words for every cycle:
    stim_net (n_stim,) and stim_words (n_cycles, n_stim, n_words)."""
    netlist = packed.netlist
    n_cycles, n_tests = next(iter(inputs.values())).shape
    stim_net = np.array(
        [n for name in inputs for n in netlist.inputs[name]], dtype=np.int32
    )
    vals = new_vals(packed, n_tests)  # scratch rows, for packing only
    stim_words = np.zeros((n_cycles, len(stim_net), vals.shape[1]), dtype=np.uint32)
    for cycle in range(n_cycles):
        pack_inputs(packed, {name: v[cycle] for name, v in inputs.items()}, vals)
        stim_words[cycle] = vals[stim_net]
    return stim_net, stim_words


def unpack_probes(packed: Packed, probe_words: np.ndarray, n_tests: int) -> dict:
    """Packed probe rows for every cycle back to outputs[port] of (n_cycles, n_tests)."""
    netlist = packed.netlist
    n_cycles = len(probe_words)
    probe_net = probe_nets(netlist)
    vals = new_vals(packed, n_tests)  # scratch rows, for unpacking only
    outputs = {
        name: np.zeros((n_cycles, n_tests), dtype=np.uint64) for name in netlist.outputs
    }
    for cycle in range(n_cycles):
        vals[probe_net] = probe_words[cycle]
        for name, values in unpack_outputs(packed, vals, n_tests).items():
            outputs[name][cycle] = values
    return outputs


def simulate(packed: Packed, inputs: dict[str, np.ndarray], gpu: bool = False) -> dict:
    """The same contract as refsim.simulate, computed by the C reference or the GPU."""
    n_tests = next(iter(inputs.values())).shape[1]
    stim_net, stim_words = pack_stimulus(packed, inputs)
    probe_words = run(packed, stim_net, stim_words, probe_nets(packed.netlist), gpu)
    return unpack_probes(packed, probe_words, n_tests)
