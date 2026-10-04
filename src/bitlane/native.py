"""The C reference and the CUDA kernel, called through ctypes.

cuda/ builds one shared library with CMake. Both simulators have the same three
calls (see cuda/bitlane.h): open copies the netlist and allocates every buffer,
run simulates one batch of tests and can be repeated, close frees everything.
Python hands over the Packed arrays as pointers and the stimulus already packed
for every cycle, so a run has no Python in its loop. `gpu=True` picks the kernel.
"""

import ctypes
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from bitlane.levels import Packed
from bitlane.refsim import new_vals, pack_inputs, unpack_outputs

ROOT = Path(__file__).resolve().parents[2]
INT = ctypes.c_int
SIM = ctypes.c_void_p  # an open simulator, as the library returned it
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
    open_args = [
        INT, INT, INT,  # n_nets, n_words, n_cycles
        INT, I32,       # n_levels, level_start
        U8, I32, I32,   # kind, in_nets, out_net
        INT, I32, I32,  # n_flops, flop_d, flop_q
        INT, I32,       # n_stim, stim_net
        INT, I32,       # n_probe, probe_net
    ]
    # fmt: on
    for suffix in ("", "_gpu"):
        getattr(lib, "bitlane_open" + suffix).restype = SIM
        getattr(lib, "bitlane_open" + suffix).argtypes = open_args
        getattr(lib, "bitlane_run" + suffix).restype = None
        getattr(lib, "bitlane_run" + suffix).argtypes = [SIM, U32, U32]
        getattr(lib, "bitlane_close" + suffix).restype = None
        getattr(lib, "bitlane_close" + suffix).argtypes = [SIM]
    lib.bitlane_wake_gpu.restype = None
    lib.bitlane_wake_gpu.argtypes = [INT]
    return lib


@dataclass
class Session:
    """An open simulator: the netlist loaded and every buffer allocated."""

    sim: int | None  # the library's handle, None once closed
    suffix: str  # "" for the C reference, "_gpu" for the kernel
    stim_shape: tuple  # (n_cycles, n_stim, n_words): what every run must be given
    probe_words: np.ndarray  # (n_cycles, n_probe, n_words): each run overwrites it


def open_session(
    packed: Packed,
    stim_net: np.ndarray,
    probe_net: np.ndarray,
    n_cycles: int,
    n_words: int,
    gpu: bool = False,
) -> Session:
    """Load the netlist and allocate the buffers for runs of n_cycles by n_words."""
    suffix = "_gpu" if gpu else ""
    # fmt: off
    sim = getattr(library(), "bitlane_open" + suffix)(
        packed.n_rows, n_words, n_cycles,
        len(packed.level_start) - 1, packed.level_start,
        packed.kind, packed.in_nets, packed.out_net,
        len(packed.flop_d), packed.flop_d, packed.flop_q,
        len(stim_net), stim_net,
        len(probe_net), probe_net,
    )
    # fmt: on
    probe_words = np.zeros((n_cycles, len(probe_net), n_words), dtype=np.uint32)
    return Session(sim, suffix, (n_cycles, len(stim_net), n_words), probe_words)


def run(session: Session, stim_words: np.ndarray) -> np.ndarray:
    """One warm run from reset: the packed outputs, (n_cycles, n_probe, n_words), for
    packed stimulus of shape (n_cycles, n_stim, n_words). The returned array belongs to
    the session and the next run overwrites it."""
    if session.sim is None:
        raise ValueError("the session is closed")
    if stim_words.shape != session.stim_shape:  # the C code trusts the shape blindly
        raise ValueError(f"stimulus {stim_words.shape}, session {session.stim_shape}")
    getattr(library(), "bitlane_run" + session.suffix)(
        session.sim, stim_words, session.probe_words
    )
    return session.probe_words


def close_session(session: Session) -> None:
    """Free the simulator's buffers. Closing twice does nothing."""
    if session.sim is not None:
        getattr(library(), "bitlane_close" + session.suffix)(session.sim)
        session.sim = None


def probe_nets(packed: Packed) -> np.ndarray:
    """The rows of every output port's nets, in port order: what the C code copies out."""
    outputs = packed.netlist.outputs
    return packed.row[[n for nets in outputs.values() for n in nets]]


def pack_stimulus(packed: Packed, inputs: dict) -> tuple[np.ndarray, np.ndarray]:
    """The rows of the driven input nets and their packed words for every cycle:
    stim_net (n_stim,) and stim_words (n_cycles, n_stim, n_words)."""
    netlist = packed.netlist
    n_cycles, n_tests = next(iter(inputs.values())).shape
    stim_net = packed.row[[n for name in inputs for n in netlist.inputs[name]]]
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
    probe_net = probe_nets(packed)
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
    n_cycles, _, n_words = stim_words.shape
    probe_net = probe_nets(packed)
    session = open_session(packed, stim_net, probe_net, n_cycles, n_words, gpu)
    outputs = unpack_probes(packed, run(session, stim_words), n_tests)
    close_session(session)
    return outputs
