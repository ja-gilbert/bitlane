"""One GPU workload for a profiler: open a design, wake the GPU, run it `runs` times.

    uv run scripts/gpu_run.py alu 1000000 1 - 3
    uv run scripts/gpu_run.py fsm 1000000 20 rst 3

The README's profiles wrap this in `ncu`; see its profiling section for the command.
"""

import sys
from pathlib import Path

from bitlane import native
from bitlane.bench import WAKE_MS
from bitlane.levels import pack
from bitlane.netlist import read_netlist
from bitlane.stimulus import random_inputs
from bitlane.synth import synth

top, n_tests, cycles, reset, runs = sys.argv[1:6]
netlist_path = Path("build") / f"{top}.json"
synth(Path("designs") / f"{top}.v", top, netlist_path)
netlist = read_netlist(netlist_path)
packed = pack(netlist)
inputs = random_inputs(
    netlist, int(cycles), int(n_tests), None if reset == "-" else reset
)
stim_net, stim_words = native.pack_stimulus(packed, inputs)
n_cycles, _, n_words = stim_words.shape
probe_net = native.probe_nets(packed)
session = native.open_session(packed, stim_net, probe_net, n_cycles, n_words, gpu=True)
native.library().bitlane_wake_gpu(WAKE_MS)  # the same wake-up the benchmark gives
for _ in range(int(runs)):
    native.run(session, stim_words)
native.close_session(session)
