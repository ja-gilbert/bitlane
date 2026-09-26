"""Sort gates into levels so that every gate's inputs are ready before it runs.

Level 0 is what is known at the start of a cycle: the input ports, the constants
and the flop outputs. A gate's level is one more than the highest level among its
inputs, so the gates of one level depend only on lower levels and can all be
evaluated at once. A gate that never becomes ready is on a combinational loop,
or is fed by one.

pack() then lays the gates out in level order as flat arrays: the form the
simulators read, and the form the binary netlist file will hold.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from bitlane.netlist import Netlist

KINDS = ["NOT", "AND", "OR", "XOR", "MUX"]  # a gate's type code is its index here


class CombLoopError(ValueError):
    pass


def gate_levels(netlist: Netlist) -> list[int]:
    """The level of each gate in netlist.gates, from 1 up (Kahn's algorithm)."""
    gates = netlist.gates
    gate_outputs = {gate.output for gate in gates}
    readers = defaultdict(list)  # net -> gates reading it, once per input pin
    waiting = [0] * len(gates)  # per gate: input pins whose driver has no level yet
    for i, gate in enumerate(gates):
        for n in gate.inputs:
            if n in gate_outputs:
                readers[n].append(i)
                waiting[i] += 1

    net_level = defaultdict(int)  # inputs, constants and flop outputs stay at 0
    level = [0] * len(gates)  # 0 means not levelled yet
    ready = deque(i for i, count in enumerate(waiting) if count == 0)
    while ready:
        i = ready.popleft()
        level[i] = 1 + max(net_level[n] for n in gates[i].inputs)
        net_level[gates[i].output] = level[i]
        for j in readers[gates[i].output]:
            waiting[j] -= 1
            if waiting[j] == 0:
                ready.append(j)

    if stuck := [gate.output for gate, lv in zip(gates, level) if lv == 0]:
        nets = ", ".join(netlist.names.get(n, f"net {n}") for n in stuck)
        raise CombLoopError(f"combinational loop: gates driving {nets}")
    return level


@dataclass
class Packed:
    """A netlist as flat arrays, gates sorted by level. Nets are the Netlist's."""

    netlist: Netlist
    kind: np.ndarray  # uint8 per gate: index into KINDS
    in_nets: np.ndarray  # int32 (n_gates, 3): input nets in port order, unused = net 0
    out_net: np.ndarray  # int32 per gate
    # int32: level k+1 is gates [level_start[k], level_start[k+1])
    level_start: np.ndarray
    flop_d: np.ndarray  # int32 per flop
    flop_q: np.ndarray


def pack(netlist: Netlist) -> Packed:
    """Sort the gates by level and pack gates and flops into arrays."""
    level = gate_levels(netlist)
    order = np.argsort(level, kind="stable")  # stable: ties keep order
    gates = [netlist.gates[i] for i in order]
    in_nets = np.zeros((len(gates), 3), dtype=np.int32)
    for row, gate in zip(in_nets, gates):
        row[: len(gate.inputs)] = gate.inputs
    return Packed(
        netlist,
        kind=np.array([KINDS.index(gate.kind) for gate in gates], dtype=np.uint8),
        in_nets=in_nets,
        out_net=np.array([gate.output for gate in gates], dtype=np.int32),
        # gates per level (index = level, none at 0), running total = start offsets
        level_start=np.cumsum(np.bincount(level, minlength=1), dtype=np.int32),
        flop_d=np.array([flop.d for flop in netlist.flops], dtype=np.int32),
        flop_q=np.array([flop.q for flop in netlist.flops], dtype=np.int32),
    )
