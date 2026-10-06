"""Sort gates into levels so that every gate's inputs are ready before it runs.

gate_levels() puts each gate one level above the highest of its inputs, so the
gates of one level can all be evaluated at once. pack() lays the gates out in
level order as flat arrays and gives every net a row of the value array. A gate's
output takes over the row of a net that no later level reads, so the rows in use
stay small enough to live in cache. The README's "How it works" explains both.
"""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from bitlane.netlist import Gate, Netlist

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
    """A netlist as flat arrays, gates sorted by level. The simulators keep one row
    of words per net in vals[row, word]; every net index below is that row."""

    netlist: Netlist
    row: np.ndarray  # int32 per net: its row of vals; nets 0 and 1 keep rows 0 and 1
    n_rows: int
    kind: np.ndarray  # uint8 per gate: index into KINDS
    in_nets: np.ndarray  # int32 (n_gates, 3): input rows in port order, unused = row 0
    out_net: np.ndarray  # int32 per gate
    # int32: level k+1 is gates [level_start[k], level_start[k+1])
    level_start: np.ndarray
    flop_d: np.ndarray  # int32 per flop
    flop_q: np.ndarray


def assign_rows(
    netlist: Netlist, gates: list[Gate], level: list[int]
) -> tuple[np.ndarray, int]:
    """A row for every net, with `gates` in level order: (row per net, number of rows).

    A gate's output reuses the row of a net whose last reader is in an earlier
    level, so no gate of a level writes a row another gate of that level reads.
    The constants, the ports and the flop D and Q nets keep their rows for good.
    """
    last_reader = {}  # net -> the highest level that reads it
    for gate, lv in zip(gates, level):
        for n in gate.inputs:
            last_reader[n] = lv  # levels only grow, so the last write is the highest
    kept = {0, 1} | {n for nets in netlist.inputs.values() for n in nets}
    kept |= {n for nets in netlist.outputs.values() for n in nets}
    kept |= {flop.d for flop in netlist.flops} | {flop.q for flop in netlist.flops}
    row = np.zeros(netlist.n_nets, dtype=np.int32)
    row[sorted(kept)] = np.arange(len(kept))  # nets 0 and 1 land on rows 0 and 1
    n_rows = len(kept)
    free = []  # rows whose net has been read for the last time
    freed_after = defaultdict(list)  # level -> rows that are free once it is done
    for gate, lv in zip(gates, level):
        free += freed_after.pop(lv - 1, [])  # only a level's first gate finds any
        if gate.output in kept:
            continue
        if free:
            row[gate.output] = free.pop()
        else:
            row[gate.output] = n_rows
            n_rows += 1
        freed_after[last_reader.get(gate.output, lv)].append(row[gate.output])
    return row, n_rows


def pack(netlist: Netlist) -> Packed:
    """Sort the gates by level, give every net a row, and pack the arrays."""
    level = gate_levels(netlist)
    # stable keeps each level in netlist order: the layout the published benchmarks used
    order = np.argsort(level, kind="stable")
    gates = [netlist.gates[i] for i in order]
    row, n_rows = assign_rows(netlist, gates, [level[i] for i in order])
    in_nets = np.zeros((len(gates), 3), dtype=np.int32)
    for ins, gate in zip(in_nets, gates):
        ins[: len(gate.inputs)] = row[gate.inputs]
    return Packed(
        netlist,
        row=row,
        n_rows=n_rows,
        kind=np.array([KINDS.index(gate.kind) for gate in gates], dtype=np.uint8),
        in_nets=in_nets,
        out_net=row[[gate.output for gate in gates]],
        # gates per level (index = level, none at 0), running total = start offsets
        level_start=np.cumsum(np.bincount(level, minlength=1), dtype=np.int32),
        flop_d=row[[flop.d for flop in netlist.flops]],
        flop_q=row[[flop.q for flop in netlist.flops]],
    )
