"""Sort gates into levels so that every gate's inputs are ready before it runs.

Level 0 is what is known at the start of a cycle: the input ports, the constants
and the flop outputs. A gate's level is one more than the highest level among its
inputs, so the gates of one level depend only on lower levels and can all be
evaluated at once. A gate that never becomes ready is on a combinational loop,
or is fed by one.
"""

from collections import defaultdict, deque

from bitlane.netlist import Netlist


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
