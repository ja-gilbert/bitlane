"""Read a Yosys JSON netlist into gates, flops and ports with dense net ids.

Net 0 is constant 0 and net 1 is constant 1, so a JSON bit "0" or "1" is a net
like any other. Every other net is numbered from 2 in order of first use.
"""

import json
from dataclasses import dataclass
from pathlib import Path

GATES = {  # Yosys cell type -> (kind, input ports in order); the output port is Y
    "$_NOT_": ("NOT", ("A",)),
    "$_AND_": ("AND", ("A", "B")),
    "$_OR_": ("OR", ("A", "B")),
    "$_XOR_": ("XOR", ("A", "B")),
    "$_MUX_": ("MUX", ("A", "B", "S")),  # Y = B if S else A
}
FLOP = "$_DFF_P_"  # ports C (clock), D, Q


@dataclass
class Gate:
    kind: str  # NOT, AND, OR, XOR or MUX
    inputs: list[int]  # net per input port, in the order GATES lists them
    output: int


@dataclass
class Flop:
    d: int
    q: int


@dataclass
class Netlist:
    n_nets: int  # nets 0 and 1 are the constants
    inputs: dict[str, list[int]]  # port name -> net per bit, bit 0 first
    outputs: dict[str, list[int]]
    clock: str | None  # the input port on every flop's C pin; None without flops
    gates: list[Gate]
    flops: list[Flop]


def read_netlist(path: Path) -> Netlist:
    modules = json.loads(path.read_text())["modules"]
    (module,) = modules.values()  # flattened, so exactly one module
    ids: dict[int, int] = {}  # Yosys wire id -> our net id

    def net(bit: int | str) -> int:
        """Our net id for a JSON bit: a Yosys wire id or the constant "0"/"1"."""
        if isinstance(bit, int):
            return ids.setdefault(bit, 2 + len(ids))
        if bit in ("0", "1"):
            return int(bit)
        raise ValueError(f"unsupported bit {bit!r}")  # "x" and "z": task 3

    inputs, outputs = {}, {}
    for name, port in module["ports"].items():
        ports = inputs if port["direction"] == "input" else outputs
        ports[name] = [net(bit) for bit in port["bits"]]

    gates, flops, clocks = [], [], set()
    for name, cell in module["cells"].items():
        pins = cell["connections"]
        if cell["type"] == FLOP:
            flops.append(Flop(d=net(pins["D"][0]), q=net(pins["Q"][0])))
            clocks.add(net(pins["C"][0]))
        elif cell["type"] in GATES:
            kind, in_ports = GATES[cell["type"]]
            gate_inputs = [net(pins[p][0]) for p in in_ports]
            gates.append(Gate(kind, gate_inputs, net(pins["Y"][0])))
        else:
            raise ValueError(f"unsupported cell type {cell['type']} in cell {name}")

    clock = None
    if clocks:
        if len(clocks) > 1:
            raise ValueError(f"flops use {len(clocks)} different clock nets")
        (clock_net,) = clocks
        clock = next((n for n, bits in inputs.items() if bits == [clock_net]), None)
        if clock is None:
            raise ValueError("the clock is not a 1-bit input port")

    return Netlist(2 + len(ids), inputs, outputs, clock, gates, flops)
