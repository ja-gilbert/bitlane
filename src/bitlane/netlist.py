"""Read a Yosys JSON netlist into gates, flops and ports with dense net ids.

Net 0 is constant 0 and net 1 is constant 1, so a JSON bit "0" or "1" is a net
like any other. Every other net is numbered from 2 in order of first use.
"""

import json
import warnings
from dataclasses import dataclass, field
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
    names: dict[int, str] = field(default_factory=dict)  # net -> public name


def bit_name(name: str, i: int, width: int) -> str:
    return name if width == 1 else f"{name}[{i}]"


def read_netlist(path: Path) -> Netlist:
    modules = json.loads(path.read_text())["modules"]
    (module,) = modules.values()  # flattened, so exactly one module
    ids: dict[int, int] = {}  # Yosys wire id -> our net id

    def net(bit: int | str, where: str) -> int:
        """Our net id for a JSON bit: a Yosys wire id or a constant "0"/"1"/"x"/"z"."""
        if isinstance(bit, int):
            return ids.setdefault(bit, 2 + len(ids))
        if bit in ("0", "1"):
            return int(bit)
        if bit == "x":
            warnings.warn(f"{where} is x; treating it as 0")
            return 0
        raise ValueError(f"{where} is {bit!r}; tri-state is not supported")

    inputs, outputs = {}, {}
    for name, port in module["ports"].items():
        ports = inputs if port["direction"] == "input" else outputs
        width = len(port["bits"])
        ports[name] = [
            net(b, bit_name(name, i, width)) for i, b in enumerate(port["bits"])
        ]

    gates, flops, clocks = [], [], set()
    for name, cell in module["cells"].items():
        pins = {
            p: net(bits[0], f"{name}.{p}") for p, bits in cell["connections"].items()
        }
        if cell["type"] == FLOP:
            flops.append(Flop(d=pins["D"], q=pins["Q"]))
            clocks.add(pins["C"])
        elif cell["type"] in GATES:
            kind, in_ports = GATES[cell["type"]]
            gates.append(Gate(kind, [pins[p] for p in in_ports], pins["Y"]))
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

    names = {}  # our net id -> public name, for messages
    for name, wire in module["netnames"].items():
        if not wire["hide_name"]:
            for i, bit in enumerate(wire["bits"]):
                if bit in ids:
                    names[ids[bit]] = bit_name(name, i, len(wire["bits"]))

    netlist = Netlist(2 + len(ids), inputs, outputs, clock, gates, flops, names)
    if undriven := sorted(undriven_nets(netlist)):
        missing = ", ".join(names.get(n, f"net {n}") for n in undriven)
        raise ValueError(f"no driver for {missing}")
    return netlist


def undriven_nets(netlist: Netlist) -> set[int]:
    """Nets something reads but nothing drives: no input, gate, flop or constant."""
    driven = {0, 1}
    driven |= {b for bits in netlist.inputs.values() for b in bits}
    driven |= {gate.output for gate in netlist.gates}
    driven |= {flop.q for flop in netlist.flops}
    used = {b for bits in netlist.outputs.values() for b in bits}
    used |= {b for gate in netlist.gates for b in gate.inputs}
    used |= {flop.d for flop in netlist.flops}
    return used - driven
