"""Edge/ground simulation: network link emulation (netem-equivalent), relay,
wire protocol, and the edge/ground node logic."""

from .link import GilbertElliott, LinkConfig, LinkModel, PacketFate, parse_netem
from .nodes import EXCHANGE_FIELDS, Exchange, GroundServer, RemoteDetector, exchange_row
from .relay import LinkRelay, high_resolution_timer, interrupt_on_ctrl_break, make_udp_socket

__all__ = [
    "EXCHANGE_FIELDS",
    "Exchange",
    "GilbertElliott",
    "GroundServer",
    "LinkConfig",
    "LinkModel",
    "LinkRelay",
    "PacketFate",
    "RemoteDetector",
    "exchange_row",
    "high_resolution_timer",
    "interrupt_on_ctrl_break",
    "make_udp_socket",
    "parse_netem",
]
