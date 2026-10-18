"""Edge-hardware simulation: compute limits for the edge process and a
real-time camera that drops frames the edge is too busy for."""

from .camera import CameraFeed, latest_finished
from .hardware import CpuMeter, EdgeHardware, apply_edge_hardware, format_cpu_list, parse_cpu_list

__all__ = ["CameraFeed", "latest_finished", "CpuMeter", "EdgeHardware", "apply_edge_hardware", "format_cpu_list", "parse_cpu_list"]
