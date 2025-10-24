"""Bacteria analysis module for micromorph."""

# Import the main Bacteria class
from .Bacteria import (
    Bacteria,
    get_bacteria_for_index,
    get_bacteria_list,
)

# Import fluorescence fitting functions
from .fluorescence_fitting import gaussian

# Define what gets exported with "from micromorph.bacteria import *"
__all__ = [
    # Main class
    "Bacteria",
    # Bacteria.py functions
    "get_bacteria_for_index",
    "get_bacteria_list",
    # # Shape analysis
    # "get_bacteria_length",
    # "get_bacteria_widths",
    # "get_bacteria_boundary",
    # "get_medial_axis",
    # "smooth_medial_axis",
    # "extend_medial_axis",
    # "extend_medial_axis_roughly",
    # # Utilities
    # "find_endpoints",
    # "find_furthest_point",
    # "find_closest_point",
    # "find_branchpoints",
    # "prune_short_branches",
    # "get_coords",
    # "trace_axis",
    # "fix_coordinates_order",
    # "get_boundary_coords",
    # "find_closest_boundary_to_axis",
    # "calculate_distance_along_line",
    # "get_width_profile_lines",
    # "apply_mask_to_image",
    # # Fluorescence fitting
    # "gaussian",
]
