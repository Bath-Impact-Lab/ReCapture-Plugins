# Gait Plugin

Author: Logan Wade, University of Bath

This pilot movement plugin contains the gait analysis, result generation, and
graph export code needed to process gait trials and prepare outputs for the
ReCapture IRIS graph schema.

## Files

- `gait_analysis.py`: gait event detection and movement-specific analysis.
- `gait_results.py`: gait result calculation and optional raw output export.
- `gait_graph_export.py`: converts raw gait output pickle files into graph JSON.
- `assets/`: movement-specific images or visual assets used by graph displays.

## Current Status

This folder is the pilot movement-plugin layout for gait. Shared utilities are
provided by `recapture_core`; movement-specific analysis, results, graph export,
and assets are kept together in this folder.
