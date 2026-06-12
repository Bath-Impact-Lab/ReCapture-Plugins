# Jump Plugin

Author: Logan Wade, University of Bath

This movement plugin contains jump segmentation and result generation code.
Graph export will be added after the gait graph schema workflow has been
finalised.

## Files

- `jump_analysis.py`: jump event detection and movement-specific analysis.
- `jump_results.py`: jump result calculation and result dictionary creation.
- `assets/`: movement-specific images or visual assets used by graph displays.

## Current Status

This plugin uses shared infrastructure from `recapture_core` and does not depend
on the old `movement_processing_main` package layout.
