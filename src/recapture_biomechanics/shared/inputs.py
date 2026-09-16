from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SCALED_MODEL_ASSET_ID = "scaled-model.osim"
MARKER_TRAJECTORIES_ASSET_ID = "marker-trajectories-augmented.trc"
INVERSE_KINEMATICS_ASSET_ID = "inverse-kinematics.mot"
PROCESSING_METADATA_ASSET_ID = "processing-metadata.json"

GAIT_REQUIRED_ASSET_IDS = [
    SCALED_MODEL_ASSET_ID,
    MARKER_TRAJECTORIES_ASSET_ID,
    INVERSE_KINEMATICS_ASSET_ID,
    PROCESSING_METADATA_ASSET_ID,
]


@dataclass(frozen=True)
class GaitInputPaths:
    """Manager-approved inputs required by the gait processor."""

    model: Path
    markers: Path
    inverse_kinematics: Path
    metadata: Path

    def validate(self) -> None:
        expected_suffixes = {
            self.model: ".osim",
            self.markers: ".trc",
            self.inverse_kinematics: ".mot",
            self.metadata: ".json",
        }
        for input_path, expected_suffix in expected_suffixes.items():
            if input_path.suffix.lower() != expected_suffix:
                raise ValueError(
                    f"Expected a {expected_suffix} analysis input."
                )
            if not input_path.is_file():
                raise ValueError("A required gait analysis input is unavailable.")
