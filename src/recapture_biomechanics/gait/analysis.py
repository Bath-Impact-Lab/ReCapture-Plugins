from __future__ import annotations

from dataclasses import dataclass

from recapture_biomechanics.gait.configuration import GaitConfiguration
from recapture_biomechanics.shared.inputs import GaitInputPaths


@dataclass(frozen=True)
class GaitIntegrationSummary:
    """SDK-independent result used while the scientific kernel is migrated."""

    model_bytes: int
    marker_bytes: int
    inverse_kinematics_bytes: int
    metadata_bytes: int
    side: str
    gait_style: str


def inspect_gait_inputs(
    inputs: GaitInputPaths,
    configuration: GaitConfiguration,
) -> GaitIntegrationSummary:
    """Validate the explicit inputs without knowing ReCapture storage layout."""

    inputs.validate()
    return GaitIntegrationSummary(
        model_bytes=inputs.model.stat().st_size,
        marker_bytes=inputs.markers.stat().st_size,
        inverse_kinematics_bytes=inputs.inverse_kinematics.stat().st_size,
        metadata_bytes=inputs.metadata.stat().st_size,
        side=configuration.side,
        gait_style=configuration.gait_style,
    )
