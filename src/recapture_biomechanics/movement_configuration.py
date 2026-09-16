"""Configuration models for non-gait movement analyses."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseMovementConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trim_start_seconds: float = Field(default=0, ge=0)
    trim_end_seconds: float = Field(default=0, ge=0)
    filter_hz: float | None = Field(default=None, gt=0)


class JumpConfiguration(BaseMovementConfiguration):
    vertical_signal: Literal["pelvis_vertical", "foot_center_vertical"] = "pelvis_vertical"


class SitToStandConfiguration(BaseMovementConfiguration):
    trunk_signal: Literal["lumbar_extension", "trunk_flexion"] = "lumbar_extension"


class RowingConfiguration(BaseMovementConfiguration):
    cycle_signal: Literal["wrist_anterior"] = "wrist_anterior"
    maximum_cycles: int | None = Field(default=None, ge=1)


class CyclingConfiguration(BaseMovementConfiguration):
    side: Literal["auto", "left", "right", "bilateral"] = "bilateral"
    cycle_signal: Literal["foot_center_vertical"] = "foot_center_vertical"
    maximum_cycles: int | None = Field(default=None, ge=1)


def common_options(configuration: BaseMovementConfiguration) -> dict[str, object]:
    return {
        "trimming_start": configuration.trim_start_seconds,
        "trimming_end": configuration.trim_end_seconds,
        "lowpass_cutoff_frequency_for_coordinate_values": (
            configuration.filter_hz if configuration.filter_hz is not None else -1
        ),
    }
