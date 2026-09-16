from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GaitConfiguration(BaseModel):
    """User-facing configuration for the first gait integration."""

    model_config = ConfigDict(extra="forbid")

    side: Literal["auto", "left", "right", "bilateral"] = "bilateral"
    gait_style: Literal["auto", "overground", "treadmill"] = "auto"
    maximum_cycles: int | None = Field(default=None, ge=1)
    trim_start_seconds: float = Field(default=0, ge=0)
    trim_end_seconds: float = Field(default=0, ge=0)
    filter_hz: float | None = Field(default=None, gt=0)
    marker_profile: Literal["auto", "opencap", "theia"] = "auto"
