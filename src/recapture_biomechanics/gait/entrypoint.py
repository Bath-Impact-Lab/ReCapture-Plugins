from __future__ import annotations

from recapture_analysis import AnalysisContext, analysis

from recapture_biomechanics import __version__
from recapture_biomechanics.gait.configuration import GaitConfiguration
from recapture_biomechanics.gait.legacy_adapter import run_gait_analysis
from recapture_biomechanics.shared.inputs import (
    GAIT_REQUIRED_ASSET_IDS,
    INVERSE_KINEMATICS_ASSET_ID,
    MARKER_TRAJECTORIES_ASSET_ID,
    PROCESSING_METADATA_ASSET_ID,
    SCALED_MODEL_ASSET_ID,
    GaitInputPaths,
)


@analysis(
    id="bath-gait",
    name="Bath gait analysis",
    version=__version__,
    description=(
        "Bilateral temporospatial and joint-kinematic gait analysis using "
        "ReCapture's processed OpenSim and marker assets."
    ),
    required_assets=GAIT_REQUIRED_ASSET_IDS,
    configuration=GaitConfiguration,
)
def run(context: AnalysisContext) -> AnalysisResult:
    context.report_progress(0.1, "Validating gait inputs")
    inputs = GaitInputPaths(
        model=context.asset(SCALED_MODEL_ASSET_ID).source_path,
        markers=context.asset(MARKER_TRAJECTORIES_ASSET_ID).source_path,
        inverse_kinematics=context.asset(
            INVERSE_KINEMATICS_ASSET_ID
        ).source_path,
        metadata=context.asset(PROCESSING_METADATA_ASSET_ID).source_path,
    )
    configuration = GaitConfiguration.model_validate(context.configuration)
    inputs.validate()
    context.report_progress(0.2, "Detecting gait cycles")
    result = run_gait_analysis(inputs, configuration)
    context.report_progress(1.0, "Gait analysis complete")
    return result
