from recapture_analysis import AnalysisContext, AnalysisResult, analysis
from recapture_biomechanics import __version__
from recapture_biomechanics.movement_configuration import SitToStandConfiguration, common_options
from recapture_biomechanics.movement_entrypoint import MOVEMENT_REQUIRED_ASSET_IDS, run_legacy_movement


@analysis(id="bath-sit-to-stand", name="Bath sit-to-stand analysis", version=__version__,
          description="Sit-to-stand segmentation and biomechanical measures.",
          required_assets=MOVEMENT_REQUIRED_ASSET_IDS, configuration=SitToStandConfiguration)
def run(context: AnalysisContext) -> AnalysisResult:
    from plugins.logan_wade_university_of_bath.sit_to_stand.sit_to_stand_analysis import sit_to_stand_analysis
    from plugins.logan_wade_university_of_bath.sit_to_stand.main import Outputs
    configuration = SitToStandConfiguration.model_validate(context.configuration)
    options = common_options(configuration)
    options["trunk_signal"] = configuration.trunk_signal
    return run_legacy_movement(context, title="Bath sit-to-stand analysis",
                               factory=sit_to_stand_analysis, output=Outputs,
                               factory_options=options)
