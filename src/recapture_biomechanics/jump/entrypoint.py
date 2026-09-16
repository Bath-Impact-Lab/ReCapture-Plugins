from recapture_analysis import AnalysisContext, AnalysisResult, analysis
from recapture_biomechanics import __version__
from recapture_biomechanics.movement_configuration import JumpConfiguration, common_options
from recapture_biomechanics.movement_entrypoint import MOVEMENT_REQUIRED_ASSET_IDS, run_legacy_movement


@analysis(id="bath-jump", name="Bath jump analysis", version=__version__,
          description="Jump segmentation and biomechanical measures.",
          required_assets=MOVEMENT_REQUIRED_ASSET_IDS, configuration=JumpConfiguration)
def run(context: AnalysisContext) -> AnalysisResult:
    from plugins.logan_wade_university_of_bath.jump.jump_analysis import jump_analysis
    from plugins.logan_wade_university_of_bath.jump.main import Outputs
    configuration = JumpConfiguration.model_validate(context.configuration)
    options = common_options(configuration)
    options["vertical_signal"] = configuration.vertical_signal
    return run_legacy_movement(context, title="Bath jump analysis", factory=jump_analysis,
                               output=Outputs, factory_options=options)
