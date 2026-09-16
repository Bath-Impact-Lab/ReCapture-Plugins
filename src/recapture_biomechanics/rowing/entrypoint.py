from recapture_analysis import AnalysisContext, AnalysisResult, analysis
from recapture_biomechanics import __version__
from recapture_biomechanics.movement_configuration import RowingConfiguration, common_options
from recapture_biomechanics.movement_entrypoint import MOVEMENT_REQUIRED_ASSET_IDS, run_legacy_movement


@analysis(id="bath-rowing", name="Bath rowing analysis", version=__version__,
          description="Rowing-cycle segmentation and biomechanical measures.",
          required_assets=MOVEMENT_REQUIRED_ASSET_IDS, configuration=RowingConfiguration)
def run(context: AnalysisContext) -> AnalysisResult:
    from plugins.logan_wade_university_of_bath.rowing.ergo_rowing_analysis import ergo_rowing_analysis
    from plugins.logan_wade_university_of_bath.rowing.main import Outputs
    configuration = RowingConfiguration.model_validate(context.configuration)
    options = common_options(configuration)
    options.update(cycle_signal=configuration.cycle_signal,
                   n_cycles=configuration.maximum_cycles or -1)
    return run_legacy_movement(context, title="Bath rowing analysis",
                               factory=ergo_rowing_analysis, output=Outputs,
                               factory_options=options)
