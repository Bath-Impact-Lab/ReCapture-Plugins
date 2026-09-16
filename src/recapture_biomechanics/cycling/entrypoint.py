from recapture_analysis import AnalysisContext, AnalysisResult, analysis
from recapture_biomechanics import __version__
from recapture_biomechanics.movement_configuration import CyclingConfiguration, common_options
from recapture_biomechanics.movement_entrypoint import MOVEMENT_REQUIRED_ASSET_IDS, run_legacy_movement


@analysis(id="bath-cycling", name="Bath cycling analysis", version=__version__,
          description="Cycling-cycle segmentation and biomechanical measures.",
          required_assets=MOVEMENT_REQUIRED_ASSET_IDS, configuration=CyclingConfiguration)
def run(context: AnalysisContext) -> AnalysisResult:
    from plugins.logan_wade_university_of_bath.cycling.ergo_cycling_analysis import ergo_cycling_analysis
    from plugins.logan_wade_university_of_bath.cycling.main import Outputs
    configuration = CyclingConfiguration.model_validate(context.configuration)
    side = {"left": "l", "right": "r", "auto": "auto", "bilateral": "auto"}[configuration.side]
    options = common_options(configuration)
    options.update(side=side, cycle_signal=configuration.cycle_signal,
                   n_cycles=configuration.maximum_cycles or -1)
    return run_legacy_movement(context, title="Bath cycling analysis",
                               factory=ergo_cycling_analysis, output=Outputs,
                               factory_options=options)
