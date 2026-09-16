from pathlib import Path

from recapture_analysis import AnalysisContext, AnalysisResult, Scalar

from recapture_biomechanics.gait import entrypoint
from recapture_biomechanics.gait.entrypoint import run
from recapture_biomechanics.shared.inputs import GAIT_REQUIRED_ASSET_IDS


def test_gait_entrypoint_declares_stable_processed_assets() -> None:
    assert run.metadata.id == "bath-gait"
    assert run.metadata.required_assets == GAIT_REQUIRED_ASSET_IDS
    assert run.metadata.configuration_schema is not None


def test_gait_entrypoint_validates_manager_approved_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = {
        "scaled-model.osim": tmp_path / "scaled-model.osim",
        "marker-trajectories-augmented.trc": (
            tmp_path / "marker-trajectories-augmented.trc"
        ),
        "inverse-kinematics.mot": tmp_path / "inverse-kinematics.mot",
        "processing-metadata.json": tmp_path / "processing-metadata.json",
    }
    for path in paths.values():
        path.write_text("x", encoding="utf-8")

    progress: list[tuple[float, str | None]] = []
    context = AnalysisContext.model_validate(
        {
            "runId": "run-1",
            "participantId": "participant-1",
            "sessionId": "session-1",
            "exerciseId": "exercise-1",
            "assets": [
                {
                    "logicalId": logical_id,
                    "sourcePath": str(source_path),
                    "sizeBytes": 1,
                }
                for logical_id, source_path in paths.items()
            ],
            "configuration": {},
        }
    )
    context._set_progress_reporter(
        lambda value, message: progress.append((value, message))
    )
    monkeypatch.setattr(
        entrypoint,
        "run_gait_analysis",
        lambda inputs, configuration: AnalysisResult(
            title="Bath gait analysis",
            outputs=[
                Scalar(
                    id="gait-speed",
                    title="Gait speed",
                    value=1.2,
                    unit="m/s",
                )
            ],
        ),
    )

    result = run(context)

    assert result.title == "Bath gait analysis"
    assert [output.id for output in result.outputs] == ["gait-speed"]
    assert progress == [
        (0.1, "Validating gait inputs"),
        (0.2, "Detecting gait cycles"),
        (1.0, "Gait analysis complete"),
    ]
