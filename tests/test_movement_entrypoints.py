from __future__ import annotations

import numpy as np
import pytest

from recapture_biomechanics.movement_entrypoint import adapt_movement_outputs


def test_rich_legacy_output_is_adapted_to_sdk_outputs():
    legacy = {
        "Movement": {
            "Count": {"value": 2, "units": "cycles", "description": "Count"},
            "Duration": {"value": np.array([1.0, 1.2]), "units": "s"},
        },
        "Angle_Results": {
            "Knee": {
                "knee": {
                    "ROM": {"LR": {"l": [20.0, 22.0], "r": [19.0, 21.0]}, "units": "deg"}
                }
            }
        },
        "Angle_Graphs": {
            "knee": {
                "units": "deg",
                "LR": {"knee": {"l": np.ones((101, 2)), "r": np.ones((101, 2)) * 2}},
            }
        },
        "Segment_Events": {
            "event_sets": {
                "cycle": {
                    "Segment_Time": np.array([[0.0, 0.5, 1.0]]),
                    "Event_Names": ["Start", "Middle", "End"],
                }
            }
        },
    }

    outputs = adapt_movement_outputs(legacy)
    kinds = {output.kind for output in outputs}
    assert {"scalar", "box-plot", "line-chart", "table"} <= kinds
    chart = next(output for output in outputs if output.kind == "line-chart")
    assert len(chart.series) == 2
    assert len(chart.series[0].points) == 101
    events = next(output for output in outputs if output.kind == "table")
    assert len(events.rows) == 3


@pytest.mark.parametrize(
    ("module_name", "plugin_id"),
    [
        ("recapture_biomechanics.jump.entrypoint", "bath-jump"),
        ("recapture_biomechanics.sit_to_stand.entrypoint", "bath-sit-to-stand"),
        ("recapture_biomechanics.rowing.entrypoint", "bath-rowing"),
        ("recapture_biomechanics.cycling.entrypoint", "bath-cycling"),
    ],
)
def test_entrypoint_metadata_and_configuration(module_name, plugin_id):
    module = __import__(module_name, fromlist=["run"])
    assert module.run.metadata.id == plugin_id
    assert module.run.metadata.required_assets == [
        "scaled-model.osim",
        "marker-trajectories-augmented.trc",
        "inverse-kinematics.mot",
    ]
    assert module.run.metadata.configuration_schema is not None
