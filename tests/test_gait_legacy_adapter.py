from recapture_biomechanics.gait.legacy_adapter import adapt_legacy_outputs


def test_adapts_scalar_bilateral_and_curve_outputs() -> None:
    outputs = adapt_legacy_outputs(
        {
            "speed": {"metric": {"var": 1.25, "units": "m/s"}},
            "Temporospatial": {
                "Step_Length": {
                    "singleLeftRight": {
                        "l": [0.5, 0.6, 0.7],
                        "r": [0.55, 0.65, 0.75],
                    },
                    "units": {
                        "y": {"label": "Step length", "unit": "m"}
                    },
                }
            },
            "jointAngle_timeseries": {
                "ankle": {
                    "multiLeftRight": {
                        "l": [[0, 2], [10, 12], [20, 22]],
                        "r": [[1, 3], [11, 13], [21, 23]],
                    },
                    "units": {
                        "x": {"label": "Gait cycle", "unit": "%"},
                        "y": {"label": "Ankle angle", "unit": "deg"},
                    },
                }
            },
        }
    )

    assert [output.id for output in outputs] == [
        "gait-speed",
        "temporospatial-step-length",
        "jointangle-timeseries-ankle",
    ]
    assert outputs[0].value == 1.25
    assert outputs[1].kind == "box-plot"
    assert outputs[1].data[0].median == 0.6
    assert outputs[2].kind == "line-chart"
    assert [point.y for point in outputs[2].series[0].points] == [
        1.0,
        11.0,
        21.0,
    ]
