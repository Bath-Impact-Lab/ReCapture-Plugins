from __future__ import annotations

import math
import re
from contextlib import redirect_stdout
from io import StringIO
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from recapture_analysis import (
    AnalysisResult,
    BoxPlot,
    BoxPlotDatum,
    LineChart,
    Notice,
    Point,
    Scalar,
    Series,
    Table,
    TableColumn,
)

from recapture_biomechanics.gait.configuration import GaitConfiguration
from recapture_biomechanics.shared.inputs import GaitInputPaths


def run_gait_analysis(
    inputs: GaitInputPaths,
    configuration: GaitConfiguration,
) -> AnalysisResult:
    """Run Logan Wade's gait engine and adapt its legacy measures to SDK v1."""
    from plugins.logan_wade_university_of_bath.gait.gait_analysis import (
        gait_analysis,
    )
    from plugins.logan_wade_university_of_bath.gait.main import Outputs

    motion_path = Path(inputs.inverse_kinematics)
    side = {
        "left": "l",
        "right": "r",
        "auto": "auto",
        "bilateral": "auto",
    }[configuration.side]
    # The subprocess protocol owns stdout. Legacy diagnostic prints must never
    # be allowed to corrupt its newline-delimited JSON messages.
    with redirect_stdout(StringIO()):
        gait = gait_analysis(
            str(motion_path.parent),
            motion_path.stem,
            str(inputs.model),
            str(inputs.markers),
            leg=side,
            lowpass_cutoff_frequency_for_coordinate_values=(
                configuration.filter_hz
                if configuration.filter_hz is not None
                else -1
            ),
            n_gait_cycles=(
                configuration.maximum_cycles
                if configuration.maximum_cycles is not None
                else -1
            ),
            gait_style=configuration.gait_style,
            trimming_start=configuration.trim_start_seconds,
            trimming_end=configuration.trim_end_seconds,
            visualize=False,
            pause_after_visualization=False,
            trial_label=motion_path.stem,
        )
        legacy = Outputs(gait, metadata={"trial_name": motion_path.stem})
    outputs = adapt_legacy_outputs(legacy)
    cycle_count = int(gait.nGaitCycles)
    outputs.insert(
        0,
        Scalar(
            id="valid-gait-cycles",
            title="Valid gait cycles",
            value=cycle_count,
            unit="cycles",
        ),
    )
    if cycle_count < 2:
        outputs.insert(
            1,
            Notice(
                id="limited-gait-cycles",
                title="Limited gait cycles",
                level="warning",
                message=(
                    "Only one complete gait cycle was detected; distribution "
                    "and symmetry results should be interpreted cautiously."
                ),
            ),
        )
    return AnalysisResult(
        title="Bath gait analysis",
        description=(
            "Bilateral temporospatial, joint-angle, pelvis, and trunk metrics "
            "computed using the University of Bath gait pipeline."
        ),
        outputs=outputs,
    )


def adapt_legacy_outputs(legacy: Mapping[str, Any]) -> list[Any]:
    outputs: list[Any] = []
    speed = legacy.get("speed")
    if isinstance(speed, Mapping):
        metric = speed.get("metric")
        if isinstance(metric, Mapping):
            value = _finite_float(metric.get("var"))
            if value is not None:
                outputs.append(
                    Scalar(
                        id="gait-speed",
                        title="Gait speed",
                        value=value,
                        unit=_unit(metric.get("units")),
                    )
                )

    for path, measure in _walk_measures(legacy):
        if path == ("speed",) or path[0] == "Segment_Events":
            continue
        output = _adapt_measure(path, measure)
        if output is not None:
            outputs.append(output)

    events = _adapt_events(legacy.get("Segment_Events"))
    if events is not None:
        outputs.append(events)
    return outputs


def _walk_measures(
    value: Mapping[str, Any],
    path: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], Mapping[str, Any]]]:
    for key, item in value.items():
        child_path = (*path, str(key))
        if not isinstance(item, Mapping):
            continue
        if any(name in item for name in ("singleLeftRight", "singleCenter", "multiLeftRight")):
            yield child_path, item
        else:
            yield from _walk_measures(item, child_path)


def _adapt_measure(path: tuple[str, ...], measure: Mapping[str, Any]) -> Any | None:
    output_id = _slug("-".join(path))
    title = _title(path[-1])
    description = _optional_text(measure.get("description"))
    units = measure.get("units")
    y_unit = _axis_value(units, "y", "unit")
    y_label = _axis_value(units, "y", "label") or title

    if "singleLeftRight" in measure:
        values = measure["singleLeftRight"]
        if not isinstance(values, Mapping):
            return None
        data = [
            datum
            for side, label in (("l", "Left"), ("r", "Right"))
            if (datum := _box_plot_datum(label, values.get(side))) is not None
        ]
        if not data:
            return None
        return BoxPlot(
            id=output_id,
            title=title,
            description=description,
            yLabel=_label_with_unit(y_label, y_unit),
            data=data,
        )

    if "singleCenter" in measure:
        datum = _box_plot_datum(title, measure["singleCenter"])
        if datum is None:
            return None
        return BoxPlot(
            id=output_id,
            title=title,
            description=description,
            yLabel=_label_with_unit(y_label, y_unit),
            data=[datum],
        )

    values = measure.get("multiLeftRight")
    if not isinstance(values, Mapping):
        return None
    series = [
        series_value
        for side, label, color in (
            ("l", "Left", "#3377cc"),
            ("r", "Right", "#d85847"),
        )
        if (
            series_value := _mean_cycle_series(
                label,
                values.get(side),
                color,
            )
        )
        is not None
    ]
    if not series:
        return None
    return LineChart(
        id=output_id,
        title=title,
        description=description,
        xLabel=_label_with_unit(
            _axis_value(units, "x", "label") or "Gait cycle",
            _axis_value(units, "x", "unit") or "%",
        ),
        yLabel=_label_with_unit(y_label, y_unit),
        series=series,
    )


def _box_plot_datum(label: str, raw: Any) -> BoxPlotDatum | None:
    values = _finite_values(raw)
    if not values:
        return None
    minimum, q1, median, q3, maximum = np.percentile(
        np.asarray(values, dtype=float),
        [0, 25, 50, 75, 100],
    )
    return BoxPlotDatum(
        label=label,
        min=float(minimum),
        q1=float(q1),
        median=float(median),
        q3=float(q3),
        max=float(maximum),
    )


def _mean_cycle_series(
    name: str,
    raw: Any,
    color: str,
) -> Series | None:
    array = np.asarray(raw, dtype=float)
    if array.size == 0:
        return None
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        return None
    mean = np.nanmean(array, axis=1)
    if not np.any(np.isfinite(mean)):
        return None
    denominator = max(len(mean) - 1, 1)
    points = [
        Point(x=index * 100 / denominator, y=float(value))
        for index, value in enumerate(mean)
        if math.isfinite(float(value))
    ]
    return Series(name=name, color=color, points=points)


def _adapt_events(raw: Any) -> Table | None:
    if not isinstance(raw, Mapping):
        return None
    event_sets = raw.get("event_sets")
    if not isinstance(event_sets, Mapping):
        return None
    rows: list[dict[str, Any]] = []
    for side, event_set in event_sets.items():
        if not isinstance(event_set, Mapping):
            continue
        times = np.asarray(event_set.get("Segment_Time", []), dtype=float)
        if times.ndim != 2:
            continue
        names = event_set.get("Event_Names") or []
        for cycle_index, cycle in enumerate(times, start=1):
            for event_index, event_time in enumerate(cycle):
                if not math.isfinite(float(event_time)):
                    continue
                name = (
                    str(names[event_index])
                    if event_index < len(names)
                    else f"Event {event_index + 1}"
                )
                rows.append(
                    {
                        "side": str(side).title(),
                        "cycle": cycle_index,
                        "event": name,
                        "time": float(event_time),
                    }
                )
    if not rows:
        return None
    return Table(
        id="gait-events",
        title="Detected gait events",
        columns=[
            TableColumn(key="side", label="Side"),
            TableColumn(key="cycle", label="Cycle"),
            TableColumn(key="event", label="Event"),
            TableColumn(key="time", label="Time", unit="s"),
        ],
        rows=rows,
    )


def _finite_values(raw: Any) -> list[float]:
    try:
        values = np.asarray(raw, dtype=float).ravel()
    except (TypeError, ValueError):
        return []
    return [float(value) for value in values if math.isfinite(float(value))]


def _finite_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _axis_value(raw: Any, axis: str, key: str) -> str | None:
    if not isinstance(raw, Mapping):
        return None
    axis_value = raw.get(axis)
    if not isinstance(axis_value, Mapping):
        return None
    return _optional_text(axis_value.get(key))


def _unit(raw: Any) -> str | None:
    if isinstance(raw, (list, tuple, np.ndarray)):
        raw = raw[0] if len(raw) else None
    return _optional_text(raw)


def _label_with_unit(label: str, unit: str | None) -> str:
    return f"{label} ({unit})" if unit else label


def _optional_text(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _title(value: str) -> str:
    return re.sub(r"[_-]+", " ", value).strip().title()
