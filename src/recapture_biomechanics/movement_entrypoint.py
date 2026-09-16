"""Shared V3 execution and rich result adaptation for movement processors."""
from __future__ import annotations

import math
import re
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from recapture_analysis import (
    AnalysisContext, AnalysisResult, BoxPlot, BoxPlotDatum, LineChart,
    Notice, Point, Scalar, Series, Table, TableColumn,
)

from recapture_biomechanics.shared.inputs import (
    INVERSE_KINEMATICS_ASSET_ID, MARKER_TRAJECTORIES_ASSET_ID,
    SCALED_MODEL_ASSET_ID,
)

MOVEMENT_REQUIRED_ASSET_IDS = [
    SCALED_MODEL_ASSET_ID, MARKER_TRAJECTORIES_ASSET_ID,
    INVERSE_KINEMATICS_ASSET_ID,
]


def run_legacy_movement(
    context: AnalysisContext, *, title: str, factory: Callable[..., Any],
    output: Callable[[Any], Mapping[str, Any]],
    factory_options: Mapping[str, Any] | None = None,
) -> AnalysisResult:
    """Run a legacy processor and translate its complete result to SDK v1."""
    model = context.asset(SCALED_MODEL_ASSET_ID).source_path
    markers = context.asset(MARKER_TRAJECTORIES_ASSET_ID).source_path
    motion = context.asset(INVERSE_KINEMATICS_ASSET_ID).source_path
    _validate_input(model, ".osim")
    _validate_input(markers, ".trc")
    _validate_input(motion, ".mot")
    context.report_progress(0.1, "Loading movement data")
    # Legacy timing diagnostics cannot share stdout with the JSONL protocol.
    with redirect_stdout(StringIO()):
        processor = factory(
            str(motion.parent), motion.stem, str(model), str(markers),
            visualize=False, pause_after_visualization=False,
            trial_label=motion.stem, **dict(factory_options or {}),
        )
        context.report_progress(0.65, "Calculating movement measures")
        legacy = output(processor)
    results = adapt_movement_outputs(legacy)
    if not results:
        results.append(Notice(
            id="no-results", title="No valid movement results", level="warning",
            message="The processor completed but produced no finite measures.",
        ))
    context.report_progress(1.0, "Analysis complete")
    return AnalysisResult(
        title=title,
        description=f"{title} using the University of Bath biomechanics pipeline.",
        outputs=results,
    )


def adapt_movement_outputs(legacy: Mapping[str, Any]) -> list[Any]:
    """Translate legacy metrics, distributions, waveforms, and events."""
    outputs: list[Any] = []
    reserved = {"Segment_Events", "Angle_Graphs", "Time_Normalized_Coordinates"}
    for path, measure in _walk_measures(legacy):
        if path[0] in reserved:
            continue
        adapted = _adapt_measure(path, measure)
        if adapted is not None:
            outputs.append(adapted)
    outputs.extend(_adapt_angle_graphs(legacy.get("Angle_Graphs")))
    events = _adapt_events(legacy.get("Segment_Events"))
    if events is not None:
        outputs.append(events)
    return _unique(outputs)


def _walk_measures(value: Mapping[str, Any], path: tuple[str, ...] = ()):
    for key, item in value.items():
        child_path = (*path, str(key))
        if not isinstance(item, Mapping):
            continue
        if any(name in item for name in ("value", "mean", "LR", "var")):
            yield child_path, item
        else:
            yield from _walk_measures(item, child_path)


def _adapt_measure(path: tuple[str, ...], measure: Mapping[str, Any]):
    output_id = _slug("-".join(path))
    title = _title(path[-1])
    description = _text(measure.get("description"))
    unit = _unit(measure.get("units"))
    if "LR" in measure and isinstance(measure["LR"], Mapping):
        data = []
        for side, label in (("l", "Left"), ("r", "Right")):
            values = _finite_values(measure["LR"].get(side))
            if values:
                data.append(_box(label, values))
        if data:
            return BoxPlot(id=output_id, title=title, description=description,
                           yLabel=_label(title, unit), data=data)
    values = _finite_values(measure.get("value", measure.get("var")))
    if len(values) == 1:
        return Scalar(id=output_id, title=title, description=description,
                      value=values[0], unit=unit)
    if values:
        return BoxPlot(id=output_id, title=title, description=description,
                       yLabel=_label(title, unit), data=[_box(title, values)])
    mean = _finite_values(measure.get("mean"))
    if mean:
        return Scalar(id=output_id, title=title, description=description,
                      value=mean[0], unit=unit)
    return None


def _adapt_angle_graphs(value: Any) -> list[Any]:
    if not isinstance(value, Mapping):
        return []
    outputs = []
    for group, group_value in value.items():
        if not isinstance(group_value, Mapping):
            continue
        unit = _unit(group_value.get("units"))
        joints = group_value.get("LR")
        if not isinstance(joints, Mapping):
            continue
        for joint, sides in joints.items():
            if not isinstance(sides, Mapping):
                continue
            chart_series = []
            for side, label, color in (("l", "Left", "#2563eb"), ("r", "Right", "#dc2626")):
                curves = _curves(sides.get(side))
                if not curves:
                    continue
                mean_curve = np.mean(np.asarray(curves, dtype=float), axis=0)
                denominator = max(len(mean_curve) - 1, 1)
                chart_series.append(Series(
                    name=label, color=color,
                    points=[Point(x=100.0 * index / denominator, y=float(y))
                            for index, y in enumerate(mean_curve)],
                ))
            if chart_series:
                outputs.append(LineChart(
                    id=_slug(f"angle-graph-{group}-{joint}"),
                    title=f"{_title(str(joint))} angle",
                    description="Mean time-normalised movement waveform by side.",
                    xLabel="Movement cycle (%)", yLabel=_label("Angle", unit),
                    series=chart_series,
                ))
    return outputs


def _adapt_events(value: Any):
    if not isinstance(value, Mapping) or not isinstance(value.get("event_sets"), Mapping):
        return None
    rows = []
    for set_name, event_set in value["event_sets"].items():
        if not isinstance(event_set, Mapping):
            continue
        names = list(event_set.get("Event_Names", []))
        times = np.asarray(event_set.get("Segment_Time", []))
        if times.ndim != 2:
            continue
        for segment_index, segment in enumerate(times, start=1):
            for event_index, event_time in enumerate(segment):
                finite = _finite_float(event_time)
                if finite is not None:
                    rows.append({
                        "eventSet": str(set_name), "segment": segment_index,
                        "event": str(names[event_index]) if event_index < len(names) else f"Event {event_index + 1}",
                        "time": finite,
                    })
    if not rows:
        return None
    return Table(
        id="detected-events", title="Detected movement events",
        description="Event timings used to segment and normalise the analysis.",
        columns=[TableColumn(key="eventSet", label="Event set"),
                 TableColumn(key="segment", label="Segment"),
                 TableColumn(key="event", label="Event"),
                 TableColumn(key="time", label="Time", unit="s")], rows=rows,
    )


def _curves(value: Any) -> list[list[float]]:
    try:
        array = np.asarray(value)
    except Exception:
        return []
    if array.ndim == 1:
        values = [_finite_float(item) for item in array]
        return [[float(item) for item in values]] if values and all(item is not None for item in values) else []
    if array.ndim != 2:
        return []
    if array.shape[0] >= array.shape[1]:
        array = array.T
    curves = []
    for row in array:
        values = [_finite_float(item) for item in row]
        if values and all(item is not None for item in values):
            curves.append([float(item) for item in values])
    return curves


def _box(label: str, values: list[float]) -> BoxPlotDatum:
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return BoxPlotDatum(label=label, min=min(values), q1=float(q1),
                        median=float(median), q3=float(q3), max=max(values),
                        values=values)


def _finite_values(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        for key in ("values", "var", "value", "mean"):
            if key in value:
                return _finite_values(value[key])
        return []
    try:
        array = np.asarray(value).reshape(-1)
    except Exception:
        return []
    return [number for item in array if (number := _finite_float(item)) is not None]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_input(path, suffix: str) -> None:
    if path.suffix.lower() != suffix or not path.is_file():
        raise ValueError(f"A required {suffix} analysis input is unavailable.")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "result"


def _title(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _unit(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _label(title: str, unit: str | None) -> str:
    return f"{title} ({unit})" if unit else title


def _unique(outputs: Iterable[Any]) -> list[Any]:
    result, counts = [], {}
    for output in outputs:
        count = counts.get(output.id, 0)
        counts[output.id] = count + 1
        if count:
            output = output.model_copy(update={"id": f"{output.id}-{count + 1}"})
        result.append(output)
    return result
