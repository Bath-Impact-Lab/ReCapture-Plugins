# -*- coding: utf-8 -*-
"""
Export gait result outputs to the ReCapture IRIS graph schema.

This script reads the raw output pickle produced by gait_results.save_outputs()
and converts selected gait outcomes into the frontend graph JSON format defined
by ReCapture-IRIS/recapture-lib/new_graph_schema.json.

The first supported outcome is:
    Temporospatial -> Step_Length
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import jsonschema


def default_schema_path():
    return (
        Path.home() /
        'Dropbox/University/GitHubClones/ReCapture-IRIS/recapture-lib/new_graph_schema.json'
    )


def load_gait_outputs(path):
    path = Path(path)
    with open(path, 'rb') as file:
        return pickle.load(file)


def export_gait_graph_payload(gait_outputs):
    payload = {}
    toe_off_vertical_lines = _toe_off_vertical_lines_from_segment_events(
        gait_outputs.get('Segment_Events', {})
    )

    ankle_graph = (
        gait_outputs
        .get('Angle_Graphs', {})
        .get('ankle')
    )
    if ankle_graph is not None:
        _set_nested_measure(
            payload,
            group='Angle_Graphs',
            measure_name='ankle',
            measure=make_multi_left_right_measure(
                values=ankle_graph.get('LR', {}),
                x_label='Gait cycle',
                x_unit='%',
                y_label='Ankle angle',
                y_unit=_normalise_unit(ankle_graph.get('units', 'deg')),
                description=ankle_graph.get(
                    'description',
                    'Ankle angle time-normalised across the gait cycle.'
                ),
                tags=[
                    {
                        'axis': 'y',
                        'text': 'Dorsiflexion',
                        'image': 'assets/ankle_dorsiflexion.png',
                        'position': 'end'
                    },
                    {
                        'axis': 'y',
                        'text': 'Plantarflexion',
                        'image': 'assets/ankle_plantarflexion.png',
                        'position': 'start'
                    },
                    {
                        'axis': 'x',
                        'text': 'Heel strike',
                        'position': 'start'
                    },
                    {
                        'axis': 'x',
                        'text': 'Next heel strike',
                        'position': 'end'
                    }
                ],
                vertical_lines=toe_off_vertical_lines
            )
        )

    step_length = (
        gait_outputs
        .get('Temporospatial', {})
        .get('Step_Length')
    )
    if step_length is not None:
        _set_nested_measure(
            payload,
            group='Temporospatial',
            measure_name='Step_Length',
            measure=make_single_left_right_measure(
                values=step_length.get('LR', {}),
                y_label='Step length',
                y_unit=_normalise_unit(step_length.get('units', 'm')),
                description=step_length.get(
                    'description',
                    'Anterior-posterior distance between heel-strike of one foot and heel-strike of the contralateral foot.'
                ),
                box_plot_name='Step length',
                tags=[
                    {
                        'axis': 'y',
                        'text': 'Longer steps',
                        'position': 'end'
                    },
                    {
                        'axis': 'y',
                        'text': 'Shorter steps',
                        'position': 'start'
                    }
                ]
            )
        )

    return payload


def make_multi_left_right_measure(values, x_label, x_unit, y_label, y_unit,
                                  description, vertical_lines=None, tags=None):
    return _drop_none({
        'multiLeftRight': {
            'r': _as_2d_float_list(values.get('r', [])),
            'l': _as_2d_float_list(values.get('l', []))
        },
        'units': {
            'x': {
                'label': x_label,
                'unit': x_unit
            },
            'y': {
                'label': y_label,
                'unit': y_unit
            }
        },
        'description': description,
        'tags': tags,
        'verticalLines': vertical_lines
    })


def make_single_left_right_measure(values, y_label, y_unit, description,
                                   box_plot_name=None, tags=None):
    return _drop_none({
        'singleLeftRight': {
            'r': _as_float_list(values.get('r', [])),
            'l': _as_float_list(values.get('l', []))
        },
        'units': {
            'y': {
                'label': y_label,
                'unit': y_unit
            }
        },
        'description': description,
        'tags': tags,
        'boxPlotName': box_plot_name
    })


def _set_nested_measure(payload, group, measure_name, measure):
    payload.setdefault(group, {})[measure_name] = measure


def _as_float_list(values):
    arr = np.asarray(values, dtype=float).ravel()
    return [
        float(value)
        for value in arr
        if np.isfinite(value)
    ]


def _as_2d_float_list(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    arr = np.where(np.isfinite(arr), arr, np.nan)
    return [
        [
            None if np.isnan(value) else float(value)
            for value in row
        ]
        for row in arr
    ]


def _toe_off_vertical_lines_from_segment_events(segment_events):
    toe_off_percent = _toe_off_percentages_from_segment_events(segment_events)
    if len(toe_off_percent) == 0:
        return None
    return [
        {
            'value': toe_off_percent,
            'label': 'Toe-off'
        }
    ]


def _toe_off_percentages_from_segment_events(segment_events):
    event_sets = segment_events.get('event_sets', {})
    toe_off_values = []

    for event_set in event_sets.values():
        segment_idx = event_set.get('Segment_Idx')
        if segment_idx is None:
            continue

        event_matrix = np.asarray(segment_idx, dtype=float)
        if event_matrix.ndim != 2 or event_matrix.shape[1] < 3:
            continue

        stride_duration = event_matrix[:, 2] - event_matrix[:, 0]
        valid = np.isfinite(stride_duration) & (stride_duration > 0)
        if not np.any(valid):
            continue

        toe_off = (
            100 *
            (event_matrix[:, 1] - event_matrix[:, 0]) /
            stride_duration
        )
        toe_off = toe_off[valid]
        toe_off = toe_off[np.isfinite(toe_off)]
        toe_off_values.extend(float(value) for value in toe_off)

    return toe_off_values


def _normalise_unit(unit):
    if isinstance(unit, (list, tuple, np.ndarray)):
        if len(unit) == 0:
            return ''
        unit = unit[0]
    if isinstance(unit, np.generic):
        unit = unit.item()
    return str(unit)


def _drop_none(value):
    return {
        key: item
        for key, item in value.items()
        if item is not None
    }


def validate_graph_payload(payload, schema_path):
    schema_path = Path(schema_path)
    with open(schema_path, 'r') as file:
        schema = json.load(file)

    jsonschema.validate(instance=payload, schema=schema)


def save_graph_payload(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as file:
        json.dump(payload, file, indent=2)
    return output_path


def main(input_pickle, output_json=None, schema_path=None, validate=True):
    input_pickle = Path(input_pickle)
    schema_path = Path(schema_path) if schema_path is not None else default_schema_path()

    if output_json is None:
        output_json = input_pickle.with_name(
            input_pickle.stem.replace('_outputs', '_graphs') + '.json'
        )
    output_json = Path(output_json)

    gait_outputs = load_gait_outputs(input_pickle)
    payload = export_gait_graph_payload(gait_outputs)

    if validate:
        validate_graph_payload(payload, schema_path)
        print(f'Validated graph payload against: {schema_path}')

    saved_path = save_graph_payload(payload, output_json)
    print(f'Saved gait graph JSON to: {saved_path}')
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export gait output pickle to ReCapture IRIS graph JSON.'
    )
    parser.add_argument('input_pickle', help='Path to gait output pickle.')
    parser.add_argument('--output-json', default=None,
                        help='Output graph JSON path.')
    parser.add_argument('--schema-path', default=None,
                        help='Path to new_graph_schema.json.')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip JSON schema validation.')
    return parser.parse_args()


if __name__ == '__main__':
    # Edit these when running directly from Spyder.
    RUN_FROM_SETTINGS = True

    if RUN_FROM_SETTINGS:
        input_pickle = r'C:\Users\z3550257\Dropbox\University\2_Bath\ReCapture\ReCapture_Release_Paper\Data\ValidationData\opencap\P02\OpenSimData\Kinematics\P02_walkPref_gait_outputs.pkl'
        output_json = None
        schema_path = None

        if not input_pickle:
            raise ValueError(
                "Set input_pickle to the gait output .pkl path before running directly."
            )

        main(
            input_pickle=input_pickle,
            output_json=output_json,
            schema_path=schema_path,
            validate=True
        )
    else:
        args = parse_args()
        main(
            input_pickle=args.input_pickle,
            output_json=args.output_json,
            schema_path=args.schema_path,
            validate=not args.no_validate
        )
