# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:13:23 2024

@author: lw2175


Code used to load and MOT file, split into each stride and calculate walking metrics

"""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import os
import sys
import json
import pickle
from time import perf_counter
import jsonschema

RECAPTURE_PLUGINS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if RECAPTURE_PLUGINS_ROOT not in sys.path:
    sys.path.insert(0, RECAPTURE_PLUGINS_ROOT)

PLUGIN_DIR = os.path.dirname(__file__)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

try:
    from .gait_analysis import gait_analysis
    from .segment_events import build_cycle_segment_events
except ImportError:
    from gait_analysis import gait_analysis
    from segment_events import build_cycle_segment_events


# %% Segment gait
def segment_gait(fpath, fname, modelPath, trcFilePath, gaitStyle='Treadmill',
                 leg='r', visualize=False, pause_after_visualization=False,
                 trial_label=None):

    # Segmentation is done in the gait_analysis class
    gait = gait_analysis(
        fpath, fname, modelPath, trcFilePath,
        n_gait_cycles=-1, leg=leg, gait_style=gaitStyle,
        visualize=visualize,
        pause_after_visualization=pause_after_visualization,
        trial_label=trial_label)
    
    return gait


def Outputs(gait, events=None, metadata=None):
    outputs_start = perf_counter()

    if events is None:
        events = gait.get_bilateral_gait_events()

    speed_outputs = compute_speed_outputs(gait)
    outcome_measures = {
        'speed': speed_outputs,
        'Temporospatial': compute_temporospatial_outputs(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'jointAngle_discrete': compute_joint_angle_discrete_outputs(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Segment_Events': compute_segment_event_outputs(events),
        'jointAngle_timeseries': compute_joint_angle_timeseries_outputs(gait, events)
    }
    _print_block_timing('total', outputs_start)

    return outcome_measures


def _print_block_timing(label, start):
    print(f"[timing] Outputs {label}: {perf_counter() - start:.3f}s")


def compute_temporospatial_outputs(gait, speed_outputs=None, metadata=None):
    block_start = perf_counter()
    outputs = {
        'Step_Length': compute_step_length_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Stride_Length': compute_stride_length_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Step_Width': compute_step_width_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Stance_Time': compute_stance_time_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Swing_Time': compute_swing_time_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Single_Support_Percent': compute_single_support_percent_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Double_Support_Percent': compute_double_support_percent_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Lateral_Pelvis_Sway': compute_lateral_pelvis_sway_output(
            gait, speed_outputs=speed_outputs, metadata=metadata),
        'Mid_Swing_Ankle_Height_Diff': (
            compute_mid_swing_ankle_height_diff_output(
                gait, speed_outputs=speed_outputs, metadata=metadata))
    }
    _print_block_timing('temporospatial', block_start)
    return outputs


def compute_joint_angle_discrete_outputs(gait, speed_outputs=None, metadata=None):
    outputs = {
        'Ankle': {},
        'Knee': {},
        'Hip': {},
        'Pelvis': {},
        'Trunk': {}
    }

    block_start = perf_counter()
    (outputs['Pelvis']['Peak_Pelvis_Obliquity'],
     outputs['Pelvis']['Average_Pelvis_Obliquity'],
     outputs['Trunk']['Peak_Lateral_Trunk_Lean'],
     outputs['Trunk']['Peak_Forward_Trunk_Lean'],
     outputs['Trunk']['Average_Forward_Trunk_Lean']) = compute_pelTrunk(gait)
    _print_block_timing('pelvis/trunk angles', block_start)

    block_start = perf_counter()
    outputAngles = compute_angleOutputs(gait)
    outputs['Ankle'] = outputAngles['ankle']
    outputs['Knee'] = outputAngles['knee']
    outputs['Hip'] = outputAngles['hip']
    _print_block_timing('angle outputs', block_start)

    return wrap_joint_angle_discrete_outputs(
        outputs, speed_outputs=speed_outputs, metadata=metadata)


def compute_joint_angle_timeseries_outputs(gait, events):
    block_start = perf_counter()
    raw_joint_angle_timeseries = compute_joint_angle_timeseries(gait, events)
    segment_events = compute_segment_event_outputs(events)
    toe_off_vertical_lines = _toe_off_vertical_lines_from_segment_events(
        segment_events)

    joint_angle_timeseries = {}
    for joint, graph in raw_joint_angle_timeseries.items():
        joint_angle_timeseries[joint] = make_joint_angle_timeseries_measure(
            joint=joint,
            graph=graph,
            vertical_lines=toe_off_vertical_lines
        )

    _print_block_timing('joint angle timeseries', block_start)
    return joint_angle_timeseries


def compute_segment_event_outputs(events):
    return build_cycle_segment_events(events, task='gait', segment_label='Stride')


def compute_speed_outputs(gait):
    block_start = perf_counter()
    gait_speed, gait_speed_units = gait.compute_pelvis_gait_speed(
        return_all=False)
    print('GAIT SPEED', gait_speed, 'm/s')
    speed_mph = gait_speed * 2.23694
    _print_block_timing('gait speed', block_start)
    return {
        'metric': {
            'var': gait_speed,
            'units': gait_speed_units
        },
        'imperial': {
            'var': speed_mph,
            'units': 'mph'
        }
    }


def _concat_lr_values(values):
    return {
        side: np.concatenate(side_values)
        for side, side_values in values.items()
    }


def compute_step_length_output(gait, speed_outputs=None, metadata=None):
    values, units = gait.compute_step_length(return_all=True)
    return make_single_left_right_measure(
        values=_concat_lr_values(values),
        y_label='Step length',
        y_unit=units,
        description='Step Length: Anterior-posterior distance between heel-strike of one leg, to heel-strike of the other leg. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.',
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
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata,
            speed_outputs=speed_outputs
        )
    )


def make_box_plot_name(metadata=None, speed_outputs=None):
    metadata = metadata or {}
    labels = [
        metadata.get('participant_name'),
        metadata.get('session_name'),
        metadata.get('trial_name')
    ]

    speed_label = _speed_label(speed_outputs)
    if speed_label:
        labels.append(speed_label)

    labels = [str(label) for label in labels if label not in [None, '']]
    return ' | '.join(labels) if labels else None


def _speed_label(speed_outputs):
    if not speed_outputs:
        return None
    try:
        speed = float(speed_outputs['metric']['var'])
        units = _normalise_unit(speed_outputs['metric']['units'])
    except (KeyError, TypeError, ValueError):
        return None
    return f'{speed:.2f} {units}'


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
                'unit': _normalise_unit(y_unit)
            }
        },
        'description': description,
        'tags': tags,
        'boxPlotName': box_plot_name
    })


def make_single_center_measure(values, y_label, y_unit, description,
                               box_plot_name=None, tags=None):
    return _drop_none({
        'singleCenter': _as_float_list(values),
        'units': {
            'y': {
                'label': y_label,
                'unit': _normalise_unit(y_unit)
            }
        },
        'description': description,
        'tags': tags,
        'boxPlotName': box_plot_name
    })


def make_multi_left_right_measure(values, x_label, x_unit, y_label, y_unit,
                                  description, vertical_lines=None,
                                  tags=None):
    return _drop_none({
        'multiLeftRight': {
            'r': _as_2d_float_list(values.get('r', [])),
            'l': _as_2d_float_list(values.get('l', []))
        },
        'units': {
            'x': {
                'label': x_label,
                'unit': _normalise_unit(x_unit)
            },
            'y': {
                'label': y_label,
                'unit': _normalise_unit(y_unit)
            }
        },
        'description': description,
        'tags': tags,
        'verticalLines': vertical_lines
    })


def make_joint_angle_timeseries_measure(joint, graph, vertical_lines=None):
    spec = _joint_angle_timeseries_spec(joint)
    return make_multi_left_right_measure(
        values=graph.get('LR', {}),
        x_label='Gait cycle',
        x_unit='%',
        y_label=spec['y_label'],
        y_unit=graph.get('units', 'deg'),
        description=graph.get('description', spec['description']),
        tags=spec['tags'],
        vertical_lines=vertical_lines
    )


def wrap_joint_angle_discrete_outputs(outputs, speed_outputs=None, metadata=None):
    wrapped_outputs = {}
    for joint_group, measures in outputs.items():
        for measure_name, measure in measures.items():
            output_name = f'{joint_group}_{_clean_measure_name(measure_name)}'
            wrapped_outputs[output_name] = make_joint_angle_discrete_measure(
                joint_group=joint_group,
                measure_name=measure_name,
                measure=measure,
                speed_outputs=speed_outputs,
                metadata=metadata
            )
    return wrapped_outputs


def _clean_measure_name(measure_name):
    return str(measure_name).replace(' ', '_').replace('-', '_')


def make_joint_angle_discrete_measure(joint_group, measure_name, measure,
                                      speed_outputs=None, metadata=None):
    values = _get_discrete_measure_values(measure)
    y_label = _joint_angle_discrete_y_label(joint_group)
    y_unit = measure.get('units', 'deg') if isinstance(measure, dict) else 'deg'
    description = (
        measure.get('description', f'{measure_name} for {joint_group}.')
        if isinstance(measure, dict) else f'{measure_name} for {joint_group}.'
    )
    tags = _joint_angle_discrete_tags(joint_group)
    box_plot_name = make_box_plot_name(
        metadata=metadata, speed_outputs=speed_outputs)

    if isinstance(values, dict) and ('r' in values or 'l' in values):
        return make_single_left_right_measure(
            values=values,
            y_label=y_label,
            y_unit=y_unit,
            description=description,
            tags=tags,
            box_plot_name=box_plot_name
        )

    return make_single_center_measure(
        values=values,
        y_label=y_label,
        y_unit=y_unit,
        description=description,
        tags=tags,
        box_plot_name=box_plot_name
    )


def _get_discrete_measure_values(measure):
    if not isinstance(measure, dict):
        return measure
    for key in ['LR', 'var', 'value', 'values']:
        if key in measure:
            return measure[key]
    return []


def _joint_angle_discrete_y_label(joint_group):
    labels = {
        'Ankle': 'Ankle angle',
        'Knee': 'Knee angle',
        'Hip': 'Hip angle',
        'Pelvis': 'Pelvis angle',
        'Trunk': 'Trunk angle'
    }
    return labels.get(joint_group, f'{joint_group} angle')


def _joint_angle_discrete_tags(joint_group):
    tags = {
        'Ankle': [
            {'axis': 'y', 'text': 'Dorsiflexion', 'position': 'end'},
            {'axis': 'y', 'text': 'Plantarflexion', 'position': 'start'}
        ],
        'Knee': [
            {'axis': 'y', 'text': 'Flexion', 'position': 'end'},
            {'axis': 'y', 'text': 'Extension', 'position': 'start'}
        ],
        'Hip': [
            {'axis': 'y', 'text': 'Flexion', 'position': 'end'},
            {'axis': 'y', 'text': 'Extension', 'position': 'start'}
        ],
        'Pelvis': [
            {'axis': 'y', 'text': 'Contralateral drop', 'position': 'end'},
            {'axis': 'y', 'text': 'Contralateral rise', 'position': 'start'}
        ],
        'Trunk': [
            {'axis': 'y', 'text': 'Greater lean', 'position': 'end'},
            {'axis': 'y', 'text': 'Lower lean', 'position': 'start'}
        ]
    }
    return tags.get(joint_group)

def _joint_angle_timeseries_spec(joint):
    specs = {
        'ankle': {
            'y_label': 'Ankle angle',
            'description': 'Ankle angle time-normalised across the gait cycle.',
            'tags': [
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
                {'axis': 'x', 'text': 'Heel strike', 'position': 'start'},
                {'axis': 'x', 'text': 'Next heel strike', 'position': 'end'}
            ]
        },
        'knee': {
            'y_label': 'Knee angle',
            'description': 'Knee angle time-normalised across the gait cycle.',
            'tags': [
                {'axis': 'y', 'text': 'Flexion', 'position': 'end'},
                {'axis': 'y', 'text': 'Extension', 'position': 'start'},
                {'axis': 'x', 'text': 'Heel strike', 'position': 'start'},
                {'axis': 'x', 'text': 'Next heel strike', 'position': 'end'}
            ]
        },
        'hip': {
            'y_label': 'Hip angle',
            'description': 'Hip angle time-normalised across the gait cycle.',
            'tags': [
                {'axis': 'y', 'text': 'Flexion', 'position': 'end'},
                {'axis': 'y', 'text': 'Extension', 'position': 'start'},
                {'axis': 'x', 'text': 'Heel strike', 'position': 'start'},
                {'axis': 'x', 'text': 'Next heel strike', 'position': 'end'}
            ]
        },
        'pelvis': {
            'y_label': 'Pelvis angle',
            'description': 'Pelvis angle time-normalised across the gait cycle.',
            'tags': [
                {'axis': 'y', 'text': 'Contralateral drop', 'position': 'end'},
                {'axis': 'y', 'text': 'Contralateral rise', 'position': 'start'},
                {'axis': 'x', 'text': 'Heel strike', 'position': 'start'},
                {'axis': 'x', 'text': 'Next heel strike', 'position': 'end'}
            ]
        }
    }
    return specs.get(joint, {
        'y_label': f'{joint.title()} angle',
        'description': f'{joint.title()} angle time-normalised across the gait cycle.',
        'tags': [
            {'axis': 'x', 'text': 'Heel strike', 'position': 'start'},
            {'axis': 'x', 'text': 'Next heel strike', 'position': 'end'}
        ]
    })


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


def compute_stride_length_output(gait, speed_outputs=None, metadata=None):
    values = {}
    values['r'], units = gait.compute_stride_length(return_all=True, side='r')
    values['l'], units = gait.compute_stride_length(return_all=True, side='l')
    return make_single_left_right_measure(
        values=_concat_lr_values(values),
        y_label='Stride length',
        y_unit=units,
        description='Stride Length: Anterior-posterior distance between heel-strike to the following heelstrike of the same leg. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.',
        tags=[
            {'axis': 'y', 'text': 'Longer strides', 'position': 'end'},
            {'axis': 'y', 'text': 'Shorter strides', 'position': 'start'}
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata,
            speed_outputs=speed_outputs
        )
    )


def compute_step_width_output(gait, speed_outputs=None, metadata=None):
    values, units = gait.compute_step_width(return_all=True)
    return make_single_center_measure(
        values=values,
        y_label='Step width',
        y_unit=units,
        description='Step Width: Medio-lateral distance between the leg at mid-stance, and the following step of the opposite leg at mid-stance',
        tags=[
            {'axis': 'y', 'text': 'Wider steps', 'position': 'end'},
            {'axis': 'y', 'text': 'Narrower steps', 'position': 'start'}
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata,
            speed_outputs=speed_outputs
        )
    )


def compute_stance_time_output(gait, speed_outputs=None, metadata=None):
    values, units = gait.compute_stance_time_by_side(return_all=True)
    return make_single_left_right_measure(
        values=_concat_lr_values(values),
        y_label='Stance time',
        y_unit=units,
        description='Stance Time: Length of time the foot is in contact with the ground. Measured from heel-strike until toe-off for all steps in each leg. The left mean/std graph values were calculated for the left stance, while the right mean/std graph values were calculated for the right stance.',
        tags=[
            {'axis': 'y', 'text': 'Longer stance', 'position': 'end'},
            {'axis': 'y', 'text': 'Shorter stance', 'position': 'start'}
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata,
            speed_outputs=speed_outputs
        )
    )


def compute_swing_time_output(gait, speed_outputs=None, metadata=None):
    values, units = gait.compute_swing_time_by_side(return_all=True)
    return make_single_left_right_measure(
        values=_concat_lr_values(values),
        y_label='Swing time',
        y_unit=units,
        description='Swing Time: Length of time the foot is in the air between steps. Measured from toe-off until heel-strike for all steps in each leg. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.',
        tags=[
            {'axis': 'y', 'text': 'Longer swing', 'position': 'end'},
            {'axis': 'y', 'text': 'Shorter swing', 'position': 'start'}
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata,
            speed_outputs=speed_outputs
        )
    )


def compute_single_support_percent_output(gait, speed_outputs=None,
                                          metadata=None):
    values = {}
    values['r'], units = gait.compute_single_support_time(
        return_all=True, side='r')
    values['l'], units = gait.compute_single_support_time(
        return_all=True, side='l')
    return make_single_left_right_measure(
        values=_concat_lr_values(values),
        y_label='Single support',
        y_unit=units,
        description='Single Support: Percent of the stride when only one foot is in contact with the ground. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.',
        tags=[
            {'axis': 'y', 'text': 'Greater single support', 'position': 'end'},
            {'axis': 'y', 'text': 'Lower single support', 'position': 'start'}
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata,
            speed_outputs=speed_outputs
        )
    )


def compute_double_support_percent_output(gait, speed_outputs=None,
                                          metadata=None):
    values = {}
    values['r'], units = gait.compute_double_support_time(
        return_all=True, side='r')
    values['l'], units = gait.compute_double_support_time(
        return_all=True, side='l')
    return make_single_left_right_measure(
        values=_concat_lr_values(values),
        y_label='Double support',
        y_unit=units,
        description='Double Support: Percent of the stride when both feet are in contact with the ground. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.',
        tags=[
            {'axis': 'y', 'text': 'Greater double support', 'position': 'end'},
            {'axis': 'y', 'text': 'Lower double support', 'position': 'start'}
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata,
            speed_outputs=speed_outputs
        )
    )


def compute_lateral_pelvis_sway_output(gait, speed_outputs=None, metadata=None):
    values, units = gait.compute_pelvis_sway_range(return_all=True)
    description = (
        'Lateral pelvis sway: maximum medio-lateral distance travelled by '
        'the pelvis over one full stride. Ensure that the calibration board '
        'is placed perpendicular to the walking direction.'
    )
    tags = [
        {'axis': 'y', 'text': 'Greater sway', 'position': 'end'},
        {'axis': 'y', 'text': 'Lower sway', 'position': 'start'}
    ]
    box_plot_name = make_box_plot_name(
        metadata=metadata, speed_outputs=speed_outputs)

    if isinstance(values, dict) and ('r' in values or 'l' in values):
        return make_single_left_right_measure(
            values=values,
            y_label='Lateral pelvis sway',
            y_unit=units,
            description=description,
            tags=tags,
            box_plot_name=box_plot_name
        )

    return make_single_center_measure(
        values=values,
        y_label='Lateral pelvis sway',
        y_unit=units,
        description=description,
        tags=tags,
        box_plot_name=box_plot_name
    )


def compute_mid_swing_ankle_height_diff_output(
        gait, speed_outputs=None, metadata=None):
    values = {}
    values['r'], units = gait.compute_midswing_ankle_heigh_dif(
        return_all=True, side='r')
    values['l'], units = gait.compute_midswing_ankle_heigh_dif(
        return_all=True, side='l')
    return make_single_left_right_measure(
        values=values,
        y_label='Mid-swing ankle height difference',
        y_unit=units,
        description=(
            'Mid-swing ankle height difference: distance between the '
            'swinging ankle and the stance ankle at mid-swing. This is a '
            'minimum foot clearance related measure.'
        ),
        tags=[
            {'axis': 'y', 'text': 'Greater clearance', 'position': 'end'},
            {'axis': 'y', 'text': 'Lower clearance', 'position': 'start'}
        ],
        box_plot_name=make_box_plot_name(
            metadata=metadata, speed_outputs=speed_outputs)
    )


def save_outputs(outcome_measures, output_path, export_format='json',
                 validate_schema_path=None, graph_only=True):
    """
    Save gait result outputs for graph display or internal pipeline use.

    JSON export is graph-only by default because the full output also contains
    internal entries such as speed and Segment_Events that are not graph schema
    measures. Pickle export preserves the full Python object for debugging or
    validation workflows.
    """
    export_format = export_format.lower()
    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if export_format in ['pickle', 'pkl']:
        with open(output_path, 'wb') as file:
            pickle.dump(outcome_measures, file)
    elif export_format == 'json':
        payload = (
            build_graph_payload(outcome_measures)
            if graph_only else outcome_measures
        )
        payload = _to_json_serialisable(payload)
        if validate_schema_path is not None:
            validate_graph_payload(payload, validate_schema_path)
        with open(output_path, 'w') as file:
            json.dump(payload, file, indent=2)
    else:
        raise ValueError(
            "export_format must be one of: 'pickle', 'pkl', or 'json'.")

    return output_path


def build_graph_payload(outcome_measures):
    graph_sections = [
        'Temporospatial',
        'jointAngle_discrete',
        'jointAngle_timeseries'
    ]
    return {
        section: outcome_measures[section]
        for section in graph_sections
        if section in outcome_measures
    }


def default_graph_schema_path():
    return os.path.join(
        os.path.expanduser('~'),
        'Dropbox', 'University', 'GitHubClones', 'ReCapture-IRIS',
        'recapture-lib', 'new_graph_schema.json'
    )


def validate_graph_payload(payload, schema_path=None):
    schema_path = schema_path or default_graph_schema_path()
    with open(schema_path, 'r') as file:
        schema = json.load(file)
    jsonschema.validate(instance=payload, schema=schema)
    return True

def _to_json_serialisable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _to_json_serialisable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_serialisable(item) for item in value]
    return value

def compute_pelTrunk(gait):
    metrics = gait.compute_pelvis_trunk_metrics()
    # Define units.
    units = 'Degrees'
    descriptionPelList = 'Peak Pelvis List (Obliquity): Peak medio-lateral list/obliquity of the pelvis accross one full stride (left and right strides). Positive values represent that the ipsilateral side is higher than the contralateral side, and negative values representing the ipsilateral side is lower than the contralateral side. Therefore, for the right side (grey dot), a positive value indicates that at one some point in the stride, the right side of the pelvis is higher than the left by that amount. See Angle_Graph of pelvis list angle over the entire stride for information across the whole stride. The left mean/std graph values were calculated during the left stride, while the right mean/std graph values were calculated during the right stride.' 
    descriptionPelListAverage = 'Average Pelvis Obliquity: Average mediolateral list/obliquity of the pelvis accross one full stride of the right leg. Therefore, a positive value indicates that on average over one stride, the right side of the pelvis was higher than the left side, while a negtive value indicates the opposite. Because this is the average value over one stride, normal gait is expected to be very close to zero due to symetrical gait, while assymetry on one side of the body will result in a postive or negative number.'
    descriptionTrunkAbsolute = 'Peak Lateral Trunk Lean: Peak medio-lateral trunk angle relative to vertical (gravity). When the value is positive, this indicates that the trunk is leaning towards the leg in contact with the ground, relative to a vertical line (gravity). Therefore, in the graph of the right side/stride, a postive value means that the trunk is leaning towards the right side, while a negative value indicates the trunk is leaning towards the left side. In the left side graph, it would be the opposite. The left mean/std graph values were calculated during the left stride, while the right mean/std graph values were calculated during the right stride.' 
    descriptionTrunkFlexABS = 'Peak Forward Trunk Lean: Peak anterior-posterior trunk angle relative to vertical (gravity). When the value is positive, this indicates that the trunk is leaning forwards, relative to a vertical line (gravity). The left mean/std graph values were calculated during the left stride, while the right mean/std graph values were calculated during the right stride.' 
    descriptionTrunkFlexAVG = 'Average Forward Trunk Lean: Average anterior-posterior trunk angle relative to vertical (gravity). When the value is positive, this indicates that on average, the trunk is leaning forwards accross the entire stride, relative to a vertical line (gravity). The left mean/std graph values were calculated during the left stride, while the right mean/std graph values were calculated during the right stride.' 
    output_pelList = {'LR': metrics['pelvis_obliquity_peak'], 'units': units,
                      'description': descriptionPelList}
    output_trunkAbsolutes = {
        'LR': metrics['trunk_lateral_peak'], 'units': units, 'description': descriptionTrunkAbsolute}
    output_pelListAvgs = {'LR': metrics['pelvis_obliquity_avg'], 'units': units,
                          'description': descriptionPelListAverage}
    output_trunkFlexABS = {
        'LR': metrics['trunk_forward_peak'], 'units': units, 'description': descriptionTrunkFlexABS}
    output_trunkFlexAVGs = {
        'LR': metrics['trunk_forward_avg'], 'units': units, 'description': descriptionTrunkFlexAVG}

    return output_pelList, output_pelListAvgs, output_trunkAbsolutes,  output_trunkFlexABS, output_trunkFlexAVGs




def compute_angleOutputs(gait):
    outputAngles, _ = gait.compute_angle_outputs()

    for joint in outputAngles:
        if joint == 'ankle':
            pos_measure = 'dorsiflexion'
            neg_measure = 'plantar flexion'
        elif joint == 'knee':
            pos_measure = 'flexion'
            neg_measure = 'extension'
        else:
            pos_measure = 'flexion'
            neg_measure = 'extension'

        for key in outputAngles[joint]:
            outputAngles[joint][key]['units'] = 'deg'

        if joint != 'knee':
            outputAngles[joint]['ROM']['description'] = f'Range of Motion (ROM): Anterior-posterior ROM of the {joint} joint over an entire gait cycle (heelstrike to heelstrike). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Heelstrike']['description'] = f'Heelstrike: Anterior-posterior joint angle of the {joint} at heelstrike. Postive values indicate {pos_measure} and negative values indicate {neg_measure}. The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Mid-stance']['description'] = f'Mid-stance: Anterior-posterior joint angle of the {joint} at mid-stance (when the ankle is directly beneath the hip joint centre). Postive values indicate {pos_measure} and negative values indicate {neg_measure}. The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Toeoff']['description'] = f'Toeoff: Anterior-posterior joint angle of the {joint} at toeoff. Postive values indicate {pos_measure} and negative values indicate {neg_measure}. The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Mid-swing']['description'] = f'Mid-swing: Anterior-posterior joint angle of the {joint} at mid-swing (when the ankle is directly beneath the hip joint centre). Postive values indicate {pos_measure} and negative values indicate {neg_measure}. The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint][f'Peak {pos_measure}']['description'] = f'Peak {pos_measure} of the {joint} over the entire gait cycle. Postive values indicates {pos_measure} and negative values indicates {neg_measure}. The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint][f'Peak {neg_measure}']['description'] = f'Peak {neg_measure} of the {joint} over the entire gait cycle. Postive values indicates {pos_measure} and negative values indicates {neg_measure}. The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
        else:
            outputAngles[joint]['ROM']['description'] = f'Range of Motion (ROM): Anterior-posterior ROM of the {joint} joint over an entire gait cycle (heelstrike to heelstrike). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Heelstrike']['description'] = f'Heelstrike: Anterior-posterior joint angle of the {joint} at heelstrike. Postive values indicates {pos_measure}. NOTE, the simulated knee is locked and can never be extended beyond 0 degress (straight leg). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Mid-stance']['description'] = f'Mid-stance: Anterior-posterior joint angle of the {joint} at mid-stance (when the ankle is directly beneath the hip joint centre). Postive values indicates {pos_measure}. NOTE, the simulated knee is locked and can never be extended beyond 0 degress (straight leg). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Toeoff']['description'] = f'Toeoff: Anterior-posterior joint angle of the {joint} at toeoff. Postive values indicates {pos_measure}. NOTE, the simulated knee is locked and can never be extended beyond 0 degress (straight leg). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint]['Mid-swing']['description'] = f'Mid-swing: Anterior-posterior joint angle of the {joint} at mid-swing (when the ankle is directly beneath the hip joint centre). Postive values indicates {pos_measure}. NOTE, the simulated knee is locked and can never be extended beyond 0 degress (straight leg). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint][f'Peak {pos_measure}']['description'] = f'Peak {pos_measure} of the {joint} over the entire gait cycle. Postive values indicates {pos_measure}. NOTE, the simulated knee is locked and can never be extended beyond 0 degress (straight leg). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'
            outputAngles[joint][f'Peak {neg_measure}']['description'] = f'Peak {neg_measure} of the {joint} over the entire gait cycle. Postive values indicates {pos_measure}. NOTE, the simulated knee is locked and can never be extended beyond 0 degress (straight leg). The left mean/std graph values were calculated on the left leg, while the right mean/std graph values were calculated on the right leg.'

    return outputAngles

def compute_joint_angle_timeseries(gait, events):
    leg, contLeg = gait.get_leg()
    tsAngles = {'ankle': {'LR': {}},
                'knee': {'LR': {}},
                'hip': {'LR': {}},
                'pelvis': {'LR': {}}}
    
    
            


    # split for each stride
    for LR in [leg, contLeg]:
        if LR in ['r']:
            side = 'ipsilateralIdx'
        else:
            side = 'contralateralIdx'

        # Event times for TO to HS
        hs_0_idx = events[side][:, 0]
        to_1_idx = events[side][:, 1]
        hs_2_idx = events[side][:, 2]

        ang = gait.get_coordinate_values()

        for joint in tsAngles:
            if joint in ['ankle']:
                jointID = 'ankle_angle'
                pos_measure = 'dorsiflexion'
                neg_measure = 'plantar flexion'
                measure = 'Anterior-posterior'
            elif joint in ['knee']:
                jointID = 'knee_angle'
                pos_measure = 'flexion'
                neg_measure = 'extension'
                measure = 'Anterior-posterior'
            elif joint in ['hip']:
                jointID = 'hip_flexion'
                pos_measure = 'flexion'
                neg_measure = 'extension'
                measure = 'Anterior-posterior'
            elif joint in ['pelvis']:
                jointID = 'pelvis_list'
                pos_measure = 'contralateral drop'
                neg_measure = 'contralateral rise'
                measure = 'Medio-lateral list/obliquity for the'

            ts = np.zeros([101, len(hs_0_idx)])

            for i in range(len(hs_0_idx)):
                if jointID not in ['pelvis_list']:
                    jointName = jointID + '_' + LR.lower()
                else:
                    jointName = jointID

                # Range of Motion
                data = ang[jointName][hs_0_idx[i]:hs_2_idx[i]]

                if jointID in ['pelvis_list'] and LR in ['r']:
                    data = data * -1  # invert pelvis right graph so that positive value indicates the contralateral side is lower

                # Normalise to 101 points
                # original time
                oTime = np.array(ang['time'][hs_0_idx[i]:hs_2_idx[i]])
                nTime = np.linspace(oTime[0], oTime[-1], 101)  # new time

                ndata = interp1d(oTime, data, kind='linear',
                                 fill_value="extrapolate")(nTime)

                ts[:, i] = ndata

            tsAngles[joint]['LR'][LR.lower()] = ts

            if joint not in ['knee']:
                tsAngles[joint]['description'] = f'{measure} {joint} angles over one stride, with positive values representing {pos_measure} and negative values representing {neg_measure}. Joint angles for the {joint} are time normalised to one full stride, so 0% on the x-axis repreesnts heel-strike, and 100% represents the following heel-strike for the same side/leg. In normal walking, toe-off will generally occur at 60% of the stride, thus stance can be viewed as between 0-60% and swing as between 60-100% on the x-axis'
            else:
                tsAngles[joint]['description'] = f'{measure} {joint} angles over one stride, with positive values representing {pos_measure} and negative values representing {neg_measure}. NOTE, the simulated knee is locked and can never be extended beyond 0 degress (straight leg). Joint angles for the {joint} are time normalised to one full stride, so 0% on the x-axis repreesnts heel-strike, and 100% represents the following heel-strike for the same side/leg. In normal walking, toe-off will generally occur at 60% of the stride, thus stance can generally be viewed as between 0-60% and swing as between 60-100% on the x-axis.'

                
            tsAngles[joint]['units'] = 'deg'

    return tsAngles


# %% Run Main
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("yeah got some command line arguments")

    # fullpath = r'C:\Users\z3550257\Dropbox\University\2_Bath\ReCapture\ReCapture_Release_Paper\Data\ValidationData\opencap\P02\OpenSimData\Kinematics\P02_walkPref.mot'
    # modelPath = r'C:\Users\z3550257\Dropbox\University\2_Bath\ReCapture\ReCapture_Release_Paper\Data\ValidationData\opencap\P02\OpenSimData\Model\LaiUhlrich2022_scaled.osim'
    # trcFilePath = r'C:\Users\z3550257\Dropbox\University\2_Bath\ReCapture\ReCapture_Release_Paper\Data\ValidationData\opencap\P02\MarkerData\P02_walkPref.trc'
    export_format = 'json'
    leg = 'r'
    gaitStyle = 'treadmill' #'treadmill or 'auto'
    participant_name = 'P02'
    session_name = 'opencap'

    fullpath = sys.argv[4] # mot file
    modelPath = sys.argv[2] # osim file
    trcFilePath = sys.argv[3] # trc file
    session_dir = sys.argv[1] # trial folder
    
    # session_dir = os.path.dirname(fullpath)
    trial_name = os.path.splitext(os.path.basename(fullpath))[0]
    output_path = os.path.join(session_dir, f'{trial_name}_gait_graphs.json')

    # set up gait class.
    gait = segment_gait(
        session_dir,
        trial_name,
        modelPath,
        trcFilePath,
        gaitStyle=gaitStyle,
        leg=leg,
        visualize=False,
        pause_after_visualization=False,
        trial_label=trial_name
    )

    # create new events metric which includes [HS, TO, HS] for both legs (instead of just ipsilateral)
    events = gait.get_gait_events()

    # Output Results
    Outcome_Measures = Outputs(
        gait,
        events,
        metadata={
            'participant_name': participant_name,
            'session_name': session_name,
            'trial_name': trial_name
        }
    )
    saved_path = save_outputs(
        Outcome_Measures,
        output_path,
        export_format=export_format,
        validate_schema_path=default_graph_schema_path(),
        graph_only=True
    )
    print(f'Saved gait outputs to: {saved_path}')

    # Step lengths - Left and right step lengths
    # Veloctiy - outputs a single velocity measure for each trial (average of each step)
    # Step widths - width between the left and right heel at heel contact
    # swingTimes - time that the foot is in the air for the left and right leg
    # pelLists - left-right tilt of the pelvis for each left and right leg during stance
    # pelListsAvg - average pelvis frontal angle (list) during each stride
    # pelTrunkSym - symmetry measure of the pelvis list (frontal angle) for each stride. number close to 0 is symmetrical, more postive or negative and the more assymetrical
    # outputAngle - Hip, Knee and Ankle angle; range of motion, heelstrike, toeoff, max and min (entire gait cycle)
