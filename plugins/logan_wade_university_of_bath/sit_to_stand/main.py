# -*- coding: utf-8 -*-
"""
Basic results wrapper for sit-to-stand trials.

TODO: Update this results script to match the graph schema now used by
gait_results.py. Outputs should be reorganised into schema-compatible
Temporospatial, jointAngle_discrete, and jointAngle_timeseries-style
sections before this plugin is used by the graphing package.
"""

import os
import sys
from time import perf_counter
import numpy as np

RECAPTURE_PLUGINS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if RECAPTURE_PLUGINS_ROOT not in sys.path:
    sys.path.insert(0, RECAPTURE_PLUGINS_ROOT)

try:
    from .sit_to_stand_analysis import sit_to_stand_analysis
except ImportError:
    from sit_to_stand_analysis import sit_to_stand_analysis


def segment_sit_to_stand(fpath, fname, modelPath, trcFilePath,
                         trunk_signal='lumbar_extension', visualize=False,
                         pause_after_visualization=False, trial_label=None):

    sts = sit_to_stand_analysis(
        fpath, fname, modelPath, trcFilePath,
        trunk_signal=trunk_signal,
        visualize=visualize,
        pause_after_visualization=pause_after_visualization,
        trial_label=trial_label)

    return sts


def Outputs(sts, events=None):
    outputs_start = perf_counter()

    if events is None:
        events = sts.get_sit_to_stand_events()

    def print_block_timing(label, start):
        print(f"[timing] Outputs {label}: {perf_counter() - start:.3f}s")

    Outcome_Measures = {
        'Sit_To_Stand': {},
        'Angle_Results': {
            'Ankle': {}, 'Knee': {}, 'Hip': {},
            'Pelvis': {}, 'Trunk': {}, 'Shoulder': {}, 'Elbow': {}
        },
        'Segment_Events': {},
        'Time_Normalized_Coordinates': {},
        'Angle_Graphs': {}
    }

    block_start = perf_counter()
    Outcome_Measures['Sit_To_Stand']['Cycle_Count'] = {
        'value': sts.get_sit_to_stand_count(),
        'units': 'cycles',
        'description': 'Number of detected full sit-stand-sit cycles in the trial.'
    }

    full_cycle_duration, full_cycle_units = sts.compute_full_cycle_duration(return_all=True)
    Outcome_Measures['Sit_To_Stand']['Full_Cycle_Duration'] = {
        'value': full_cycle_duration,
        'mean': float(full_cycle_duration.mean()),
        'units': full_cycle_units,
        'description': 'Duration from sitting start to sitting end for each full sit-stand-sit cycle.'
    }

    sit_to_stand_duration, sit_to_stand_units = sts.compute_sit_to_stand_duration(return_all=True)
    Outcome_Measures['Sit_To_Stand']['Sit_To_Stand_Duration'] = {
        'value': sit_to_stand_duration,
        'mean': float(sit_to_stand_duration.mean()),
        'units': sit_to_stand_units,
        'description': 'Duration from sitting start to standing start.'
    }

    standing_duration, standing_units = sts.compute_standing_duration(return_all=True)
    Outcome_Measures['Sit_To_Stand']['Standing_Duration'] = {
        'value': standing_duration,
        'mean': float(standing_duration.mean()),
        'units': standing_units,
        'description': 'Duration from standing start to standing end.'
    }

    stand_to_sit_duration, stand_to_sit_units = sts.compute_stand_to_sit_duration(return_all=True)
    Outcome_Measures['Sit_To_Stand']['Stand_To_Sit_Duration'] = {
        'value': stand_to_sit_duration,
        'mean': float(stand_to_sit_duration.mean()),
        'units': stand_to_sit_units,
        'description': 'Duration from standing end to sitting end.'
    }
    print_block_timing('sit-to-stand scalars', block_start)

    block_start = perf_counter()
    Outcome_Measures['Segment_Events'] = build_sit_to_stand_segment_events(events)
    print_block_timing('sit-to-stand events', block_start)

    block_start = perf_counter()
    outputAngles = compute_angleOutputs(sts)
    Outcome_Measures['Angle_Results']['Ankle'] = outputAngles['ankle']
    Outcome_Measures['Angle_Results']['Knee'] = outputAngles['knee']
    Outcome_Measures['Angle_Results']['Hip'] = outputAngles['hip']
    Outcome_Measures['Angle_Results']['Pelvis'] = outputAngles['pelvis']
    Outcome_Measures['Angle_Results']['Trunk'] = outputAngles['trunk']
    Outcome_Measures['Angle_Results']['Shoulder'] = outputAngles['shoulder']
    Outcome_Measures['Angle_Results']['Elbow'] = outputAngles['elbow']
    print_block_timing('angle outputs', block_start)

    block_start = perf_counter()
    Outcome_Measures['Time_Normalized_Coordinates'] = sts.get_coordinates_normalized_time()
    print_block_timing('time-normalized coordinates', block_start)

    block_start = perf_counter()
    Outcome_Measures['Angle_Graphs'] = compute_AngleGraphs(sts)
    print_block_timing('angle graphs', block_start)

    print_block_timing('total', outputs_start)
    return Outcome_Measures


def build_sit_to_stand_segment_events(events):
    event_sets = {
        'sit_stand_full': {
            'Segment_Idx': np.asarray(events['eventIdx']),
            'Segment_Time': np.asarray(events['eventTime']),
            'Event_Names': list(events.get('eventNames', [])),
            'Event_Side': 'bilateral'
        }
    }

    return {
        'schema_version': 1,
        'task': 'sit_to_stand',
        'segment_label': 'Sit-to-Stand',
        'default_event_set': 'sit_stand_full',
        'event_sets': event_sets
    }


def compute_angleOutputs(sts):
    joint_groups = {
        'ankle': ['ankle'],
        'knee': ['knee'],
        'hip': ['hip'],
        'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
        'trunk': ['trunk_flexion', 'trunk_bending', 'trunk_rotation'],
        'shoulder': ['shoulder_flexion', 'shoulder_adduction', 'shoulder_rotation'],
        'elbow': ['elbow']
    }
    joint_labels = {
        'ankle': 'ankle',
        'knee': 'knee',
        'hip': 'hip',
        'pelvis_tilt': 'pelvis tilt',
        'pelvis_list': 'pelvis list',
        'pelvis_rotation': 'pelvis rotation',
        'trunk_flexion': 'trunk flexion',
        'trunk_bending': 'trunk bending',
        'trunk_rotation': 'trunk rotation',
        'shoulder_flexion': 'shoulder flexion',
        'shoulder_adduction': 'shoulder adduction',
        'shoulder_rotation': 'shoulder rotation',
        'elbow': 'elbow'
    }

    rawAngles = sts.compute_angle_outputs()
    outputAngles = {group: {} for group in joint_groups}

    event_names = [
        'SittingEnd',
        'StandingStart',
        'StandingEnd',
        'SittingStart'
    ]

    for group_name, joint_names in joint_groups.items():
        for joint_name in joint_names:
            outputAngles[group_name][joint_name] = rawAngles[joint_name]
            for key in outputAngles[group_name][joint_name]:
                outputAngles[group_name][joint_name][key]['units'] = 'deg'
                if key == 'ROM':
                    description = f'Range of motion of the {joint_labels[joint_name]} over the sit-to-stand window.'
                elif key in event_names:
                    description = f'{key}: Joint angle of the {joint_labels[joint_name]} at the {key.lower()} event.'
                else:
                    description = f'{key} of the {joint_labels[joint_name]} over the sit-to-stand window.'
                outputAngles[group_name][joint_name][key]['description'] = description

    return outputAngles


def compute_AngleGraphs(sts):
    graph_groups = {
        'ankle': ['ankle'],
        'knee': ['knee'],
        'hip': ['hip'],
        'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
        'trunk': ['trunk_flexion', 'trunk_bending', 'trunk_rotation'],
        'shoulder': ['shoulder_flexion', 'shoulder_adduction', 'shoulder_rotation'],
        'elbow': ['elbow']
    }

    raw_graphs = sts.compute_angle_graphs()
    angle_graphs = {}

    for group_name, joint_names in graph_groups.items():
        angle_graphs[group_name] = {'LR': {}, 'units': 'deg'}
        for joint_name in joint_names:
            angle_graphs[group_name]['LR'][joint_name] = raw_graphs[joint_name]['LR']

    return angle_graphs
