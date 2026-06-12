# -*- coding: utf-8 -*-
"""
Basic results wrapper for ergometer rowing trials.
"""

import os
import sys
from time import perf_counter

RECAPTURE_PLUGINS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if RECAPTURE_PLUGINS_ROOT not in sys.path:
    sys.path.insert(0, RECAPTURE_PLUGINS_ROOT)

try:
    from .ergo_rowing_analysis import ergo_rowing_analysis
except ImportError:
    from ergo_rowing_analysis import ergo_rowing_analysis

from recapture_core.events.segment_events import build_bilateral_segment_events_from_cycle


def segment_rowing(fpath, fname, modelPath, trcFilePath,
                   cycle_signal='wrist_anterior', visualize=False,
                   pause_after_visualization=False, trial_label=None):

    rowing = ergo_rowing_analysis(
        fpath, fname, modelPath, trcFilePath,
        n_cycles=-1, cycle_signal=cycle_signal,
        visualize=visualize,
        pause_after_visualization=pause_after_visualization,
        trial_label=trial_label)

    return rowing


def Outputs(rowing, events=None):
    outputs_start = perf_counter()

    if events is None:
        events = rowing.get_cycle_events()

    def print_block_timing(label, start):
        print(f"[timing] Outputs {label}: {perf_counter() - start:.3f}s")

    Outcome_Measures = {
        'Rowing': {},
        'Angle_Results': {
            'Ankle': {}, 'Knee': {}, 'Hip': {},
            'Pelvis': {}, 'Trunk': {}, 'Shoulder': {}, 'Elbow': {}
        },
        'Segment_Events': {},
        'Time_Normalized_Coordinates': {},
        'Angle_Graphs': {}
    }

    block_start = perf_counter()
    cycle_counts = rowing.get_cycle_counts()
    Outcome_Measures['Rowing']['Cycle_Count'] = {
        'LR': cycle_counts,
        'units': 'cycles',
        'description': 'Number of detected rowing strokes using surrogate catch events from anterior hand position.'
    }

    stroke_rate_values, stroke_rate_units = rowing.compute_stroke_rate_by_side(return_all=True)
    Outcome_Measures['Rowing']['Stroke_Rate'] = {
        'LR': stroke_rate_values,
        'units': stroke_rate_units,
        'description': 'Rowing stroke rate estimated from consecutive surrogate catch events.'
    }

    cycle_duration_values, cycle_duration_units = rowing.compute_cycle_duration_by_side(return_all=True)
    Outcome_Measures['Rowing']['Cycle_Duration'] = {
        'LR': cycle_duration_values,
        'units': cycle_duration_units,
        'description': 'Duration of each detected rowing stroke.'
    }
    print_block_timing('rowing scalars', block_start)

    block_start = perf_counter()
    Outcome_Measures['Segment_Events'] = build_bilateral_segment_events_from_cycle(
        events, task='rowing', segment_label='Stroke')
    print_block_timing('cycle events', block_start)

    block_start = perf_counter()
    outputAngles = compute_angleOutputs(rowing)
    Outcome_Measures['Angle_Results']['Ankle'] = outputAngles['ankle']
    Outcome_Measures['Angle_Results']['Knee'] = outputAngles['knee']
    Outcome_Measures['Angle_Results']['Hip'] = outputAngles['hip']
    Outcome_Measures['Angle_Results']['Pelvis'] = outputAngles['pelvis']
    Outcome_Measures['Angle_Results']['Trunk'] = outputAngles['trunk']
    Outcome_Measures['Angle_Results']['Shoulder'] = outputAngles['shoulder']
    Outcome_Measures['Angle_Results']['Elbow'] = outputAngles['elbow']
    print_block_timing('angle outputs', block_start)

    block_start = perf_counter()
    Outcome_Measures['Time_Normalized_Coordinates'] = rowing.get_coordinates_normalized_time()
    print_block_timing('time-normalized coordinates', block_start)

    block_start = perf_counter()
    Outcome_Measures['Angle_Graphs'] = compute_AngleGraphs(rowing)
    print_block_timing('angle graphs', block_start)

    print_block_timing('total', outputs_start)
    return Outcome_Measures


def compute_angleOutputs(rowing):
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

    rawAngles = rowing.compute_angle_outputs()
    outputAngles = {group: {} for group in joint_groups}

    for group_name, joint_names in joint_groups.items():
        for joint_name in joint_names:
            outputAngles[group_name][joint_name] = rawAngles[joint_name]
            for key in outputAngles[group_name][joint_name]:
                outputAngles[group_name][joint_name][key]['units'] = 'deg'
                if key == 'ROM':
                    description = f'Range of motion of the {joint_labels[joint_name]} over one rowing stroke.'
                elif key in ['Catch', 'Finish']:
                    description = f'{key}: Joint angle of the {joint_labels[joint_name]} at surrogate {key.lower()}.'
                else:
                    description = f'{key} of the {joint_labels[joint_name]} over one rowing stroke.'
                outputAngles[group_name][joint_name][key]['description'] = description

    return outputAngles


def compute_AngleGraphs(rowing):
    graph_groups = {
        'ankle': ['ankle'],
        'knee': ['knee'],
        'hip': ['hip'],
        'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
        'trunk': ['trunk_flexion', 'trunk_bending', 'trunk_rotation'],
        'shoulder': ['shoulder_flexion', 'shoulder_adduction', 'shoulder_rotation'],
        'elbow': ['elbow']
    }

    raw_graphs = rowing.compute_angle_graphs()
    angle_graphs = {}

    for group_name, joint_names in graph_groups.items():
        angle_graphs[group_name] = {'LR': {}, 'units': 'deg'}
        for joint_name in joint_names:
            angle_graphs[group_name]['LR'][joint_name] = raw_graphs[joint_name]['LR']

    return angle_graphs
