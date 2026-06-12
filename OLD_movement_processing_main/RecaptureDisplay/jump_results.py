# -*- coding: utf-8 -*-
"""
Basic results wrapper for jump trials.
"""

from time import perf_counter
import numpy as np

try:
    from ..ActivityAnalyses.jump_analysis import jump_analysis
except ImportError:
    from movement_processing_main.ActivityAnalyses.jump_analysis import jump_analysis


def segment_jump(fpath, fname, modelPath, trcFilePath,
                 vertical_signal='pelvis_vertical', visualize=False,
                 pause_after_visualization=False, trial_label=None):

    jump = jump_analysis(
        fpath, fname, modelPath, trcFilePath,
        vertical_signal=vertical_signal,
        visualize=visualize,
        pause_after_visualization=pause_after_visualization,
        trial_label=trial_label)

    return jump


def Outputs(jump, events=None):
    outputs_start = perf_counter()

    if events is None:
        events = jump.get_jump_events()

    def print_block_timing(label, start):
        print(f"[timing] Outputs {label}: {perf_counter() - start:.3f}s")

    Outcome_Measures = {
        'Jumping': {},
        'Angle_Results': {
            'Ankle': {}, 'Knee': {}, 'Hip': {},
            'Pelvis': {}, 'Trunk': {}, 'Shoulder': {}, 'Elbow': {}
        },
        'Segment_Events': {},
        'Time_Normalized_Coordinates': {},
        'Angle_Graphs': {}
    }

    block_start = perf_counter()
    Outcome_Measures['Jumping']['Jump_Count'] = {
        'value': jump.get_jump_count(),
        'units': 'jumps',
        'description': 'Number of detected jumps in the trial.'
    }

    jump_duration, jump_duration_units = jump.compute_jump_duration(return_all=True)
    Outcome_Measures['Jumping']['Jump_Duration'] = {
        'value': jump_duration,
        'mean': float(jump_duration.mean()),
        'units': jump_duration_units,
        'description': 'Duration of each jump analysis window from initial quiet standing to the start of post-landing stillness.'
    }

    time_to_takeoff, time_to_takeoff_units = jump.compute_time_to_takeoff(return_all=True)
    Outcome_Measures['Jumping']['Time_To_TakeOff'] = {
        'value': time_to_takeoff,
        'mean': float(time_to_takeoff.mean()),
        'units': time_to_takeoff_units,
        'description': 'Time from initial quiet standing to take-off for each detected jump.'
    }

    flight_time, flight_time_units = jump.compute_flight_time(return_all=True)
    Outcome_Measures['Jumping']['Flight_Time'] = {
        'value': flight_time,
        'mean': float(flight_time.mean()),
        'units': flight_time_units,
        'description': 'Time between take-off and landing for each detected jump based on foot marker flight.'
    }
    print_block_timing('jump scalars', block_start)

    block_start = perf_counter()
    Outcome_Measures['Segment_Events'] = build_jump_segment_events(events)
    print_block_timing('jump events', block_start)

    block_start = perf_counter()
    outputAngles = compute_angleOutputs(jump)
    Outcome_Measures['Angle_Results']['Ankle'] = outputAngles['ankle']
    Outcome_Measures['Angle_Results']['Knee'] = outputAngles['knee']
    Outcome_Measures['Angle_Results']['Hip'] = outputAngles['hip']
    Outcome_Measures['Angle_Results']['Pelvis'] = outputAngles['pelvis']
    Outcome_Measures['Angle_Results']['Trunk'] = outputAngles['trunk']
    Outcome_Measures['Angle_Results']['Shoulder'] = outputAngles['shoulder']
    Outcome_Measures['Angle_Results']['Elbow'] = outputAngles['elbow']
    print_block_timing('angle outputs', block_start)

    block_start = perf_counter()
    Outcome_Measures['Time_Normalized_Coordinates'] = jump.get_coordinates_normalized_time()
    print_block_timing('time-normalized coordinates', block_start)

    block_start = perf_counter()
    Outcome_Measures['Angle_Graphs'] = compute_AngleGraphs(jump)
    print_block_timing('angle graphs', block_start)

    print_block_timing('total', outputs_start)
    return Outcome_Measures


def build_jump_segment_events(events):
    event_idx = np.asarray(events['eventIdx'])
    event_time = np.asarray(events['eventTime'])
    window_idx = np.asarray(events['windowIdx'])
    window_time = np.asarray(events['windowTime'])

    return {
        'schema_version': 1,
        'task': 'jump',
        'segment_label': 'Jump',
        'default_event_set': 'takeoff_movement',
        'event_sets': {
            'takeoff_movement': {
                'Segment_Idx': event_idx[:, [0, 1]],
                'Segment_Time': event_time[:, [0, 1]],
                'Event_Names': ['MovementStart', 'TakeOff'],
                'Event_Side': 'bilateral'
            },
            'takeoff_window': {
                'Segment_Idx': window_idx[:, [0, 1]],
                'Segment_Time': window_time[:, [0, 1]],
                'Event_Names': ['Start', 'TakeOff'],
                'Event_Side': 'bilateral'
            },
            'landing': {
                'Segment_Idx': event_idx[:, [2, 3]],
                'Segment_Time': event_time[:, [2, 3]],
                'Event_Names': ['Landing', 'PostLandingStillStart'],
                'Event_Side': 'bilateral'
            }
        }
    }


def compute_angleOutputs(jump):
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

    rawAngles = jump.compute_angle_outputs()
    outputAngles = {group: {} for group in joint_groups}

    for group_name, joint_names in joint_groups.items():
        for joint_name in joint_names:
            outputAngles[group_name][joint_name] = rawAngles[joint_name]
            for key in outputAngles[group_name][joint_name]:
                outputAngles[group_name][joint_name][key]['units'] = 'deg'
                if key == 'ROM':
                    description = f'Range of motion of the {joint_labels[joint_name]} over the jump window.'
                elif key in ['Start', 'TakeOff', 'Landing', 'PostLandingStillStart']:
                    description = f'{key}: Joint angle of the {joint_labels[joint_name]} at the {key.lower()} event.'
                else:
                    description = f'{key} of the {joint_labels[joint_name]} over the jump window.'
                outputAngles[group_name][joint_name][key]['description'] = description

    return outputAngles


def compute_AngleGraphs(jump):
    graph_groups = {
        'ankle': ['ankle'],
        'knee': ['knee'],
        'hip': ['hip'],
        'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
        'trunk': ['trunk_flexion', 'trunk_bending', 'trunk_rotation'],
        'shoulder': ['shoulder_flexion', 'shoulder_adduction', 'shoulder_rotation'],
        'elbow': ['elbow']
    }

    raw_graphs = jump.compute_angle_graphs()
    angle_graphs = {}

    for group_name, joint_names in graph_groups.items():
        angle_graphs[group_name] = {'LR': {}, 'units': 'deg'}
        for joint_name in joint_names:
            angle_graphs[group_name]['LR'][joint_name] = raw_graphs[joint_name]['LR']

    return angle_graphs
