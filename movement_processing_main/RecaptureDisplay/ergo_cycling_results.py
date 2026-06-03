# -*- coding: utf-8 -*-
"""
Basic results wrapper for ergometer cycling trials.
"""

from time import perf_counter

try:
    from ..ActivityAnalyses.ergo_cycling_analysis import ergo_cycling_analysis
    from .segment_events import build_cycle_segment_events
except ImportError:
    from movement_processing_main.ActivityAnalyses.ergo_cycling_analysis import ergo_cycling_analysis
    from movement_processing_main.RecaptureDisplay.segment_events import build_cycle_segment_events


def segment_cycling(fpath, fname, modelPath, trcFilePath, side='auto',
                    cycle_signal='foot_center_vertical', visualize=False,
                    pause_after_visualization=False, trial_label=None):

    cycling = ergo_cycling_analysis(
        fpath, fname, modelPath, trcFilePath,
        side=side, n_cycles=-1, cycle_signal=cycle_signal,
        visualize=visualize,
        pause_after_visualization=pause_after_visualization,
        trial_label=trial_label)

    return cycling


def Outputs(cycling, events=None):
    outputs_start = perf_counter()

    if events is None:
        events = cycling.get_cycle_events()

    def print_block_timing(label, start):
        print(f"[timing] Outputs {label}: {perf_counter() - start:.3f}s")

    Outcome_Measures = {
        'Cycling': {},
        'Angle_Results': {
            'Ankle': {}, 'Knee': {}, 'Hip': {},
            'Pelvis': {}, 'Trunk': {}, 'Shoulder': {}, 'Elbow': {}
        },
        'Segment_Events': {},
        'Time_Normalized_Coordinates': {},
        'Angle_Graphs': {}
    }

    block_start = perf_counter()
    cycle_counts = cycling.get_cycle_counts()
    Outcome_Measures['Cycling']['Cycle_Count'] = {
        'LR': cycle_counts,
        'units': 'cycles',
        'description': 'Number of detected cycling revolutions for each side using surrogate top-dead-centre events from the foot-centre vertical signal.'
    }

    cadence_values, cadence_units = cycling.compute_cadence_by_side(return_all=True)
    Outcome_Measures['Cycling']['Cadence'] = {
        'LR': cadence_values,
        'units': cadence_units,
        'description': 'Cycling cadence estimated from consecutive surrogate top-dead-centre events for each side.'
    }

    cycle_duration_values, cycle_duration_units = cycling.compute_cycle_duration_by_side(return_all=True)
    Outcome_Measures['Cycling']['Cycle_Duration'] = {
        'LR': cycle_duration_values,
        'units': cycle_duration_units,
        'description': 'Duration of each detected cycling revolution for each side.'
    }
    print_block_timing('cycling scalars', block_start)

    block_start = perf_counter()
    Outcome_Measures['Segment_Events'] = build_cycle_segment_events(
        events, task='cycling', segment_label='Revolution')
    print_block_timing('cycle events', block_start)

    block_start = perf_counter()
    outputAngles = compute_angleOutputs(cycling)
    Outcome_Measures['Angle_Results']['Ankle'] = outputAngles['ankle']
    Outcome_Measures['Angle_Results']['Knee'] = outputAngles['knee']
    Outcome_Measures['Angle_Results']['Hip'] = outputAngles['hip']
    Outcome_Measures['Angle_Results']['Pelvis'] = outputAngles['pelvis']
    Outcome_Measures['Angle_Results']['Trunk'] = outputAngles['trunk']
    Outcome_Measures['Angle_Results']['Shoulder'] = outputAngles['shoulder']
    Outcome_Measures['Angle_Results']['Elbow'] = outputAngles['elbow']
    print_block_timing('angle outputs', block_start)

    block_start = perf_counter()
    Outcome_Measures['Time_Normalized_Coordinates'] = cycling.get_coordinates_normalized_time()
    print_block_timing('time-normalized coordinates', block_start)

    block_start = perf_counter()
    Outcome_Measures['Angle_Graphs'] = compute_AngleGraphs(cycling)
    print_block_timing('angle graphs', block_start)

    print_block_timing('total', outputs_start)
    return Outcome_Measures


def compute_angleOutputs(cycling):
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

    rawAngles = cycling.compute_angle_outputs()
    outputAngles = {group: {} for group in joint_groups}

    for group_name, joint_names in joint_groups.items():
        for joint_name in joint_names:
            outputAngles[group_name][joint_name] = rawAngles[joint_name]
            for key in outputAngles[group_name][joint_name]:
                outputAngles[group_name][joint_name][key]['units'] = 'deg'
                if key == 'ROM':
                    description = f'Range of motion of the {joint_labels[joint_name]} over one cycling revolution.'
                elif key in ['TDC', 'BDC']:
                    description = f'{key}: Joint angle of the {joint_labels[joint_name]} at surrogate {"top-dead-centre" if key == "TDC" else "bottom-dead-centre"}.'
                else:
                    description = f'{key} of the {joint_labels[joint_name]} over one cycling revolution.'
                outputAngles[group_name][joint_name][key]['description'] = description

    return outputAngles


def compute_AngleGraphs(cycling):
    graph_groups = {
        'ankle': ['ankle'],
        'knee': ['knee'],
        'hip': ['hip'],
        'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
        'trunk': ['trunk_flexion', 'trunk_bending', 'trunk_rotation'],
        'shoulder': ['shoulder_flexion', 'shoulder_adduction', 'shoulder_rotation'],
        'elbow': ['elbow']
    }

    raw_graphs = cycling.compute_angle_graphs()
    angle_graphs = {}

    for group_name, joint_names in graph_groups.items():
        angle_graphs[group_name] = {'LR': {}, 'units': 'deg'}
        for joint_name in joint_names:
            angle_graphs[group_name]['LR'][joint_name] = raw_graphs[joint_name]['LR']

    return angle_graphs
