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

RECAPTURE_PLUGINS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
if RECAPTURE_PLUGINS_ROOT not in sys.path:
    sys.path.insert(0, RECAPTURE_PLUGINS_ROOT)

try:
    from ..ActivityAnalyses.gait_analysis import gait_analysis
    from .segment_events import build_cycle_segment_events
except ImportError:
    from movement_processing_main.ActivityAnalyses.gait_analysis import gait_analysis
    from movement_processing_main.RecaptureDisplay.segment_events import build_cycle_segment_events


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


def Outputs(gait, events=None):
    outputs_start = perf_counter()

    if events is None:
        events = gait.get_bilateral_gait_events()

    def print_block_timing(label, start):
        print(f"[timing] Outputs {label}: {perf_counter() - start:.3f}s")

    Outcome_Measures = {'speed': {'metric': {}, 'imperial': {}}, 'Temporospatial': {}, 'Distance': {
    }, 'Angle_Results': {'Ankle': {}, 'Knee': {}, 'Hip': {}, 'Pelvis': {}, 'Trunk': {}}, 'Segment_Events': {}, 'Angle_Graphs': {}}

    block_start = perf_counter()

    # Extract Temporospatial measures
    # Step
    stepLengths = {'LR': {}, 'units': [], 'description': 'Step Length: Anterior-posterior distance between heel-strike of one leg, to heel-strike of the other leg. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.'}
    stepLengths['LR'], stepLengths['units'] = gait.compute_step_length(return_all=True)
    for side in stepLengths['LR']:  # fix how the data is stored
        stepLengths['LR'][side] = np.concatenate(stepLengths['LR'][side])
    Outcome_Measures['Temporospatial']['Step_Length'] = stepLengths
    
    #Stride
    strideLengths = {'LR': {}, 'units': [], 'description': 'Stride Length: Anterior-posterior distance between heel-strike to the following heelstrike of the same leg. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.'}
    strideLengths['LR']['r'], strideLengths['units'] = gait.compute_stride_length(return_all=True, side='r')
    strideLengths['LR']['l'], strideLengths['units'] = gait.compute_stride_length(return_all=True, side='l')
    for side in strideLengths['LR']:  # fix how the data is stored
        strideLengths['LR'][side] = np.concatenate(strideLengths['LR'][side])
    Outcome_Measures['Temporospatial']['Stride_Length'] = strideLengths

    # Step Width
    stepWidths = {'LR': {}, 'units': [], 'description': 'Step Width: Medio-lateral distance between the leg at mid-stance, and the following step of the opposite leg at mid-stance'}
    stepWidths['LR'], stepWidths['units'] = gait.compute_step_width(return_all=True)
    Outcome_Measures['Temporospatial']['Step_Width'] = stepWidths

    # Stance Time
    stanceTimes = {'LR': {}, 'units': [], 'description': []}
    stanceTimes['LR'], stanceTimes['units'] = gait.compute_stance_time_by_side(return_all=True)
    stanceTimes['description'] = 'Stance Time: Length of time the foot is in contact with the ground. Measured from heel-strike until toe-off for all steps in each leg. The left mean/std graph values were calculated for the left stance, while the right mean/std graph values were calculated for the right stance.'
    for side in stanceTimes['LR']:  # fix how the data is stored
        stanceTimes['LR'][side] = np.concatenate(stanceTimes['LR'][side])
    Outcome_Measures['Temporospatial']['Stance_Time'] = stanceTimes

    # Swing Time
    swingTimes = {'LR': {}, 'units': [], 'description': []}
    swingTimes['LR'], swingTimes['units'] = gait.compute_swing_time_by_side(return_all=True)
    swingTimes['description'] = 'Swing Time: Length of time the foot is in the air between steps. Measured from toe-off until heel-strike for all steps in each leg. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.'
    for side in swingTimes['LR']:  # fix how the data is stored
        swingTimes['LR'][side] = np.concatenate(swingTimes['LR'][side])
    Outcome_Measures['Temporospatial']['Swing_Time'] = swingTimes
    
    # Single Support Time
    singleSupportTimes = {'LR': {}, 'units': [], 'description': 'Single Support: Percent of the stride when only one foot is in contact with the ground. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.'}
    singleSupportTimes['LR']['r'], singleSupportTimes['units'] = gait.compute_single_support_time(return_all=True, side='r')
    singleSupportTimes['LR']['l'], singleSupportTimes['units'] = gait.compute_single_support_time(return_all=True, side='l')
    for side in  singleSupportTimes['LR']:  # fix how the data is stored
        singleSupportTimes['LR'][side] = np.concatenate(singleSupportTimes['LR'][side])
    Outcome_Measures['Temporospatial']['Single_Support_Percent'] = singleSupportTimes
    
    # Double Support Time
    doubleSupportTimes = {'LR': {}, 'units': [], 'description': 'Double Support: Percent of the stride when both feet are in contact with the ground. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.'}
    doubleSupportTimes['LR']['r'], doubleSupportTimes['units'] = gait.compute_double_support_time(return_all=True, side='r')
    doubleSupportTimes['LR']['l'], doubleSupportTimes['units'] = gait.compute_double_support_time(return_all=True, side='l')
    for side in  doubleSupportTimes['LR']:  # fix how the data is stored
        doubleSupportTimes['LR'][side] = np.concatenate(doubleSupportTimes['LR'][side])
    Outcome_Measures['Temporospatial']['Double_Support_Percent'] = doubleSupportTimes
    print_block_timing('temporospatial', block_start)
    
    
    
    #Distance Measures
    # Foot clearance height during swing
    # Outcome_Measures['Distance']['Min_Toe_Clearance_DropFoot_Only'] = compute_minimum_toe_clearance(
    #     gait, events, return_all=True)

    # Pelvis Sway
    block_start = perf_counter()
    pelvis_sway_values, pelvis_sway_units = gait.compute_pelvis_sway_range(
        return_all=True)
    Outcome_Measures['Distance']['Lateral_Pelvis_Sway'] = {
        'LR': pelvis_sway_values,
        'units': pelvis_sway_units,
        'description': 'Lateral Pelvis Sway: Maximum medio-lateral distance travelled by the pelvis over one full stride. This value is calculated by measuring the maximal distance travelled medio-laterally by the pelvis over one stride. ****Note - Ensure that calibration board is placed perpendicular to the walking direction****'
    }
    print_block_timing('pelvis sway', block_start)
    
    # Mid Swing Ankle Height Difference
    # compute vertical clearance of the swing ankle above the stance ankle at the time when the ankles pass by one another
    block_start = perf_counter()
    midSwingAnkleHeightDiff = {'LR': {}, 'units': [], 'description': 'Mid-swing Ankle Height Difference: Distance between the swinging angkle and the stance angle at midstance. This is a better indicator of minimum foot clearance for participants who do not have drop foot. The left mean/std graph values were calculated for the left leg, while the right mean/std graph values were calculated for the right leg.'}
    midSwingAnkleHeightDiff['LR']['r'], midSwingAnkleHeightDiff['units'] = gait.compute_midswing_ankle_heigh_dif(return_all=True, side='r')
    midSwingAnkleHeightDiff['LR']['l'], midSwingAnkleHeightDiff['units'] = gait.compute_midswing_ankle_heigh_dif(return_all=True, side='l')
    Outcome_Measures['Distance']['Mid_Swing_Ankle_Height_Diff'] = midSwingAnkleHeightDiff
    print_block_timing('mid-swing ankle height', block_start)



    #Angle Measures
    # Frontal Angle outputs
    block_start = perf_counter()
    (Outcome_Measures['Angle_Results']['Pelvis']['Peak_Pelvis_Obliquity'],
     Outcome_Measures['Angle_Results']['Pelvis']['Average_Pelvis_Obliquity'],
     Outcome_Measures['Angle_Results']['Trunk']['Peak_Lateral_Trunk_Lean'],
     Outcome_Measures['Angle_Results']['Trunk']['Peak_Forward_Trunk_Lean'],
     Outcome_Measures['Angle_Results']['Trunk']['Average_Forward_Trunk_Lean']) = compute_pelTrunk(gait)
    print_block_timing('pelvis/trunk angles', block_start)
    
    # compute outputs of the hip, knee and ankle sagittal plane angles at specific timepoints
    block_start = perf_counter()
    outputAngles = compute_angleOutputs(gait)
    Outcome_Measures['Angle_Results']['Ankle'] = outputAngles['ankle']
    Outcome_Measures['Angle_Results']['Knee'] = outputAngles['knee']
    Outcome_Measures['Angle_Results']['Hip'] = outputAngles['hip']
    print_block_timing('angle outputs', block_start)
    




    # compute and time normalise graphs of each gait cycle for the hip, knee and ankle
    block_start = perf_counter()
    Outcome_Measures['Angle_Graphs'] = compute_AngleGraphs(gait, events)
    print_block_timing('angle graphs', block_start)
    Outcome_Measures['Segment_Events'] = build_cycle_segment_events(
        events, task='gait', segment_label='Stride')
    #print(Outcome_Measures['Angle_Graphs'], flush=True)
    
    # Correlations between joint angles on each side of the body
    # corr, corr_mean = gait.compute_correlations(visualize=True, return_all=True) # unsure how this is working and cannot get is to run without error
    
    
    
    
    # Gait speed
    block_start = perf_counter()
    gait_speed, gait_speed_units = gait.compute_pelvis_gait_speed(return_all=False)
    print('GAIT SPEED', gait_speed, 'm/s')
    Outcome_Measures['speed']['metric']['var'] = gait_speed
    Outcome_Measures['speed']['metric']['units'] = gait_speed_units
    # Convert to miles per hour (1 m/s = 2.23694 mph)
    speed_mph = Outcome_Measures['speed']['metric']['var'] * 2.23694
    # Create a separate variable for the converted speed
    Outcome_Measures['speed']['imperial']['var'], Outcome_Measures['speed']['imperial']['units'] = (
        speed_mph, "mph")
    print_block_timing('gait speed', block_start)
    print_block_timing('total', outputs_start)

    return Outcome_Measures


def save_outputs(outcome_measures, output_path, export_format='pickle'):
    """
    Save raw gait result outputs for later graph export or pipeline use.

    This function is intentionally separate from Outputs() so the analysis code
    remains side-effect free by default. Use pickle for internal Python handoff,
    especially while outputs still contain NumPy arrays. JSON is available for
    simple inspection/debugging and converts NumPy values to Python lists.
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
        with open(output_path, 'w') as file:
            json.dump(_to_json_serialisable(outcome_measures), file, indent=2)
    else:
        raise ValueError(
            "export_format must be one of: 'pickle', 'pkl', or 'json'.")

    return output_path


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

def compute_AngleGraphs(gait, events):
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

    fullpath = r'C:\Users\z3550257\Dropbox\University\2_Bath\ReCapture\ReCapture_Release_Paper\Data\ValidationData\opencap\P02\OpenSimData\Kinematics\P02_walkPref.mot'
    modelPath = r'C:\Users\z3550257\Dropbox\University\2_Bath\ReCapture\ReCapture_Release_Paper\Data\ValidationData\opencap\P02\OpenSimData\Model\LaiUhlrich2022_scaled.osim'
    trcFilePath = r'C:\Users\z3550257\Dropbox\University\2_Bath\ReCapture\ReCapture_Release_Paper\Data\ValidationData\opencap\P02\MarkerData\P02_walkPref.trc'
    export_format = 'pickle'
    leg = 'r'
    gaitStyle = 'treadmill' #'treadmill or 'auto'
    
    session_dir = os.path.dirname(fullpath)
    trial_name = os.path.splitext(os.path.basename(fullpath))[0]
    output_path = os.path.join(session_dir, f'{trial_name}_gait_outputs.pkl')

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
    Outcome_Measures = Outputs(gait, events)
    saved_path = save_outputs(
        Outcome_Measures,
        output_path,
        export_format=export_format
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
