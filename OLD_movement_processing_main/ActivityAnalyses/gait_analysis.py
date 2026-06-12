"""
    ---------------------------------------------------------------------------
    OpenCap processing: gaitAnalysis.py
    ---------------------------------------------------------------------------

    Copyright 2023 Stanford University and the Authors
    
    Author(s): Antoine Falisse, Scott Uhlrich
    
    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
--------------------------------------------------------------------------------
    Modified by Logan Wade, 2026
--------------------------------------------------------------------------------
"""
 
import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import copy
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from utilsKinematics import kinematics


class gait_analysis(kinematics):
    
    def __init__(self, session_dir, trial_name, model_path, trcFilePath, leg='auto',
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_gait_cycles=-1, gait_style='auto', trimming_start=0, 
                 trimming_end=0, visualize=False,
                 pause_after_visualization=False, trial_label=None):
        
        # Inherit init from kinematics class.
        super().__init__(
            session_dir, 
            trial_name,
            model_path,
            trcFilePath,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)
        
        # We might want to trim the start/end of the trial to remove bad data. 
        # For example, this might be needed with HRNet during overground 
        # walking, where, at the end, the subject is leaving the field of view 
        # but HRNet returns relatively high confidence values. As a result,
        # the trial is not well trimmed. Here, we provide the option to
        # manually trim the start and end of the trial.
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
        self.pause_after_visualization = pause_after_visualization
        self.trial_label = trial_label
                        
        # Marker data load and filter.
        self.markerDict = self.get_marker_dict(session_dir, trial_name, trcFilePath,
            lowpass_cutoff_frequency = lowpass_cutoff_frequency_for_coordinate_values)

        # Coordinate values.
        self.coordinateValues = self.get_coordinate_values()
        
        # OSIM model file path
        self.model_path = model_path
        
        # Trim marker data and coordinate values.
        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start,6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:,:]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]
        
        if self.trimming_end > 0:
            self.idx_trim_end = np.where(np.round(self.markerDict['time'],6) <= np.round(self.markerDict['time'][-1] - self.trimming_end,6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end,:]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]
        
        # Rotate marker data so x is forward (not using for now, but could be useful for some analyses).
        self.rotation_about_y, self.markerDictRotated = self.rotate_x_forward()

        # Segment gait cycles.
        all_gait_events = self.segment_walking(
            n_gait_cycles=n_gait_cycles,
            leg=leg,
            return_all_legs=True,
            visualize=visualize)
        self.primaryLeg = all_gait_events['default_leg']
        self.gaitEventsBySide = {
            'r': all_gait_events['r'],
            'l': all_gait_events['l']
        }
        self.gaitEventsBilateral = self.get_bilateral_gait_events(reference_leg=self.primaryLeg)
        # Backward-compatible aliases for older one-leg callers.
        self.defaultLeg = self.primaryLeg
        self.gaitEvents = self.gaitEventsBySide[self.primaryLeg]
        self.nGaitCycles = np.shape(self.gaitEvents['ipsilateralIdx'])[0]
        
        # Determine treadmill speed (0 if overground).
        self.treadmillSpeed,_ = self.compute_treadmill_speed(gait_style=gait_style)
        
        # Initialize variables to be lazy loaded.
        self._comValues = None
        self._R_world_to_gait = None
        self._R_world_to_gait_by_side = {}
        self._leg_length = None

        # Rotate marker data with a per gait cycle rotation
        self.markerDictRotatedPerGaitCycle = self.rotate_vector_into_gait_frame()
    
    # Compute COM trajectory.
    def comValues(self,rotate=None,filt_freq=-1):
        if rotate == None:
            if self._comValues is None or filt_freq != -1:
                self._comValues = self.get_center_of_mass_values(lowpass_cutoff_frequency = filt_freq)
                if self.trimming_start > 0:
                    self._comValues = self._comValues.iloc[self.idx_trim_start:]            
                if self.trimming_end > 0:
                    self._comValues = self._comValues.iloc[:self.idx_trim_end]
            return self._comValues

        if rotate == 'gaitCycle':
            if self._comValuesRotatedPerGaitCycle is None or filt_freq!=-1:
                comUnrotated = self.comValues(filt_freq=filt_freq)
                comRotated = self.rotate_vector_into_gait_frame(comUnrotated[['x', 'y', 'z']].to_numpy())
                # turn back into a dataframe with time as first column
                self._comValuesRotatedPerGaitCycle = pd.DataFrame(data=np.concatenate((np.expand_dims(comUnrotated['time'].to_numpy(), axis=1), comRotated),axis=1),
                                           columns=['time','x','y','z'])        
                if self.trimming_start > 0:
                    self._comValuesRotatedPerGaitCycle = self._comValuesRotatedPerGaitCycle.iloc[self.idx_trim_start:]            
                if self.trimming_end > 0:
                    self._comValuesRotatedPerGaitCycle = self._comValuesRotatedPerGaitCycle.iloc[:self.idx_trim_end]   
            return self._comValuesRotatedPerGaitCycle
        
        if rotate == 'y': # need to initialize self.rotation_about_y -- currently commented in the init function
            if self._comValuesRotated is None or filt_freq!=-1:
                self._comValuesRotated = self.rotate_com(self.comValues(filt_freq=filt_freq),{'y':self.rotation_about_y})
                if self.trimming_start > 0:
                    self._comValuesRotated = self._comValuesRotated.iloc[self.idx_trim_start:]            
                if self.trimming_end > 0:
                    self._comValuesRotated = self._comValuesRotated.iloc[:self.idx_trim_end]   
            return self._comValuesRotated
    
    # Compute gait frame.
    def R_world_to_gait(self, side=None):
        side_key = self._normalize_side(side)
        if side_key == self.primaryLeg:
            if self._R_world_to_gait is None:
                self._R_world_to_gait = self.compute_gait_frame(side=side_key)
            return self._R_world_to_gait

        if side_key not in self._R_world_to_gait_by_side:
            self._R_world_to_gait_by_side[side_key] = self.compute_gait_frame(side=side_key)
        return self._R_world_to_gait_by_side[side_key]
    
    def get_gait_events(self, side='both'):
        if side in [None, 'both', 'bilateral']:
            return copy.deepcopy(self.gaitEventsBilateral)
        if side in ['primary', 'default']:
            return copy.deepcopy(self.gaitEventsBySide[self.primaryLeg])
        return copy.deepcopy(self.gaitEventsBySide[self._normalize_side(side)])

    def get_primary_gait_events(self):
        return self.get_gait_events(side='primary')

    def get_gait_events_by_side(self):
        return {side: copy.deepcopy(events)
                for side, events in self.gaitEventsBySide.items()}

    def get_stride_counts(self):
        return {
            side: np.shape(events['ipsilateralIdx'])[0]
            for side, events in self.gaitEventsBySide.items()
        }

    def get_bilateral_gait_events(self, reference_leg=None):
        reference_leg = self._normalize_side(reference_leg)
        contralateral_leg = 'l' if reference_leg == 'r' else 'r'

        ipsilateral_events = copy.deepcopy(self.gaitEventsBySide[reference_leg])
        contralateral_events = self.gaitEventsBySide[contralateral_leg]

        return {
            'ipsilateralIdx': ipsilateral_events['ipsilateralIdx'],
            'contralateralIdx': contralateral_events['ipsilateralIdx'],
            'ipsilateralTime': ipsilateral_events['ipsilateralTime'],
            'contralateralTime': contralateral_events['ipsilateralTime'],
            'eventNamesIpsilateral': ipsilateral_events['eventNamesIpsilateral'],
            'eventNamesContralateral': contralateral_events['eventNamesIpsilateral'],
            'ipsilateralLeg': ipsilateral_events['ipsilateralLeg']
        }

    def _get_side_gait_events(self, side=None):
        return self.gaitEventsBySide[self._normalize_side(side)]

    def _build_gait_events_for_leg(self, hsIps, toIps, hsCont, toCont, leg,
                                   n_gait_cycles=-1):
        if len(hsIps) < 2:
            raise Exception('Not enough gait cycles found.')

        available_cycles = len(hsIps) - 1
        if available_cycles < n_gait_cycles:
            print('You requested {} gait cycles, but only {} were found. '
                  'Proceeding with this number.'.format(n_gait_cycles, available_cycles))
            n_gait_cycles = available_cycles
        if n_gait_cycles == -1:
            n_gait_cycles = available_cycles
            print('Processing {} gait cycles, leg: '.format(n_gait_cycles) + leg + '.')

        gaitEvents_ips = np.zeros((n_gait_cycles, 3), dtype=int)
        gaitEvents_cont = np.zeros((n_gait_cycles, 2), dtype=int)
        if n_gait_cycles < 1:
            raise Exception('Not enough gait cycles found.')

        for i in range(n_gait_cycles):
            gaitEvents_ips[i, 0] = hsIps[-i-2]
            gaitEvents_ips[i, 2] = hsIps[-i-1]

            toIpsFound = False
            for j in range(len(toIps)):
                if toIps[-j-1] > gaitEvents_ips[i, 0] and toIps[-j-1] < gaitEvents_ips[i, 2] and not toIpsFound:
                    gaitEvents_ips[i, 1] = toIps[-j-1]
                    toIpsFound = True

            hsContFound = False
            toContFound = False
            for j in range(len(toCont)):
                if toCont[-j-1] > gaitEvents_ips[i, 0] and toCont[-j-1] < gaitEvents_ips[i, 2] and not toContFound:
                    gaitEvents_cont[i, 0] = toCont[-j-1]
                    toContFound = True

            for j in range(len(hsCont)):
                if hsCont[-j-1] > gaitEvents_ips[i, 0] and hsCont[-j-1] < gaitEvents_ips[i, 2] and not hsContFound:
                    gaitEvents_cont[i, 1] = hsCont[-j-1]
                    hsContFound = True

            ipsilateral_order_valid = (
                gaitEvents_ips[i, 0] < gaitEvents_ips[i, 1] <
                gaitEvents_ips[i, 2]
            )
            contralateral_order_valid = (
                gaitEvents_ips[i, 0] < gaitEvents_cont[i, 0] <
                gaitEvents_ips[i, 2] and
                gaitEvents_ips[i, 0] < gaitEvents_cont[i, 1] <
                gaitEvents_ips[i, 2]
            )

            if (
                not toIpsFound or not toContFound or not hsContFound or
                not ipsilateral_order_valid or
                not contralateral_order_valid
            ):
                print('Could not find contralateral gait event within ' +
                               'ipsilateral gait event range ' + str(i+1) +
                               ' steps until the end. Skipping this step.')
                gaitEvents_cont[i, :] = -1
                gaitEvents_ips[i, :] = -1

        mask_ips = (gaitEvents_ips == -1).any(axis=1)
        if all(mask_ips):
            raise Exception('No good steps for ' + leg + ' leg.')
        gaitEvents_ips = gaitEvents_ips[~mask_ips]
        gaitEvents_cont = gaitEvents_cont[~mask_ips]

        gaitEventTimes_ips = self.markerDict['time'][gaitEvents_ips]
        gaitEventTimes_cont = self.markerDict['time'][gaitEvents_cont]

        return {'ipsilateralIdx': gaitEvents_ips,
                'contralateralIdx': gaitEvents_cont,
                'ipsilateralTime': gaitEventTimes_ips,
                'contralateralTime': gaitEventTimes_cont,
                'eventNamesIpsilateral': ['HS', 'TO', 'HS'],
                'eventNamesContralateral': ['TO', 'HS'],
                'ipsilateralLeg': leg}
    
    def rotate_x_forward(self):
        
        #Adjusted to account for theia data
        if 'r.PSIS_study' in self.markerDict['markers']:
            # Find the midpoint of the PSIS markers
            psis_midpoint = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study']) / 2

            # Find the midpoint of the ASIS markers
            asis_midpoint = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study']) / 2
        else:
            # Find the midpoint of the PSIS markers
            psis_midpoint = self.markerDict['markers']['PELVIS1']

            # Find the midpoint of the ASIS markers
            asis_midpoint = (self.markerDict['markers']['RHIP'] + self.markerDict['markers']['LHIP']) / 2


        # Compute the vector pointing from the PSIS midpoint to the ASIS midpoint
        heading_vector = asis_midpoint - psis_midpoint

        # Compute the angle between the heading vector projected onto x-z plane and x-axis
        angle = np.unwrap(np.arctan2(heading_vector[:,2], heading_vector[:,0]))

        # compute average angle during middle 50% of the trial
        n_frames = len(self.markerDict['time'])
        start_index = int(n_frames * 0.25)
        end_index = int(n_frames * 0.75)
        angle = np.degrees(np.mean(angle[start_index:end_index], axis=0))

        # Apply the rotation to the marker data
        marker_dict_rotated = self.rotate_marker_dict(self.markerDict, {'y':angle})

        return angle, marker_dict_rotated
    
        
    def leg_length(self):

        if self._leg_length is None:

            leg, contLeg = self.get_leg()
            # compute the midpoint between the knee and m_knee markers
            kjc = (self.markerDict['markers'][leg + '_knee_study'] + 
                            self.markerDict['markers'][leg + '_mknee_study']) / 2
            ajc = (self.markerDict['markers'][leg + '_ankle_study'] +
                                self.markerDict['markers'][leg + '_mankle_study']) / 2
            hjc = self.markerDict['markers'][leg.upper() + 'HJC_study']
                
            # compute the femur vector from hjc to kjc, then find the average of its norm
            femur_vector = kjc - hjc
            femur_length = np.mean(np.linalg.norm(femur_vector, axis=1))

            # compute the tibia vector from kjc to ajc, then find the average of its norm
            tibia_vector = ajc - kjc
            tibia_length = np.mean(np.linalg.norm(tibia_vector, axis=1))

            # sum the femur and tibia lengths to get the leg length
            _leg_length = {'ipsilateral':femur_length + tibia_length}

            # repeat for contraolateral leg
            kjc = (self.markerDict['markers'][contLeg + '_knee_study'] + 
                            self.markerDict['markers'][contLeg + '_mknee_study']) / 2
            ajc = (self.markerDict['markers'][contLeg + '_ankle_study'] +
                                self.markerDict['markers'][contLeg + '_mankle_study']) / 2
            hjc = self.markerDict['markers'][contLeg.upper() + 'HJC_study']
                   
            femur_vector = kjc - hjc
            femur_length = np.mean(np.linalg.norm(femur_vector, axis=1))

            tibia_vector = ajc - kjc
            tibia_length = np.mean(np.linalg.norm(tibia_vector, axis=1))

            _leg_length['contralateral'] = femur_length + tibia_length
        
        return _leg_length
    
    
    def compute_scalars(self,scalarNames,return_all=False):
               
        # Verify that scalarNames are methods in gait_analysis.
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if 'compute_' in entry]
        
        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*possibleMethods)
            return
        
        nonexistant_methods = [entry for entry in scalarNames if 'compute_' + entry not in method_names]
        
        if len(nonexistant_methods) > 0:
            raise Exception(str(['compute_' + a for a in nonexistant_methods]) + ' does not exist in gait_analysis class.')
        
        scalarDict = {}
        for scalarName in scalarNames:
            thisFunction = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'],
                scalarDict[scalarName]['units']) = thisFunction(return_all=return_all)
        
        return scalarDict
    
    
    def compute_stride_length(self,return_all=False, side=None):
        
        leg,_ = self.get_leg(side=side)
        gait_events = self._get_side_gait_events(side=side)
        
        if leg + '_calc_study' in self.markerDict['markers']:
            calc_position = self.rotate_vector_into_gait_frame(
                self.markerDict['markers'][leg + '_calc_study'].copy(),
                side=side)

        else:
            calc_position = self.rotate_vector_into_gait_frame(
                self.markerDict['markers'][leg.upper() + 'ANK'].copy(),
                side=side)


        # On treadmill, the stride length is the difference in ipsilateral
        # calcaneus position at heel strike + treadmill speed * time.
        strideLengths = (
                - calc_position[gait_events['ipsilateralIdx'][:,:1],0] +
                calc_position[gait_events['ipsilateralIdx'][:,2:3],0] + 
                self.treadmillSpeed * np.diff(gait_events['ipsilateralTime'][:,(0,2)]))       
        
        # Average across all strides.
        strideLength = np.mean(strideLengths)
        
        # Define units.
        units = 'm'
        
        if return_all:
            return strideLengths,units
        else: 
            return strideLength, units
        
    
    def compute_step_length(self,return_all=False):
        leg, contLeg = self.get_leg()
        step_lengths = {}
        
        if leg + '_calc_study' in self.markerDictRotated['markers']:
            ipsMarker = leg + '_calc_study' 
            contMarker = contLeg + '_calc_study'
        else:
            ipsMarker = leg.upper() + 'ANK' 
            contMarker = contLeg.upper() + 'ANK'

        
        step_lengths[contLeg.lower()] = (
            - self.markerDictRotated['markers'][ipsMarker][self.gaitEvents['ipsilateralIdx'][:,:1],0] + 
            self.markerDictRotated['markers'][contMarker][self.gaitEvents['contralateralIdx'][:,1:2],0] + 
            self.treadmillSpeed * (self.gaitEvents['contralateralTime'][:,1:2] -
                                   self.gaitEvents['ipsilateralTime'][:,:1]))
        
        step_lengths[leg.lower()]  = (
            self.markerDictRotated['markers'][ipsMarker][self.gaitEvents['ipsilateralIdx'][:,2:],0] - 
            self.markerDictRotated['markers'][contMarker][self.gaitEvents['contralateralIdx'][:,1:2],0] + 
            self.treadmillSpeed * (-self.gaitEvents['contralateralTime'][:,1:2] +
                                   self.gaitEvents['ipsilateralTime'][:,2:]))
               
        # Average across all strides.
        step_length = {key: np.mean(values) for key, values in step_lengths.items()}
        
        # Define units.
        units = 'm'
        
        # some functions depend on having values for each step, otherwise return average
        if return_all:
            return step_lengths, units
        else:
            return step_length, units
        
        
    def compute_step_length_symmetry(self,return_all=False):
        step_lengths,units = self.compute_step_length(return_all=True)
        
        step_length_symmetry_all = step_lengths['r'] / step_lengths['l'] * 100
        
        # Average across strides
        step_length_symmetry = np.mean(step_length_symmetry_all)
        
        # define units 
        units = '% (R/L)'
        
        if return_all:
            return step_length_symmetry_all, units
        else:
            return step_length_symmetry, units
        
    
    def compute_gait_speed(self,return_all=False):
        # Legacy COM-based 3D gait speed. This is slow because it computes
        # whole-body COM through OpenSim.
        #could improve this by only running this on every 5th frame instead of every frame?
                            
        comValuesArray = np.vstack((self.comValues()['x'],self.comValues()['y'],self.comValues()['z'])).T
        gait_speeds = (
            np.linalg.norm(
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:,:1]] -
                comValuesArray[self.gaitEvents['ipsilateralIdx'][:,2:3]], axis=2) /
                np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)]) + self.treadmillSpeed) 
        
        # Average across all strides.
        gait_speed = np.mean(gait_speeds)
        
        # Define units.
        units = 'm/s'
        
        if return_all:
            return gait_speeds,units
        else:
            return gait_speed, units

    def compute_pelvis_gait_speed(self, return_all=False):
        # 3D pelvis-based gait speed. This mirrors compute_gait_speed but uses
        # pelvis marker center instead of whole-body COM.
        if 'r.ASIS_study' in self.markerDict['markers']:
            pelvisMarkerNames = ['r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study']
        else:
            pelvisMarkerNames = ['PELVIS1', 'RHIP', 'LHIP']

        pelvisMarkers = [self.markerDict['markers'][mkr] for mkr in pelvisMarkerNames]
        pelvisCenter = np.mean(np.array(pelvisMarkers), axis=0)

        gait_speeds = (
            np.linalg.norm(
                pelvisCenter[self.gaitEvents['ipsilateralIdx'][:,:1]] -
                pelvisCenter[self.gaitEvents['ipsilateralIdx'][:,2:3]], axis=2) /
            np.diff(self.gaitEvents['ipsilateralTime'][:, (0, 2)]) +
            self.treadmillSpeed)
        gait_speed = np.mean(gait_speeds)
        units = 'm/s'

        if return_all:
            return gait_speeds, units
        else:
            return gait_speed, units
    
    def compute_cadence(self,return_all=False):
        
        # In steps per minute.
        cadence_all = 60*2/np.diff(self.gaitEvents['ipsilateralTime'][:,(0,2)])
        
        # Average across all strides.
        cadence = np.mean(cadence_all)
        
        # Define units.
        units = 'steps/min'
        
        if return_all:
            return cadence_all,units
        else:
            return cadence, units
        
    def compute_treadmill_speed(self, overground_speed_threshold=0.3,
                               gait_style='auto', return_all=False):
    
        # Heuristic to determine if overground or treadmill.
        if gait_style == 'auto' or gait_style == 'treadmill':
            leg, _ = self.get_leg()
    
            # ADJUST FOR THEIA VALIDATION
            foot_marker_name = leg + '_ankle_study'
            if foot_marker_name in self.markerDict['markers']:
                foot_position = self.markerDict['markers'][foot_marker_name]
            else:
                foot_marker_name = leg.upper() + 'ANK'
                foot_position = self.markerDict['markers'][foot_marker_name]
    
            stanceTimeLength = np.round(np.diff(self.gaitEvents['ipsilateralIdx'][:, :2]))
            startIdx = np.round(self.gaitEvents['ipsilateralIdx'][:, :1] + 0.1 * stanceTimeLength).astype(int)
            endIdx = np.round(self.gaitEvents['ipsilateralIdx'][:, 1:2] - 0.3 * stanceTimeLength).astype(int)
    
            # Average instantaneous velocities per gait cycle
            dt = np.diff(self.markerDict['time'][:2])[0]
            treadmillSpeeds = np.zeros((self.nGaitCycles,))
    
            for i in range(self.nGaitCycles):
                segment = foot_position[startIdx[i, 0]:endIdx[i, 0], :]
                if segment.shape[0] > 1:  # avoid empty or single-frame slices
                    vel = np.diff(segment, axis=0) / dt
                    treadmillSpeeds[i] = np.linalg.norm(np.mean(vel, axis=0))
                else:
                    treadmillSpeeds[i] = np.nan  # mark bad cycles
    
            # Remove NaNs before filtering
            valid = ~np.isnan(treadmillSpeeds)
            speeds_valid = treadmillSpeeds[valid]
    
            if len(speeds_valid) > 0:
                # --- MAD OUTLIER FILTERING ---
                median = np.median(speeds_valid)
                mad = np.median(np.abs(speeds_valid - median))
    
                if mad > 0:
                    modified_z = 0.6745 * (speeds_valid - median) / mad
                    keep_mask = np.abs(modified_z) < 3.5
                    filtered_speeds = speeds_valid[keep_mask]
    
                    # fallback if everything removed
                    if len(filtered_speeds) == 0:
                        filtered_speeds = speeds_valid
                else:
                    # all values identical (or nearly)
                    filtered_speeds = speeds_valid
    
                treadmillSpeed = np.mean(filtered_speeds)
            else:
                treadmillSpeed = 0
                filtered_speeds = np.array([])
    
            # Overground detection
            if treadmillSpeed < overground_speed_threshold and gait_style != 'treadmill':
                treadmillSpeed = 0
                treadmillSpeeds = np.zeros(self.nGaitCycles)
            else:
                # optionally return filtered speeds mapped back to original size
                treadmillSpeeds = filtered_speeds
    
        elif gait_style == 'overground':
            treadmillSpeed = 0
            treadmillSpeeds = np.zeros(self.nGaitCycles)
    
        units = 'm/s'
    
        if return_all:
            return treadmillSpeeds, units
        else:
            return treadmillSpeed, units
    
    def compute_step_width(self,return_all=False):
        
        leg,contLeg = self.get_leg()
        
        #ADJUSTED THEIA VALIDATION
        # Get ankle joint center positions.
        if leg + '_ankle_study' in self.markerDict['markers']:
            ankle_position_ips = (
                self.markerDict['markers'][leg + '_ankle_study'] + 
                self.markerDict['markers'][leg + '_mankle_study'])/2
            ankle_position_cont = (
                self.markerDict['markers'][contLeg + '_ankle_study'] + 
                self.markerDict['markers'][contLeg + '_mankle_study'])/2     
        else:
            ankle_position_ips = self.markerDict['markers'][leg.upper() + 'ANK']
            ankle_position_cont = self.markerDict['markers'][contLeg.upper() + 'ANK']
                
        
        # Find indices of 40-60% of the stance phase
        ips_stance_length = np.diff(self.gaitEvents['ipsilateralIdx'][:,(0,1)])
        cont_stance_length = (self.gaitEvents['contralateralIdx'][:,0] - 
                              self.gaitEvents['ipsilateralIdx'][:,0] +
                              self.gaitEvents['ipsilateralIdx'][:,2]-
                              self.gaitEvents['contralateralIdx'][:,1])
        
        midstanceIdx_ips = [range(self.gaitEvents['ipsilateralIdx'][i,0] + 
                                  int(np.round(.4*ips_stance_length[i])),
                                  self.gaitEvents['ipsilateralIdx'][i,0] + 
                                  int(np.round(.6*ips_stance_length[i]))) 
                                  for i in range(self.nGaitCycles)]
        
        midstanceIdx_cont = [range(np.min((self.gaitEvents['contralateralIdx'][i,1] + 
                                  int(np.round(.4*cont_stance_length[i])),
                                  self.gaitEvents['ipsilateralIdx'][i,2]-1)),
                                  np.min((self.gaitEvents['contralateralIdx'][i,1] + 
                                  int(np.round(.6*cont_stance_length[i])),
                                  self.gaitEvents['ipsilateralIdx'][i,2]))) 
                                  for i in range(self.nGaitCycles)]   
        
        ankleVector = np.zeros((self.nGaitCycles,3))
        for i in range(self.nGaitCycles):
            ankleVector[i,:] = (
                np.mean(ankle_position_cont[midstanceIdx_cont[i],:],axis=0) - 
                np.mean(ankle_position_ips[midstanceIdx_ips[i],:],axis=0))
                     
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector[i,:], self.R_world_to_gait()[i,:,:]) 
            for i in range(self.nGaitCycles)])
        
        # Step width is z distance.
        stepWidths = np.abs(ankleVector_inGaitFrame[:,2])
        
        # Average across all strides.
        stepWidth = np.mean(stepWidths)
        
        # Define units.
        units = 'm'
        
        if return_all:
            return stepWidths, units
        else:
            return stepWidth, units
    
    def compute_stance_time(self, return_all=False, side=None):
        
        gait_events = self._get_side_gait_events(side=side)
        stanceTimes = np.diff(gait_events['ipsilateralTime'][:,:2])
        
        # Average across all strides.
        stanceTime = np.mean(stanceTimes)
        
        # Define units.
        units = 's'
        
        if return_all:
            return stanceTimes, units
        else:
            return stanceTime, units

    def compute_stance_time_by_side(self, return_all=False):
        stance_times = {
            'r': np.diff(self.gaitEventsBySide['r']['ipsilateralTime'][:, :2]),
            'l': np.diff(self.gaitEventsBySide['l']['ipsilateralTime'][:, :2])
        }

        units = 's'

        if return_all:
            return stance_times, units
        else:
            return {side: np.mean(values) for side, values in stance_times.items()}, units
    
    def compute_swing_time(self, return_all=False, side=None):
        
        gait_events = self._get_side_gait_events(side=side)
        swingTimes = np.diff(gait_events['ipsilateralTime'][:,1:])
        
        # Average across all strides.
        swingTime = np.mean(swingTimes)
        
        # Define units.
        units = 's'
        
        if return_all:
            return swingTimes, units
        else:  
            return swingTime, units

    def compute_swing_time_by_side(self, return_all=False):
        swing_times = {
            'r': np.diff(self.gaitEventsBySide['r']['ipsilateralTime'][:, 1:]),
            'l': np.diff(self.gaitEventsBySide['l']['ipsilateralTime'][:, 1:])
        }

        units = 's'

        if return_all:
            return swing_times, units
        else:
            return {side: np.mean(values) for side, values in swing_times.items()}, units
    
    def compute_single_support_time(self,return_all=False, side=None):
        
        double_support_time,_ = self.compute_double_support_time(return_all=True, side=side) 

        singleSupportTimes = 100 - double_support_time    
        
        # Average across all strides.
        singleSupportTime = np.mean(singleSupportTimes)
        
        # Define units.
        units = '%'
        
        if return_all:
            return singleSupportTimes,units
        else:
            return singleSupportTime, units
        
    def compute_double_support_time(self,return_all=False, side=None):
        
        gait_events = self._get_side_gait_events(side=side)

        # Ipsilateral stance time - contralateral swing time.
        doubleSupportTimes = (
            (np.diff(gait_events['ipsilateralTime'][:,:2]) - 
            np.diff(gait_events['contralateralTime'][:,:2])) /
            np.diff(gait_events['ipsilateralTime'][:,(0,2)])) * 100
                            
        # Average across all strides.
        doubleSupportTime = np.mean(doubleSupportTimes)
        
        # Define units.
        units = '%'
        
        if return_all:
            return doubleSupportTimes, units
        else:
            return doubleSupportTime, units
        
    def compute_midswing_dorsiflexion_angle(self,return_all=False, side=None):
        # compute ankle dorsiflexion angle during midstance
        gait_events = self._get_side_gait_events(side=side)
        to_1_idx = gait_events['ipsilateralIdx'][:,1]
        hs_2_idx = gait_events['ipsilateralIdx'][:,2]
        
        # ankle markers
        leg,contLeg = self.get_leg(side=side)
        ankleVector = (self.markerDict['markers'][leg + '_ankle_study'] - 
                       self.markerDict['markers'][contLeg + '_ankle_study'])
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector, self.R_world_to_gait(side=side)[i,:,:]) 
              for i in range(len(to_1_idx))])                                          
        
        swingDfAngles = np.zeros((to_1_idx.shape))
        
        for i in range(len(to_1_idx)):
            # find index within a swing phase with the smallest z distance between ankles
            idx_midSwing = np.argmin(np.abs(ankleVector_inGaitFrame[
                                     i,to_1_idx[i]:hs_2_idx[i],0]))+to_1_idx[i]
            
            swingDfAngles[i] = np.mean(self.coordinateValues['ankle_angle_' + 
                                gait_events['ipsilateralLeg']].to_numpy()[idx_midSwing])          
        
        # Average across all strides.
        swingDfAngle = np.mean(swingDfAngles)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return swingDfAngles, units
        else:
            return swingDfAngle, units
        
    def compute_midswing_ankle_heigh_dif(self,return_all=False, side=None):
        # compute vertical clearance of the swing ankle above the stance ankle
        # at the time when the ankles pass by one another
        gait_events = self._get_side_gait_events(side=side)
        to_1_idx = gait_events['ipsilateralIdx'][:,1]
        hs_2_idx = gait_events['ipsilateralIdx'][:,2]
        
        # ankle markers
        leg,contLeg = self.get_leg(side=side)
        # ADJUSTED THEIA VALIDATION
        if leg + '_ankle_study' in self.markerDict['markers']:
            ankleVector = (self.markerDict['markers'][leg + '_ankle_study'] - 
                           self.markerDict['markers'][contLeg + '_ankle_study'])   
        else:
            ankleVector = (self.markerDict['markers'][leg.upper() + 'ANK'] - 
                           self.markerDict['markers'][contLeg.upper() + 'ANK']) 
        
        ankleVector_inGaitFrame = np.array(
            [np.dot(ankleVector, self.R_world_to_gait(side=side)[i,:,:]) 
              for i in range(len(to_1_idx))])                                          
        
        swingAnkleHeighDiffs = np.zeros((to_1_idx.shape))
        
        for i in range(len(to_1_idx)):
            # find index within a swing phase with the smallest z distance between ankles
            idx_midSwing = np.argmin(np.abs(ankleVector_inGaitFrame[
                                     i,to_1_idx[i]:hs_2_idx[i],0]))+to_1_idx[i]
            
            swingAnkleHeighDiffs[i] = ankleVector_inGaitFrame[i,idx_midSwing,1]  
        
        # Average across all strides.
        swingAnkleHeighDiff = np.mean(swingAnkleHeighDiffs)
        
        # Define units.
        units = 'm'
        
        if return_all:
            return swingAnkleHeighDiffs, units
        else:
            return swingAnkleHeighDiff, units

    def compute_minimum_toe_clearance(self, return_all=False):
        toe_clearances = {}

        for side in ['r', 'l']:
            gait_events = self._get_side_gait_events(side=side)

            if side + '_toe_study' in self.markerDict['markers']:
                toeVector = self.markerDict['markers'][side + '_toe_study'][:, 1]
            else:
                toeVector = self.markerDict['markers'][side.upper() + 'TOE'][:, 1]

            invToeVector = toeVector * -1
            peaks, _ = find_peaks(invToeVector, prominence=0.001)
            thresh = np.mean([np.sort(invToeVector[peaks])[-len(gait_events['ipsilateralIdx']):]])
            toeVector_ground = toeVector - (thresh * -1)

            to_1_idx = gait_events['ipsilateralIdx'][:, 1]
            hs_2_idx = gait_events['ipsilateralIdx'][:, 2]
            swingFootClearance = np.zeros((to_1_idx.shape))

            for i in range(len(to_1_idx)):
                toeCycle = toeVector_ground[to_1_idx[i]:hs_2_idx[i]]
                swingFootClearance[i] = np.min(
                    toeCycle[int(25 * (len(toeCycle)/100)):-1])

            toe_clearances[side] = swingFootClearance

        units = 'm'

        if return_all:
            return toe_clearances, units
        else:
            return {side: np.mean(values) for side, values in toe_clearances.items()}, units

    def compute_pelvis_sway_range(self, return_all=False, side=None):
        coord = self.get_coordinate_values()
        pel = coord['pelvis_tz']
        gait_events = self._get_side_gait_events(side=side)

        hs_0_idx = gait_events['ipsilateralIdx'][:, 0]
        hs_2_idx = gait_events['ipsilateralIdx'][:, 2]

        sway = np.zeros((hs_0_idx.shape))
        for i in range(len(hs_0_idx)):
            sway[i] = np.ptp(pel[hs_0_idx[i]:hs_2_idx[i]])

        sway_mean = np.mean(sway)
        units = 'm'

        if return_all:
            return sway, units
        else:
            return sway_mean, units

    def compute_pelvis_trunk_metrics(self):
        pelLists = {}
        pelListAvgs = None
        trunkAbsolutes = {}
        trunkFlexABSs = {}
        trunkFlexAVGs = {}

        coord = self.get_coordinate_values()
        pel_raw = coord['pelvis_list']
        lum_raw = coord['lumbar_bending']
        lum_sag = coord['lumbar_extension']
        pel_sag = coord['pelvis_tilt']
        pel_forwardBend = (-pel_sag) + (-lum_sag)

        for side in ['r', 'l']:
            gait_events = self._get_side_gait_events(side=side)
            hs_0_idx = gait_events['ipsilateralIdx'][:, 0]
            to_1_idx = gait_events['ipsilateralIdx'][:, 1]
            hs_2_idx = gait_events['ipsilateralIdx'][:, 2]

            pel = pel_raw.copy()
            lum = lum_raw.copy()
            if side == 'r':
                pel = pel * -1
            else:
                lum = lum * -1

            pelList = np.zeros((hs_0_idx.shape))
            pelListAvg = np.zeros((hs_0_idx.shape))
            trunkAbsolute = np.zeros((hs_0_idx.shape))
            trunkFlexABS = np.zeros((hs_0_idx.shape))
            trunkFlexAVG = np.zeros((hs_0_idx.shape))

            for i in range(len(hs_0_idx)):
                pelList[i] = np.max(pel[hs_0_idx[i]:hs_2_idx[i]])
                trunkIdx = np.argmax(lum[hs_0_idx[i]:to_1_idx[i]]) + hs_0_idx[i]
                trunkFlexABS[i] = np.max(pel_forwardBend[hs_0_idx[i]:to_1_idx[i]])
                trunkFlexAVG[i] = np.average(pel_forwardBend[hs_0_idx[i]:to_1_idx[i]])

                if side == 'r':
                    pelListAvg[i] = np.average(pel[hs_0_idx[i]:hs_2_idx[i]])
                    trunkAbsolute[i] = (pel[trunkIdx] * -1) + (lum[trunkIdx])
                else:
                    trunkAbsolute[i] = (pel[trunkIdx]) + (lum[trunkIdx] * -1)

            pelLists[side] = pelList
            trunkFlexABSs[side] = trunkFlexABS
            trunkFlexAVGs[side] = trunkFlexAVG
            if side == 'r':
                trunkAbsolutes[side] = trunkAbsolute
                pelListAvgs = pelListAvg
            else:
                trunkAbsolutes[side] = trunkAbsolute * -1

        return {
            'pelvis_obliquity_peak': pelLists,
            'pelvis_obliquity_avg': pelListAvgs,
            'trunk_lateral_peak': trunkAbsolutes,
            'trunk_forward_peak': trunkFlexABSs,
            'trunk_forward_avg': trunkFlexAVGs
        }

    def compute_angle_outputs(self):
        outputAngles = {
            'ankle': {'Heelstrike': {'LR': {}}, 'Mid-stance': {'LR': {}}, 'Toeoff': {'LR': {}},
                      'Mid-swing': {'LR': {}}, 'Peak dorsiflexion': {'LR': {}},
                      'Peak plantar flexion': {'LR': {}}, 'ROM': {'LR': {}}},
            'knee': {'Heelstrike': {'LR': {}}, 'Mid-stance': {'LR': {}}, 'Toeoff': {'LR': {}},
                     'Mid-swing': {'LR': {}}, 'Peak flexion': {'LR': {}},
                     'Peak extension': {'LR': {}}, 'ROM': {'LR': {}}},
            'hip': {'Heelstrike': {'LR': {}}, 'Mid-stance': {'LR': {}}, 'Toeoff': {'LR': {}},
                    'Mid-swing': {'LR': {}}, 'Peak flexion': {'LR': {}},
                    'Peak extension': {'LR': {}}, 'ROM': {'LR': {}}}
        }
        mid_events = {}
        ang = self.get_coordinate_values()

        for side in ['r', 'l']:
            gait_events = self._get_side_gait_events(side=side)
            hs_0_idx = gait_events['ipsilateralIdx'][:, 0]
            to_1_idx = gait_events['ipsilateralIdx'][:, 1]
            hs_2_idx = gait_events['ipsilateralIdx'][:, 2]

            mid_events[side] = self.compute_mid_events(side=side)
            mst_0_idx = mid_events[side][:, 0]
            msw_1_idx = mid_events[side][:, 1]

            for joint in outputAngles:
                if joint == 'ankle':
                    jointID = 'ankle_angle'
                    pos_measure = 'dorsiflexion'
                    neg_measure = 'plantar flexion'
                elif joint == 'knee':
                    jointID = 'knee_angle'
                    pos_measure = 'flexion'
                    neg_measure = 'extension'
                else:
                    jointID = 'hip_flexion'
                    pos_measure = 'flexion'
                    neg_measure = 'extension'

                rom = np.zeros(len(hs_0_idx))
                heelstrike = np.zeros(len(hs_0_idx))
                toeoff = np.zeros(len(hs_0_idx))
                peak = np.zeros(len(hs_0_idx))
                trough = np.zeros(len(hs_0_idx))
                midStance = np.zeros(len(hs_0_idx))
                midSwing = np.zeros(len(hs_0_idx))

                joint_col = jointID + '_' + side
                for i in range(len(hs_0_idx)):
                    rom[i] = np.ptp(ang[joint_col][hs_0_idx[i]:hs_2_idx[i]])
                    heelstrike[i] = ang[joint_col][hs_0_idx[i]]
                    toeoff[i] = ang[joint_col][to_1_idx[i]]
                    peak[i] = np.max(ang[joint_col][hs_0_idx[i]:hs_2_idx[i]])
                    trough[i] = np.min(ang[joint_col][hs_0_idx[i]:hs_2_idx[i]])
                    midStance[i] = ang[joint_col][mst_0_idx[i]]
                    midSwing[i] = ang[joint_col][msw_1_idx[i]]

                outputAngles[joint]['ROM']['LR'][side] = rom
                outputAngles[joint]['Heelstrike']['LR'][side] = heelstrike
                outputAngles[joint]['Mid-stance']['LR'][side] = midStance
                outputAngles[joint]['Toeoff']['LR'][side] = toeoff
                outputAngles[joint]['Mid-swing']['LR'][side] = midSwing
                outputAngles[joint][f'Peak {pos_measure}']['LR'][side] = peak
                outputAngles[joint][f'Peak {neg_measure}']['LR'][side] = trough

        return outputAngles, mid_events
        
    def compute_peak_angle(self,dof,start_idx,end_idx,return_all=False):
        # start_idx and end_idx are 1xnGaitCycles        
        
        peakAngles = np.zeros((self.nGaitCycles))
        
        for i in range(self.nGaitCycles):                       
            peakAngles[i] = np.max(self.coordinateValues[dof + '_' +
                                self.gaitEvents['ipsilateralLeg']][start_idx[i]:end_idx[i]])
        
        # Average across all strides.
        peakAngle = np.mean(peakAngles)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return peakAngles, units
        else:
            return peakAngle, units
        
    def compute_rom(self,dof,start_idx,end_idx,return_all=False):
        # start_idx and end_idx are 1xnGaitCycles        
        
        roms = np.zeros((self.nGaitCycles))
        
        for i in range(self.nGaitCycles):                       
            roms[i] = np.ptp(self.coordinateValues[dof + '_' +
                                self.gaitEvents['ipsilateralLeg']][start_idx[i]:end_idx[i]])
        
        # Average across all strides.
        rom = np.mean(roms)
        
        # Define units.
        units = 'deg'
        
        if return_all:
            return roms, units
        else:
            return rom, units
                        
    def compute_correlations(self, cols_to_compare=None, visualize=False,
                             return_all=False):
        # this computes a weighted correlation between either side's dofs. 
        # the weighting is based on mean absolute percent error. In effect,
        # this penalizes both shape and magnitude differences.
        
        leg,contLeg = self.get_leg(lower=True)
               
        correlations_all_cycles = []
        mean_correlation_all_cycles = np.zeros((self.nGaitCycles,1))
        
        for i in range(self.nGaitCycles):

            
            hs_ind_1 = self.gaitEvents['ipsilateralIdx'][i,0]
            hs_ind_cont = self.gaitEvents['contralateralIdx'][i,1]
            hs_ind_2 = self.gaitEvents['ipsilateralIdx'][i,2]
            
            df1 = pd.DataFrame()
            df2 = pd.DataFrame()
            
            if cols_to_compare is None:
                cols_to_compare = df1.columns
            
            # create a dataframe of coords for this gait cycle
            for col in self.coordinateValues.columns:
                if col.endswith('_' + leg):
                    df1[col] = self.coordinateValues[col][hs_ind_1:hs_ind_2]
                elif col.endswith('_' + contLeg):
                    df2[col] = np.concatenate((self.coordinateValues[col][hs_ind_cont:hs_ind_2],
                                               self.coordinateValues[col][hs_ind_1:hs_ind_cont]))
            df1 = df1.reset_index(drop=True)
            df2 = df2.reset_index(drop=True)
                    
            # Interpolating both dataframes to have 101 rows for each column
            df1_interpolated = df1.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
            df2_interpolated = df2.interpolate(method='linear', limit_direction='both', limit_area='inside', limit=100)
        
            # Computing the correlation between appropriate columns in both dataframes
            correlations = {}
            total_weighted_correlation = 0
            # total_weight = 0
        
            for col1 in df1_interpolated.columns:
                if any(col1.startswith(col_compare) for col_compare in cols_to_compare):
                    if col1.endswith('_r'):   
                        corresponding_col = col1[:-2] + '_l'
                    elif col1.endswith('_l'):
                        corresponding_col = col1[:-2] + '_r'
                            
                    if corresponding_col in df2_interpolated.columns:
                        signal1 = df1_interpolated[col1]
                        signal2 = df2_interpolated[corresponding_col]
        
                        max_range_signal1 = np.ptp(signal1)
                        max_range_signal2 = np.ptp(signal2)
                        max_range = max(max_range_signal1, max_range_signal2)
        
                        mean_abs_error = np.mean(np.abs(signal1 - signal2)) / max_range
        
                        correlation = signal1.corr(signal2)
                        weight = 1 - mean_abs_error
        
                        weighted_correlation = correlation * weight
                        correlations[col1] = weighted_correlation
        
                        total_weighted_correlation += weighted_correlation
        
                        # Plotting the signals if visualize is True
                        if visualize:
                            plt.figure(figsize=(8, 5))
                            plt.plot(signal1, label='df1')
                            plt.plot(signal2, label='df2')
                            plt.title(f"Comparison between {col1} and {corresponding_col} with weighted correlation {weighted_correlation}")
                            plt.legend()
                            plt.show()
        
            mean_correlation_all_cycles[i] = total_weighted_correlation / len(correlations)
            correlations_all_cycles.append(correlations)
            
        if not return_all:
            mean_correlation_all_cycles = np.mean(mean_correlation_all_cycles)
            correlations_all_cycles =  {key: sum(d[key] for d in correlations_all_cycles) / 
                                        len(correlations_all_cycles) for key in correlations_all_cycles[0]}
            
        return correlations_all_cycles, mean_correlation_all_cycles

    def compute_gait_frame(self, side=None):

        # Create frame for each gait cycle with x: pelvis heading, 
        # z: average vector between ASIS during gait cycle, y: cross.
        
        #ADJUST FOR THEIA VALIDATION
        # Pelvis center trajectory (for overground heading vector).
        if 'r.ASIS_study' in self.markerDict['markers']:
            pelvisMarkerNames = ['r.ASIS_study','L.ASIS_study','r.PSIS_study','L.PSIS_study']
        else:
            pelvisMarkerNames = ['PELVIS1','RHIP','LHIP']

        pelvisMarkers = [self.markerDict['markers'][mkr]  for mkr in pelvisMarkerNames]
        pelvisCenter = np.mean(np.array(pelvisMarkers),axis=0)
        
        # Ankle trajectory (for treadmill heading vector).
        gait_events = self._get_side_gait_events(side=side)
        n_gait_cycles = np.shape(gait_events['ipsilateralIdx'])[0]
        leg = gait_events['ipsilateralLeg']
        if leg == 'l': leg='L'
        #ADJUST THEIA VALIDATION
        if 'r.ASIS_study' in self.markerDict['markers']:
            anklePos = self.markerDict['markers'][leg + '_ankle_study']
        else:
            anklePos = self.markerDict['markers'][leg.upper() + 'ANK']

        
        # Vector from left ASIS to right ASIS (for mediolateral direction).
        #ADJUST THEIA VALIDATION
        if 'r.ASIS_study' in self.markerDict['markers']:
            asisMarkerNames = ['L.ASIS_study','r.ASIS_study']
        else:
            asisMarkerNames = ['LHIP','RHIP']

        asisMarkers = [self.markerDict['markers'][mkr]  for mkr in asisMarkerNames]
        asisVector = np.squeeze(np.diff(np.array(asisMarkers),axis=0))
        
        # Heading vector per gait cycle.
        # If overground, use pelvis center trajectory; treadmill: ankle trajectory.
        if self.treadmillSpeed == 0:
            x = np.diff(pelvisCenter[gait_events['ipsilateralIdx'][:,(0,2)],:],axis=1)[:,0,:]
            x = x / np.linalg.norm(x,axis=1,keepdims=True)
        else: 
            x = np.zeros((n_gait_cycles,3))
            for i in range(n_gait_cycles):
                x[i,:] = anklePos[gait_events['ipsilateralIdx'][i,2]] - \
                         anklePos[gait_events['ipsilateralIdx'][i,1]]
            x = x / np.linalg.norm(x,axis=1,keepdims=True)
            
        # Mean ASIS vector over gait cycle.
        z_temp = np.zeros((n_gait_cycles,3))
        for i in range(n_gait_cycles):
            z_temp[i,:] = np.mean(asisVector[gait_events['ipsilateralIdx'][i,0]: \
                             gait_events['ipsilateralIdx'][i,2]],axis=0)
        z_temp = z_temp / np.linalg.norm(z_temp,axis=1,keepdims=True)
        
        # Cross to get y.
        y = np.cross(z_temp,x)
        
        z = np.cross(x,y)
        
        # 3x3xnSteps.
        R_lab_to_gait = np.stack((x.T,y.T,z.T),axis=1).transpose((2, 0, 1))
        
        return R_lab_to_gait
    
    def rotate_vector_into_gait_frame(self,vectorArray=None, side=None):
        # vectorArray is a nFramesx3 array
        # This takes a vector array and rotates it into the gait frame, per gait frame. Thus,
        # the data in the vector array is not expressed all in the same frame. This data should
        # only be used on gait cycle, by gait cycle data. Note, the second heel strike data gets overwritten
        # by subsequent gait cycles (since it is the same index as the first heel strike in the subsequent
        # gait cycle). We assume that the gait frame doesn't change dramatically from step to step.

        gait_events = self._get_side_gait_events(side=side)
        n_gait_cycles = np.shape(gait_events['ipsilateralIdx'])[0]

        def rotate_vec(vec,R):
            return np.dot(vec,R)
        
        if vectorArray is None: # rotate each marker in the entire markerDict
            markerDict_rotated_per_step = copy.deepcopy(self.markerDict)
            for marker_name,marker in markerDict_rotated_per_step['markers'].items():
                for i in range(n_gait_cycles):
                    markerDict_rotated_per_step['markers'][marker_name][gait_events['ipsilateralIdx'][i,0]:
                                                                        gait_events['ipsilateralIdx'][i,2],:] = rotate_vec(
                    marker[gait_events['ipsilateralIdx'][i,0]:gait_events['ipsilateralIdx'][i,2],:],
                    self.R_world_to_gait(side=side)[i,:,:])
            return markerDict_rotated_per_step
            
        else:
            for i in range(n_gait_cycles):
                vectorArray[gait_events['ipsilateralIdx'][i,0]:gait_events['ipsilateralIdx'][i,2],:] = rotate_vec(
                    vectorArray[gait_events['ipsilateralIdx'][i,0]:gait_events['ipsilateralIdx'][i,2],:],
                        self.R_world_to_gait(side=side)[i,:,:])

            return vectorArray
    
    def get_leg(self,lower=False, side=None):

        if self._normalize_side(side) == 'r':
            leg = 'r'
            contLeg = 'L'
        else:
            leg = 'L'
            contLeg = 'r'
        
        if lower:
            return leg.lower(), contLeg.lower()
        else:
            return leg, contLeg
    
    def get_coordinates_normalized_time(self, side='both'):
        if side in ['both', 'bilateral', None]:
            return {
                gait_side: self.get_coordinates_normalized_time(side=gait_side)
                for gait_side in ['r', 'l']
            }
        
        gait_events = self._get_side_gait_events(side=side)
        n_gait_cycles = np.shape(gait_events['ipsilateralIdx'])[0]
        colNames = self.coordinateValues.columns
        data = self.coordinateValues.to_numpy(copy=True)
        coordValuesNorm = []
        for i in range(n_gait_cycles):
            coordValues = data[gait_events['ipsilateralIdx'][i,0]:gait_events['ipsilateralIdx'][i,2]+1]
            coordValuesNorm.append(np.stack([np.interp(np.linspace(0,100,101),
                                   np.linspace(0,100,len(coordValues)),coordValues[:,i]) \
                                   for i in range(coordValues.shape[1])],axis=1))
             
        coordinateValuesTimeNormalized = {}
        coordVals_mean = np.mean(np.array(coordValuesNorm),axis=0)
        coordinateValuesTimeNormalized['mean'] = pd.DataFrame(data=coordVals_mean, columns=colNames)
        
        if n_gait_cycles > 2:
            coordVals_sd = np.std(np.array(coordValuesNorm), axis=0)
            coordinateValuesTimeNormalized['sd'] = pd.DataFrame(data=coordVals_sd, columns=colNames)
        else:
            coordinateValuesTimeNormalized['sd'] = None
        
        coordinateValuesTimeNormalized['indiv'] = [pd.DataFrame(data=d, columns=colNames) for d in coordValuesNorm]
        
        return coordinateValuesTimeNormalized

    def compute_mid_events(self, side=None):
        gait_events = self._get_side_gait_events(side=side)
        hs_0_idx = gait_events['ipsilateralIdx'][:,0]
        to_1_idx = gait_events['ipsilateralIdx'][:,1]
        hs_2_idx = gait_events['ipsilateralIdx'][:,2]

        leg,contLeg = self.get_leg(side=side)
        if leg + '_ankle_study' in self.markerDict['markers']:
            ankleHipVector = (self.markerDict['markers'][leg + '_ankle_study'] - 
                           self.markerDict['markers'][leg.upper() + 'HJC_study'])
        else: 
            ankleHipVector = (self.markerDict['markers'][leg.upper() + 'ANK'] - 
                           self.markerDict['markers'][leg.upper() + 'HIP'])
                
        ankleHipVector_inGaitFrame = np.array(
            [np.dot(ankleHipVector, self.R_world_to_gait(side=side)[i,:,:]) 
              for i in range(len(to_1_idx))])                                          
        
        swingIdx = np.zeros((to_1_idx.shape), dtype=int)
        stanceIdx = np.zeros((to_1_idx.shape), dtype=int)
        
        for i in range(len(to_1_idx)):
            idx_midSwing = np.argmin(np.abs(ankleHipVector_inGaitFrame[
                                     i,to_1_idx[i]:hs_2_idx[i],0]))+to_1_idx[i]
            
            idx_midStance = np.argmin(np.abs(ankleHipVector_inGaitFrame[
                                     i,hs_0_idx[i]:to_1_idx[i],0]))+hs_0_idx[i]

            swingIdx[i] = idx_midSwing
            stanceIdx[i] = idx_midStance
            
        return np.concatenate([stanceIdx.reshape(-1, 1), swingIdx.reshape(-1, 1)], axis=1)

    def segment_walking(self, n_gait_cycles=-1, leg='auto', visualize=False,
                        return_all_legs=False):

        # n_gait_cycles = -1 finds all accessible gait cycles. Otherwise, it 
        # finds that many gait cycles, working backwards from end of trial.
               
        # Helper functions
        def detect_gait_peaks(r_calc_rel_x,
                              l_calc_rel_x,
                              r_toe_rel_x,
                              l_toe_rel_x,
                              prominence = 0.3):
            # Find HS.
            rHS, _ = find_peaks(r_calc_rel_x, prominence=prominence)
            lHS, _ = find_peaks(l_calc_rel_x, prominence=prominence)
            
            # Find TO.
            rTO, _ = find_peaks(-r_toe_rel_x, prominence=prominence)
            lTO, _ = find_peaks(-l_toe_rel_x, prominence=prominence)
            
            return rHS,lHS,rTO,lTO
        
        def detect_correct_order(rHS, rTO, lHS, lTO):
            # checks if the peaks are in the right order
                    
            expectedOrder = {'rHS': 'lTO',
                             'lTO': 'lHS',
                             'lHS': 'rTO',
                             'rTO': 'rHS'}
                    
            # Identify vector that has the smallest value in it. Put this vector name
            # in vName1
            vectors = {'rHS': rHS, 'rTO': rTO, 'lHS': lHS, 'lTO': lTO}
            non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
        
            # Check if there are any non-empty vectors
            if not non_empty_vectors:
                return True  # All vectors are empty, consider it correct order
        
            vName1 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
        
            # While there are any values in any of the vectors (rHS, rTO, lHS, or lTO)
            while any([len(vName) > 0 for vName in vectors.values()]):
                # Delete the smallest value from the vName1
                vectors[vName1] = np.delete(vectors[vName1], 0)
        
                # Then find the vector with the next smallest value. Define vName2 as the
                # name of this vector
                non_empty_vectors = {k: v for k, v in vectors.items() if len(v) > 0}
                
                # Check if there are any non-empty vectors
                if not non_empty_vectors:
                    break  # All vectors are empty, consider it correct order
        
                vName2 = min(non_empty_vectors, key=lambda k: non_empty_vectors[k][0])
        
                # If vName2 != expectedOrder[vName1], return False
                if vName2 != expectedOrder[vName1]:
                    return False
        
                # Set vName1 equal to vName2 and clear vName2
                vName1, vName2 = vName2, ''
        
            return True
        

        def detect_correct_order_robust(rHS, rTO, lHS, lTO, min_full_cycles=2, debug=False):
            """
            Robust gait event order checker and cleaner.
        
            Accepts extra/missing events at the trial edges and only fails when
            there is no sufficiently long valid alternating sequence.
        
            Expected repeating order:
                rHS -> lTO -> lHS -> rTO -> rHS -> ...
        
            Parameters
            ----------
            rHS, rTO, lHS, lTO : array-like
                Arrays of event indices.
            min_full_cycles : int
                Minimum number of usable full cycles required.
            debug : bool
                If True, prints diagnostic info.
        
            Returns
            -------
            rHS_f, rTO_f, lHS_f, lTO_f : np.ndarray
                Cleaned event arrays containing only the best valid sequence.
            is_usable : bool
                True if enough valid cycles were found.
            usable_cycles : int
                Number of full valid cycles found.
            """
        
            expected_next = {
                'rHS': 'lTO',
                'lTO': 'lHS',
                'lHS': 'rTO',
                'rTO': 'rHS'
            }

            input_counts = {
                'rHS': len(rHS),
                'rTO': len(rTO),
                'lHS': len(lHS),
                'lTO': len(lTO)
            }
        
            # Combine all events into one sorted list
            events = []
            events.extend((int(x), 'rHS') for x in rHS)
            events.extend((int(x), 'rTO') for x in rTO)
            events.extend((int(x), 'lHS') for x in lHS)
            events.extend((int(x), 'lTO') for x in lTO)
            events.sort(key=lambda x: x[0])
        
            if len(events) == 0:
                if debug:
                    print("No events found.")
                return (
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    False,
                    0
                )
        
            # Build the longest valid subsequence from each possible start point
            best_sequence = []
        
            for start_idx in range(len(events)):
                seq = [events[start_idx]]
                expected = expected_next[events[start_idx][1]]
        
                for j in range(start_idx + 1, len(events)):
                    _, label_j = events[j]
        
                    if label_j == expected:
                        seq.append(events[j])
                        expected = expected_next[label_j]
                    else:
                        # Ignore out-of-order extras rather than failing immediately
                        continue
        
                if len(seq) > len(best_sequence):
                    best_sequence = seq
        
            if len(best_sequence) == 0:
                if debug:
                    print("No valid ordered subsequence found.")
                return (
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    np.array([], dtype=int),
                    False,
                    0
                )
        
            # Count complete cycles inside the best sequence
            cycle_count_r = 0
            for i in range(len(best_sequence) - 4):
                labels = [x[1] for x in best_sequence[i:i+5]]
                if labels == ['rHS', 'lTO', 'lHS', 'rTO', 'rHS']:
                    cycle_count_r += 1
        
            cycle_count_l = 0
            for i in range(len(best_sequence) - 4):
                labels = [x[1] for x in best_sequence[i:i+5]]
                if labels == ['lHS', 'rTO', 'rHS', 'lTO', 'lHS']:
                    cycle_count_l += 1
        
            usable_cycles = max(cycle_count_r, cycle_count_l)
            is_usable = usable_cycles >= min_full_cycles
        
            # Rebuild cleaned event arrays from best valid sequence
            rHS_f = np.array([idx for idx, label in best_sequence if label == 'rHS'], dtype=int)
            rTO_f = np.array([idx for idx, label in best_sequence if label == 'rTO'], dtype=int)
            lHS_f = np.array([idx for idx, label in best_sequence if label == 'lHS'], dtype=int)
            lTO_f = np.array([idx for idx, label in best_sequence if label == 'lTO'], dtype=int)
        
            if debug:
                output_counts = {
                    'rHS': len(rHS_f),
                    'rTO': len(rTO_f),
                    'lHS': len(lHS_f),
                    'lTO': len(lTO_f)
                }
                best_set = set(best_sequence)
                rejected_events = [
                    event for event in events if event not in best_set
                ]
                rejected_counts = {
                    label: sum(1 for _, event_label in rejected_events
                               if event_label == label)
                    for label in expected_next
                }

                print("\nGait robust-order debug")
                if self.trial_label:
                    print(f"  trial: {self.trial_label}")
                print(f"  input counts: {input_counts}")
                print(f"  output counts: {output_counts}")
                print(f"  rejected counts: {rejected_counts}")
                print(f"  usable full cycles found: {usable_cycles}")
                print(f"  usable: {is_usable}")
                if rejected_events:
                    print("  rejected events:")
                    for idx, label in rejected_events:
                        event_time = self.markerDict['time'][idx]
                        print(f"    {label} idx={idx} time={event_time:.3f}s")
                print("  cleaned arrays:")
                print("    rHS:", rHS_f)
                print("    rTO:", rTO_f)
                print("    lHS:", lHS_f)
                print("    lTO:", lTO_f)
        
            return rHS_f, rTO_f, lHS_f, lTO_f, is_usable, usable_cycles

        def clean_events_by_stride_windows(rHS, rTO, lHS, lTO,
                                           r_calc_rel_x, r_toe_rel_x,
                                           l_calc_rel_x, l_toe_rel_x,
                                           min_full_cycles=2, debug=False):
            time = self.markerDict['time']
            dt = float(np.median(np.diff(time)))
            min_event_gap = max(1, int(round(0.15 / dt)))
            min_stride_samples = max(2, int(round(0.30 / dt)))
            max_stride_samples = max(min_stride_samples + 1,
                                     int(round(2.0 / dt)))

            def deduplicate_events(events, signal, keep_max=True):
                events = np.asarray(events, dtype=int)
                if len(events) <= 1:
                    return events

                selected = []
                group = [events[0]]
                for event_idx in events[1:]:
                    if event_idx - group[-1] <= min_event_gap:
                        group.append(event_idx)
                    else:
                        group_signal = signal[group]
                        best_local = (
                            int(np.argmax(group_signal)) if keep_max
                            else int(np.argmin(group_signal))
                        )
                        selected.append(group[best_local])
                        group = [event_idx]

                group_signal = signal[group]
                best_local = (
                    int(np.argmax(group_signal)) if keep_max
                    else int(np.argmin(group_signal))
                )
                selected.append(group[best_local])
                return np.asarray(selected, dtype=int)

            def choose_event(candidates, start_idx, end_idx, phase):
                candidates = np.asarray([
                    idx for idx in candidates if start_idx < idx < end_idx
                ], dtype=int)
                if len(candidates) == 0:
                    return None

                target_idx = start_idx + phase * (end_idx - start_idx)
                return int(candidates[np.argmin(np.abs(candidates - target_idx))])

            rHS_d = deduplicate_events(rHS, r_calc_rel_x, keep_max=True)
            lHS_d = deduplicate_events(lHS, l_calc_rel_x, keep_max=True)
            rTO_d = deduplicate_events(rTO, r_toe_rel_x, keep_max=False)
            lTO_d = deduplicate_events(lTO, l_toe_rel_x, keep_max=False)

            kept_rHS = set()
            kept_rTO = set()
            kept_lHS = set()
            kept_lTO = set()
            valid_right_cycles = 0
            valid_left_cycles = 0

            def add_valid_cycles(hs_ips, to_ips, hs_cont, to_cont,
                                 ips_label):
                nonlocal valid_right_cycles, valid_left_cycles
                for hs_start, hs_end in zip(hs_ips[:-1], hs_ips[1:]):
                    stride_duration = hs_end - hs_start
                    if (
                        stride_duration < min_stride_samples or
                        stride_duration > max_stride_samples
                    ):
                        continue

                    ips_to = choose_event(to_ips, hs_start, hs_end, 0.60)
                    cont_to = choose_event(to_cont, hs_start, hs_end, 0.15)
                    cont_hs = choose_event(hs_cont, hs_start, hs_end, 0.50)
                    if ips_to is None or cont_to is None or cont_hs is None:
                        continue

                    if ips_label == 'r':
                        kept_rHS.update([int(hs_start), int(hs_end)])
                        kept_rTO.add(ips_to)
                        kept_lTO.add(cont_to)
                        kept_lHS.add(cont_hs)
                        valid_right_cycles += 1
                    else:
                        kept_lHS.update([int(hs_start), int(hs_end)])
                        kept_lTO.add(ips_to)
                        kept_rTO.add(cont_to)
                        kept_rHS.add(cont_hs)
                        valid_left_cycles += 1

            add_valid_cycles(rHS_d, rTO_d, lHS_d, lTO_d, 'r')
            add_valid_cycles(lHS_d, lTO_d, rHS_d, rTO_d, 'l')
            usable_cycles = max(valid_right_cycles, valid_left_cycles)
            ok = usable_cycles >= min_full_cycles

            rHS_f = np.asarray(sorted(kept_rHS), dtype=int)
            rTO_f = np.asarray(sorted(kept_rTO), dtype=int)
            lHS_f = np.asarray(sorted(kept_lHS), dtype=int)
            lTO_f = np.asarray(sorted(kept_lTO), dtype=int)

            if debug:
                print("\nGait stride-window fallback debug")
                print(f"  deduplicated counts: "
                      f"rHS={len(rHS_d)}, rTO={len(rTO_d)}, "
                      f"lHS={len(lHS_d)}, lTO={len(lTO_d)}")
                print(f"  valid right cycles: {valid_right_cycles}")
                print(f"  valid left cycles: {valid_left_cycles}")
                print(f"  output counts: "
                      f"rHS={len(rHS_f)}, rTO={len(rTO_f)}, "
                      f"lHS={len(lHS_f)}, lTO={len(lTO_f)}")

            return rHS_f, rTO_f, lHS_f, lTO_f, ok, usable_cycles
        
        # Subtract sacrum from foot.
        # It looks like the position-based approach will be more robust.
        # 
        # adjustement for theia validation
        if 'r_calc_study' in self.markerDict['markers']:
            r_calcMarker = self.markerDict['markers']['r_calc_study']
            r_pelMarker = self.markerDict['markers']['r.PSIS_study']
            r_toeMarker = self.markerDict['markers']['r_toe_study']
            l_calcMarker = self.markerDict['markers']['L_calc_study']
            l_pelMarker = self.markerDict['markers']['L.PSIS_study']
            l_toeMarker = self.markerDict['markers']['L_toe_study']
        else:
            r_calcMarker = self.markerDict['markers']['RANK']
            r_pelMarker = self.markerDict['markers']['RHIP']
            r_toeMarker = self.markerDict['markers']['RTOE']
            l_calcMarker = self.markerDict['markers']['LANK']
            l_pelMarker = self.markerDict['markers']['LHIP']
            l_toeMarker = self.markerDict['markers']['LTOE']
        #         
        r_calc_rel = (r_calcMarker - r_pelMarker)
        r_toe_rel = (r_toeMarker - r_pelMarker)
        r_toe_rel_x = r_toe_rel[:,0]
        # Repeat for left.
        l_calc_rel = (l_calcMarker - l_pelMarker)
        l_toe_rel = (l_toeMarker - l_pelMarker)
            
        # Identify which direction the subject is walking.
        #AJUSTMENT FOR THEIA VALIDATION
        if 'r.PSIS_study' in self.markerDict['markers']:
            mid_psis = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study'])/2
            mid_asis = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study'])/2
        else:
            mid_psis = self.markerDict['markers']['PELVIS1']
            mid_asis = (self.markerDict['markers']['RHIP'] + self.markerDict['markers']['LHIP'])/2

        mid_dir = mid_asis - mid_psis
        mid_dir_floor = np.copy(mid_dir)
        mid_dir_floor[:,1] = 0
        mid_dir_floor = mid_dir_floor / np.linalg.norm(mid_dir_floor,axis=1,keepdims=True)
        
        # Dot product projections   
        r_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_calc_rel)
        l_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_calc_rel)
        r_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,r_toe_rel)
        l_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor,l_toe_rel)
        
        # OLD APPROACH THAT NEEDS ALL events to be detected               
        # Detect peaks, check if they're in the right order, if not reduce prominence.
        # the peaks can be less prominent with pathological or slower gait patterns
        prominences = [0.3, 0.25, 0.2]
        robust_rejection_fallback_threshold = 0.18
        
        # for i,prom in enumerate(prominences):
        #     rHS,lHS,rTO,lTO = detect_gait_peaks(r_calc_rel_x=r_calc_rel_x,
        #                           l_calc_rel_x=l_calc_rel_x,
        #                           r_toe_rel_x=r_toe_rel_x,
        #                           l_toe_rel_x=l_toe_rel_x,
        #                           prominence=prom)
        #     if not detect_correct_order(rHS=rHS, rTO=rTO, lHS=lHS, lTO=lTO):
        #         if prom == prominences[-1]:
        #             raise ValueError('The ordering of gait events is not correct. Consider trimming your trial using the trimming_start and trimming_end options.')
        #         else:
        #             print('The gait events were not in the correct order. Trying peak detection again ' +
        #               'with prominence = ' + str(prominences[i+1]) + '.')
        #     else:
        #         # everything was in the correct order. continue.
        #         break
    
        for i, prom in enumerate(prominences):
            rHS, lHS, rTO, lTO = detect_gait_peaks(
                r_calc_rel_x=r_calc_rel_x,
                l_calc_rel_x=l_calc_rel_x,
                r_toe_rel_x=r_toe_rel_x,
                l_toe_rel_x=l_toe_rel_x,
                prominence=prom
            )

            raw_rHS = np.asarray(rHS, dtype=int)
            raw_rTO = np.asarray(rTO, dtype=int)
            raw_lHS = np.asarray(lHS, dtype=int)
            raw_lTO = np.asarray(lTO, dtype=int)
            raw_event_count = (
                len(raw_rHS) + len(raw_rTO) + len(raw_lHS) + len(raw_lTO)
            )
        
            rHS, rTO, lHS, lTO, ok, usable_cycles = detect_correct_order_robust(
                rHS, rTO, lHS, lTO,
                min_full_cycles=2,
                debug=visualize
            )
            cleaned_event_count = len(rHS) + len(rTO) + len(lHS) + len(lTO)
            rejected_fraction = (
                (raw_event_count - cleaned_event_count) / raw_event_count
                if raw_event_count > 0 else 0
            )
        
            if not ok:
                if prom == prominences[-1]:
                    raise ValueError(
                        'Could not find a sufficiently long valid gait event sequence. '
                        'Trial may need trimming or has poor marker quality.'
                    )
                else:
                    print(
                        'Gait events were noisy or incomplete. Trying peak detection again '
                        'with prominence = ' + str(prominences[i+1]) + '.'
                    )
            else:
                if rejected_fraction > robust_rejection_fallback_threshold:
                    fallback = clean_events_by_stride_windows(
                        raw_rHS, raw_rTO, raw_lHS, raw_lTO,
                        r_calc_rel_x, r_toe_rel_x,
                        l_calc_rel_x, l_toe_rel_x,
                        min_full_cycles=2,
                        debug=visualize
                    )
                    if fallback[4]:
                        print(
                            'Robust gait ordering removed '
                            f'{100 * rejected_fraction:.1f}% of candidate events. '
                            'Using stride-window fallback events.'
                        )
                        rHS, rTO, lHS, lTO, _, usable_cycles = fallback
                    else:
                        print(
                            'Robust gait ordering removed '
                            f'{100 * rejected_fraction:.1f}% of candidate events, '
                            'but stride-window fallback did not find enough valid cycles. '
                            'Keeping robust-order events.'
                        )
                print(f'Using {usable_cycles} valid gait cycles with prominence = {prom}.')
                break
        
        if visualize:
            import matplotlib.pyplot as plt
            plt.close('all')
            title_suffix = f": {self.trial_label}" if self.trial_label else ""
            plt.figure(1)
            plt.plot(self.markerDict['time'],r_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],r_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][rHS], r_calc_rel_x[rHS], color='red', label='rHS')
            plt.scatter(self.markerDict['time'][rTO], r_toe_rel_x[rTO], color='blue', label='rTO')
            plt.legend()
            plt.title(f'Gait event detection: right{title_suffix}')

            plt.figure(2)
            plt.plot(self.markerDict['time'],l_toe_rel_x,label='toe')
            plt.plot(self.markerDict['time'],l_calc_rel_x,label='calc')
            plt.scatter(self.markerDict['time'][lHS], l_calc_rel_x[lHS], color='red', label='lHS')
            plt.scatter(self.markerDict['time'][lTO], l_toe_rel_x[lTO], color='blue', label='lTO')
            plt.legend()
            plt.title(f'Gait event detection: left{title_suffix}')
            if self.pause_after_visualization:
                plt.show(block=True)

        # Find the number of gait cycles for the foot of interest.
        if leg=='auto':
            # Find the last HS of either foot.
            if rHS[-1] > lHS[-1]:
                default_leg = 'r'
            else:
                default_leg = 'l'
        else:
            default_leg = leg.lower()

        all_gait_events = {
            'default_leg': default_leg,
            'r': self._build_gait_events_for_leg(rHS, rTO, lHS, lTO, 'r',
                                                 n_gait_cycles=n_gait_cycles),
            'l': self._build_gait_events_for_leg(lHS, lTO, rHS, rTO, 'l',
                                                 n_gait_cycles=n_gait_cycles)
        }

        if return_all_legs:
            return all_gait_events
        
        return all_gait_events[default_leg]
