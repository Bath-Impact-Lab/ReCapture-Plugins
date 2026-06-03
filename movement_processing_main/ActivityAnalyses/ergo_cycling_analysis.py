"""
    ---------------------------------------------------------------------------
    ReCapture processing: ergo_cycling_analysis.py
    ---------------------------------------------------------------------------

    Initial scaffold for cycle segmentation of ergometer cycling trials using
    ankle-marker-based top-dead-center surrogate events.
--------------------------------------------------------------------------------
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import copy
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from utilsKinematics import kinematics


class ergo_cycling_analysis(kinematics):

    def __init__(self, session_dir, trial_name, model_path, trcFilePath,
                 side='auto', lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_cycles=-1, trimming_start=0, trimming_end=0,
                 cycle_signal='foot_center_vertical', visualize=False,
                 pause_after_visualization=False, trial_label=None):

        super().__init__(
            session_dir,
            trial_name,
            model_path,
            trcFilePath,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)

        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
        self.cycle_signal = cycle_signal
        self.pause_after_visualization = pause_after_visualization
        self.trial_label = trial_label

        self.markerDict = self.get_marker_dict(
            session_dir, trial_name, trcFilePath,
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values)
        self.coordinateValues = self.get_coordinate_values()
        self.model_path = model_path

        if self.trimming_start > 0:
            self.idx_trim_start = np.where(
                np.round(self.markerDict['time'] - self.trimming_start, 6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:, ]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = (
                    self.markerDict['markers'][marker][self.idx_trim_start:, :])
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]

        if self.trimming_end > 0:
            self.idx_trim_end = np.where(
                np.round(self.markerDict['time'], 6) <=
                np.round(self.markerDict['time'][-1] - self.trimming_end, 6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end, ]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = (
                    self.markerDict['markers'][marker][:self.idx_trim_end, :])
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]

        all_cycle_events = self.segment_cycling(
            n_cycles=n_cycles,
            side=side,
            return_all_sides=True,
            visualize=visualize)
        self.primarySide = all_cycle_events['default_side']
        self.cycleEventsBySide = {
            'r': all_cycle_events['r'],
            'l': all_cycle_events['l']
        }
        self.cycleEventsBilateral = self.get_bilateral_cycle_events(
            reference_side=self.primarySide)

        # Backward-compatible aliases for one-side callers if needed later.
        self.defaultSide = self.primarySide
        self.cycleEvents = self.cycleEventsBySide[self.primarySide]
        self.nCycles = np.shape(self.cycleEvents['ipsilateralIdx'])[0]

    def get_cycle_events(self, side='both'):
        if side in [None, 'both', 'bilateral']:
            return copy.deepcopy(self.cycleEventsBilateral)
        if side in ['primary', 'default']:
            return copy.deepcopy(self.cycleEventsBySide[self.primarySide])
        return copy.deepcopy(self.cycleEventsBySide[self._normalize_side(side)])

    def get_primary_cycle_events(self):
        return self.get_cycle_events(side='primary')

    def get_cycle_events_by_side(self):
        return {side: copy.deepcopy(events)
                for side, events in self.cycleEventsBySide.items()}

    def get_cycle_counts(self):
        return {
            side: np.shape(events['ipsilateralIdx'])[0]
            for side, events in self.cycleEventsBySide.items()
        }

    def get_bilateral_cycle_events(self, reference_side=None):
        reference_side = self._normalize_side(reference_side)
        contralateral_side = 'l' if reference_side == 'r' else 'r'

        ipsilateral_events = copy.deepcopy(self.cycleEventsBySide[reference_side])
        contralateral_events = self.cycleEventsBySide[contralateral_side]

        return {
            'ipsilateralIdx': ipsilateral_events['ipsilateralIdx'],
            'contralateralIdx': contralateral_events['ipsilateralIdx'],
            'ipsilateralTime': ipsilateral_events['ipsilateralTime'],
            'contralateralTime': contralateral_events['ipsilateralTime'],
            'eventNamesIpsilateral': ipsilateral_events['eventNamesIpsilateral'],
            'eventNamesContralateral': contralateral_events['eventNamesIpsilateral'],
            'ipsilateralLeg': ipsilateral_events['ipsilateralLeg']
        }

    def get_side(self, lower=False, side=None):
        if self._normalize_side(side) == 'r':
            ipsilateral = 'r'
            contralateral = 'L'
        else:
            ipsilateral = 'L'
            contralateral = 'r'

        if lower:
            return ipsilateral.lower(), contralateral.lower()
        return ipsilateral, contralateral

    def get_coordinates_normalized_time(self, side='both'):
        if side in ['both', 'bilateral', None]:
            return {
                cycle_side: self.get_coordinates_normalized_time(side=cycle_side)
                for cycle_side in ['r', 'l']
            }

        cycle_events = self._get_side_cycle_events(side=side)
        n_cycles = np.shape(cycle_events['ipsilateralIdx'])[0]
        colNames = self.coordinateValues.columns
        data = self.coordinateValues.to_numpy(copy=True)
        coordValuesNorm = []

        for i in range(n_cycles):
            coordValues = data[
                cycle_events['ipsilateralIdx'][i, 0]:
                cycle_events['ipsilateralIdx'][i, 2] + 1]
            coordValuesNorm.append(np.stack([
                np.interp(
                    np.linspace(0, 100, 101),
                    np.linspace(0, 100, len(coordValues)),
                    coordValues[:, j])
                for j in range(coordValues.shape[1])
            ], axis=1))

        coordinateValuesTimeNormalized = {}
        coordVals_mean = np.mean(np.array(coordValuesNorm), axis=0)
        coordinateValuesTimeNormalized['mean'] = pd.DataFrame(
            data=coordVals_mean, columns=colNames)

        if n_cycles > 2:
            coordVals_sd = np.std(np.array(coordValuesNorm), axis=0)
            coordinateValuesTimeNormalized['sd'] = pd.DataFrame(
                data=coordVals_sd, columns=colNames)
        else:
            coordinateValuesTimeNormalized['sd'] = None

        coordinateValuesTimeNormalized['indiv'] = [
            pd.DataFrame(data=d, columns=colNames) for d in coordValuesNorm]

        return coordinateValuesTimeNormalized

    def compute_cycle_duration(self, return_all=False, side=None):
        cycle_events = self._get_side_cycle_events(side=side)
        cycle_durations = np.diff(cycle_events['ipsilateralTime'][:, (0, 2)])
        cycle_duration = np.mean(cycle_durations)
        units = 's'

        if return_all:
            return cycle_durations, units
        return cycle_duration, units

    def compute_cycle_duration_by_side(self, return_all=False):
        cycle_durations = {
            'r': np.diff(self.cycleEventsBySide['r']['ipsilateralTime'][:, (0, 2)]),
            'l': np.diff(self.cycleEventsBySide['l']['ipsilateralTime'][:, (0, 2)])
        }
        units = 's'

        if return_all:
            return cycle_durations, units
        return {side: np.mean(values) for side, values in cycle_durations.items()}, units

    def compute_cadence(self, return_all=False, side=None):
        cycle_duration, _ = self.compute_cycle_duration(return_all=True, side=side)
        cadence = 60 / cycle_duration
        cadence_mean = np.mean(cadence)
        units = 'rpm'

        if return_all:
            return cadence, units
        return cadence_mean, units

    def compute_cadence_by_side(self, return_all=False):
        cadence_by_side = {}
        for side in ['r', 'l']:
            cycle_duration, _ = self.compute_cycle_duration(return_all=True, side=side)
            cadence_by_side[side] = 60 / cycle_duration

        units = 'rpm'
        if return_all:
            return cadence_by_side, units
        return {side: np.mean(values) for side, values in cadence_by_side.items()}, units

    def compute_angle_outputs(self):
        outputAngles = {}

        for joint_name, joint_config in self._get_angle_definitions().items():
            outputAngles[joint_name] = {
                'TDC': {'LR': {}},
                'BDC': {'LR': {}},
                f"Peak {joint_config['positive_label']}": {'LR': {}},
                f"Peak {joint_config['negative_label']}": {'LR': {}},
                'ROM': {'LR': {}}
            }

            for side in ['r', 'l']:
                cycle_events = self._get_side_cycle_events(side=side)
                start_idx = cycle_events['ipsilateralIdx'][:, 0]
                half_idx = cycle_events['ipsilateralIdx'][:, 1]
                end_idx = cycle_events['ipsilateralIdx'][:, 2]
                coord_values = self._get_joint_coordinate_series(
                    joint_config, side=side).to_numpy(copy=True)

                tdc = np.zeros(len(start_idx))
                bdc = np.zeros(len(start_idx))
                peak = np.zeros(len(start_idx))
                trough = np.zeros(len(start_idx))
                rom = np.zeros(len(start_idx))

                for i in range(len(start_idx)):
                    cycle_data = coord_values[start_idx[i]:end_idx[i]]
                    tdc[i] = coord_values[start_idx[i]]
                    bdc[i] = coord_values[half_idx[i]]
                    peak[i] = np.max(cycle_data)
                    trough[i] = np.min(cycle_data)
                    rom[i] = np.ptp(cycle_data)

                outputAngles[joint_name]['TDC']['LR'][side] = tdc
                outputAngles[joint_name]['BDC']['LR'][side] = bdc
                outputAngles[joint_name][f"Peak {joint_config['positive_label']}"]['LR'][side] = peak
                outputAngles[joint_name][f"Peak {joint_config['negative_label']}"]['LR'][side] = trough
                outputAngles[joint_name]['ROM']['LR'][side] = rom

        return outputAngles

    def compute_angle_graphs(self):
        tsAngles = {}

        for joint_name, joint_config in self._get_angle_definitions().items():
            tsAngles[joint_name] = {'LR': {}}

            for side in ['r', 'l']:
                cycle_events = self._get_side_cycle_events(side=side)
                start_idx = cycle_events['ipsilateralIdx'][:, 0]
                end_idx = cycle_events['ipsilateralIdx'][:, 2]
                coord_values = self._get_joint_coordinate_series(
                    joint_config, side=side).to_numpy(copy=True)

                ts = np.zeros((101, len(start_idx)))
                for i in range(len(start_idx)):
                    cycle_data = coord_values[start_idx[i]:end_idx[i] + 1]
                    ts[:, i] = np.interp(
                        np.linspace(0, 100, 101),
                        np.linspace(0, 100, len(cycle_data)),
                        cycle_data
                    )

                tsAngles[joint_name]['LR'][side] = ts

        return tsAngles

    def segment_cycling(self, n_cycles=-1, side='auto', return_all_sides=False,
                        visualize=False):

        prominence_levels = [0.02, 0.015, 0.01]
        side_events = {}

        detected_peaks = {}
        for cycle_side in ['r', 'l']:
            signal = self._get_cycle_signal(cycle_side)
            min_distance = self._estimate_min_peak_distance(signal)
            valid_peaks = None

            for prominence in prominence_levels:
                peaks, _ = find_peaks(signal, prominence=prominence,
                                      distance=min_distance)
                if len(peaks) >= 2:
                    valid_peaks = peaks
                    break

            if valid_peaks is None:
                raise ValueError(
                    f'Could not detect enough ankle-based cycling events for side {cycle_side}.')

            detected_peaks[cycle_side] = valid_peaks

        detected_peaks['r'], detected_peaks['l'], ok, usable_cycles = (
            self._enforce_alternating_tdc_sequence(
                detected_peaks['r'], detected_peaks['l'], min_cycles=2))

        if not ok:
            raise ValueError(
                'Could not find a sufficiently long valid alternating cycling event sequence.')

        print(f'Using {usable_cycles} valid alternating cycling cycles.')

        for cycle_side in ['r', 'l']:
            side_events[cycle_side] = self._build_cycle_events_for_side(
                detected_peaks[cycle_side], cycle_side, n_cycles=n_cycles)

        if side == 'auto':
            default_side = 'r' if len(detected_peaks['r']) >= len(detected_peaks['l']) else 'l'
        else:
            default_side = side.lower()

        if visualize:
            self._visualize_cycle_detection(detected_peaks)
            if self.pause_after_visualization:
                plt.show(block=True)

        all_cycle_events = {
            'default_side': default_side,
            'r': side_events['r'],
            'l': side_events['l']
        }

        if return_all_sides:
            return all_cycle_events
        return all_cycle_events[default_side]

    def _get_side_cycle_events(self, side=None):
        return self.cycleEventsBySide[self._normalize_side(side)]

    def _get_cycle_signal(self, side):
        if self.cycle_signal == 'ankle_vertical':
            return self._get_ankle_vertical_signal(side)
        return self._get_foot_center_vertical_signal(side)

    def _get_ankle_vertical_signal(self, side):
        ankle_position = self._get_ankle_marker(side)
        reference_position = self._get_hip_reference(side)
        relative_position = ankle_position - reference_position
        return relative_position[:, 1]

    def _get_foot_center_vertical_signal(self, side):
        ankle_position = self._get_ankle_marker(side)
        toe_position = self._get_toe_marker(side)

        if toe_position is None:
            return self._get_ankle_vertical_signal(side)

        foot_center = (ankle_position + toe_position) / 2
        reference_position = self._get_hip_reference(side)
        relative_position = foot_center - reference_position
        return relative_position[:, 1]

    def _get_ankle_marker(self, side):
        if side == 'r':
            lateral_ankle = self._get_first_available_marker(
                ['r_ankle_study', 'R_ankle_study'])
            medial_ankle = self._get_first_available_marker(
                ['r_mankle_study', 'R_mankle_study'])
            fallback_ankle = self._get_first_available_marker(['RANK'])
        else:
            lateral_ankle = self._get_first_available_marker(
                ['L_ankle_study', 'l_ankle_study'])
            medial_ankle = self._get_first_available_marker(
                ['L_mankle_study', 'l_mankle_study'])
            fallback_ankle = self._get_first_available_marker(['LANK'])

        if lateral_ankle is not None and medial_ankle is not None:
            ankle_position = (lateral_ankle + medial_ankle) / 2
        elif lateral_ankle is not None:
            ankle_position = lateral_ankle
        elif fallback_ankle is not None:
            ankle_position = fallback_ankle
        else:
            raise KeyError(f'Could not find ankle marker for side {side}.')
        return ankle_position

    def _get_toe_marker(self, side):
        if side == 'r':
            return self._get_first_available_marker(
                ['r_toe_study', 'R_toe_study', 'RTOE'])
        return self._get_first_available_marker(
            ['L_toe_study', 'l_toe_study', 'LTOE'])

    def _get_hip_reference(self, side):
        if side == 'r':
            hjc_marker = self._get_first_available_marker(['RHJC_study', 'RHJC'])
            hip_marker = self._get_first_available_marker(['RHIP'])
        else:
            hjc_marker = self._get_first_available_marker(['LHJC_study', 'LHJC'])
            hip_marker = self._get_first_available_marker(['LHIP'])

        if hjc_marker is not None:
            return hjc_marker
        if hip_marker is not None:
            return hip_marker

        if 'r.ASIS_study' in self.markerDict['markers']:
            pelvis_markers = [
                self.markerDict['markers']['r.ASIS_study'],
                self.markerDict['markers']['L.ASIS_study'],
                self.markerDict['markers']['r.PSIS_study'],
                self.markerDict['markers']['L.PSIS_study']
            ]
        else:
            pelvis_markers = [
                self.markerDict['markers']['PELVIS1'],
                self.markerDict['markers']['RHIP'],
                self.markerDict['markers']['LHIP']
            ]
        return np.mean(np.array(pelvis_markers), axis=0)

    def _estimate_min_peak_distance(self, signal):
        dt = np.diff(self.markerDict['time'][:2])[0]
        if dt <= 0:
            return 10
        # Conservative default for cycling cadence up to ~120 rpm.
        min_cycle_seconds = 0.45
        return max(10, int(np.round(min_cycle_seconds / dt)))

    def _enforce_alternating_tdc_sequence(self, r_peaks, l_peaks, min_cycles=2):
        expected_next = {'rTDC': 'lTDC', 'lTDC': 'rTDC'}

        events = []
        events.extend((int(x), 'rTDC') for x in r_peaks)
        events.extend((int(x), 'lTDC') for x in l_peaks)
        events.sort(key=lambda x: x[0])

        if len(events) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), False, 0

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
                    continue

            if len(seq) > len(best_sequence):
                best_sequence = seq

        if len(best_sequence) == 0:
            return np.array([], dtype=int), np.array([], dtype=int), False, 0

        r_clean = np.array([idx for idx, label in best_sequence if label == 'rTDC'], dtype=int)
        l_clean = np.array([idx for idx, label in best_sequence if label == 'lTDC'], dtype=int)
        usable_cycles = min(max(len(r_clean) - 1, 0), max(len(l_clean) - 1, 0))
        is_usable = usable_cycles >= min_cycles

        return r_clean, l_clean, is_usable, usable_cycles

    def _build_cycle_events_for_side(self, tdc_peaks, side, n_cycles=-1):
        if len(tdc_peaks) < 2:
            raise Exception(f'Not enough cycling cycles found for side {side}.')

        available_cycles = len(tdc_peaks) - 1
        if available_cycles < n_cycles:
            print('You requested {} cycles, but only {} were found. '
                  'Proceeding with this number.'.format(n_cycles, available_cycles))
            n_cycles = available_cycles
        if n_cycles == -1:
            n_cycles = available_cycles
            print('Processing {} cycling cycles, side: {}.'.format(n_cycles, side))

        cycle_events = np.zeros((n_cycles, 3), dtype=int)
        for i in range(n_cycles):
            start_idx = tdc_peaks[-i-2]
            end_idx = tdc_peaks[-i-1]
            half_idx = int(np.round((start_idx + end_idx) / 2))
            cycle_events[i, :] = [start_idx, half_idx, end_idx]

        cycle_times = self.markerDict['time'][cycle_events]

        return {
            'ipsilateralIdx': cycle_events,
            'contralateralIdx': None,
            'ipsilateralTime': cycle_times,
            'contralateralTime': None,
            'eventNamesIpsilateral': ['TDC', 'BDC', 'TDC'],
            'eventNamesContralateral': None,
            'ipsilateralLeg': side
        }

    def _visualize_cycle_detection(self, detected_peaks):
        plt.close('all')
        for fig_idx, side in enumerate(['r', 'l'], start=1):
            signal = self._get_cycle_signal(side)
            peaks = detected_peaks[side]
            plt.figure(fig_idx)
            plt.plot(self.markerDict['time'], signal, label=f'{side} cycle signal')
            plt.scatter(self.markerDict['time'][peaks], signal[peaks],
                        color='red', label='TDC surrogate')
            plt.legend()
            title = f'Cycling event detection: {side}'
            if self.trial_label:
                title = f'{title}: {self.trial_label}'
            plt.title(title)
