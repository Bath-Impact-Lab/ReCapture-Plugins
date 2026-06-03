"""
    ---------------------------------------------------------------------------
    ReCapture processing: ergo_rowing_analysis.py
    ---------------------------------------------------------------------------

    Initial scaffold for ergometer rowing cycle segmentation using the most
    anterior hand position as a surrogate catch event.
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
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt

from utilsKinematics import kinematics


class ergo_rowing_analysis(kinematics):

    def __init__(self, session_dir, trial_name, model_path, trcFilePath,
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_cycles=-1, trimming_start=0, trimming_end=0,
                 cycle_signal='wrist_anterior', visualize=False,
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

        all_cycle_events = self.segment_rowing(
            n_cycles=n_cycles,
            return_all_sides=True,
            visualize=visualize)
        self.primarySide = all_cycle_events['default_side']
        self.cycleEventsBySide = {
            'r': all_cycle_events['r'],
            'l': all_cycle_events['l']
        }
        self.cycleEventsBilateral = self.get_bilateral_cycle_events(
            reference_side=self.primarySide)

        self.defaultSide = self.primarySide
        self.cycleEvents = self.cycleEventsBySide[self.primarySide]
        self.nCycles = np.shape(self.cycleEvents['ipsilateralIdx'])[0]

    def get_cycle_events(self, side='both'):
        if side in [None, 'both', 'bilateral']:
            return copy.deepcopy(self.cycleEventsBilateral)
        if side in ['primary', 'default']:
            return copy.deepcopy(self.cycleEventsBySide[self.primarySide])
        return copy.deepcopy(self.cycleEventsBySide[self._normalize_side(side)])

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

    def compute_stroke_rate(self, return_all=False, side=None):
        cycle_duration, _ = self.compute_cycle_duration(return_all=True, side=side)
        stroke_rate = 60 / cycle_duration
        stroke_rate_mean = np.mean(stroke_rate)
        units = 'spm'

        if return_all:
            return stroke_rate, units
        return stroke_rate_mean, units

    def compute_stroke_rate_by_side(self, return_all=False):
        stroke_rate_by_side = {}
        for side in ['r', 'l']:
            cycle_duration, _ = self.compute_cycle_duration(return_all=True, side=side)
            stroke_rate_by_side[side] = 60 / cycle_duration

        units = 'spm'
        if return_all:
            return stroke_rate_by_side, units
        return {side: np.mean(values) for side, values in stroke_rate_by_side.items()}, units

    def compute_angle_outputs(self):
        outputAngles = {}

        for joint_name, joint_config in self._get_angle_definitions().items():
            outputAngles[joint_name] = {
                'Catch': {'LR': {}},
                'Finish': {'LR': {}},
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

                catch = np.zeros(len(start_idx))
                finish = np.zeros(len(start_idx))
                peak = np.zeros(len(start_idx))
                trough = np.zeros(len(start_idx))
                rom = np.zeros(len(start_idx))

                for i in range(len(start_idx)):
                    cycle_data = coord_values[start_idx[i]:end_idx[i]]
                    catch[i] = coord_values[start_idx[i]]
                    finish[i] = coord_values[half_idx[i]]
                    peak[i] = np.max(cycle_data)
                    trough[i] = np.min(cycle_data)
                    rom[i] = np.ptp(cycle_data)

                outputAngles[joint_name]['Catch']['LR'][side] = catch
                outputAngles[joint_name]['Finish']['LR'][side] = finish
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

    def segment_rowing(self, n_cycles=-1, return_all_sides=False, visualize=False):
        wrist_signal = self._get_wrist_anterior_signal()
        wrist_signal = self._prepare_detection_signal(wrist_signal)
        min_distance = self._estimate_min_peak_distance(wrist_signal)

        prominence_levels = [0.1, 0.05]
        catch_peaks = None        
        for prominence in prominence_levels:
            peaks, param = find_peaks(wrist_signal, prominence=prominence, distance=min_distance, width=min_distance/3)
            peaks = self._confirm_peaks_with_velocity_zero_crossing(wrist_signal, peaks)
            
            #filter if variables are way lower than the three max peaks
            peak_values = wrist_signal[peaks]
            top3 = np.sort(peak_values)[-3:]
            avg_top3 = np.mean(top3)
            threshold = 0.5 * avg_top3
            valid_mask = peak_values >= threshold
            filtered_peaks = peaks[valid_mask]
            
            if len(peaks) >= 3:
                catch_peaks = filtered_peaks
                break

        if catch_peaks is None or len(catch_peaks) <=2:
            raise ValueError('Could not detect enough rowing catch events from wrist position.')

        print(
            f"[debug rowing] trial={self.trialName if hasattr(self, 'trialName') else 'unknown'} "
            f"marker_len={len(self.markerDict['time'])} "
            f"coord_len={len(self.coordinateValues)} "
            f"n_peaks={len(catch_peaks)} "
            f"max_peak={int(np.max(catch_peaks))}"
        )

        side_events = {}
        for side in ['r', 'l']:
            side_events[side] = self._build_cycle_events_for_side(
                catch_peaks, side, n_cycles=n_cycles)

        if visualize:
            self._visualize_cycle_detection(wrist_signal, catch_peaks)
            if self.pause_after_visualization:
                plt.show(block=True)

        all_cycle_events = {
            'default_side': 'r',
            'r': side_events['r'],
            'l': side_events['l']
        }

        if return_all_sides:
            return all_cycle_events
        return all_cycle_events['r']

    def _get_side_cycle_events(self, side=None):
        return self.cycleEventsBySide[self._normalize_side(side)]

    def _get_wrist_anterior_signal(self):
        right_wrist = self._get_first_available_marker([
            'r_mwrist_study', 'r_lwrist_study',
            'RWRIST'
        ])
        left_wrist = self._get_first_available_marker([
            'L_mwrist_study', 'L_lwrist_study',
            'LWRIST'
        ])

        if right_wrist is None and left_wrist is None:
            raise KeyError('Could not find wrist markers for rowing detection.')

        signals = []
        if self.cycle_signal == 'wrist_relative_to_toe':
            right_toe = self._get_first_available_marker(['r_toe_study', 'RTOE'])
            left_toe = self._get_first_available_marker(['L_toe_study', 'LTOE'])

            if right_wrist is not None and right_toe is not None:
                signals.append(right_wrist[:, 0] - right_toe[:, 0])
            elif right_wrist is not None and left_toe is not None:
                signals.append(right_wrist[:, 0] - left_toe[:, 0])

            if left_wrist is not None and left_toe is not None:
                signals.append(left_wrist[:, 0] - left_toe[:, 0])
            elif left_wrist is not None and right_toe is not None:
                signals.append(left_wrist[:, 0] - right_toe[:, 0])

            if not signals:
                raise KeyError('Could not find toe markers to pair with wrist markers for rowing detection.')
        else:
            if right_wrist is not None:
                signals.append(right_wrist[:, 0])
            if left_wrist is not None:
                signals.append(left_wrist[:, 0])

        return np.mean(np.vstack(signals), axis=0)

    def _prepare_detection_signal(self, signal):
        signal = np.asarray(signal, dtype=float)

        win = min(len(signal) - 1 if len(signal) % 2 == 0 else len(signal), 31)
        if win < 5:
            win = 5
        if win % 2 == 0:
            win += 1
        signal_smooth = savgol_filter(signal, window_length=win, polyorder=2)

        baseline_win = max(101, min(len(signal) // 10 * 2 + 1, 301))
        if baseline_win % 2 == 0:
            baseline_win += 1
        baseline = median_filter(signal_smooth, size=baseline_win, mode='nearest')

        return signal_smooth - baseline

    def _confirm_peaks_with_velocity_zero_crossing(self, signal, candidate_peaks,
                                                   search_radius=8,
                                                   velocity_tolerance_ratio=0.2):
        if len(candidate_peaks) == 0:
            return np.array([], dtype=int)

        velocity = np.gradient(signal)
        max_abs_velocity = np.max(np.abs(velocity))
        velocity_tolerance = max_abs_velocity * velocity_tolerance_ratio
        confirmed_peaks = []

        for peak_idx in candidate_peaks:
            left = max(1, peak_idx - search_radius)
            right = min(len(signal) - 2, peak_idx + search_radius)
            best_idx = None
            best_score = np.inf

            for idx in range(left, right + 1):
                sign_change = velocity[idx - 1] > 0 and velocity[idx + 1] < 0
                near_zero_velocity = abs(velocity[idx]) <= velocity_tolerance

                if sign_change and near_zero_velocity:
                    score = abs(velocity[idx])
                    if score < best_score:
                        best_score = score
                        best_idx = idx

            if best_idx is None:
                local_vel = velocity[left:right + 1]
                candidate_idx = left + int(np.argmin(np.abs(local_vel)))
                sign_change = (
                    velocity[max(candidate_idx - 1, 0)] > 0 and
                    velocity[min(candidate_idx + 1, len(velocity) - 1)] < 0
                )
                if not sign_change:
                    continue
                best_idx = candidate_idx

            confirmed_peaks.append(best_idx)

        if not confirmed_peaks:
            return np.array([], dtype=int)

        return np.array(sorted(set(confirmed_peaks)), dtype=int)

    def _estimate_min_peak_distance(self, signal):
        dt = np.diff(self.markerDict['time'][:2])[0]
        if dt <= 0:
            return 10
        min_cycle_seconds = 0.8
        return max(10, int(np.round(min_cycle_seconds / dt)))

    def _build_cycle_events_for_side(self, catch_peaks, side, n_cycles=-1):
        if len(catch_peaks) < 2:
            raise Exception(f'Not enough rowing cycles found for side {side}.')

        available_cycles = len(catch_peaks) - 1
        if available_cycles < n_cycles:
            print('You requested {} cycles, but only {} were found. '
                  'Proceeding with this number.'.format(n_cycles, available_cycles))
            n_cycles = available_cycles
        if n_cycles == -1:
            n_cycles = available_cycles
            print('Processing {} rowing cycles, side: {}.'.format(n_cycles, side))

        cycle_events = np.zeros((n_cycles, 3), dtype=int)
        for i in range(n_cycles):
            start_idx = catch_peaks[-i-2]
            end_idx = catch_peaks[-i-1]
            half_idx = int(np.round((start_idx + end_idx) / 2))
            cycle_events[i, :] = [start_idx, half_idx, end_idx]

        cycle_times = self.markerDict['time'][cycle_events]

        return {
            'ipsilateralIdx': cycle_events,
            'contralateralIdx': None,
            'ipsilateralTime': cycle_times,
            'contralateralTime': None,
            'eventNamesIpsilateral': ['Catch', 'Finish', 'Catch'],
            'eventNamesContralateral': None,
            'ipsilateralLeg': side
        }

    def _visualize_cycle_detection(self, signal, peaks):
        plt.close('all')
        plt.figure(1)
        plt.plot(self.markerDict['time'], signal, label='wrist anterior signal')
        plt.scatter(self.markerDict['time'][peaks], signal[peaks],
                    color='red', label='catch surrogate')
        plt.legend()
        title = 'Rowing event detection'
        if self.trial_label:
            title = f'{title}: {self.trial_label}'
        plt.title(title)
