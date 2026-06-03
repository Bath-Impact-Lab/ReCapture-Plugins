"""
    ---------------------------------------------------------------------------
    ReCapture processing: sit_to_stand_analysis.py
    ---------------------------------------------------------------------------

    Sit-to-stand segmentation using trunk flexion onset, pelvis rise onset,
    and post-rise stillness. No chair/contact marker is assumed.
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
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

from utilsKinematics import kinematics


class sit_to_stand_analysis(kinematics):

    def __init__(self, session_dir, trial_name, model_path, trcFilePath,
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 trimming_start=0, trimming_end=0,
                 trunk_signal='lumbar_extension', visualize=False,
                 pause_after_visualization=False, trial_label=None):

        super().__init__(
            session_dir,
            trial_name,
            model_path,
            trcFilePath,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)

        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
        self.trunk_signal = trunk_signal
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

        self.stsEvents = self.segment_sit_to_stand(visualize=visualize)
        self.primarySide = 'r'
        self.defaultSide = self.primarySide

    def get_sit_to_stand_events(self):
        return copy.deepcopy(self.stsEvents)

    def get_cycle_events(self, side='both'):
        del side
        return self.get_sit_to_stand_events()

    def get_coordinates_normalized_time(self, side='both'):
        del side

        colNames = self.coordinateValues.columns
        coordValuesNorm = []
        for window_idx in self.stsEvents['windowIdx']:
            start_idx, _, _, end_idx = window_idx
            coordValues = self.coordinateValues.iloc[start_idx:end_idx + 1].to_numpy(copy=True)
            coordValuesNorm.append(np.stack([
                np.interp(
                    np.linspace(0, 100, 101),
                    np.linspace(0, 100, len(coordValues)),
                    coordValues[:, j])
                for j in range(coordValues.shape[1])
            ], axis=1))

        coordVals_mean = np.mean(np.array(coordValuesNorm), axis=0)
        coordVals_sd = None
        if len(coordValuesNorm) > 1:
            coordVals_sd = pd.DataFrame(
                data=np.std(np.array(coordValuesNorm), axis=0), columns=colNames)

        return {
            'mean': pd.DataFrame(data=coordVals_mean, columns=colNames),
            'sd': coordVals_sd,
            'indiv': [pd.DataFrame(data=d, columns=colNames) for d in coordValuesNorm]
        }

    def get_sit_to_stand_count(self):
        return int(self.stsEvents['windowIdx'].shape[0])

    def compute_full_cycle_duration(self, return_all=False):
        duration = self.stsEvents['windowTime'][:, 3] - self.stsEvents['windowTime'][:, 0]
        if return_all:
            return duration, 's'
        return float(np.mean(duration)), 's'

    def compute_sit_to_stand_duration(self, return_all=False):
        duration = self.stsEvents['windowTime'][:, 1] - self.stsEvents['windowTime'][:, 0]
        if return_all:
            return duration, 's'
        return float(np.mean(duration)), 's'

    def compute_standing_duration(self, return_all=False):
        duration = self.stsEvents['windowTime'][:, 2] - self.stsEvents['windowTime'][:, 1]
        if return_all:
            return duration, 's'
        return float(np.mean(duration)), 's'

    def compute_stand_to_sit_duration(self, return_all=False):
        duration = self.stsEvents['windowTime'][:, 3] - self.stsEvents['windowTime'][:, 2]
        if return_all:
            return duration, 's'
        return float(np.mean(duration)), 's'

    def compute_angle_outputs(self):
        outputAngles = {}
        event_positions = self.stsEvents['windowIdx']

        for joint_name, joint_config in self._get_angle_definitions().items():
            outputAngles[joint_name] = {
                'SittingEnd': {'LR': {}},
                'StandingStart': {'LR': {}},
                'StandingEnd': {'LR': {}},
                'SittingStart': {'LR': {}},
                f"Peak {joint_config['positive_label']}": {'LR': {}},
                f"Peak {joint_config['negative_label']}": {'LR': {}},
                'ROM': {'LR': {}}
            }

            for side in ['r', 'l']:
                coord_values = self._get_joint_coordinate_series(
                    joint_config, side=side).to_numpy(copy=True)
                start_vals = np.zeros(len(event_positions))
                trunk_start_vals = np.zeros(len(event_positions))
                pelvis_rise_vals = np.zeros(len(event_positions))
                standing_vals = np.zeros(len(event_positions))
                peak_vals = np.zeros(len(event_positions))
                trough_vals = np.zeros(len(event_positions))
                rom_vals = np.zeros(len(event_positions))

                for i, (sitting_end_idx, standing_start_idx, standing_end_idx, sitting_start_idx) in enumerate(event_positions):
                    window_data = coord_values[sitting_end_idx:sitting_start_idx + 1]
                    start_vals[i] = coord_values[sitting_end_idx]
                    trunk_start_vals[i] = coord_values[standing_start_idx]
                    pelvis_rise_vals[i] = coord_values[standing_end_idx]
                    standing_vals[i] = coord_values[sitting_start_idx]
                    peak_vals[i] = np.max(window_data)
                    trough_vals[i] = np.min(window_data)
                    rom_vals[i] = np.ptp(window_data)

                outputAngles[joint_name]['SittingEnd']['LR'][side] = start_vals
                outputAngles[joint_name]['StandingStart']['LR'][side] = trunk_start_vals
                outputAngles[joint_name]['StandingEnd']['LR'][side] = pelvis_rise_vals
                outputAngles[joint_name]['SittingStart']['LR'][side] = standing_vals
                outputAngles[joint_name][f"Peak {joint_config['positive_label']}"]['LR'][side] = peak_vals
                outputAngles[joint_name][f"Peak {joint_config['negative_label']}"]['LR'][side] = trough_vals
                outputAngles[joint_name]['ROM']['LR'][side] = rom_vals

        return outputAngles

    def compute_angle_graphs(self):
        tsAngles = {}

        for joint_name, joint_config in self._get_angle_definitions().items():
            tsAngles[joint_name] = {'LR': {}}

            for side in ['r', 'l']:
                coord_values = self._get_joint_coordinate_series(
                    joint_config, side=side).to_numpy(copy=True)
                ts = np.zeros((101, len(self.stsEvents['windowIdx'])))
                for i, (start_idx, _, _, end_idx) in enumerate(self.stsEvents['windowIdx']):
                    window_data = coord_values[start_idx:end_idx + 1]
                    ts[:, i] = np.interp(
                        np.linspace(0, 100, 101),
                        np.linspace(0, 100, len(window_data)),
                        window_data
                    )
                tsAngles[joint_name]['LR'][side] = ts

        return tsAngles

    def segment_sit_to_stand(self, visualize=False):
        time = self.coordinateValues['time'].to_numpy(dtype=float)
        dt = float(np.median(np.diff(time)))
        if dt <= 0:
            raise ValueError('Coordinate time vector is not strictly increasing.')

        trunk_angle = self._smooth_signal(self._get_trunk_flexion_signal())
        pelvis_vertical = self._smooth_signal(self._get_pelvis_vertical_signal())
        trunk_velocity = np.gradient(trunk_angle, time)
        pelvis_velocity = np.gradient(pelvis_vertical, time)

        window_seconds = 0.25
        window_samples = max(5, int(np.round(window_seconds / dt)))
        standing_plateau_window_samples = max(5, int(np.round(0.20 / dt)))
        standing_plateau_search_samples = max(
            standing_plateau_window_samples,
            int(np.round(2.0 / dt))
        )
        standing_exit_window_samples = max(5, int(np.round(0.15 / dt)))
        baseline_samples = min(len(time), max(window_samples, int(np.round(1.0 / dt))))
        min_sustain = max(4, int(np.round(0.08 / dt)))
        min_rep_gap = max(20, int(np.round(0.7 / dt)))
        seated_trunk_baseline = float(np.median(trunk_angle[:baseline_samples]))
        seated_pelvis_baseline = float(np.median(pelvis_vertical[:baseline_samples]))

        pelvis_range = float(np.percentile(pelvis_vertical, 95) - np.percentile(pelvis_vertical, 5))
        if pelvis_range < 0.05:
            raise ValueError('Pelvis vertical displacement is too small to detect sit-to-stand.')

        seated_level = float(np.percentile(pelvis_vertical, 10))
        standing_level = float(np.percentile(pelvis_vertical, 90))
        high_pelvis_threshold = seated_level + 0.65 * (standing_level - seated_level)

        positive_pelvis_velocity = pelvis_velocity[pelvis_velocity > 0]
        if len(positive_pelvis_velocity) == 0:
            raise ValueError('Could not detect upward pelvis motion for sit-to-stand.')
        pelvis_velocity_threshold = max(
            0.04,
            0.25 * float(np.percentile(positive_pelvis_velocity, 90))
        )
        negative_pelvis_velocity = pelvis_velocity[pelvis_velocity < 0]
        pelvis_descent_velocity_threshold = (
            min(-0.04, 0.25 * float(np.percentile(negative_pelvis_velocity, 10)))
            if len(negative_pelvis_velocity) > 0 else -0.04
        )

        trunk_velocity_abs = np.abs(trunk_velocity)
        trunk_velocity_threshold = max(
            4.0,
            0.25 * float(np.percentile(trunk_velocity_abs, 90))
        )

        pelvis_rise_runs = self._extract_sustained_runs(
            pelvis_velocity > pelvis_velocity_threshold,
            min_sustain
        )
        pelvis_rise_starts = self._deduplicate_event_starts(
            [run[0] for run in pelvis_rise_runs],
            min_gap=min_rep_gap
        )
        pelvis_rise_starts = self._filter_valid_pelvis_rise_starts(
            pelvis_rise_starts,
            pelvis_vertical,
            seated_level,
            high_pelvis_threshold,
            pelvis_range,
            dt
        )

        quiet_metric = (
            np.abs(pelvis_velocity) / max(pelvis_velocity_threshold, 1e-8) 
            )
        still_runs = self._extract_still_runs(quiet_metric, window_samples)
        if len(still_runs) == 0:
            still_runs = [(0, min(len(time) - 1, window_samples - 1))]

        event_rows = []
        window_rows = []
        skipped_reasons = []
        last_sitting_start_idx = -1
        for rep_idx, pelvis_rise_idx in enumerate(pelvis_rise_starts):
            next_pelvis_rise_idx = (
                pelvis_rise_starts[rep_idx + 1]
                if rep_idx + 1 < len(pelvis_rise_starts) else len(time) - 1
            )
            start_still_idx, start_still_end_idx = self._find_preceding_still_run(
                still_runs, pelvis_rise_idx)
            if last_sitting_start_idx < 0:
                sitting_end_idx = self._refine_sitting_end_with_trunk_flexion(
                    trunk_angle,
                    rough_sitting_end_idx=start_still_end_idx,
                    pelvis_rise_idx=pelvis_rise_idx,
                    seated_trunk_baseline=seated_trunk_baseline,
                    dt=dt
                )
            else:
                sitting_end_idx = self._find_repeated_sitting_end_from_trunk(
                    trunk_angle,
                    previous_sitting_start_idx=last_sitting_start_idx,
                    pelvis_rise_idx=pelvis_rise_idx
                )
                if sitting_end_idx is None:
                    sitting_end_idx = last_sitting_start_idx + 1

            if sitting_end_idx <= last_sitting_start_idx:
                skipped_reasons.append(
                    f"rep {rep_idx + 1}: SittingEnd={sitting_end_idx} "
                    f"overlaps previous SittingStart={last_sitting_start_idx}; "
                    f"using {last_sitting_start_idx + 1}"
                )
                sitting_end_idx = last_sitting_start_idx + 1
                start_still_end_idx = max(start_still_end_idx, sitting_end_idx)

            trunk_extension_idx = self._find_trunk_extension_peak(
                trunk_angle,
                pelvis_rise_idx,
                seated_trunk_baseline,
                dt
            )

            if trunk_extension_idx < start_still_end_idx:
                trunk_extension_idx = start_still_end_idx

            standing_still_idx = self._find_following_standing_still_start(
                still_runs,
                pelvis_vertical,
                pelvis_rise_idx,
                high_pelvis_threshold,
                window_samples,
                default_end=min(len(time) - 1, pelvis_rise_idx + 2 * window_samples),
                plateau_window_samples=standing_plateau_window_samples,
                plateau_search_samples=standing_plateau_search_samples,
                plateau_range_threshold=max(0.01, 0.03 * pelvis_range),
                plateau_slope_threshold=0.05,
                dt=dt
            )

            if standing_still_idx <= pelvis_rise_idx:
                skipped_reasons.append(
                    f"rep {rep_idx + 1}: StandingStart not after pelvis rise "
                    f"(pelvis_rise_idx={pelvis_rise_idx}, standing_idx={standing_still_idx})"
                )
                continue

            stand_to_sit_idx, stand_to_sit_end_idx = self._find_stand_to_sit_events(
                trunk_angle,
                pelvis_vertical,
                pelvis_velocity,
                standing_still_idx,
                next_pelvis_rise_idx,
                seated_trunk_baseline,
                seated_pelvis_baseline,
                pelvis_descent_velocity_threshold,
                window_samples,
                standing_exit_window_samples,
                min_sustain,
                pelvis_range,
                dt
            )
            if stand_to_sit_idx is None:
                skipped_reasons.append(
                    f"rep {rep_idx + 1}: StandingEnd/StandToSitStart not detected "
                    f"after StandingStart={standing_still_idx}"
                )
                continue
            if stand_to_sit_end_idx is None:
                skipped_reasons.append(
                    f"rep {rep_idx + 1}: SittingStart not detected after "
                    f"StandingEnd={stand_to_sit_idx}"
                )
                continue
            if not (sitting_end_idx < standing_still_idx < stand_to_sit_idx < stand_to_sit_end_idx):
                skipped_reasons.append(
                    f"rep {rep_idx + 1}: full-cycle ordering failed "
                    f"({sitting_end_idx}, {standing_still_idx}, "
                    f"{stand_to_sit_idx}, {stand_to_sit_end_idx})"
                )
                continue

            event_rows.append([
                sitting_end_idx,
                standing_still_idx,
                stand_to_sit_idx,
                stand_to_sit_end_idx
            ])
            window_rows.append([
                sitting_end_idx,
                standing_still_idx,
                stand_to_sit_idx,
                stand_to_sit_end_idx
            ])
            last_sitting_start_idx = stand_to_sit_end_idx

        if len(event_rows) == 0:
            if skipped_reasons:
                print('Sit-stand full-cycle candidates skipped:')
                for reason in skipped_reasons:
                    print(f'  {reason}')
            raise ValueError('Could not construct valid sit-to-stand windows.')

        sts_events = {
            'eventIdx': np.asarray(event_rows, dtype=int),
            'eventTime': time[np.asarray(event_rows, dtype=int)],
            'eventNames': [
                'SittingEnd',
                'StandingStart',
                'StandingEnd',
                'SittingStart'
            ],
            'windowIdx': np.asarray(window_rows, dtype=int),
            'windowTime': time[np.asarray(window_rows, dtype=int)],
            'windowNames': [
                'SittingEnd',
                'StandingStart',
                'StandingEnd',
                'SittingStart'
            ]
        }

        if visualize:
            self._visualize_sit_to_stand_detection(
                time, trunk_angle, pelvis_vertical, sts_events,
                high_pelvis_threshold, pelvis_velocity_threshold)
            if self.pause_after_visualization:
                plt.show(block=True)
                
        test = 1

        return sts_events

    def _find_trunk_extension_peak(self, trunk_angle, pelvis_rise_idx,
                                   seated_trunk_baseline, dt):
        search_start = max(0, pelvis_rise_idx - int(round(0.8 / dt)))
        search_end = min(
            len(trunk_angle) - 1,
            pelvis_rise_idx + int(round(0.3 / dt))
        )
        if search_end <= search_start:
            return pelvis_rise_idx

        flexion_signal = trunk_angle - seated_trunk_baseline
        flexion_min_idx = search_start + int(
            np.argmin(flexion_signal[search_start:search_end + 1])
        )

        if flexion_min_idx <= search_start:
            return search_start

        extension_search = flexion_signal[search_start:flexion_min_idx + 1]
        return search_start + int(np.argmax(extension_search))

    def _refine_sitting_end_with_trunk_flexion(self, trunk_angle,
                                               rough_sitting_end_idx,
                                               pelvis_rise_idx,
                                               seated_trunk_baseline, dt):
        rough_sitting_end_idx = int(max(0, rough_sitting_end_idx))
        pelvis_rise_idx = int(min(len(trunk_angle) - 1, pelvis_rise_idx))
        if pelvis_rise_idx <= rough_sitting_end_idx:
            return rough_sitting_end_idx

        search_start = max(
            0,
            min(rough_sitting_end_idx, pelvis_rise_idx - int(round(0.8 / dt)))
        )
        search_end = pelvis_rise_idx
        trunk_segment = trunk_angle[search_start:search_end + 1]
        if len(trunk_segment) < 3:
            return rough_sitting_end_idx

        trunk_min_idx = search_start + int(np.argmin(trunk_segment))
        trunk_drop = float(seated_trunk_baseline - trunk_angle[trunk_min_idx])
        local_range = float(
            np.percentile(trunk_segment, 95) - np.percentile(trunk_segment, 5))
        min_trunk_drop = max(3.0, 0.20 * local_range)
        if trunk_drop < min_trunk_drop:
            return rough_sitting_end_idx

        onset_threshold = seated_trunk_baseline - max(1.0, 0.20 * trunk_drop)
        onset_idx = trunk_min_idx
        for idx in range(trunk_min_idx, search_start, -1):
            if trunk_angle[idx - 1] >= onset_threshold:
                onset_idx = idx
                break

        return int(min(onset_idx, rough_sitting_end_idx))

    def _filter_valid_pelvis_rise_starts(self, pelvis_rise_starts,
                                         pelvis_vertical, seated_level,
                                         high_pelvis_threshold,
                                         pelvis_range, dt):
        valid_starts = []
        future_samples = max(5, int(round(1.5 / dt)))
        seated_start_threshold = seated_level + 0.35 * pelvis_range
        min_rise_amplitude = 0.45 * pelvis_range

        for rise_idx in pelvis_rise_starts:
            rise_idx = int(rise_idx)
            future_end = min(len(pelvis_vertical) - 1,
                             rise_idx + future_samples)
            if future_end <= rise_idx:
                continue

            start_height = float(pelvis_vertical[rise_idx])
            future_peak = float(np.max(pelvis_vertical[rise_idx:future_end + 1]))
            starts_low = start_height <= seated_start_threshold
            reaches_high = future_peak >= high_pelvis_threshold
            sufficient_rise = (
                future_peak - start_height >= min_rise_amplitude
            )

            if starts_low and reaches_high and sufficient_rise:
                valid_starts.append(rise_idx)

        return valid_starts

    def _find_repeated_sitting_end_from_trunk(self, trunk_angle,
                                             previous_sitting_start_idx,
                                             pelvis_rise_idx):
        search_start = int(max(0, previous_sitting_start_idx + 1))
        search_end = int(min(len(trunk_angle) - 1, pelvis_rise_idx))
        if search_end <= search_start + 2:
            return None

        trunk_segment = trunk_angle[search_start:search_end + 1]
        local_range = float(
            np.percentile(trunk_segment, 95) - np.percentile(trunk_segment, 5))
        trunk_min_idx = search_start + int(np.argmin(trunk_segment))
        if trunk_min_idx <= search_start:
            return None

        pre_flexion_segment = trunk_angle[search_start:trunk_min_idx + 1]
        trunk_peak_idx = search_start + int(np.argmax(pre_flexion_segment))
        trunk_drop = float(trunk_angle[trunk_peak_idx] - trunk_angle[trunk_min_idx])
        min_trunk_drop = max(3.0, 0.20 * local_range)
        if trunk_drop < min_trunk_drop:
            return None

        return int(trunk_peak_idx)

    def _find_preceding_still_run(self, still_runs, movement_idx):
        candidates = [run for run in still_runs if run[1] < movement_idx]
        if not candidates:
            fallback_end = max(0, movement_idx - 1)
            fallback_start = max(0, fallback_end - 5)
            return fallback_start, fallback_end
        return candidates[-1]

    def _find_following_standing_still_start(self, still_runs, pelvis_vertical,
                                             pelvis_rise_idx, high_pelvis_threshold,
                                             window_samples, default_end,
                                             plateau_window_samples,
                                             plateau_search_samples,
                                             plateau_range_threshold,
                                             plateau_slope_threshold,
                                             dt):
        search_end = min(
            len(pelvis_vertical) - 1,
            max(default_end, pelvis_rise_idx + plateau_search_samples)
        )
        local_pelvis = pelvis_vertical[pelvis_rise_idx:search_end + 1]
        if len(local_pelvis) == 0:
            return default_end

        local_peak = float(np.max(local_pelvis))
        rise_amplitude = max(0.0, local_peak - float(pelvis_vertical[pelvis_rise_idx]))
        plateau_tolerance = max(0.01, 0.05 * rise_amplitude)
        plateau_threshold = max(high_pelvis_threshold, local_peak - plateau_tolerance)

        plateau_idx = self._find_standing_pelvis_plateau_start(
            pelvis_vertical,
            start_idx=pelvis_rise_idx,
            end_idx=search_end,
            height_threshold=high_pelvis_threshold,
            window_samples=plateau_window_samples,
            range_threshold=plateau_range_threshold,
            slope_threshold=plateau_slope_threshold,
            dt=dt
        )
        if plateau_idx is not None:
            return int(plateau_idx)

        candidates = [
            run for run in still_runs
            if run[0] > pelvis_rise_idx and
            run[0] <= search_end and
            np.max(pelvis_vertical[run[0]:run[1] + 1]) >= plateau_threshold and
            np.median(pelvis_vertical[run[0]:run[1] + 1]) >= high_pelvis_threshold
        ]
        if not candidates:
            plateau_candidates = np.where(
                pelvis_vertical[pelvis_rise_idx:search_end + 1] >= plateau_threshold
            )[0]
            if len(plateau_candidates) > 0:
                return pelvis_rise_idx + int(plateau_candidates[0])
            return default_end
        return candidates[0][0]

    def _find_standing_pelvis_plateau_start(self, pelvis_vertical, start_idx,
                                            end_idx, height_threshold,
                                            window_samples, range_threshold,
                                            slope_threshold, dt):
        start_idx = int(max(0, start_idx))
        end_idx = int(min(len(pelvis_vertical) - 1, end_idx))
        if end_idx - start_idx + 1 < window_samples:
            return None

        for idx in range(start_idx, end_idx - window_samples + 2):
            window = np.asarray(
                pelvis_vertical[idx:idx + window_samples], dtype=float)
            if np.median(window) < height_threshold:
                continue

            pelvis_range = float(np.max(window) - np.min(window))
            pelvis_slope = float((window[-1] - window[0]) / ((len(window) - 1) * dt))
            if (
                pelvis_range <= range_threshold and
                abs(pelvis_slope) <= slope_threshold
            ):
                return idx

        return None

    def _find_stand_to_sit_events(self, trunk_angle, pelvis_vertical,
                                  pelvis_velocity, standing_still_idx,
                                  search_end_idx, seated_trunk_baseline,
                                  seated_pelvis_baseline,
                                  pelvis_descent_velocity_threshold,
                                  window_samples, standing_exit_window_samples,
                                  min_sustain, pelvis_range, dt):
        search_start = int(min(len(pelvis_vertical) - 1, standing_still_idx + 1))
        search_end = int(min(len(pelvis_vertical) - 1, search_end_idx))
        if search_end <= search_start:
            return None, None

        stand_to_sit_start = self._find_standing_plateau_exit(
            pelvis_vertical,
            standing_still_idx=standing_still_idx,
            search_end_idx=search_end,
            window_samples=standing_exit_window_samples,
            drop_threshold=max(0.01, 0.03 * pelvis_range)
        )
        if stand_to_sit_start is None:
            stand_to_sit_start = self._first_sustained_true(
                (np.arange(len(pelvis_velocity)) >= search_start) &
                (np.arange(len(pelvis_velocity)) <= search_end) &
                (pelvis_velocity <= pelvis_descent_velocity_threshold),
                min_sustain
            )
        if stand_to_sit_start is None:
            return None, None

        pelvis_return_idx = self._find_pelvis_return_to_sitting_height(
            pelvis_vertical,
            start_idx=stand_to_sit_start,
            end_idx=search_end,
            seated_pelvis_baseline=seated_pelvis_baseline,
            pelvis_range=pelvis_range,
            window_samples=window_samples
        )
        if pelvis_return_idx is None:
            return stand_to_sit_start, None

        trunk_return_search_start = min(
            search_end,
            pelvis_return_idx + int(round(0.3 / dt))
        )
        stand_to_sit_end = self._find_stand_to_sit_trunk_return(
            trunk_angle,
            start_idx=trunk_return_search_start,
            end_idx=search_end,
            seated_trunk_baseline=seated_trunk_baseline,
            dt=dt
        )

        return stand_to_sit_start, stand_to_sit_end

    def _find_standing_plateau_exit(self, pelvis_vertical, standing_still_idx,
                                    search_end_idx, window_samples,
                                    drop_threshold):
        standing_still_idx = int(max(0, standing_still_idx))
        search_end_idx = int(min(len(pelvis_vertical) - 1, search_end_idx))
        if search_end_idx <= standing_still_idx + window_samples:
            return None

        plateau_end = min(
            len(pelvis_vertical),
            standing_still_idx + window_samples
        )
        plateau_height = float(np.median(
            pelvis_vertical[standing_still_idx:plateau_end]))
        exit_level = plateau_height - drop_threshold
        search_start = standing_still_idx + 1
        search_end = max(search_start, search_end_idx - window_samples)

        for idx in range(search_start, search_end + 1):
            window = np.asarray(
                pelvis_vertical[idx:idx + window_samples], dtype=float)
            if len(window) < window_samples:
                break
            if np.mean(window <= exit_level) >= 0.6:
                first_below = int(np.where(window <= exit_level)[0][0])
                return idx + first_below

        return None

    def _find_pelvis_return_to_sitting_height(self, pelvis_vertical, start_idx,
                                             end_idx, seated_pelvis_baseline,
                                             pelvis_range, window_samples):
        start_idx = int(max(0, start_idx))
        end_idx = int(min(len(pelvis_vertical) - 1, end_idx))
        if end_idx <= start_idx:
            return None

        sitting_tolerance = max(0.015, 0.08 * pelvis_range)
        sitting_height_threshold = seated_pelvis_baseline + sitting_tolerance
        search_end = max(start_idx, end_idx - window_samples)

        for idx in range(start_idx, search_end + 1):
            window = np.asarray(
                pelvis_vertical[idx:idx + window_samples], dtype=float)
            if len(window) < window_samples:
                break
            if np.mean(window <= sitting_height_threshold) >= 0.6:
                first_below = int(np.where(window <= sitting_height_threshold)[0][0])
                return idx + first_below

        pelvis_segment = pelvis_vertical[start_idx:end_idx + 1]
        closest_idx = start_idx + int(np.argmin(
            np.abs(pelvis_segment - seated_pelvis_baseline)))
        if pelvis_vertical[closest_idx] <= sitting_height_threshold:
            return closest_idx

        return None

    def _find_stand_to_sit_trunk_return(self, trunk_angle, start_idx, end_idx,
                                        seated_trunk_baseline, dt):
        start_idx = int(max(0, start_idx))
        end_idx = int(min(len(trunk_angle) - 1, end_idx))
        if end_idx <= start_idx:
            return None

        plateau_window_samples = max(5, int(round(0.18 / dt)))
        trunk_segment = trunk_angle[start_idx:end_idx + 1]
        trunk_range = float(np.percentile(trunk_segment, 95) -
                            np.percentile(trunk_segment, 5))
        plateau_range_threshold = max(2.0, 0.20 * trunk_range)
        plateau_slope_threshold = 12.0
        baseline_tolerance = max(2.0, 0.15 * trunk_range)
        trunk_return_threshold = seated_trunk_baseline - baseline_tolerance
        trunk_return_peak_idx = start_idx + int(np.argmax(trunk_segment))
        plateau_search_end = max(start_idx, trunk_return_peak_idx)

        for idx in range(start_idx, plateau_search_end + 1):
            window = np.asarray(
                trunk_angle[idx:idx + plateau_window_samples], dtype=float)
            if len(window) < plateau_window_samples:
                break
            if np.median(window) < trunk_return_threshold:
                continue

            window_range = float(np.max(window) - np.min(window))
            window_slope = float(
                (window[-1] - window[0]) / ((len(window) - 1) * dt))
            if (
                window_range <= plateau_range_threshold and
                abs(window_slope) <= plateau_slope_threshold
            ):
                return idx

        return trunk_return_peak_idx

    def _extract_still_runs(self, quiet_metric, window_samples):
        if len(quiet_metric) <= window_samples:
            return [(0, len(quiet_metric) - 1)]

        rolling_metric = np.array([
            np.median(quiet_metric[i:i + window_samples])
            for i in range(len(quiet_metric) - window_samples + 1)
        ])
        threshold = np.quantile(rolling_metric, 0.35)
        quiet_windows = rolling_metric <= threshold
        quiet_window_runs = self._extract_sustained_runs(quiet_windows, 1)

        still_runs = []
        for start_idx, end_idx in quiet_window_runs:
            run_start = start_idx
            run_end = min(len(quiet_metric) - 1, end_idx + window_samples - 1)
            if run_end - run_start + 1 >= window_samples:
                still_runs.append((run_start, run_end))

        return still_runs

    def _extract_sustained_runs(self, mask, min_samples):
        runs = []
        run_start = None

        for idx, is_true in enumerate(np.asarray(mask, dtype=bool)):
            if is_true and run_start is None:
                run_start = idx
            elif not is_true and run_start is not None:
                if idx - run_start >= min_samples:
                    runs.append((run_start, idx - 1))
                run_start = None

        if run_start is not None and len(mask) - run_start >= min_samples:
            runs.append((run_start, len(mask) - 1))

        return runs

    def _first_sustained_true(self, mask, min_samples):
        run = 0
        for idx, is_true in enumerate(np.asarray(mask, dtype=bool)):
            if is_true:
                run += 1
                if run >= min_samples:
                    return idx - min_samples + 1
            else:
                run = 0
        return None

    def _deduplicate_event_starts(self, starts, min_gap):
        deduped = []
        for start in starts:
            if not deduped or start - deduped[-1] >= min_gap:
                deduped.append(start)
        return deduped

    def _smooth_signal(self, signal):
        signal = np.asarray(signal, dtype=float)
        if len(signal) < 5:
            return signal

        window = min(len(signal) - 1 if len(signal) % 2 == 0 else len(signal), 31)
        if window < 5:
            window = 5
        if window % 2 == 0:
            window += 1

        return savgol_filter(signal, window_length=window, polyorder=2)

    def _get_trunk_flexion_signal(self):
        if self.trunk_signal not in self.coordinateValues.columns:
            raise KeyError(f"Could not find trunk signal '{self.trunk_signal}'.")
        return self.coordinateValues[self.trunk_signal].to_numpy(copy=True)

    def _get_pelvis_vertical_signal(self):
        if 'pelvis_ty' in self.coordinateValues.columns:
            return self.coordinateValues['pelvis_ty'].to_numpy(copy=True)
        return self._get_pelvis_reference()[:, 1]

    def _visualize_sit_to_stand_detection(self, time, trunk_angle, pelvis_vertical,
                                          sts_events, high_pelvis_threshold,
                                          pelvis_velocity_threshold):
        del pelvis_velocity_threshold
        plt.close('all')
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        display_events = self._build_sit_stand_display_events(sts_events)

        axes[0].plot(time, trunk_angle, label='trunk angle')
        for display_event in display_events:
            idx = display_event['idx']
            label = display_event['label']
            color = display_event['color']
            axes[0].axvline(time[idx], color=color, linestyle='--', alpha=0.45)
            axes[0].text(time[idx], trunk_angle[idx], label, fontsize=8)
        axes[0].legend()
        title = 'Sit-to-stand event detection'
        if self.trial_label:
            title = f'{title}: {self.trial_label}'
        axes[0].set_title(title)

        axes[1].plot(time, pelvis_vertical, label='pelvis vertical')
        axes[1].axhline(high_pelvis_threshold, color='red', linestyle=':',
                        label='standing pelvis threshold')
        for display_event in display_events:
            idx = display_event['idx']
            label = display_event['label']
            color = display_event['color']
            axes[1].axvline(time[idx], color=color, linestyle='--', alpha=0.45)
            axes[1].text(time[idx], pelvis_vertical[idx], label, fontsize=8)
        axes[1].legend()
        axes[1].set_xlabel('Time (s)')

    def _build_sit_stand_display_events(self, sts_events):
        display_events = []
        event_names = sts_events.get('eventNames', [])

        if event_names == ['SittingEnd', 'StandingStart', 'StandingEnd', 'SittingStart']:
            label_map = {
                'SittingEnd': ('Sitting end', 'tab:green'),
                'StandingStart': ('Standing start', 'tab:blue'),
                'StandingEnd': ('Standing end', 'tab:orange'),
                'SittingStart': ('Sitting start', 'tab:red')
            }
            for rep_idx, event_row in enumerate(sts_events.get('eventIdx', []), start=1):
                for name, idx in zip(event_names, event_row):
                    if name not in label_map:
                        continue
                    label, color = label_map[name]
                    display_events.append({
                        'idx': int(idx),
                        'label': f'R{rep_idx} {label}',
                        'color': color
                    })
            return sorted(display_events, key=lambda event: event['idx'])

        return sorted(display_events, key=lambda event: event['idx'])


def segment_STS(*args, **kwargs):
    raise NotImplementedError(
        'segment_STS has been replaced by sit_to_stand_analysis. '
        'Use RecaptureDisplay.sit_to_stand_results.segment_sit_to_stand instead.'
    )
