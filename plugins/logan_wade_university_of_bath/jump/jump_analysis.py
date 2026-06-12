"""
    ---------------------------------------------------------------------------
    ReCapture processing: jump_analysis.py
    ---------------------------------------------------------------------------

    Marker-based jump segmentation using quiet-standing, take-off, landing,
    and post-landing stabilization events.
--------------------------------------------------------------------------------
"""

import sys
import os

RECAPTURE_PLUGINS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if RECAPTURE_PLUGINS_ROOT not in sys.path:
    sys.path.insert(0, RECAPTURE_PLUGINS_ROOT)

import copy
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt

from recapture_core.utils import kinematics


class jump_analysis(kinematics):

    def __init__(self, session_dir, trial_name, model_path, trcFilePath,
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 trimming_start=0, trimming_end=0,
                 vertical_signal='pelvis_vertical', visualize=False,
                 pause_after_visualization=False, trial_label=None):

        super().__init__(
            session_dir,
            trial_name,
            model_path,
            trcFilePath,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values)

        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
        self.vertical_signal = vertical_signal
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

        self.jumpEvents = self.segment_jump(visualize=visualize)
        self.primarySide = 'r'
        self.defaultSide = self.primarySide

    def get_jump_events(self):
        return copy.deepcopy(self.jumpEvents)

    def get_cycle_events(self, side='both'):
        return self.get_jump_events()

    def get_coordinates_normalized_time(self, side='both'):
        del side

        colNames = self.coordinateValues.columns
        coordValuesNorm = []
        for window_idx in self.jumpEvents['windowIdx']:
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

    def get_jump_count(self):
        return int(self.jumpEvents['windowIdx'].shape[0])

    def compute_jump_duration(self, return_all=False):
        jump_duration = self.jumpEvents['windowTime'][:, 3] - self.jumpEvents['windowTime'][:, 0]
        if return_all:
            return jump_duration, 's'
        return float(np.mean(jump_duration)), 's'

    def compute_time_to_takeoff(self, return_all=False):
        time_to_takeoff = self.jumpEvents['windowTime'][:, 1] - self.jumpEvents['windowTime'][:, 0]
        if return_all:
            return time_to_takeoff, 's'
        return float(np.mean(time_to_takeoff)), 's'

    def compute_flight_time(self, return_all=False):
        flight_time = self.jumpEvents['windowTime'][:, 2] - self.jumpEvents['windowTime'][:, 1]
        if return_all:
            return flight_time, 's'
        return float(np.mean(flight_time)), 's'

    def compute_angle_outputs(self):
        outputAngles = {}
        event_positions = self.jumpEvents['windowIdx']

        for joint_name, joint_config in self._get_angle_definitions().items():
            outputAngles[joint_name] = {
                'Start': {'LR': {}},
                'TakeOff': {'LR': {}},
                'Landing': {'LR': {}},
                'PostLandingStillStart': {'LR': {}},
                f"Peak {joint_config['positive_label']}": {'LR': {}},
                f"Peak {joint_config['negative_label']}": {'LR': {}},
                'ROM': {'LR': {}}
            }

            for side in ['r', 'l']:
                coord_values = self._get_joint_coordinate_series(
                    joint_config, side=side).to_numpy(copy=True)
                start_vals = np.zeros(len(event_positions))
                takeoff_vals = np.zeros(len(event_positions))
                landing_vals = np.zeros(len(event_positions))
                end_vals = np.zeros(len(event_positions))
                peak_vals = np.zeros(len(event_positions))
                trough_vals = np.zeros(len(event_positions))
                rom_vals = np.zeros(len(event_positions))

                for i, (start_idx, takeoff_idx, landing_idx, end_idx) in enumerate(event_positions):
                    window_data = coord_values[start_idx:end_idx + 1]
                    start_vals[i] = coord_values[start_idx]
                    takeoff_vals[i] = coord_values[takeoff_idx]
                    landing_vals[i] = coord_values[landing_idx]
                    end_vals[i] = coord_values[end_idx]
                    peak_vals[i] = np.max(window_data)
                    trough_vals[i] = np.min(window_data)
                    rom_vals[i] = np.ptp(window_data)

                outputAngles[joint_name]['Start']['LR'][side] = start_vals
                outputAngles[joint_name]['TakeOff']['LR'][side] = takeoff_vals
                outputAngles[joint_name]['Landing']['LR'][side] = landing_vals
                outputAngles[joint_name]['PostLandingStillStart']['LR'][side] = end_vals
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
                ts = np.zeros((101, len(self.jumpEvents['windowIdx'])))
                for i, (start_idx, _, _, end_idx) in enumerate(self.jumpEvents['windowIdx']):
                    window_data = coord_values[start_idx:end_idx + 1]
                    ts[:, i] = np.interp(
                        np.linspace(0, 100, 101),
                        np.linspace(0, 100, len(window_data)),
                        window_data
                    )
                tsAngles[joint_name]['LR'][side] = ts

        return tsAngles

    def segment_jump(self, visualize=False):
        """Detect jump events from pelvis and foot vertical marker trajectories."""
        time = self.markerDict['time']
        dt = float(np.median(np.diff(time)))
        if dt <= 0:
            raise ValueError('Marker time vector is not strictly increasing.')

        # Smooth the two signals used for event detection:
        # pelvis vertical motion confirms a real jump, while foot height defines
        # take-off and landing from baseline crossing.
        pelvis_vertical = self._smooth_signal(self._get_vertical_body_signal())
        toe_vertical = self._smooth_signal(self._get_vertical_foot_signal())
        toe_velocity = np.gradient(toe_vertical, time)
        pelvis_velocity = np.gradient(pelvis_vertical, time)

        # Convert detection durations into samples so thresholds scale with the
        # trial sample rate.
        window_seconds = 0.20
        window_samples = max(5, int(np.round(window_seconds / dt)))
        landing_quiet_window_samples = max(4, int(np.round(0.12 / dt)))
        landing_quiet_search_samples = max(
            landing_quiet_window_samples,
            int(np.round(1.5 / dt))
        )
        landing_plateau_window_samples = max(5, int(np.round(0.20 / dt)))
        landing_plateau_search_samples = max(
            landing_plateau_window_samples,
            int(np.round(1.5 / dt))
        )
        takeoff_exit_window_samples = max(5, int(np.round(0.10 / dt)))
        min_sustain = max(4, int(np.round(0.05 / dt)))
        min_flight_samples = max(6, int(np.round(0.08 / dt)))

        # Quiet periods are detected from a combined pelvis/foot velocity
        # metric; foot velocity is down-weighted because it is usually noisier.
        quiet_metric = (
            np.abs(pelvis_velocity) +
            0.5 * np.abs(toe_velocity)
        )
        landing_quiet_metric = (
            np.abs(pelvis_velocity) +
            0.15 * np.abs(toe_velocity)
        )

        # Baselines are taken from the start of the trial, assuming the
        # participant begins standing still before the first jump.
        baseline_slice = slice(0, min(len(time), window_samples))
        pelvis_baseline = float(np.median(pelvis_vertical[baseline_slice]))
        pelvis_noise = float(np.std(pelvis_vertical[baseline_slice]))
        foot_baseline = float(np.median(toe_vertical[baseline_slice]))
        foot_noise = float(np.std(toe_vertical[baseline_slice]))

        # Thresholds are noise-adaptive, with practical minimum values to avoid
        # overfitting to very quiet recordings.
        min_pelvis_rise = max(0.03, 4 * pelvis_noise)
        pelvis_recovery_tolerance = max(0.01, 2 * pelvis_noise)
        downward_threshold = min(-0.03, 0.08 * np.min(pelvis_velocity))
        baseline_crossing_threshold = foot_baseline + max(0.005, 2 * foot_noise)
        airborne_threshold = foot_baseline + max(0.02, 6 * foot_noise)
        airborne_mask = toe_vertical >= airborne_threshold
        landing_threshold = foot_baseline + max(0.01, 3 * foot_noise)

        airborne_runs = self._extract_sustained_runs(airborne_mask, min_flight_samples)
        if len(airborne_runs) == 0:
            raise ValueError('Could not detect any jump take-off events from foot marker height.')

        # Still runs are used to anchor pre-jump standing and post-landing
        # stabilization. A slightly more permissive version is used after
        # landing because participants often settle with small residual motion.
        still_runs = self._extract_still_runs(quiet_metric, window_samples)
        if len(still_runs) == 0:
            still_runs = [(0, min(len(time) - 1, window_samples - 1))]
        relaxed_still_runs = self._extract_still_runs(
            quiet_metric, window_samples, quiet_quantile=0.68)
        if len(relaxed_still_runs) == 0:
            relaxed_still_runs = still_runs

        event_rows = []
        window_rows = []
        movement_start_methods = []
        landing_end_methods = []
        for airborne_start_idx, airborne_end_idx in airborne_runs:
            # The airborne mask is intentionally conservative. These refinements
            # move take-off/landing back to the foot baseline crossing.
            takeoff_idx = self._refine_takeoff_to_baseline_crossing(
                toe_vertical, airborne_start_idx, baseline_crossing_threshold)
            landing_idx = self._refine_landing_to_baseline_crossing(
                toe_vertical, airborne_end_idx, baseline_crossing_threshold)

            # Require the foot to settle back near baseline after landing before
            # accepting the flight candidate.
            sustained_landing_idx = self._first_sustained_condition(
                np.arange(len(time)) >= landing_idx,
                toe_vertical <= landing_threshold,
                min_sustain
            )
            if sustained_landing_idx is None:
                continue

            pelvis_rise = (
                np.max(pelvis_vertical[takeoff_idx:landing_idx + 1]) -
                pelvis_baseline
            )
            if pelvis_rise < min_pelvis_rise:
                continue

            # Start of the analysis window is quiet standing. MovementStart is
            # preferably the point where the pelvis exits that pre-jump plateau
            # downward. The velocity method is retained as fallback for trials
            # where the plateau exit is not clear.
            start_still_idx, start_still_end_idx = self._find_preceding_still_run(
                still_runs, takeoff_idx)
            movement_start_idx = self._find_takeoff_plateau_exit(
                pelvis_vertical,
                start_still_idx,
                start_still_end_idx,
                takeoff_idx,
                drop_threshold=max(0.01, 2 * pelvis_noise),
                window_samples=takeoff_exit_window_samples,
                max_plateau_range=max(0.05, 6 * pelvis_noise),
                max_plateau_slope=0.10,
                exit_slope_threshold=0.00,
                max_search_seconds=1.0,
                dt=dt
            )
            movement_start_method = 'plateau_exit'
            if movement_start_idx is None:
                movement_start_idx = self._first_sustained_condition(
                    (np.arange(len(time)) >= start_still_end_idx) &
                    (np.arange(len(time)) < takeoff_idx),
                    pelvis_velocity <= downward_threshold,
                    min_sustain
                )
                movement_start_method = 'velocity'
            if movement_start_idx is None:
                movement_start_idx = start_still_end_idx
                movement_start_method = 'still_end'
            elif movement_start_idx == start_still_end_idx:
                movement_start_method = 'still_end'

            # Landing end is tied to post-impact pelvis recovery: after landing,
            # pelvis must drop below the MovementStart height, return toward
            # that height, then enter a local low-motion phase. The local quiet
            # check is intentionally more tolerant than the trial-wide still
            # detector because foot signals can remain noisy after landing.
            end_still_idx = self._find_landing_recovery_still_start(
                pelvis_vertical,
                landing_quiet_metric,
                relaxed_still_runs,
                landing_idx,
                movement_start_idx,
                default_end=min(len(time) - 1, landing_idx + window_samples),
                baseline_tolerance=pelvis_recovery_tolerance,
                local_quiet_window_samples=landing_quiet_window_samples,
                local_quiet_search_samples=landing_quiet_search_samples,
                plateau_window_samples=landing_plateau_window_samples,
                plateau_search_samples=landing_plateau_search_samples,
                plateau_range_threshold=max(0.02, 3 * pelvis_noise),
                plateau_slope_threshold=0.05,
                dt=dt
            )

            event_rows.append([movement_start_idx, takeoff_idx, landing_idx, end_still_idx])
            window_rows.append([start_still_idx, takeoff_idx, landing_idx, end_still_idx])
            movement_start_methods.append(movement_start_method)
            landing_end_methods.append(self._last_landing_end_method)

        if len(event_rows) == 0:
            raise ValueError('Detected airborne phases, but could not construct valid jump windows.')

        jump_events = {
            'eventIdx': np.asarray(event_rows, dtype=int),
            'eventTime': time[np.asarray(event_rows, dtype=int)],
            'eventNames': ['MovementStart', 'TakeOff', 'Landing', 'PostLandingStillStart'],
            'windowIdx': np.asarray(window_rows, dtype=int),
            'windowTime': time[np.asarray(window_rows, dtype=int)],
            'windowNames': ['Start', 'TakeOff', 'Landing', 'PostLandingStillStart'],
            'eventDetectionMethods': {
                'MovementStart': movement_start_methods,
                'PostLandingStillStart': landing_end_methods
            }
        }

        if visualize:
            self._visualize_jump_detection(
                pelvis_vertical, toe_vertical, jump_events, foot_baseline, airborne_threshold)
            if self.pause_after_visualization:
                plt.show(block=True)

        return jump_events

    def _first_sustained_condition(self, mask_a, mask_b, min_samples):
        mask = np.asarray(mask_a, dtype=bool) & np.asarray(mask_b, dtype=bool)
        run = 0
        for idx, is_true in enumerate(mask):
            if is_true:
                run += 1
                if run >= min_samples:
                    return idx - min_samples + 1
            else:
                run = 0
        return None

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

    def _refine_takeoff_to_baseline_crossing(self, toe_vertical, airborne_start_idx,
                                             baseline_crossing_threshold):
        idx = int(airborne_start_idx)
        while idx > 0 and toe_vertical[idx - 1] > baseline_crossing_threshold:
            idx -= 1
        return idx

    def _refine_landing_to_baseline_crossing(self, toe_vertical, airborne_end_idx,
                                             baseline_crossing_threshold):
        idx = int(airborne_end_idx)
        while (
            idx < len(toe_vertical) - 1 and
            toe_vertical[idx + 1] > baseline_crossing_threshold
        ):
            idx += 1
        return idx

    def _extract_still_runs(self, quiet_metric, window_samples,
                            quiet_quantile=0.60):
        if len(quiet_metric) <= window_samples:
            return [(0, len(quiet_metric) - 1)]

        rolling_metric = np.array([
            np.median(quiet_metric[i:i + window_samples])
            for i in range(len(quiet_metric) - window_samples + 1)
        ])
        threshold = np.quantile(rolling_metric, quiet_quantile)
        quiet_windows = rolling_metric <= threshold
        quiet_window_runs = self._extract_sustained_runs(quiet_windows, 1)

        still_runs = []
        for start_idx, end_idx in quiet_window_runs:
            run_start = start_idx
            run_end = min(len(quiet_metric) - 1, end_idx + window_samples - 1)
            if run_end - run_start + 1 >= window_samples:
                still_runs.append((run_start, run_end))

        return still_runs

    def _find_preceding_still_run(self, still_runs, takeoff_idx):
        candidates = [run for run in still_runs if run[1] < takeoff_idx]
        if not candidates:
            fallback_end = max(0, takeoff_idx - 1)
            fallback_start = max(0, fallback_end - 5)
            return fallback_start, fallback_end
        return candidates[-1]

    def _find_following_still_start(self, still_runs, landing_idx, default_end):
        candidates = [run for run in still_runs if run[0] > landing_idx]
        if not candidates:
            return default_end
        return candidates[0][0]

    def _find_takeoff_plateau_exit(self, pelvis_vertical, still_start_idx,
                                   still_end_idx, takeoff_idx,
                                   drop_threshold, window_samples,
                                   max_plateau_range, max_plateau_slope,
                                   exit_slope_threshold, max_search_seconds,
                                   dt):
        still_start_idx = int(max(0, still_start_idx))
        still_end_idx = int(min(len(pelvis_vertical) - 1, still_end_idx))
        takeoff_idx = int(min(len(pelvis_vertical) - 1, takeoff_idx))

        if takeoff_idx <= still_end_idx:
            return None

        plateau_window = np.asarray(
            pelvis_vertical[still_start_idx:still_end_idx + 1], dtype=float)
        if len(plateau_window) < 2:
            return None

        plateau_range = float(np.max(plateau_window) - np.min(plateau_window))
        plateau_slope = float(
            (plateau_window[-1] - plateau_window[0]) /
            ((len(plateau_window) - 1) * dt)
        )
        if (
            plateau_range > max_plateau_range or
            abs(plateau_slope) > max_plateau_slope
        ):
            return None

        plateau_height = float(np.median(plateau_window))
        exit_level = plateau_height - drop_threshold
        search_start = max(
            still_end_idx + 1,
            takeoff_idx - int(np.round(max_search_seconds / dt))
        )
        search_end = max(search_start, takeoff_idx - window_samples)

        for idx in range(search_start, search_end + 1):
            window = np.asarray(
                pelvis_vertical[idx:idx + window_samples], dtype=float)
            if len(window) < window_samples:
                break
            exit_slope = float((window[-1] - window[0]) / ((len(window) - 1) * dt))
            if np.mean(window <= exit_level) >= 0.6 and exit_slope <= exit_slope_threshold:
                first_below = int(np.where(window <= exit_level)[0][0])
                return idx + first_below

        return None

    def _find_landing_recovery_still_start(self, pelvis_vertical,
                                           landing_quiet_metric, still_runs,
                                           landing_idx, movement_start_idx,
                                           default_end, baseline_tolerance,
                                           local_quiet_window_samples,
                                           local_quiet_search_samples,
                                           plateau_window_samples,
                                           plateau_search_samples,
                                           plateau_range_threshold,
                                           plateau_slope_threshold,
                                           dt):
        movement_start_height = float(pelvis_vertical[movement_start_idx])
        recovery_level = movement_start_height - baseline_tolerance

        dropped_idx = None
        for idx in range(int(landing_idx), len(pelvis_vertical)):
            if pelvis_vertical[idx] < recovery_level:
                dropped_idx = idx
                break

        if dropped_idx is None:
            recovery_idx = int(landing_idx)
        else:
            recovery_idx = None
            for idx in range(dropped_idx, len(pelvis_vertical)):
                if pelvis_vertical[idx] >= recovery_level:
                    recovery_idx = idx
                    break
            if recovery_idx is None:
                self._last_landing_end_method = 'default_end'
                return int(default_end)

        impact_min_idx = int(landing_idx + np.argmin(
            pelvis_vertical[int(landing_idx):min(
                len(pelvis_vertical),
                int(landing_idx) + plateau_search_samples + 1
            )]
        ))
        plateau_idx = self._find_pelvis_plateau_start(
            pelvis_vertical,
            start_idx=max(recovery_idx, impact_min_idx),
            end_idx=min(
                len(pelvis_vertical) - 1,
                int(landing_idx) + plateau_search_samples
            ),
            window_samples=plateau_window_samples,
            range_threshold=plateau_range_threshold,
            slope_threshold=plateau_slope_threshold,
            dt=dt
        )
        if plateau_idx is not None:
            self._last_landing_end_method = 'plateau'
            return int(plateau_idx)

        local_quiet_idx = self._find_local_quiet_start(
            landing_quiet_metric,
            start_idx=recovery_idx,
            end_idx=min(
                len(landing_quiet_metric) - 1,
                recovery_idx + local_quiet_search_samples
            ),
            window_samples=local_quiet_window_samples,
            quiet_quantile=0.65
        )
        if local_quiet_idx is not None:
            self._last_landing_end_method = 'local_quiet'
            return int(local_quiet_idx)

        candidates = [
            run for run in still_runs
            if (
                run[1] >= recovery_idx and
                run[0] > landing_idx and
                run[0] <= recovery_idx + local_quiet_search_samples
            )
        ]
        if not candidates:
            self._last_landing_end_method = 'recovery'
            return int(recovery_idx)

        first_run = candidates[0]
        self._last_landing_end_method = 'relaxed_still'
        return int(max(first_run[0], recovery_idx))

    def _find_pelvis_plateau_start(self, pelvis_vertical, start_idx, end_idx,
                                   window_samples, range_threshold,
                                   slope_threshold, dt):
        start_idx = int(max(0, start_idx))
        end_idx = int(min(len(pelvis_vertical) - 1, end_idx))
        if end_idx - start_idx + 1 < window_samples:
            return None

        for idx in range(start_idx, end_idx - window_samples + 2):
            window = np.asarray(
                pelvis_vertical[idx:idx + window_samples], dtype=float)
            pelvis_range = float(np.max(window) - np.min(window))
            pelvis_slope = float((window[-1] - window[0]) / ((len(window) - 1) * dt))
            if (
                pelvis_range <= range_threshold and
                abs(pelvis_slope) <= slope_threshold
            ):
                return idx

        return None

    def _find_local_quiet_start(self, quiet_metric, start_idx, end_idx,
                                window_samples, quiet_quantile=0.65):
        start_idx = int(max(0, start_idx))
        end_idx = int(min(len(quiet_metric) - 1, end_idx))
        if end_idx <= start_idx:
            return start_idx

        local_metric = np.asarray(quiet_metric[start_idx:end_idx + 1], dtype=float)
        if len(local_metric) <= window_samples:
            return start_idx

        rolling_metric = np.array([
            np.median(local_metric[i:i + window_samples])
            for i in range(len(local_metric) - window_samples + 1)
        ])
        threshold = np.quantile(rolling_metric, quiet_quantile)
        quiet_windows = rolling_metric <= threshold
        quiet_idx = self._first_sustained_condition(
            np.ones(len(quiet_windows), dtype=bool),
            quiet_windows,
            min_samples=1
        )
        if quiet_idx is None:
            return None
        return start_idx + int(quiet_idx)

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

    def _get_vertical_body_signal(self):
        if self.vertical_signal == 'hip_center_vertical':
            return self._get_hip_center()[:, 1]
        return self._get_pelvis_reference()[:, 1]

    def _get_vertical_foot_signal(self):
        toe_markers = []
        heel_markers = []

        right_toe = self._get_first_available_marker(['r_toe_study', 'R_toe_study', 'RTOE'])
        left_toe = self._get_first_available_marker(['L_toe_study', 'l_toe_study', 'LTOE'])
        right_heel = self._get_first_available_marker(['r_heel_study', 'RHEEL'])
        left_heel = self._get_first_available_marker(['L_heel_study', 'l_heel_study', 'LHEEL'])

        for marker in [right_toe, left_toe]:
            if marker is not None:
                toe_markers.append(marker[:, 1])
        for marker in [right_heel, left_heel]:
            if marker is not None:
                heel_markers.append(marker[:, 1])

        if toe_markers:
            foot_signal = np.mean(np.vstack(toe_markers), axis=0)
        elif heel_markers:
            foot_signal = np.mean(np.vstack(heel_markers), axis=0)
        else:
            ankle_markers = [
                self._get_first_available_marker(['r_ankle_study', 'R_ankle_study', 'RANK']),
                self._get_first_available_marker(['L_ankle_study', 'l_ankle_study', 'LANK'])
            ]
            ankle_vertical = [marker[:, 1] for marker in ankle_markers if marker is not None]
            if not ankle_vertical:
                raise KeyError('Could not find foot markers for jump event detection.')
            foot_signal = np.mean(np.vstack(ankle_vertical), axis=0)

        return foot_signal

    def _visualize_jump_detection(self, pelvis_vertical, toe_vertical, jump_events,
                                  foot_baseline, airborne_threshold):
        plt.close('all')
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        time = self.markerDict['time']
        event_idx = jump_events['eventIdx']

        axes[0].plot(time, pelvis_vertical, label='pelvis vertical')
        for jump_idx, jump_event_row in enumerate(event_idx, start=1):
            for idx, label in zip(jump_event_row, jump_events['eventNames']):
                axes[0].axvline(time[idx], color='k', linestyle='--', alpha=0.35)
                axes[0].text(time[idx], pelvis_vertical[idx], f'J{jump_idx}-{label}', fontsize=7)
        axes[0].legend()
        title = 'Jump event detection'
        if self.trial_label:
            title = f'{title}: {self.trial_label}'
        axes[0].set_title(title)

        axes[1].plot(time, toe_vertical, label='foot vertical')
        axes[1].axhline(foot_baseline, color='gray', linestyle=':', label='standing baseline')
        axes[1].axhline(airborne_threshold, color='red', linestyle=':', label='airborne threshold')
        for jump_idx, jump_event_row in enumerate(event_idx, start=1):
            for idx, label in zip(jump_event_row, jump_events['eventNames']):
                axes[1].axvline(time[idx], color='k', linestyle='--', alpha=0.35)
                axes[1].text(time[idx], toe_vertical[idx], f'J{jump_idx}-{label}', fontsize=7)
        axes[1].legend()
        axes[1].set_xlabel('Time (s)')
