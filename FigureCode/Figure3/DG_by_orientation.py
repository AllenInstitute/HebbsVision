from difflib import diff_bytes
import h5py
import v1dd_physiology.data_fetching as daf
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.stats
import pandas as pd
import csv
import pickle
import math
# import cv2


# nwb_f = h5py.File('nwbs/processed/M416296_12_20181210.nwb', 'r')
nwb_1 = h5py.File('behavior_session_869319414.nwb', 'r')
nwb_2 = h5py.File('behavior_session_917500256.nwb')
# nwb_test = h5py.File('behavior_ophys_experiment_1048483611.nwb')
# nwb_f = h5py.File('nwbs/processed/M409828_14_20181214.nwb', 'r')

# sess_id = daf.get_session_id(nwb_f=nwb_f)
# print(sess_id)

# plane_ns = daf.get_plane_names(nwb_f=nwb_f)
# print("Planes: ", plane_ns)

# for plane_n in plane_ns:
#     depth = daf.get_plane_depth(nwb_f=nwb_f, plane_n=plane_n)
#     print(f'depth of {plane_n}: {depth} um')

trial_fluorescence = []
presentation = nwb_f['stimulus']['presentation']
nm_timestamps = np.array(
    presentation['natural_movie'].get('timestamps'))
nm_data = np.array(presentation['natural_movie'].get('data'))
new_clips = np.where(nm_data[:, 2] == 0)
clip_duration = 300  # new_clips[0][1]-1
for repeat_id in range(new_clips[0].shape[0]):
    frames_to_capture = np.where(f_ts >= nm_timestamps[new_clips[0][repeat_id]])[
        0][0:clip_duration]
    trial_fluorescence.append(f[frames_to_capture, roi_n])
trial_fluorescence_np = np.array(trial_fluorescence)
for trial_idx in range(trial_fluorescence_np.shape[0]):
    removed_trial = trial_fluorescence_np[trial_idx]
    remaining_trials = np.delete(
        trial_fluorescence_np, trial_idx, 0)
    r, p = scipy.stats.pearsonr(
        removed_trial, np.mean(remaining_trials, 0))
    movie_oracle_r_values[roi_n, trial_idx] = r

    get_assembly_time_trace(
        trial_fluorescence_np.T, movie_oracle_r_values[roi_n, :], name=f'r_value_over_each_natural_movie_assembly{roi_n}')

    # total_movie_oracle_r_values = np.append(
    #     total_movie_oracle_r_values, movie_oracle_r_values, 0)
# Plot Movie Oracles
mean_over_holdouts = np.mean(movie_oracle_r_values, 1)
fig = plt.figure()
plt.title('Assembly natural movie oracle score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.hist(mean_over_holdouts[:], bins=50)
plt.savefig(
    'oracle_dists/assemblies_esteps_150000__affinity_04_session'+str(13)+'_movies.png')
plt.close()
np.save('oracle_dists/assemblies_esteps_150000_affinity_04_session'+str(13) +
        '_natural_movie_oracle_r_values.npy', movie_oracle_r_values)
np.save('oracle_dists/assemblies_esteps_150000_affinity_04_session'+str(13) +
        '_natural_movie_oracle_scores.npy', mean_over_holdouts)


# Get repeated drifting gratings
presentation = nwb_f['stimulus']['presentation']
dgc_onsets = np.array(
    presentation['drifting_gratings_windowed'].get('timestamps'))
dgc_data = np.array(presentation['drifting_gratings_windowed'].get('data'))
num_samples = presentation['drifting_gratings_windowed'].get('num_samples')
# stims = daf.get_stim_list(nwb_f=nwb_f)
# dgc_onsets = daf.get_dgc_onset_times(nwb_f, dgc_type='windowed')
# presentation = nwb_f['stimulus']['presentation']
# num_samples = np.array(
#     presentation['drifting_gratings_windowed'].get('num_samples'))
duration_sec = 2
grating_number = 0


# Get Tuning Curves and Oracles from Drfiting Gratings
trial_responses_by_assembly_and_orientation = {}
mean_response_by_assembly_and_orientation = {}
oracle_by_assembly_and_orientation = {}
for assembly_n in range(passing_roi_count):
    trial_responses_by_assembly_and_orientation[assembly_n] = {}
    mean_response_by_assembly_and_orientation[assembly_n] = []
    oracle_by_assembly_and_orientation[assembly_n] = []
    for orientation in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 330]:
        trial_responses_by_assembly_and_orientation[assembly_n][orientation] = [
        ]
        trials = np.where(dgc_data[:, 4] == orientation)[0]
        for trial_id in trials:
            mask = (f_ts < dgc_data[trial_id, 1]) * \
                (f_ts >= dgc_data[trial_id, 0])
            frames_to_capture = np.where(mask)[0]
            if frames_to_capture.shape[0] > 10:
                frames_to_capture = frames_to_capture[0:10]
            trial_responses_by_assembly_and_orientation[assembly_n][orientation].append(
                f[frames_to_capture, assembly_n])
        # Now compute oracle
        fluorescence_across_trials_np = np.array(
            trial_responses_by_assembly_and_orientation[assembly_n][orientation])
        r_sum = 0
        for trial_idx in range(len(trials)):
            removed_trial = fluorescence_across_trials_np[trial_idx]
            remaining_trials = np.delete(
                fluorescence_across_trials_np, trial_idx, 0)
            if assembly_n == 3:
                print(removed_trial)
                print(np.mean(remaining_trials, 0))
            r, p = scipy.stats.pearsonr(
                removed_trial, np.mean(remaining_trials, 0))
            r_sum += np.nan_to_num(r)
        oracle_by_assembly_and_orientation[assembly_n].append(
            r_sum / len(trials))
        # Now compute mean response
        mean_response_by_assembly_and_orientation[assembly_n].append(np.mean(
            np.array(trial_responses_by_assembly_and_orientation[assembly_n][orientation])))

degree_orientations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 330]

# Plot Assembly Orientation Mean Responses
rows = passing_roi_count // 4
if passing_roi_count % 4 > 0:
    rows += 1
fig, axes = plt.subplots(4, rows, figsize=(15, 15))
fig.suptitle('Assembly Tuning Curves: DG Mean Coactivity')
for assembly_n in range(passing_roi_count):
    row = assembly_n // 4
    column = assembly_n % 4
    # axes[row, column].set_theta_direction(-1)
    # axes[row, column].set_theta_offset(np.pi / 2.0)
    axes[row, column].plot(
        degree_orientations, mean_response_by_assembly_and_orientation[assembly_n])
    axes[row, column].set_title(f'Assembly {assembly_n}')
plt.savefig('oracle_dists/assemblies_esteps_150000_affinity_04_session' +
            str(13)+'_tuning_curves_DG_windowed_mean_coactivity.png')
plt.close()

# , subplot_kw={'projection': 'polar'}
# Plot Assembly Orientation Oracle Values
rows = passing_roi_count // 4
if passing_roi_count % 4 > 0:
    rows += 1
fig, axes = plt.subplots(4, rows, figsize=(15, 15))
fig.suptitle('Assembly Oracle Score By Orientation: DG Coactivity')
for assembly_n in range(passing_roi_count):
    row = assembly_n // 4
    column = assembly_n % 4
    # axes[row, column].set_theta_direction(-1)
    # axes[row, column].set_theta_offset(np.pi / 2.0)
    axes[row, column].plot(degree_orientations,
                           oracle_by_assembly_and_orientation[assembly_n])
    axes[row, column].set_title(f'Assembly {assembly_n}')
plt.savefig('oracle_dists/assemblies_esteps_150000_affinity_04_session' +
            str(13)+'_oracle_by_orientation_DG_windowed_coactivity.png')
plt.close()


# Get repeated drifting fullscreen gratings
presentation = nwb_f['stimulus']['presentation']
dgc_onsets = np.array(
    presentation['drifting_gratings_full'].get('timestamps'))
dgc_data = np.array(presentation['drifting_gratings_full'].get('data'))
num_samples = presentation['drifting_gratings_full'].get('num_samples')
# stims = daf.get_stim_list(nwb_f=nwb_f)
# dgc_onsets = daf.get_dgc_onset_times(nwb_f, dgc_type='windowed')
# presentation = nwb_f['stimulus']['presentation']
# num_samples = np.array(
#     presentation['drifting_gratings_windowed'].get('num_samples'))
duration_sec = 2
grating_number = 0

# Get Tuning Curves and Oracles from Drfiting Gratings
trial_responses_by_assembly_and_orientation = {}
mean_response_by_assembly_and_orientation = {}
oracle_by_assembly_and_orientation = {}
for assembly_n in range(passing_roi_count):
    trial_responses_by_assembly_and_orientation[assembly_n] = {}
    mean_response_by_assembly_and_orientation[assembly_n] = []
    oracle_by_assembly_and_orientation[assembly_n] = []
    for orientation in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 330]:
        trial_responses_by_assembly_and_orientation[assembly_n][orientation] = [
        ]
        trials = np.where(dgc_data[:, 4] == orientation)[0]
        for trial_id in trials:
            mask = (f_ts < dgc_data[trial_id, 1]) * \
                (f_ts >= dgc_data[trial_id, 0])
            frames_to_capture = np.where(mask)[0]
            if frames_to_capture.shape[0] > 10:
                frames_to_capture = frames_to_capture[0:10]
            trial_responses_by_assembly_and_orientation[assembly_n][orientation].append(
                f[frames_to_capture, assembly_n])
        # Now compute oracle
        fluorescence_across_trials_np = np.array(
            trial_responses_by_assembly_and_orientation[assembly_n][orientation])
        r_sum = 0
        for trial_idx in range(len(trials)):
            removed_trial = fluorescence_across_trials_np[trial_idx]
            remaining_trials = np.delete(
                fluorescence_across_trials_np, trial_idx, 0)
            if assembly_n == 3:
                print(removed_trial)
                print(np.mean(remaining_trials, 0))
            r, p = scipy.stats.pearsonr(
                removed_trial, np.mean(remaining_trials, 0))
            r_sum += np.nan_to_num(r)
        oracle_by_assembly_and_orientation[assembly_n].append(
            r_sum / len(trials))
        # Now compute mean response
        mean_response_by_assembly_and_orientation[assembly_n].append(np.mean(
            np.array(trial_responses_by_assembly_and_orientation[assembly_n][orientation])))

degree_orientations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 330]

# Plot Assembly Orientation Mean Responses
rows = passing_roi_count // 4
if passing_roi_count % 4 > 0:
    rows += 1
fig, axes = plt.subplots(4, rows, figsize=(15, 15))
fig.suptitle('Assembly Tuning Curves: DG Mean Coactivity')
for assembly_n in range(passing_roi_count):
    row = assembly_n // 4
    column = assembly_n % 4
    # axes[row, column].set_theta_direction(-1)
    # axes[row, column].set_theta_offset(np.pi / 2.0)
    axes[row, column].plot(
        degree_orientations, mean_response_by_assembly_and_orientation[assembly_n])
    axes[row, column].set_title(f'Assembly {assembly_n}')
plt.savefig('oracle_dists/assemblies_esteps_150000_affinity_04_session' +
            str(13)+'_tuning_curves_DG_fullscreen_mean_coactivity.png')
plt.close()

# , subplot_kw={'projection': 'polar'}
# Plot Assembly Orientation Oracle Values
rows = passing_roi_count // 4
if passing_roi_count % 4 > 0:
    rows += 1
fig, axes = plt.subplots(4, rows, figsize=(15, 15))
fig.suptitle('Assembly Oracle Score By Orientation: DG Coactivity')
for assembly_n in range(passing_roi_count):
    row = assembly_n // 4
    column = assembly_n % 4
    # axes[row, column].set_theta_direction(-1)
    # axes[row, column].set_theta_offset(np.pi / 2.0)
    axes[row, column].plot(degree_orientations,
                           oracle_by_assembly_and_orientation[assembly_n])
    axes[row, column].set_title(f'Assembly {assembly_n}')
plt.savefig('oracle_dists/assemblies_esteps_150000_affinity_04_session' +
            str(13)+'_oracle_by_orientation_DG_fullscreen_coactivity.png')
plt.close()

print("Done!")