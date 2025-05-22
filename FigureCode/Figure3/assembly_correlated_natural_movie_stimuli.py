# from difflib import diff_bytes
import h5py
# import v1dd_physiology.data_fetching as daf
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.stats
import pandas as pd
import csv
import pickle
import math
import tifffile as tf
import seaborn as sns
import scipy.signal
# import cv2
from matplotlib.colors import LinearSegmentedColormap

colors = ['grey', (.4, .6, .8, .5), 'white']
greymap = LinearSegmentedColormap.from_list(
        "Custom", colors, N=80)

# Get the activity time trace
ACTIVITY_RASTER = scipy.io.loadmat(
    "/home/julian/scan13/hyperparameter_tuning/esteps_150000_affinity_04_sessionM409828_13_ACTIVITY-RASTER.mat", struct_as_record=True, squeeze_me=True)
SGC_ASSEMBLIES = scipy.io.loadmat(
    "/home/julian/scan13/hyperparameter_tuning/esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)

# Take the first scan time of any ROI in the frame as the starting time.  And the last as th eend
assembly_timesteps_start = np.load('/home/julian/scan13/scan13_extracted_d/sessionM409828_13_f_ts.npy')[:,0]
assembly_timesteps_end = np.load('/home/julian/scan13/scan13_extracted_d/sessionM409828_13_f_ts.npy')[:,2707]

# print(ACTIVITY_RASTER.keys())

activity_raster = ACTIVITY_RASTER['activity_raster']
activity_raster_peaks = ACTIVITY_RASTER['activity_raster_peaks']

# print(activity_raster_peaks.shape)

coactivity_trace = activity_raster.mean(axis=1)

assemblies = SGC_ASSEMBLIES['assemblies']
# print(assemblies)
assembly_coactivity_trace = np.vstack(
    [activity_raster[:, A-1].mean(axis=1) for A in assemblies]).T

# assembly_peak_times = np.vstack([activity_raster_peaks[:, A-1] for A in assemblies]).T

# Get Experiment Data
nwb_f = h5py.File('/media/berteau/Elements/nwbs/processed/M409828_13_20181213.nwb', 'r')
tiff_file = tf.imread('/media/berteau/Elements/data/stim_movies/stim_movie_long.tif')

mean_trigger_stim_variances = []
normalized_mean_trigger_stim_variances = []
assembly_numbers = []

assembly_mapping = pd.read_pickle('/home/julian/scan13/map_ordered_to_sgc_output.pickle')

for assembly_id in range(assembly_coactivity_trace.shape[1]):
    assembly_number = assembly_mapping[assembly_id+1]
    assembly_numbers.append(assembly_number)
    num_activity_points_in_natural_movies = 0
    assembly = assembly_coactivity_trace[:,assembly_id]
    assembly_activity_max = np.max(assembly)
    assembly_high_activity_points, _ = scipy.signal.find_peaks(assembly, height=assembly_activity_max/5, threshold=assembly_activity_max/20, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    print(f'Assembly {assembly_number}: {len(assembly_high_activity_points)} high activity points')

    # Get nm stimulus times and data
    presentation = nwb_f['stimulus']['presentation']
    nm_timestamps = np.array(
        presentation['natural_movie'].get('timestamps'))
    nm_data = np.array(presentation['natural_movie'].get('data'))
    #nm_data has the start time, end time, and tiff frame.

    # Create frame storage
    frame_storage = np.zeros((len(assembly_high_activity_points), 304, 608))

    for assembly_active_point_idx in range(len(assembly_high_activity_points)):
        assembly_active_point = assembly_high_activity_points[assembly_active_point_idx]
        
        point_start_ts = assembly_timesteps_start[assembly_active_point]
        point_end_ts = assembly_timesteps_end[assembly_active_point]

        frames_to_capture = np.where(np.logical_and(nm_timestamps >= point_start_ts, nm_timestamps <= point_end_ts))[0]

        if len(frames_to_capture) > 0:
            num_activity_points_in_natural_movies += 1
            tiff_frames = nm_data[frames_to_capture][:,2].astype(int)
            frame_storage[assembly_active_point_idx,:,:] = tiff_file[tiff_frames[0], :, :]
            plt.figure(figsize=(12,4))
            im = sns.heatmap(frame_storage[assembly_active_point_idx,:,:], cmap='Greys_r')
            # plt.colorbar(fraction=0.025, pad=0.04)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'Assembly {assembly_number} Trigger Frame {assembly_active_point_idx}', fontsize=24)
            plt.savefig(f'/home/julian/scan13/assembly_NM_trigger_frames/Assembly_{assembly_number}_redux_find_peaks_20_percent_threshold_trigger_frame_{assembly_active_point_idx}.png', dpi=1200)
            plt.close()

    print(f'{num_activity_points_in_natural_movies} high activity points in natural movies')
        

    mean_image = np.mean(frame_storage, 0)
    plt.figure(figsize=(12,4))
    im = sns.heatmap(mean_image, cmap='Greys_r')
    # plt.colorbar(fraction=0.025, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'A {assembly_number} Mean Triggering Movie Frame', fontsize=24)
    plt.savefig(f'/home/julian/scan13/assembly_NM_trigger_frames/Assembly_{assembly_number}_redux_find_peaks_20_percent_threshold_mean_trigger_frame.png', dpi=1200)
    plt.close()

    trigger_stim_variance = np.var(frame_storage, axis=0)
    plt.figure(figsize=(12,4))
    im = sns.heatmap(trigger_stim_variance, vmin=np.min(trigger_stim_variance), vmax=np.max(trigger_stim_variance), cmap='Greys_r')
    # plt.colorbar(fraction=0.025, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Assembly {assembly_number} trigger frame variance', fontsize=24)
    plt.savefig(f'/home/julian/scan13/assembly_NM_trigger_frames/Assembly_{assembly_number}_redux_find_peaks_20_percent_threshold_trigger_frame_variance_per_pixel.png', dpi=1200)
    plt.close()

    normalized_variance = trigger_stim_variance / num_activity_points_in_natural_movies
    plt.figure(figsize=(12,4))
    im = sns.heatmap(normalized_variance, vmin=np.min(normalized_variance), vmax=np.max(normalized_variance), cmap='Greys_r')
    # plt.colorbar(fraction=0.025, pad=0.04)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Normalized (by activation count) Assembly {assembly_number} trigger frame variance', fontsize=24)
    plt.savefig(f'/home/julian/scan13/assembly_NM_trigger_frames/Assembly_{assembly_number}_redux_find_peaks_20_percent_threshold_trigger_frame_normalized_variance_per_pixel.png', dpi=1200)
    plt.close()

    mean_trigger_stim_variances.append(np.mean(trigger_stim_variance))
    normalized_mean_trigger_stim_variances.append(np.mean(trigger_stim_variance) / num_activity_points_in_natural_movies)
    print(f'Assembly {assembly_number} mean trigger frame variance: {np.mean(trigger_stim_variance)}')
    print(f'Normalized by number of activity points: {np.mean(trigger_stim_variance) / num_activity_points_in_natural_movies}')

numbers, mean_values  = zip(*sorted(zip(assembly_numbers, mean_trigger_stim_variances)))

plt.figure()
plt.plot(numbers, mean_values)
plt.xlabel('Assembly ID')
plt.ylabel('Mean Variance')
plt.title(f'Mean Activating Frame Variance by Assembly')
plt.savefig(f'/home/julian/scan13/assembly_NM_trigger_frames/trigger_frame_mean_variance_per_assembly.png', dpi=1200)
plt.close()

numbers, mean_values  = zip(*sorted(zip(assembly_numbers, normalized_mean_trigger_stim_variances)))

plt.figure()
plt.plot(numbers, mean_values)
plt.xlabel('Assembly ID')
plt.ylabel('Normalized Mean Variance')
plt.title(f'Normalized Mean Activating Frame Variance by Assembly')
plt.savefig(f'/home/julian/scan13/assembly_NM_trigger_frames/trigger_frame_mean_variance_per_assembly_normalized.png', dpi=1200)
plt.close()
