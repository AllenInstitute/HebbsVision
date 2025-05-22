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
from tqdm import tqdm

# Function to process assemblies and return results
def process_assemblies(use_random_ensembles, assembly_timesteps_start, assembly_timesteps_end, activity_raster, ACTIVITY_RASTER, SGC_ASSEMBLIES, nwb_f, tiff_file, percent_threshold, fig_descriptor):
    if use_random_ensembles:
        random_ensembles = pickle.load(open('/home/julian/scan13/random_ensembles.pkl', 'rb'))
        assemblies = random_ensembles
        assembly_coactivity_trace = np.vstack(
            [activity_raster[:, A-1].mean(axis=1) for A in random_ensembles]).T
        titleword = 'Random Ensemble'
    else:
        assemblies = SGC_ASSEMBLIES['assemblies']
        assembly_coactivity_trace = np.vstack(
            [activity_raster[:, A-1].mean(axis=1) for A in assemblies]).T
        titleword = 'Assembly'

    scaling_value = percent_threshold / 100
    assembly_mapping = pd.read_pickle('/home/julian/scan13/map_ordered_to_sgc_output.pickle')

    mean_images = []
    normalized_mean_trigger_stim_variances = []
    frame_storage_list = []

    for assembly_id in tqdm(range(assembly_coactivity_trace.shape[1])):
        
        assembly_number = assembly_mapping[assembly_id+1]
        assembly = assembly_coactivity_trace[:, assembly_id]
        assembly_activity_max = np.max(assembly)
        assembly_high_activity_points, _ = scipy.signal.find_peaks(
            assembly, height=assembly_activity_max*scaling_value)

        presentation = nwb_f['stimulus']['presentation']
        nm_timestamps = np.array(presentation['natural_movie'].get('timestamps'))
        nm_data = np.array(presentation['natural_movie'].get('data'))

        frame_storage = np.zeros((len(assembly_high_activity_points), 304, 608))

        for assembly_active_point_idx, assembly_active_point in enumerate(assembly_high_activity_points):
            point_start_ts = assembly_timesteps_start[assembly_active_point]
            point_end_ts = assembly_timesteps_end[assembly_active_point]

            frames_to_capture = np.where(np.logical_and(nm_timestamps >= point_start_ts, nm_timestamps <= point_end_ts))[0]

            if len(frames_to_capture) > 0:
                tiff_frames = nm_data[frames_to_capture][:, 2].astype(int)
                frame_storage[assembly_active_point_idx, :, :] = tiff_file[tiff_frames[0], :, :]

        frame_storage_list.append(frame_storage)

        mean_image = np.mean(frame_storage, axis=0)
        trigger_stim_variance = np.var(frame_storage, axis=0)
        normalized_variance = trigger_stim_variance / len(assembly_high_activity_points)

        # Save individual images
        plt.figure(figsize=(12, 4))
        sns.heatmap(mean_image, cmap='Greys_r')
        plt.title(f'{titleword} {assembly_number} Mean Trigger Frame ({fig_descriptor})')
        plt.savefig(f'./julian_scan13_pngs/Assembly_{assembly_number}_{fig_descriptor}_mean_trigger_frame.png', dpi=1200)
        plt.close()

        plt.figure(figsize=(12, 4))
        sns.heatmap(normalized_variance, cmap='Greys_r')
        plt.title(f'{titleword} {assembly_number} Normalized Variance ({fig_descriptor})')
        plt.savefig(f'./julian_scan13_pngs/Assembly_{assembly_number}_{fig_descriptor}_normalized_variance.png', dpi=1200)
        plt.close()

        mean_images.append(mean_image)
        normalized_mean_trigger_stim_variances.append(normalized_variance)

    return mean_images, normalized_mean_trigger_stim_variances, frame_storage_list

# Main script


percent_threshold = 30

# Get the activity time trace
ACTIVITY_RASTER = scipy.io.loadmat(
    "/home/julian/scan13/hyperparameter_tuning/esteps_150000_affinity_04_sessionM409828_13_ACTIVITY-RASTER.mat", struct_as_record=True, squeeze_me=True)
SGC_ASSEMBLIES = scipy.io.loadmat(
    "/home/julian/scan13/hyperparameter_tuning/esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)
assembly_timesteps_start = np.load('/home/julian/scan13/scan13_extracted_d/sessionM409828_13_f_ts.npy')[:, 0]
assembly_timesteps_end = np.load('/home/julian/scan13/scan13_extracted_d/sessionM409828_13_f_ts.npy')[:, 2707]
activity_raster = ACTIVITY_RASTER['activity_raster']

nwb_f = h5py.File('/media/berteau/Elements/nwbs/processed/M409828_13_20181213.nwb', 'r')
tiff_file = tf.imread('/media/berteau/Elements/data/stim_movies/stim_movie_long.tif')

# Run with use_random_ensembles = True
mean_random, normalized_random, _ = process_assemblies(
    True, assembly_timesteps_start, assembly_timesteps_end, activity_raster, ACTIVITY_RASTER, SGC_ASSEMBLIES, nwb_f, tiff_file, percent_threshold, "random_ensembles")

# Run with use_random_ensembles = False
mean_real, normalized_real, _ = process_assemblies(
    False, assembly_timesteps_start, assembly_timesteps_end, activity_raster, ACTIVITY_RASTER, SGC_ASSEMBLIES, nwb_f, tiff_file, percent_threshold, "real_assemblies")

# Compute differences and save images
for i, (mean_random, mean_real_assemblies, norm_r, norm_real_r) in enumerate(zip(mean_random, mean_real, normalized_random, normalized_real)):
    diff_mean = mean_real_assemblies - mean_random
    diff_normalized = norm_real_r - norm_r

    # plt.figure(figsize=(12,4))
    # im = sns.heatmap(mean_image, cmap='Greys_r') #, vmin=0, vmax=256)
    # # plt.colorbar(fraction=0.025, pad=0.04)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title(f'Assembly {assembly_number} Mean Triggering Frame', fontsize=20)
    # plt.savefig(f'./julian_scan13_pngs/Assembly_{assembly_number}_{fig_descriptor}_find_peaks_{percent_threshold}_percent_threshold_mean_trigger_frame.png', dpi=1200)
    # plt.close()

# diff_mean = mean_real - mean_random
# diff_normalized = normalized_real - normalized_random
    print(f'\nAssembly {i+1}')
    print(np.max(mean_real_assemblies), np.min(mean_real_assemblies))
    print(np.max(mean_random), np.min(mean_random))
    plt.figure(figsize=(12, 4))
    sns.heatmap(diff_mean, cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Mean Trigger Frame: Assembly {i+1} minus Random Ensemble {i+1}', fontsize=20)
    plt.savefig(f'./Trigger_Frame_Assembly_{i+1}_difference_in_mean_trigger_frame_assembly_minus_random.png', dpi=1200)
    plt.close()

    plt.figure(figsize=(12, 4))
    sns.heatmap(diff_normalized, cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Normalized Variance: Assembly {i+1} minus Random Ensemble {i+1}', fontsize=20)
    plt.savefig(f'./Trigger_Frame_Assembly_{i+1}_difference_normalized_variance_assembly_minus_random.png', dpi=1200)
    plt.close()