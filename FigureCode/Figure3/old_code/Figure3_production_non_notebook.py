# %% [markdown]
# ## Figure 4 Code to Produce Figures
# 
# This figure will focus on the presentation of Further Analysis of Extracted Assemblies.

# %%
# importing packages
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import pickle
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform, pdist
import scipy.io
from scipy import stats
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from difflib import diff_bytes
import seaborn as sns
import h5py
import scipy
import os
import upsetplot
import ptitprince
import matplotlib.cm as cm
plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (10,10)

import v1dd_physiology.data_fetching as daf
from statannotations.Annotator import Annotator

from matplotlib.colors import LinearSegmentedColormap

paired_blue0 = cm.get_cmap('Paired')(0)
paired_blue1 = cm.get_cmap('Paired')(1)

colors1 = ['grey', (.4, .6, .8, .5)]
colors2 = ['grey', (.2, .3, .5, .5), (.4, .6, .8, .5), 'white']
colors3 = [(.2, .3, .5, .5), 'white']
colors4 = ['grey', (.2, .3, .5, .5), (.4, .6, .8, .5),]
colors5 = [(.2, .3, .5, .5), (.4, .6, .8, .5), 'white']
colors6 = ['grey', 'steelblue', (.4, .6, .8, .5),]
colors7 = ['grey', 'steelblue']
colors8 = ['grey', 'cornflowerblue']
colors9 = ['grey', paired_blue1]
colors10 = ['dimgrey', paired_blue0]
greymap = LinearSegmentedColormap.from_list(
        "Custom", colors10, N=80) 

# %%
# dF_trace = np.load("/media/berteau/Elements/sessionM409828_12_nm_dff.npy")[0,:,:]
dF_trace = np.load("../Data/Session13/sessionM409828_13_dff.npy") # stefan extracted
print(dF_trace.shape)
dF_trace_clipped = dF_trace[16000:16600, 1300:1600].T
plt.figure(figsize=(4,16))
sns.heatmap(dF_trace_clipped, cmap=greymap, cbar=False, yticklabels=False, xticklabels=False)
plt.show()

# %%
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %% [markdown]
# ### Oracle Scores Assemblies

# %%
### Functions relevant for Oracle Score Analysis

def distance(x, y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

# def process(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_canny = cv2.Canny(img_gray, 0, 50)
#     img_dilate = cv2.dilate(img_canny, None, iterations=1)
#     img_erode = cv2.erode(img_dilate, None, iterations=1)
#     return img_erode

def get_assembly_time_trace(coactivity_trace, scores_in_order, name='scored_time_trace'):
    # Set up three subplots
    num_assemblies = coactivity_trace.shape[1]
    fig, ax = plt.subplots(num_assemblies, 1, figsize=(12, 12))

    # plot
    for i in range(num_assemblies):
        ax[i].plot(coactivity_trace[:, i], color='green')
        ax[i].set_ylabel(f"{scores_in_order[i]:.2}")
        ax[i].set_xlabel("Time Steps")
        ax[i].grid()

    fig.suptitle("Assembly Time Trace")
    #plt.savefig(f"oracle_dists2/assemblies_esteps_150000_affinity_04_{name}.png", dpi = 1200)

# %%
ACTIVITY_RASTER = scipy.io.loadmat(
    "../Data/Session13/Assembly_Files/esteps_150000_affinity_04_sessionM409828_13_ACTIVITY-RASTER.mat", struct_as_record=True, squeeze_me=True)
SGC_ASSEMBLIES = scipy.io.loadmat(
    "../Data/Session13/Assembly_Files/esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)

#print(ACTIVITY_RASTER.keys())

activity_raster = ACTIVITY_RASTER['activity_raster']
activity_raster_peaks = ACTIVITY_RASTER['activity_raster_peaks']

coactivity_trace = activity_raster.mean(axis=1)

assemblies = SGC_ASSEMBLIES['assemblies']

# Ensure correct indexing (convert 1-based to 0-based)
assemblies = [np.array(A) - 1 for A in assemblies]
assemblies.sort(key = len)
assemblies.reverse()

# for assembly in assemblies:
#     print(np.max(assembly))
#     print(np.min)

assembly_coactivity_trace = np.vstack(
    [activity_raster[:, A].mean(axis=1) for A in assemblies]).T

scores_in_order = np.load('oracle_dists/assemblies_esteps_150000_affinity_04_session13_natural_movie_oracle_scores.npy')
get_assembly_time_trace(assembly_coactivity_trace, scores_in_order)

# %%
plt.figure()
print(activity_raster.shape)
sns.heatmap(activity_raster[0:60,0:100].T, cmap='PuBu')
plt.title('Cellular Activity Raster')
plt.xlabel('Time')
plt.ylabel('Cell')
plt.show

# %%
nwb_f = h5py.File('M409828_13_20181213.nwb', 'r')
import v1dd_physiology.data_fetching as daf

sess_id = daf.get_session_id(nwb_f=nwb_f)
print(sess_id)

# plane_ns = daf.get_plane_names(nwb_f=nwb_f)
# print("Planes: ", plane_ns)

# for plane_n in plane_ns:
#     depth = daf.get_plane_depth(nwb_f=nwb_f, plane_n=plane_n)
#     print(f'depth of {plane_n}: {depth} um')

fs = []
dffs = []
events = []
locomotions = []
rois = []
pika_rois = []
coords = []

# Uncomment this if you want the actual fluorescence activity trace
# f = np.array(nwb_f['processing']['rois_and_traces_plane0']
#              ['Fluorescence']['f_raw_subtracted'].get('data'))

# If you want DFF, you need to compute it using the allensdk, or load saved traces from /home/berteau/v1DD

f = assembly_coactivity_trace
# print(coactivity_trace)
f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
                ['Fluorescence']['f_raw_subtracted'].get('timestamps'))
# events
clip_ts = f.shape[1]

total_movie_oracle_r_values = np.zeros((0, 9))

passing_roi_count = f.shape[1]

oracle_r_values = np.zeros((passing_roi_count, 25, 8))
movie_oracle_r_values = np.zeros((passing_roi_count, 9))

# %%
plane_ns = daf.get_plane_names(nwb_f=nwb_f)
plane_ns
rois_dict = {}
for plane_n in plane_ns:
    roi_ns = daf.get_roi_ns(nwb_f=nwb_f, plane_n=plane_n)
    print(f'there are {len(roi_ns)} in {plane_n} of session: {sess_id}:')
    print('\nnames of first 100 rois:\n')
    print(roi_ns[0:100])
    rois_dict[plane_n] = roi_ns

# %%
assembly_neurons = set()
for assembly in assemblies:
    assembly_neurons.update(set(assembly))
assembly_neurons = np.array(list(assembly_neurons))
len(assembly_neurons)

# # %%
pika_rois_all_dict = {}
pika_rois_in_assembly_dict = {}
pika_rois_no_assembly_dict = {}
for plane_n, roi_ns in rois_dict.items():
    pika_rois = []
    pika_rois_in_assembly = []
    pika_rois_no_assembly = []
    # print(plane_n)
    for roi_n in roi_ns:
        score = daf.get_pika_classifier_score(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        if score > 0.5:  # Using the threshold from team PIKA, per https://github.com/zhuangjun1981/v1dd_physiology/blob/main/v1dd_physiology/example_notebooks/2022-06-27-data-fetching-basic.ipynb
            pika_rois.append(roi_n)
            if int(roi_n[4:]) in assembly_neurons:
                pika_rois_in_assembly.append(roi_n)
            else:
                pika_rois_no_assembly.append(roi_n)
    pika_rois_all_dict[plane_n] = pika_rois
    pika_rois_in_assembly_dict[plane_n] = pika_rois_in_assembly
    pika_rois_no_assembly_dict[plane_n] = pika_rois_no_assembly

total_rois = np.sum([len(val) for val in pika_rois_all_dict.values()])
neuron_all_movie_oracle_r_values = np.zeros((total_rois, 9))

total_rois = np.sum([len(val) for val in pika_rois_in_assembly_dict.values()])
neuron_in_assembly_movie_oracle_r_values = np.zeros((total_rois, 9))

total_rois = np.sum([len(val) for val in pika_rois_no_assembly_dict.values()])
neuron_no_assembly_movie_oracle_r_values = np.zeros((total_rois, 9))
trial_dff = []
count_n_all = -1
count_n_in_assembly = -1
count_n_no_assembly = -1
for curr_dict, oracle_array, c in zip([pika_rois_all_dict, pika_rois_in_assembly_dict, pika_rois_no_assembly_dict], 
        [neuron_all_movie_oracle_r_values, neuron_in_assembly_movie_oracle_r_values, neuron_no_assembly_movie_oracle_r_values],
        [1,2,3]):
    for plane_n, pika_roi_ns in curr_dict.items():
        for roi_n in pika_roi_ns:
            if c == 1:
                count_n_all += 1
                current_count = count_n_all
            elif c == 2:
                count_n_in_assembly += 1
                current_count = count_n_in_assembly
            elif c == 3:
                count_n_no_assembly += 1
                current_count = count_n_no_assembly
            ### Get Time Trace
            f, f_ts = daf.get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type='dff')
            f_binary_raster = activity_raster[:, int(roi_n[4:])-1]

            # Get Repeated Natural Movies
            trial_fluorescence = []
            # trial_dff = []
            presentation = nwb_f['stimulus']['presentation']
            nm_timestamps = np.array(
                presentation['natural_movie'].get('timestamps'))
            nm_data = np.array(presentation['natural_movie'].get('data'))
            new_clips = np.where(nm_data[:, 2] == 0)[0]
            clip_duration = 300  # new_clips[1]-1
            for repeat_id in range(new_clips.shape[0]):
                frames_to_capture = np.where(f_ts >= nm_timestamps[new_clips[repeat_id]])[
                    0][0:clip_duration]
                trial_fluorescence.append(f_binary_raster[frames_to_capture])
                trial_dff.append(f[frames_to_capture])
            trial_fluorescence_np = np.array(trial_fluorescence)
            for trial_idx in range(trial_fluorescence_np.shape[0]):
                removed_trial = trial_fluorescence_np[trial_idx]
                remaining_trials = np.delete(
                    trial_fluorescence_np, trial_idx, 0)
                r, p = scipy.stats.pearsonr(
                    removed_trial, np.mean(remaining_trials, 0))
                oracle_array[current_count, trial_idx] = r

groups_p_values = ['T-test: {:.3g}'.format(stats.ttest_ind(np.array(neuron_all_movie_oracle_r_values).flatten(), np.array(neuron_in_assembly_movie_oracle_r_values).flatten(), equal_var = False).pvalue, 5),
                    'T-test: {:.3g}'.format(stats.ttest_ind(np.array(neuron_all_movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten(), equal_var = False).pvalue, 5)]

all_arr = [np.array(neuron_all_movie_oracle_r_values).flatten(),
            np.array(neuron_in_assembly_movie_oracle_r_values).flatten(),
            np.array(neuron_no_assembly_movie_oracle_r_values).flatten()]
sns.set_theme(style="whitegrid")
ax = sns.boxplot(data=all_arr)
ax.set_xticklabels(["All Neurons", "Neurons In Assembly", "Neurons Out Assembly"], size = 16)
ax.set_title('Natural Movie Oracle Score', size = 22)
ax.set_ylabel('Oracle Score', size = 16)

# print(groups, p_values)

# %%
# mask = np.mean(np.vstack(trial_dff), axis=0) > 0.5
# plt.figure(figsize=(4*8,12*8))
# # sns.heatmap(np.vstack(trial_dff)[120*1:120*4, 40*5:300], cmap='viridis', cbar=False)
# sns.heatmap(np.vstack(trial_dff)[0:300, mask], cmap=greymap, cbar=False)
# plt.xticks([])
# plt.yticks([])

# %%
movie_oracle_r_values = np.zeros((passing_roi_count, 9))
f = assembly_coactivity_trace
# print(coactivity_trace)
f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
                ['Fluorescence']['f_raw_subtracted'].get('timestamps'))
for roi_n in range(passing_roi_count):

    # Get Repeated Natural Movies
    trial_fluorescence = []
    presentation = nwb_f['stimulus']['presentation']
    nm_timestamps = np.array(
        presentation['natural_movie'].get('timestamps'))
    nm_data = np.array(presentation['natural_movie'].get('data'))
    new_clips = np.where(nm_data[:, 2] == 0)[0]
    clip_duration = 300  # new_clips[1]-1
    for repeat_id in range(new_clips.shape[0]):
        frames_to_capture = np.where(f_ts >= nm_timestamps[new_clips[repeat_id]])[
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

    # get_assembly_time_trace(trial_fluorescence_np.T, movie_oracle_r_values[roi_n, :], name=f'r_value_over_each_natural_movie_assembly{roi_n}')

    # total_movie_oracle_r_values = np.append(
    #     total_movie_oracle_r_values, movie_oracle_r_values, 0)
# Plot Movie Oracles
mean_over_holdouts = np.mean(movie_oracle_r_values, 1)
fig = plt.figure()
plt.title('Assembly natural movie oracle score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.hist(mean_over_holdouts[:], bins=50)
#plt.savefig('./figure4_plots/oracle_dists2/assemblies_esteps_150000__affinity_04_session'+str(13)+'_movies.png')
plt.show()


all_arr = [np.array(movie_oracle_r_values).flatten(),
            np.array(neuron_all_movie_oracle_r_values).flatten(),
            np.array(neuron_in_assembly_movie_oracle_r_values).flatten(),
            np.array(neuron_no_assembly_movie_oracle_r_values).flatten()]

sns.set_theme(style="whitegrid")
ax = sns.boxplot(data=all_arr,
                notch=True, showcaps=True,
                flierprops={"marker": "x"},
                boxprops={"facecolor": (.4, .6, .8, .5)},
                medianprops={"color": "coral"},
            )
ax.set_xticklabels(["Assemblies", "All Neurons", "Neurons In Assembly", "Neurons Out Assembly"], size = 12)
ax.set_title('Natural Movie Oracle Score', size = 20)
ax.set_ylabel('Oracle Score', size = 14)

medians = np.array(
    [np.median(np.array(movie_oracle_r_values).flatten()),
     np.median(np.array(neuron_all_movie_oracle_r_values).flatten()),
     np.median(np.array(neuron_in_assembly_movie_oracle_r_values).flatten()),
     np.median(np.array(neuron_no_assembly_movie_oracle_r_values).flatten())]
)

vertical_offset = medians * 0.2 # offset from median for display
p_values = [np.nan,
            'T-test: {:.3g}'.format(stats.ttest_ind(np.array(movie_oracle_r_values).flatten(), np.array(neuron_all_movie_oracle_r_values).flatten(), equal_var = False).pvalue, 5),
            'T-test: {:.3g}'.format(stats.ttest_ind(np.array(movie_oracle_r_values).flatten(), np.array(neuron_in_assembly_movie_oracle_r_values).flatten(), equal_var = False).pvalue, 5),
            'T-test: {:.3g}'.format(stats.ttest_ind(np.array(movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten(), equal_var = False).pvalue, 5)]

for xtick in ax.get_xticks():
    if xtick != 0:
        ax.text(xtick, medians[xtick] + vertical_offset[xtick], p_values[xtick], 
                horizontalalignment='center', size='small', color='black', weight='semibold')

#plt.savefig('./figure4_plots/oracle_scores_histogram_f_subtracted.png', dpi = 1200)
plt.show()

plt.close()
np.save('oracle_dists2/assemblies_esteps_150000_affinity_04_session'+str(13)+
        '_natural_movie_oracle_r_values.npy', movie_oracle_r_values)
np.save('oracle_dists2/assemblies_esteps_150000_affinity_04_session'+str(13)+
        '_natural_movie_oracle_scores.npy', mean_over_holdouts)

# %%


# # %%
pika_rois_all_dict = {}
pika_rois_in_assembly_dict = {}
pika_rois_no_assembly_dict = {}
no_assembly_neurons = []
# for plane_n, roi_ns in rois_dict.items():
    # pika_rois = []
    # pika_rois_in_assembly = []
    # pika_rois_no_assembly = []
    # # print(plane_n)
    # for roi_n in roi_ns:
    #     score = daf.get_pika_classifier_score(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
    #     if score > 0.5:  # Using the threshold from team PIKA, per https://github.com/zhuangjun1981/v1dd_physiology/blob/main/v1dd_physiology/example_notebooks/2022-06-27-data-fetching-basic.ipynb
    #         pika_rois.append(roi_n)
    #         if int(roi_n[4:]) in assembly_neurons:
    #             pika_rois_in_assembly.append(roi_n)
    #         else:
    #             no_assembly_neurons.append(int(roi_n[4:]))
    #             pika_rois_no_assembly.append(roi_n)
    # pika_rois_all_dict[plane_n] = pika_rois
    # pika_rois_in_assembly_dict[plane_n] = pika_rois_in_assembly
    # pika_rois_no_assembly_dict[plane_n] = pika_rois_no_assembly

# total_rois = np.sum([len(val) for val in pika_rois_all_dict.values()])
# neuron_all_movie_oracle_r_values = np.zeros((total_rois, 9))

# total_rois = np.sum([len(val) for val in pika_rois_in_assembly_dict.values()])
# neuron_in_assembly_movie_oracle_r_values = np.zeros((total_rois, 9))

# total_rois = np.sum([len(val) for val in pika_rois_no_assembly_dict.values()])
# neuron_no_assembly_movie_oracle_r_values = np.zeros((total_rois, 9))

# count_n_all = -1
# count_n_in_assembly = -1
# count_n_no_assembly = -1
# for curr_dict, oracle_array, c in zip([pika_rois_all_dict, pika_rois_in_assembly_dict, pika_rois_no_assembly_dict], 
#         [neuron_all_movie_oracle_r_values, neuron_in_assembly_movie_oracle_r_values, neuron_no_assembly_movie_oracle_r_values],
#         [1,2,3]):
#     for plane_n, pika_roi_ns in curr_dict.items():
#         for roi_n in pika_roi_ns:
#             if c == 1:
#                 count_n_all += 1
#                 current_count = count_n_all
#             elif c == 2:
#                 count_n_in_assembly += 1
#                 current_count = count_n_in_assembly
#             elif c == 3:
#                 count_n_no_assembly += 1
#                 current_count = count_n_no_assembly
#             ### Get Time Trace
#             dff, dff_ts = daf.get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type='dff')
#             f_binary_raster = activity_raster[:, int(roi_n[4:])-1]
            
#             # Get Repeated Natural Movies
#             trial_fluorescence = []
#             presentation = nwb_f['stimulus']['presentation']
#             nm_timestamps = np.array(
#                 presentation['natural_movie'].get('timestamps'))
#             nm_data = np.array(presentation['natural_movie'].get('data'))
#             new_clips = np.where(nm_data[:, 2] == 0)[0]
#             clip_duration = 300  # new_clips[1]-1
#             for repeat_id in range(new_clips.shape[0]):
#                 frames_to_capture = np.where(dff_ts >= nm_timestamps[new_clips[repeat_id]])[
#                     0][0:clip_duration]
#                 trial_fluorescence.append(f_binary_raster[frames_to_capture])
#             trial_fluorescence_np = np.array(trial_fluorescence)
#             for trial_idx in range(trial_fluorescence_np.shape[0]):
#                 removed_trial = trial_fluorescence_np[trial_idx]
#                 remaining_trials = np.delete(
#                     trial_fluorescence_np, trial_idx, 0)
#                 r, p = scipy.stats.pearsonr(
#                     removed_trial, np.mean(remaining_trials, 0))
#                 oracle_array[current_count, trial_idx] = r

# movie_oracle_r_values = np.zeros((passing_roi_count, 9))
# f = assembly_coactivity_trace
# # print(coactivity_trace)
# f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
#                 ['Fluorescence']['f_raw_subtracted'].get('timestamps'))
# for roi_n in range(passing_roi_count):

#     # Get Repeated Natural Movies
#     trial_fluorescence = []
#     presentation = nwb_f['stimulus']['presentation']
#     nm_timestamps = np.array(
#         presentation['natural_movie'].get('timestamps'))
#     nm_data = np.array(presentation['natural_movie'].get('data'))
#     new_clips = np.where(nm_data[:, 2] == 0)[0]
#     clip_duration = 300  # new_clips[1]-1
#     for repeat_id in range(new_clips.shape[0]):
#         frames_to_capture = np.where(f_ts >= nm_timestamps[new_clips[repeat_id]])[
#             0][0:clip_duration]
#         trial_fluorescence.append(f[frames_to_capture, roi_n])
#     trial_fluorescence_np = np.array(trial_fluorescence)
#     for trial_idx in range(trial_fluorescence_np.shape[0]):
#         removed_trial = trial_fluorescence_np[trial_idx]
#         remaining_trials = np.delete(
#             trial_fluorescence_np, trial_idx, 0)
#         r, p = scipy.stats.pearsonr(
#             removed_trial, np.mean(remaining_trials, 0))
#         movie_oracle_r_values[roi_n, trial_idx] = r

#     # get_assembly_time_trace(trial_fluorescence_np.T, movie_oracle_r_values[roi_n, :], name=f'r_value_over_each_natural_movie_assembly{roi_n}')

#     # total_movie_oracle_r_values = np.append(
#     #     total_movie_oracle_r_values, movie_oracle_r_values, 0)
# # Plot Movie Oracles
# mean_over_holdouts = np.mean(movie_oracle_r_values, 1)
# fig = plt.figure()
# plt.title('Assembly natural movie oracle score')
# plt.xlabel('Score')
# plt.ylabel('Frequency')
# plt.hist(mean_over_holdouts[:], bins=50)
# #plt.savefig('./figure4_plots/oracle_dists2/assemblies_esteps_150000__affinity_04_session'+str(13)+'_movies.png')
# plt.show()


# plt.figure(figsize=(14, 10))
# all_arr = [np.array(movie_oracle_r_values).flatten(),
#             np.array(neuron_all_movie_oracle_r_values).flatten(),
#             np.array(neuron_in_assembly_movie_oracle_r_values).flatten(),
#             np.array(neuron_no_assembly_movie_oracle_r_values).flatten()]
# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(data=all_arr,
#                 notch=True, showcaps=True,
#                 flierprops={"marker": "x"},
#                 boxprops={"facecolor": (.4, .6, .8, .5)},
#                 medianprops={"color": "coral"},
#             )
# ax.set_xticklabels(["Assemblies", "All Neurons", "Assembly Cells", "Non-Assembly Cells"], size = 12)
# ax.set_title('Natural Movie Oracle Score', size = 20)
# ax.set_ylabel('Oracle Score', size = 15)

# medians = np.array(
#     [np.median(np.array(movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_all_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_in_assembly_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_no_assembly_movie_oracle_r_values).flatten())]
# )

# vertical_offset = medians * 0.2 # offset from median for display
# p_values = [np.nan,
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(movie_oracle_r_values).flatten(), np.array(neuron_all_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(movie_oracle_r_values).flatten(), np.array(neuron_in_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5)]

# for xtick in ax.get_xticks():
#     if xtick != 0:
#         ax.text(xtick, medians[xtick] + vertical_offset[xtick], p_values[xtick], 
#                 horizontalalignment='center', size='small', color='black', weight='semibold')

# plt.savefig('./figure4_plots/oracle_scores_histogram_dff.png', dpi = 1200)
# plt.show()

# # %%
# plt.figure(figsize=(14, 10))
# all_arr = [np.array(movie_oracle_r_values).flatten(),
#             np.array(neuron_all_movie_oracle_r_values).flatten(),
#             np.array(neuron_in_assembly_movie_oracle_r_values).flatten(),
#             np.array(neuron_no_assembly_movie_oracle_r_values).flatten()]
# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(data=all_arr,
#                 notch=True, showcaps=True,
#                 flierprops={"marker": "x"},
#                 boxprops={"facecolor": (.4, .6, .8, .5)},
#                 medianprops={"color": "coral"},
#             )
# ax.set_xticklabels(["Assemblies", "All Cells", "Assembly Cells", "Non-Assembly Cells"], size = 16)
# ax.set_title('Natural Movie Oracle Score', size = 20)
# ax.set_ylabel('Oracle Score', size = 18)
# plt.yticks(fontsize=16)

# medians = np.array(
#     [np.median(np.array(movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_all_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_in_assembly_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_no_assembly_movie_oracle_r_values).flatten())]
# )

# vertical_offset = medians * 0.2 # offset from median for display
# p_values = [np.nan,
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(movie_oracle_r_values).flatten(), np.array(neuron_all_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(movie_oracle_r_values).flatten(), np.array(neuron_in_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5)]

# for xtick in ax.get_xticks():
#     if xtick != 0:
#         ax.text(xtick, medians[xtick] + vertical_offset[xtick], p_values[xtick], 
#                 horizontalalignment='center', size= 12, color='black', weight='semibold')

# plt.savefig('./figure4_plots/oracle_scores_histogram_dff.png', dpi = 1200)
# plt.show()

# # %%
# # Get repeated drifting fullscreen gratings
# presentation = nwb_f['stimulus']['presentation']
# dgc_onsets = np.array(
#     presentation['drifting_gratings_full'].get('timestamps'))
# dgc_data = np.array(presentation['drifting_gratings_full'].get('data'))
# num_samples = presentation['drifting_gratings_full'].get('num_samples')
# # stims = daf.get_stim_list(nwb_f=nwb_f)
# # dgc_onsets = daf.get_dgc_onset_times(nwb_f, dgc_type='windowed')
# # presentation = nwb_f['stimulus']['presentation']
# # num_samples = np.array(
# #     presentation['drifting_gratings_windowed'].get('num_samples'))
# duration_sec = 2
# grating_number = 0

# # Get Tuning Curves and Oracles from Drfiting Gratings
# trial_responses_by_assembly_and_orientation = {}
# mean_response_by_assembly_and_orientation = {}
# oracle_by_assembly_and_orientation = {}
# for assembly_n in range(passing_roi_count):
#     trial_responses_by_assembly_and_orientation[assembly_n] = {}
#     mean_response_by_assembly_and_orientation[assembly_n] = []
#     oracle_by_assembly_and_orientation[assembly_n] = []
#     for orientation in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 330]:
#         trial_responses_by_assembly_and_orientation[assembly_n][orientation] = []
#         trials = np.where(dgc_data[:,4] == orientation)[0]
#         for trial_id in trials:
#             mask = (f_ts < dgc_data[trial_id,1]) * (f_ts >= dgc_data[trial_id,0])
#             frames_to_capture = np.where(mask)[0]
#             if frames_to_capture.shape[0] > 10:
#                 frames_to_capture = frames_to_capture[0:10]
#             trial_responses_by_assembly_and_orientation[assembly_n][orientation].append(f[frames_to_capture, assembly_n])
#         # Now compute oracle
#         fluorescence_across_trials_np = np.array(trial_responses_by_assembly_and_orientation[assembly_n][orientation])
#         r_sum = 0
#         for trial_idx in range(len(trials)):
#             removed_trial = fluorescence_across_trials_np[trial_idx]
#             remaining_trials = np.delete(
#                 fluorescence_across_trials_np, trial_idx, 0)
#             if assembly_n == 3:
#                 print(removed_trial)
#                 print(np.mean(remaining_trials, 0))
#             r, p = scipy.stats.pearsonr(
#                 removed_trial, np.mean(remaining_trials, 0))
#             r_sum += np.nan_to_num(r)
#         oracle_by_assembly_and_orientation[assembly_n].append(r_sum / len(trials))
#         # Now compute mean response
#         mean_response_by_assembly_and_orientation[assembly_n].append(np.mean(np.array(trial_responses_by_assembly_and_orientation[assembly_n][orientation])))

# degree_orientations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 330]

# # Plot Assembly Orientation Mean Responses
# rows = passing_roi_count // 4
# if passing_roi_count % 4 > 0:
#     rows += 1
# fig, axes = plt.subplots(4, rows, figsize=(15,15))
# fig.suptitle('Assembly Tuning Curves: DG Mean Coactivity')
# for assembly_n in range(passing_roi_count):
#     row = assembly_n // 4
#     column = assembly_n % 4
#     # axes[row, column].set_theta_direction(-1)
#     # axes[row, column].set_theta_offset(np.pi / 2.0)
#     axes[row, column].plot(degree_orientations, mean_response_by_assembly_and_orientation[assembly_n])
#     axes[row, column].set_title(f'Assembly {assembly_n+1}')
# plt.savefig('./figure4_plots/oracle_dists2/assemblies_esteps_150000_affinity_04_session'+str(13)+'_tuning_curves_DG_fullscreen_mean_coactivity.png')
# plt.close()

# # , subplot_kw={'projection': 'polar'}
# # Plot Assembly Orientation Oracle Values
# rows = passing_roi_count // 4
# if passing_roi_count % 4 > 0:
#     rows += 1
# fig, axes = plt.subplots(4, rows, figsize=(15,15))
# fig.suptitle('Assembly Oracle Score By Orientation: DG Coactivity')
# for assembly_n in range(passing_roi_count):
#     row = assembly_n // 4
#     column = assembly_n % 4
#     # axes[row, column].set_theta_direction(-1)
#     # axes[row, column].set_theta_offset(np.pi / 2.0)
#     axes[row, column].plot(degree_orientations, oracle_by_assembly_and_orientation[assembly_n])
#     axes[row, column].set_title(f'Assembly {assembly_n+1}')
# plt.savefig('./figure4_plots/oracle_dists2/assemblies_esteps_150000_affinity_04_session'+str(13)+'_oracle_by_orientation_DG_fullscreen_coactivity.png')
# plt.close()


# print("Done!")

# %% [markdown]
# ### Sparsity Analysis

# %%
def get_assembly_time_trace(coactivity_trace):
    # Set up subplots
    num_assemblies = coactivity_trace.shape[1]
    fig, ax = plt.subplots(num_assemblies, 1, figsize=(12, 12))

    # plot
    for i in range(num_assemblies):
        ax[i].plot(coactivity_trace[:, i], color='green')
        ax[i].set_ylabel("A_{}".format(i+1))
        ax[i].set_xlabel("Time Steps")
        ax[i].grid()

    fig.suptitle("Assembly Time Trace")
    plt.savefig("stefan_time_trace.png")

def gini(x):
    """
    Calculate the Gini coefficient for a NumPy array of values.

    Args:
        x (numpy.ndarray): 1D array of values.

    Returns:
        float: Gini coefficient (ranging from 0 to 1).
    """

    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))

def plot_ginis(coactivity_trace):
    num_assemblies = coactivity_trace.shape[1]
    print("Old version ", coactivity_trace[1].shape)
    print("New Version ", coactivity_trace[:,1].shape)
    gini_values = [gini(coactivity_trace[:,i]) for i in range(num_assemblies)]
    labels = [f'A {i+1}' for i in range(num_assemblies)]

    df = pd.DataFrame({'gini': gini_values, 'labels': labels})

    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x = df.labels,
                     y = df.gini,
                     color= (.4, .6, .8, .5)
                     )
    ax.set_title('Sparsity of Assembly Co-Activity', size = 20)
    ax.set_xticklabels(labels, size = 12)
    ax.set_ylabel('Gini Coefficient', size = 15)
    ax.set_xlabel('Assemblies', size = 15)

    # # Create a base bar plot
    # plt.figure()
    # plt.bar(x=np.arange(len(gini_values)), height=gini_values, tick_label=labels)

    # # Apply Labels:
    # plt.title("Assembly Sparsity")            # Add a title
    # plt.xlabel("Assemblies")                 # Label the x-axis
    # plt.ylabel("Gini Coefficient")                     # Label the y-axis

    # # Remove chartjunk (redundant elements)
    # plt.tick_params(axis='both', which='both', length=0)  # Hide ticks
    plt.gca().spines['top'].set_visible(False)            # Hide top spine
    plt.gca().spines['right'].set_visible(False)          # Hide right spine

    # Add annotations for clarity (optional)
    for i, value in enumerate(gini_values):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')
    plt.savefig('./figure4_plots/sparsity_with_Gini_coefficient_by_assembly.png', dpi = 1200)

# %%
with open('../Figure3/map_ordered_to_sgc_output.pickle', 'rb') as f:
    map_ordered_to_sgc_output = pickle.load(f)
map_ordered_to_sgc_output

# %%
ACTIVITY_RASTER = scipy.io.loadmat(
    "../Data/Session13/Assembly_Files/esteps_150000_affinity_04_sessionM409828_13_ACTIVITY-RASTER.mat", struct_as_record=True, squeeze_me=True)
SGC_ASSEMBLIES = scipy.io.loadmat(
    "../Data/Session13/Assembly_Files/esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)

#print(ACTIVITY_RASTER.keys())

activity_raster = ACTIVITY_RASTER['activity_raster']
activity_raster_peaks = ACTIVITY_RASTER['activity_raster_peaks']

coactivity_trace = activity_raster.mean(axis=1)

assemblies = SGC_ASSEMBLIES['assemblies']
#print(assemblies)
assembly_coactivity_trace = np.vstack(
    [activity_raster[:, A-1].mean(axis=1) for A in assemblies]).T

scores_in_order = np.load('oracle_dists/assemblies_esteps_150000_affinity_04_session13_natural_movie_oracle_scores.npy')

correct_order = []
for i in map_ordered_to_sgc_output.values():
    correct_order.append(i)
correct_order = np.array(correct_order)


ordered_assembly_coactivity_trace = assembly_coactivity_trace[:,correct_order - 1]

# %%
assemblies = SGC_ASSEMBLIES['assemblies']
ordered_assemblies = assemblies[correct_order- 1]
for A in ordered_assemblies:
    print(len(A))

# %%
# new_ordered_assembly_coactivity_trace = np.vstack(
#     [activity_raster[:, A-1].mean(axis=1) for A in ordered_assemblies]).T
# new_ordered_assembly_coactivity_trace.shape

#get_assembly_time_trace(assembly_coactivity_trace)
plot_ginis(ordered_assembly_coactivity_trace)

# %% [markdown]
# ### Setting up Natural Movie Work

# %%
import v1dd_physiology.data_fetching as daf
nwb_f = h5py.File('M409828_13_20181213.nwb', 'r')

# %%
sess_id = daf.get_session_id(nwb_f=nwb_f)
print(sess_id)

plane_ns = daf.get_plane_names(nwb_f=nwb_f)
print("Planes: ", plane_ns)

for plane_n in plane_ns:
    depth = daf.get_plane_depth(nwb_f=nwb_f, plane_n=plane_n)
    print(f'depth of {plane_n}: {depth} um')

# %%
# Get Repeated Natural Movies (11 NATURAL MOVIES SHOWN 9 TIMES)
# trial_fluorescence = []
presentation = nwb_f['stimulus']['presentation']
nm_timestamps = np.array(
    presentation['natural_movie'].get('timestamps'))
nm_data = np.array(presentation['natural_movie'].get('data')) # columns name dataset, gives you the start time, the end time, and the frame number for the natural movie data that was presented
new_clips = np.where(nm_data[:, 2] == 0)[0] # get the index where a new clip rotation begins (frame numbers repeat)
clip_duration = 300  

# new_clips[1]-1
# for repeat_id in range(new_clips.shape[0]):
#     frames_to_capture = np.where(f_ts >= nm_timestamps[new_clips[repeat_id]])[
#         0][0:clip_duration]
#     trial_fluorescence.append(f[frames_to_capture])
#     trial_fluorescence_np = np.array(trial_fluorescence)
#     for trial_idx in range(trial_fluorescence_np.shape[0]):
#         removed_trial = trial_fluorescence_np[trial_idx]
#         remaining_trials = np.delete(
#             trial_fluorescence_np, trial_idx, 0)
#         r, p = scipy.stats.pearsonr(
#             removed_trial, np.mean(remaining_trials, 0))
#         oracle_array[current_count, trial_idx] = r

# %%
f = ordered_assembly_coactivity_trace
passing_roi_count = f.shape[1] # 15

coactivity_during_movie = np.zeros((passing_roi_count, 300))
results = {}
# print(coactivity_trace)
f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
                ['Fluorescence']['f_raw_subtracted'].get('timestamps'))

# Get Repeated Natural Movies
trial_fluorescence = []
presentation = nwb_f['stimulus']['presentation']
nm_timestamps = np.array(
    presentation['natural_movie'].get('timestamps'))
nm_data = np.array(presentation['natural_movie'].get('data'))[:-900] # don't use the last 900 frames, these are the "short" nms which are unrelated
new_clips = np.where(nm_data[:, 2] == 0)[0]
clip_duration = 300  # new_clips[1]-1

clip_ids = []
assembly_coactivity_time_traces = []
start_of_nms = np.where(f_ts > nm_data[0,0])[0][0] # find the frames where the natural movies start to be presented
end_of_nms = np.where(f_ts > nm_data[-1,1])[0][0] # find the frames where the natural movies finish presenting
for time_idx in range(start_of_nms, end_of_nms):
    idx_nm = np.where(nm_data[:,1] > f_ts[time_idx])[0][0] # get the first index
    total_frame_presented = nm_data[idx_nm, 2]
    within_repeats_frame_num = total_frame_presented % 3600
    clip_id = within_repeats_frame_num // 300
    clip_ids.append(clip_id)
    assembly_coactivity_time_traces.append(ordered_assembly_coactivity_trace[time_idx,:])

clip_ids = np.array(clip_ids).reshape(-1,1)
assembly_coactivity_time_traces = np.array(assembly_coactivity_time_traces)

####print(trial_fluorescence_np.shape)
# for trial_idx in range(trial_fluorescence_np.shape[0]):
#     removed_trial = trial_fluorescence_np[trial_idx]
#     remaining_trials = np.delete(
#         trial_fluorescence_np, trial_idx, 0)
#     r, p = scipy.stats.pearsonr(
#         removed_trial, np.mean(remaining_trials, 0))
#     movie_oracle_r_values[roi_n, trial_idx] = r

# %%
clip_ids.shape, assembly_coactivity_time_traces.shape

# %%
nm_dff = np.load("../Data/Session13/sessionM409828_13_nm_dff.npy") # stefan extracted
nm_events = np.load("../Data/Session13/sessionM409828_13_nm_events.npy") # stefan extracted
nm_stimulus = np.load("../Data/Session13/sessionM409828_13_nm_stimulus.npy") 
# nm_stimulus is not time locked to the fluorescence presentation, should not use it for defining clip_ids. 

# %% [markdown]
# ### Decoding of the Natural Movies

# %%
np.unique(clip_ids)

# %%
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# # Initialize lists to store accuracy for each neuron
# accuracies = []

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(assembly_coactivity_time_traces, clip_ids, test_size=0.2, random_state=74)

# # Initialize and train the Lasso Regression model
# alpha = 0.1  # Regularization parameter
# lasso_model = Ridge(alpha=alpha)
# lasso_model.fit(X_train, y_train)

# # Make predictions
# predictions = lasso_model.predict(X_test)

# # Round the predictions to the nearest integer to get the decoded clip IDs
# decoded_clip_ids = predictions.round().astype(int)

# # Print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, decoded_clip_ids))


# # # Create a dataframe to store the results
# # results_df = pd.DataFrame({'Assembly': range(1, num_assemblies + 1), 'Accuracy': accuracies})

# # # Plot the results using Seaborn
# # plt.figure(figsize=(10, 6))
# # sns.barplot(x='Assembly', y='Accuracy', data=results_df, palette='viridis')
# # plt.title('Accuracy of Decoding Natural Movie Clip IDs for Each Assembly')
# # plt.xlabel('Neuron')
# # plt.ylabel('Accuracy')
# # plt.ylim(0, 1)  # Set y-axis limit to [0, 1] for accuracy range
# # plt.show()

# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import confusion_matrix

# # # assuming clip_ids and assembly_coactivity_time_traces are your numpy arrays
# # clip_ids = np.random.randint(0, 12, size=(13087, 1)) 
# # assembly_coactivity_time_traces = np.random.rand(13087, 15)

# # split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(assembly_coactivity_time_traces, clip_ids.ravel(), test_size=0.2, random_state=74)

# # train the model
# clf = RandomForestClassifier(n_estimators=100, random_state=74)
# clf.fit(X_train, y_train)

# # predict the clip_ids
# y_pred = clf.predict(X_test)

# # print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, y_pred))

# # plot the confusion matrix
# # The columns represent the original or expected class distribution, and the 
# # rows represent the predicted or output distribution by the classifier.
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, vmax = 100)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.yticks(rotation=0)
# plt.show()

# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import confusion_matrix

# # split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(assembly_coactivity_time_traces, clip_ids.ravel(), test_size=0.2, random_state=74)

# # define the model
# clf = RandomForestClassifier(random_state=74)

# # define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # create the grid search object
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

# # fit the grid search
# grid_search.fit(X_train, y_train)

# # get the best model
# best_clf = grid_search.best_estimator_

# # predict the clip_ids
# y_pred = best_clf.predict(X_test)

# # print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, y_pred))

# # plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, vmax = 100,cmap=greymap, annot_kws={"size": 35 / np.sqrt(len(cm))})
# plt.title("Assembly Clip ID Classification with Random Forrest Classifier", fontsize=16)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.yticks(rotation=0)
# plt.show()

# %%
# from sklearn.svm import SVC

# # split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(assembly_coactivity_time_traces, clip_ids.ravel(), test_size=0.2, random_state=74)

# #SVMs work best when the data is scaled
# from sklearn.preprocessing import StandardScaler
# # fit the scaler to the training data and transform it, and apply it to test
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # define the model
# clf = SVC(random_state=74)

# # define the parameter grid
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'poly', 'sigmoid']
# }

# # create the grid search object
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

# # fit the grid search
# grid_search.fit(X_train_scaled, y_train)

# # get the best model
# best_clf = grid_search.best_estimator_

# # predict the clip_ids
# y_pred = best_clf.predict(X_test_scaled)

# # print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, y_pred))

# # plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, vmax = 100, annot_kws={"size": 35 / np.sqrt(len(cm))})
# plt.title("Assembly Clip ID Classification with Support Vector Machine", fontsize=16)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.yticks(rotation=0)
# plt.show()

# %%
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler

# # split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(assembly_coactivity_time_traces, clip_ids.ravel(), test_size=0.2, random_state=74)

# # scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # define the model
# clf = MLPClassifier(random_state=74)

# # define the parameter grid: checked on all and relu provided best score (makes results comparable between assembly and null sets as well)
# param_grid = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (50, 50, 100), (100, 100)],
#     'activation': ['relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
#     'learning_rate': ['constant','adaptive', 'invscaling'],
#     'batch_size': [64, 128, 256, 512, 1024],
#     'max_iter': [500]
# }

# # create the grid search object
# grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=1)

# # fit the grid search
# grid_search.fit(X_train_scaled, y_train)

# # get the best model
# best_clf = grid_search.best_estimator_

# # predict the clip_ids
# y_pred = best_clf.predict(X_test_scaled)

# # print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, y_pred))

# # plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, vmax = 100,cmap=greymap, annot_kws={"size": 35 / np.sqrt(len(cm))})
# plt.title("Assembly Clip ID Classification with MLPClassifier", fontsize=16)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.savefig('./figure4_plots/assembly_clip_id_decoder_MLPClassifier.png', dpi = 1200)
# plt.show()

# %%
# print('\n Best estimator:')
# print(grid_search.best_estimator_)
# print('\n Best hyperparameters:')
# print(grid_search.best_params_)

# %%
# #### Develop normalized Heatmap
# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(10,7))
# sns.heatmap(cm_norm, annot=True, vmax = 1, cmap=greymap, fmt=".3f", annot_kws={"size": 35 / np.sqrt(len(cm))})
# plt.title("Assembly Clip ID Classification Percentage with MLPClassifier", fontsize=16)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.yticks(rotation=0)
# plt.savefig('./figure4_plots/assembly_clip_id_percentage_decoder_MLPClassifier.png', dpi = 1200)
# plt.show()

# %%
plt.hist(clip_ids)

# %% [markdown]
# ### Producing Decoder Null Model
# Null Model to Assemblies will be Random 'Ensemble' of Neurons of the same sizes
# 
# Framework:
# 1. Produce 'random_ensembles' with ids similar to the setup of assemblies
# 2. Check the raster plots produced by first algorithmic step of SGC to see if we can use that
# 3. Calculate a co-activity trace of each 'random_ensemble', check the trace to those of assemblies to see differences
# 4. Prepare Decoding Framework by setting the time scale for coactiivity traces to the same as the clip ids
# 5. Run decoding framework in the same way that was produced with assemblies and compare results

# %%


# %%
### STEP 1: Produce 'random_ensembles': Random Collections of Neurons that are same sizes as the assemblies. 

### Set the seed for reproducability, outside of the loop.
random.seed(47)
np.random.seed(47)

### Overlap is fine so we don't have to worry about that.
SGC_ASSEMBLIES = scipy.io.loadmat(
    "../Data/Session13/Assembly_Files/esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)
assemblies = SGC_ASSEMBLIES['assemblies']

dF_trace = np.load("../Data/Session13/sessionM409828_13_CALCIUM-FLUORESCENCE.npy")
num_neurons = dF_trace.shape[1]

random_ensembles = []
for A in assemblies:
    curr_length = len(A)
    # get random ids, make sure there are no repeats in the ids for that specific ensemble
    random_ensembles.append(np.sort(np.array(random.sample(range(num_neurons), curr_length))))

# Order the random ensembles by size
random_ensembles.sort(key = len)
random_ensembles.reverse()

# %%
with open('random_ensembles.pkl', 'wb') as f:
    pickle.dump(random_ensembles, f)
random_ensembles

# %%
print("Assembly Lengths")
assemblies = list(assemblies)
assemblies.sort(key = len)
assemblies.reverse()
print([len(A) for A in assemblies])

print("Random Ensemble Lengths")
print([len(r) for r in random_ensembles])

# %%
### STEP 2: Use the raster plots produced by SGC
### STEP 3: Produce a co-activity trace of each random_ensemble
ACTIVITY_RASTER = scipy.io.loadmat(
    "../Data/Session13/Assembly_Files/esteps_150000_affinity_04_sessionM409828_13_ACTIVITY-RASTER.mat", struct_as_record=True, squeeze_me=True)

activity_raster = ACTIVITY_RASTER['activity_raster']
#activity_raster_peaks = ACTIVITY_RASTER['activity_raster_peaks']

coactivity_trace = activity_raster.mean(axis=1)

# Assembly Coactivity Trace
assembly_coactivity_trace = np.vstack(
    [activity_raster[:, A-1].mean(axis=1) for A in assemblies]).T

# Random Ensemble Coactivity Trace
random_ensembles_coactivity_trace = np.vstack(
    [activity_raster[:, A-1].mean(axis=1) for A in random_ensembles]).T

# %%
def get_ensemble_time_trace(coactivity_trace, title):
    # Set up three subplots
    num_assemblies = coactivity_trace.shape[1]
    fig, ax = plt.subplots(num_assemblies, 1, figsize=(12, 12))

    # plot
    for i in range(num_assemblies):
        ax[i].plot(coactivity_trace[:, i], color=(.4, .6, .8, .5))
        ax[i].set_ylabel("Co-A")
        ax[i].set_xlabel("Time Steps")
        ax[i].grid()

    fig.suptitle(title)
    #plt.savefig(f"oracle_dists2/assemblies_esteps_150000_affinity_04_{name}.png", dpi = 1200)

# %%
### Plot the assembly coactivity traces in order
# scores_in_order = np.load('oracle_dists/assemblies_esteps_150000_affinity_04_session13_natural_movie_oracle_scores.npy')
# get_ensemble_time_trace(assembly_coactivity_trace, title = "Co-Activity Time Trace")

# %%
### Plot the random ensemble coactivity traces in order
scores_in_order = np.load('oracle_dists/assemblies_esteps_150000_affinity_04_session13_natural_movie_oracle_scores.npy')
get_ensemble_time_trace(random_ensembles_coactivity_trace, title = "Random Ensemble Time Trace")

# %%
# Check UpSet Plot as a sanity check that we got a proper random sample
RE = {}
for i, ensemble in enumerate(random_ensembles):
    print(f'RE {i + 1} Size: {len(ensemble)} Neurons')
    RE[f"RE {i + 1}"] = ensemble - 1 # Correct the IDs for Python Indexing
all_sets = upsetplot.from_contents(RE)

ax_dict = upsetplot.UpSet(all_sets, subset_size='count', min_subset_size= 10, 
                            show_counts = True, show_percentages = False,
                            sort_by = 'cardinality', facecolor="grey").plot()
plt.title("Intersection of Random Ensembles in Scan 1.3 of V1DD", size = 24)

# %%
from tqdm import tqdm

### Compare the assembly coactivity traces to the random ensemble coactivity traces
assembly_traces = assembly_coactivity_trace
random_ensemble_traces = random_ensembles_coactivity_trace
# scores_in_order
# def correlate_assembly_traces_vs_random_ensembles(assembly_traces, random_ensemble_traces, scores_in_order, name='compared_time_traces'):
# Set up three subplots
num_assemblies = assembly_traces.shape[1]
# fig, ax = plt.subplots(num_assemblies, 1, figsize=(12, 12))

# plot
# for i in range(num_assemblies):
#     ax[i].plot(assembly_traces[:, i], color='green', label='Assembly')
#     ax[i].plot(random_ensemble_traces[:, i], color='red', label='Random Ensemble')
#     ax[i].set_ylabel(f"{scores_in_order[i]:.2}")
#     ax[i].set_xlabel("Time Steps")
#     ax[i].grid()
#     ax[i].legend()

# fig.suptitle("Assembly Time Trace vs Random Ensemble Time Trace")
# Calculate mean r score between all pairs of assemblies
assembly_r_scores = []
for i in range(num_assemblies):
    for j in range(num_assemblies):
        if i != j:
            assembly_r_scores.append(stats.pearsonr(assembly_traces[:, i], assembly_traces[:, j])[0])
print(f"Mean R Score: {np.mean(assembly_r_scores)}")
# Cacuate mean r score between all pairs of random ensembles
null_r_scores = []
for i in range(num_assemblies):
    for j in range(num_assemblies):
        if i != j:
            null_r_scores.append(stats.pearsonr(random_ensemble_traces[:, i], random_ensemble_traces[:, j])[0])
    # print(stats.pearsonr(random_ensemble_traces[:, i], random_ensemble_traces[:, i])[0])
    # print()
    # null_r_scores.append(stats.pearsonr(random_ensemble_traces[:, i], random_ensemble_traces[:, j])[0])
print(f"Mean R Score Random Ensemble: {np.mean(null_r_scores)}")
#perform a wicoxen ranksum test on the two sets of r scores
print(stats.ranksums(assembly_r_scores, null_r_scores))
print(stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(null_r_scores).flatten(), alternative = 'less').pvalue)


def calculate_pearson_matrix(data):
    # Subtract the mean of each time series
    data_mean_subtracted = data - np.mean(data, axis=1, keepdims=True)
    
    # Compute the covariance matrix
    covariance_matrix = np.dot(data_mean_subtracted, data_mean_subtracted.T)
    
    # Compute the standard deviation for each time series
    std_devs = np.sqrt(np.sum(data_mean_subtracted**2, axis=1))
    
    # Compute the Pearson correlation matrix
    pearson_matrix = covariance_matrix / np.outer(std_devs, std_devs)
    
    return pearson_matrix

a_cell_r_scores = []
no_a_cell_r_scores = []
all_cell_r_scores = []
print("Activity raster shape:", activity_raster.shape)
raster_pearson = calculate_pearson_matrix(activity_raster.T)
for i in tqdm(range(activity_raster.shape[1])):
    for j in range(activity_raster.shape[1]):
        if i != j:
            r = raster_pearson[i, j]
            all_cell_r_scores.append(r)
            if i in assembly_neurons and j in assembly_neurons:
                a_cell_r_scores.append(r)
            elif i not in assembly_neurons and j not in assembly_neurons:
                no_a_cell_r_scores.append(r)

# a_cell_r_scores = []
# no_a_cell_r_scores = []
# all_cell_r_scores = []
# for i in tqdm(range(activity_raster.shape[1])):
#     for j in range(activity_raster.shape[1]):
#         if i != j:
#             r = stats.pearsonr(activity_raster[:, i], activity_raster[:, j])[0]
#             all_cell_r_scores.append(r)
#             if i in assembly_neurons and j in assembly_neurons:
#                 a_cell_r_scores.append(r)
#             elif i not in assembly_neurons and j not in assembly_neurons:
#                 no_a_cell_r_scores.append(r)

#plt

# plt.figure(figsize=(14, 10))

# fig, ax = plt.subplots(figsize=(14, 10))
# # Flatten the arrays
# all_arr = [np.array(assembly_r_scores).flatten(), np.array(null_r_scores).flatten()]

# # Set the theme to whitegrid for simplicity
# sns.set_theme(style="whitegrid")

# # Get the Blues_r color palette and shuffle it
# palette = sns.color_palette("Blues_r", n_colors=len(all_arr))
# random.shuffle(palette)

# # Violin Plot
# ax = sns.violinplot(data=all_arr, palette=palette, saturation=0.5, linewidth=3, native_scale=True, common_norm=True, inner='box', inner_kws=dict(box_width=5, whis_width=2))

# # Remove top and right spines
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# # Set the labels and title
# ax.set_xticklabels(["Assemblies", "Random Ensembles"], size=20, weight='bold')
# ax.set_title('Correlation of Assemblies and Random Ensembles', size=24, weight='bold')
# ax.set_ylabel('Correlation (R)', size=22, weight='bold')
# plt.yticks(fontsize=20, weight='bold')

# # Calculate medians and vertical offset
# medians = np.array([np.median(np.array(assembly_r_scores).flatten()), np.median(np.array(null_r_scores).flatten())])
# vertical_offset = medians * 0.01

np.save('oracle_dists/assembly_r_scores.npy', assembly_r_scores)
np.save('oracle_dists/null_r_scores.npy', null_r_scores)
np.save('oracle_dists/raster_pearson.npy', raster_pearson)

# # Calculate p-value
p_value_assembly_null = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(null_r_scores).flatten(), alternative='two-sided').pvalue
p_value_assembly_all = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(all_cell_r_scores).flatten(), alternative='two-sided').pvalue
p_value_assembly_a = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(a_cell_r_scores).flatten(), alternative='two-sided').pvalue
p_value_assembly_no_a = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(no_a_cell_r_scores).flatten(), alternative='two-sided').pvalue
p_value_a_no_a = stats.ranksums(np.array(a_cell_r_scores).flatten(), np.array(no_a_cell_r_scores).flatten(), alternative='two-sided').pvalue





# %%

import ptitprince as pt
# Create a figure
plt.figure(figsize=(12,8))
sns.set_theme(style="whitegrid")

# Prepare data for raincloud plot
# print(all_arr)
print(len(assembly_r_scores))
print(len(null_r_scores))
data = pd.DataFrame({
    "Values": np.concatenate((np.array(assembly_r_scores), np.array(null_r_scores))),
    "Group": [f"Assembly\n(n={len(assembly_r_scores)})"] * len(assembly_r_scores) + \
                [f"Null Ensembles\n(n={len(null_r_scores)})"] * len(null_r_scores)
})

# Prepare data for raincloud plot
data = pd.DataFrame({
    "Activity Correlations": np.concatenate((np.array(assembly_r_scores), np.array(null_r_scores), np.array(all_cell_r_scores), np.array(a_cell_r_scores), np.array(no_a_cell_r_scores))),
    "Cell Grouping": [f"Assemblies\nn={len(assembly_r_scores)}"] * len(assembly_r_scores) + \
                [f"Random\nEnsembles\nn={len(null_r_scores)}"] * len(null_r_scores) + \
                [f"All\nCells\nn={len(all_cell_r_scores)}"] * len(all_cell_r_scores) + \
                [f"Assembly\nCells\nn={len(a_cell_r_scores)}"] * len(a_cell_r_scores) + \
                [f"Non-Assembly\nCells\nn={len(no_a_cell_r_scores)}"] * len(no_a_cell_r_scores)
})

# x_labels = [f"Assemblies\nn={len(assembly_r_scores)}", f"Random\nEnsembles\nn={len(null_r_scores)}", f"All\nCells\nn={len(all_cell_r_scores)}", f"Assembly\nCells\nn={len(a_cell_r_scores)}", f"Non-Assembly\nCells\nn={len(no_a_cell_r_scores)}"]

print('Raincloud...')
# Create the raincloud plot
ax = pt.RainCloud(
    y="Activity Correlations",
    x="Cell Grouping",
    data=data,
    palette=[(.4, .6, .8, .5), 'grey'],
    width_viol=0.6,  # Adjust violin width
    alpha=0.8,  # Transparency of the cloud
    move=0.1,  # Adjust position of violins
    point_size = 6,
    orient="v",  # Horizontal orientation
    linewidth=2
)

print('P-Values')
# # Calculate p-value
p_value_assembly_null = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(null_r_scores).flatten(), alternative='two-sided').pvalue
p_value_assembly_all = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(all_cell_r_scores).flatten(), alternative='two-sided').pvalue
p_value_assembly_a = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(a_cell_r_scores).flatten(), alternative='two-sided').pvalue
p_value_assembly_no_a = stats.ranksums(np.array(assembly_r_scores).flatten(), np.array(no_a_cell_r_scores).flatten(), alternative='two-sided').pvalue
print('Between A and No A')
p_value_a_no_a = stats.ranksums(np.array(a_cell_r_scores).flatten(), np.array(no_a_cell_r_scores).flatten(), alternative='two-sided').pvalue
print('Between All and A')
p_value_all_v_a = stats.ranksums(np.array(all_cell_r_scores).flatten(), np.array(a_cell_r_scores).flatten(), alternative='two-sided').pvalue

plt.ylim(-0.1, 1.2)  # Set y-axis limit to [0, 1] for accuracy range

# # Add significance stars
for p_index, p_value in enumerate([p_value_assembly_null, p_value_assembly_all, p_value_assembly_a, p_value_assembly_no_a, p_value_all_v_a, p_value_a_no_a]):
    if p_value < 0.001:
        significance = 'P < 0.001\n***'
    elif p_value < 0.01:
        significance = 'P < 0.01\n**'
    elif p_value < 0.05:
        significance = 'P < 0.05\n*'
    else:
        significance = 'ns'

    # Add p-value annotation
    if p_index == 0:
        p_offset = 0.5
        p_height = 1.05
    elif p_index > 0 and p_index < 4:
        p_offset = 1.0 + p_index
        p_height = 1.05
    elif p_index == 4:
        p_offset = 2.5
        p_height = 0.5
    elif p_index == 5:
        p_offset = 3.5
        p_height = 0.5

    ax.text(p_offset, p_height, f'{significance}', 
                color='black', ha='center', va='bottom', fontsize=16, weight='bold')

# ax.annotate(f'P-Val: {p_value:.3g}', xy=(1, medians[1] + vertical_offset[1]), xytext=(1, medians[1] + vertical_offset[1] + 0.05),
#             arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='center', size=18, color='black', weight='semibold')

# BREADCRUMB: RESTORE BELOW TO END OF CELL
# print('Finishing off figure...')
# # Add a multiline title to include the p-value, add y_label
# title = f'Correlation'
# plt.title(title, size=24)
# plt.xlabel('Correlation (r)', size=20)
# plt.xticks(fontsize=20)  # Adjust size of xticks
# plt.yticks(fontsize=20)  # Adjust size of yticks
# plt.ylabel("Groups", size=20)

# plt.tight_layout()
# plt.savefig('./figure4_plots/correlations_assemblies_vs_random_ensembles_raincould_plot.png', dpi=1200)
# plt.savefig('./figure4_plots/correlations_assemblies_vs_random_ensembles_raincould_plot.pdf', dpi=1200)
# plt.savefig('./figure4_plots/correlations_assemblies_vs_random_ensembles_raincould_plot.svg', dpi=1200)
# plt.close()

# Early Exit Breakoff Breadcrumb
# exit()

# %%
# Compare correlation of asesmblies and random ensembles
# correlate_assembly_traces_vs_random_ensembles(assembly_coactivity_trace, random_ensembles_coactivity_trace, scores_in_order, name='assemblies_vs_ensembles')

# %%
### Step 4: Prepare Decoding Framework in the same way as Assemblies
print('Decoding...')
nwb_f = h5py.File('M409828_13_20181213.nwb', 'r')

# Get Repeated Natural Movies (12 NATURAL MOVIES SHOWN 9 TIMES)
# trial_fluorescence = []
presentation = nwb_f['stimulus']['presentation']
nm_timestamps = np.array(
    presentation['natural_movie'].get('timestamps'))
nm_data = np.array(presentation['natural_movie'].get('data')) # columns name dataset, gives you the start time, the end time, and the frame number for the natural movie data that was presented
new_clips = np.where(nm_data[:, 2] == 0)[0] # get the index where a new clip rotation begins (frame numbers repeat)
clip_duration = 300  

f = random_ensembles_coactivity_trace
passing_roi_count = f.shape[1] # 15

coactivity_during_movie = np.zeros((passing_roi_count, 300))
results = {}
# print(coactivity_trace)
f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
                ['Fluorescence']['f_raw_subtracted'].get('timestamps'))

# Get Repeated Natural Movies
trial_fluorescence = []
presentation = nwb_f['stimulus']['presentation']
nm_timestamps = np.array(
    presentation['natural_movie'].get('timestamps'))
nm_data = np.array(presentation['natural_movie'].get('data'))[:-900] # don't use the last 900 frames, these are the "short" nms which are unrelated
new_clips = np.where(nm_data[:, 2] == 0)[0]
clip_duration = 300  # new_clips[1]-1

clip_ids = []
random_ensemble_coactivity_time_traces = []
start_of_nms = np.where(f_ts > nm_data[0,0])[0][0] # find the frames where the natural movies start to be presented
end_of_nms = np.where(f_ts > nm_data[-1,1])[0][0] # find the frames where the natural movies finish presenting
for time_idx in tqdm(range(start_of_nms, end_of_nms)):
    idx_nm = np.where(nm_data[:,1] > f_ts[time_idx])[0][0] # get the first index
    total_frame_presented = nm_data[idx_nm, 2]
    within_repeats_frame_num = total_frame_presented % 3600
    clip_id = within_repeats_frame_num // 300
    clip_ids.append(clip_id)
    random_ensemble_coactivity_time_traces.append(random_ensembles_coactivity_trace[time_idx,:])

clip_ids = np.array(clip_ids).reshape(-1,1)
random_ensemble_coactivity_time_traces = np.array(random_ensemble_coactivity_time_traces)

print(clip_ids.shape, random_ensemble_coactivity_time_traces.shape)

# %%
# ### Step 5: Run Decoding Metric with Random Ensembles
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score


# # Initialize lists to store accuracy for each neuron
# accuracies = []

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(random_ensemble_coactivity_time_traces, clip_ids, test_size=0.2, random_state=74)

# # Initialize and train the Lasso Regression model
# alpha = 0.1  # Regularization parameter
# lasso_model = Ridge(alpha=alpha)
# lasso_model.fit(X_train, y_train)

# # Make predictions
# predictions = lasso_model.predict(X_test)

# # Round the predictions to the nearest integer to get the decoded clip IDs
# decoded_clip_ids = predictions.round().astype(int)

# # Print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, decoded_clip_ids))


# # # Create a dataframe to store the results
# # results_df = pd.DataFrame({'Assembly': range(1, num_assemblies + 1), 'Accuracy': accuracies})

# # # Plot the results using Seaborn
# # plt.figure(figsize=(10, 6))
# # sns.barplot(x='Assembly', y='Accuracy', data=results_df, palette='viridis')
# # plt.title('Accuracy of Decoding Natural Movie Clip IDs for Each Assembly')
# # plt.xlabel('Neuron')
# # plt.ylabel('Accuracy')
# # plt.ylim(0, 1)  # Set y-axis limit to [0, 1] for accuracy range
# # plt.show()

# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import confusion_matrix

# # split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(random_ensemble_coactivity_time_traces, clip_ids.ravel(), test_size=0.2, random_state=74)

# # define the model
# clf = RandomForestClassifier(random_state=74)

# # define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # create the grid search object
# grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)

# # fit the grid search
# grid_search.fit(X_train, y_train)

# # get the best model
# best_clf = grid_search.best_estimator_

# # predict the clip_ids
# y_pred = best_clf.predict(X_test)

# # print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, y_pred))

# # plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, vmax = 100,cmap=greymap, annot_kws={"size": 35 / np.sqrt(len(cm))})
# plt.title("Random Ensemble Clip ID Classification with Random Forrest Classifier", fontsize=16)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.yticks(rotation=0)
# plt.show()

# %%
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import StandardScaler

# # split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(random_ensemble_coactivity_time_traces, clip_ids.ravel(), test_size=0.2, random_state=74)

# # scale the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # define the model
# clf = MLPClassifier(random_state=74)

# # define the parameter grid: checked on all and relu provided best score (makes results comparable between assembly and null sets as well)
# param_grid = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (50, 50, 100), (100, 100)],
#     'activation': ['relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
#     'learning_rate': ['constant','adaptive', 'invscaling'],
#     'batch_size': [64, 128, 256, 512, 1024],
#     'max_iter': [500]
# }

# # create the grid search object
# grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=1)

# # fit the grid search
# grid_search.fit(X_train_scaled, y_train)

# # get the best model
# best_clf = grid_search.best_estimator_

# # predict the clip_ids
# y_pred = best_clf.predict(X_test_scaled)

# # print out the accuracy
# print("Accuracy Score of", accuracy_score(y_test, y_pred))

# # plot the confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,7))
# sns.heatmap(cm, annot=True, vmax = 100,cmap=greymap, annot_kws={"size": 35 / np.sqrt(len(cm))})
# plt.title("Random Ensemble Clip ID Classification with MLPClassifier", fontsize=16)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.yticks(rotation=0)
# plt.savefig('./figure4_plots/random_ensemble_clip_id_decoder_MLPClassifier.png', dpi = 1200)
# plt.show()

# %%
# print('\n Best estimator:')
# print(grid_search.best_estimator_)
# print('\n Best hyperparameters:')
# print(grid_search.best_params_)

# %%


# %%
# #### Develop normalized Heatmap
# cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(10,7))
# sns.heatmap(cm_norm, annot=True, cmap=greymap, vmax = 1, fmt=".3f", annot_kws={"size": 35 / np.sqrt(len(cm))})
# plt.title("Random Ensemble Clip ID Classification Percentage with MLPClassifier", fontsize=16)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.yticks(rotation=0)
# plt.savefig('./figure4_plots/random_ensemble_clip_id_percentage_decoder_MLPClassifier.png', dpi = 1200)
# plt.show()

# %% [markdown]
# ### Reproduce with Balanced Clip IDs

# %%
clip_ids.shape, assembly_coactivity_time_traces.shape

# %%
# Get unique clip_ids and their counts
unique_clip_ids, counts = np.unique(clip_ids, return_counts=True)

# Determine the minimum count of any clip_id
min_count = np.min(counts)

# Create new lists for balanced clip_ids and corresponding assembly_coactivations
balanced_clip_ids = []
balanced_assembly_coactivations = []

# Sample min_count indices for each clip_id
for clip_id in unique_clip_ids:
    # Get indices of the current clip_id
    indices = np.where(clip_ids == clip_id)[0]
    # Randomly sample min_count indices
    np.random.seed(747)
    sampled_indices = np.random.choice(indices, min_count, replace=False)
    # Append the sampled indices' values to the new lists
    balanced_clip_ids.extend(clip_ids[sampled_indices])
    balanced_assembly_coactivations.extend(assembly_coactivity_time_traces[sampled_indices])

# Convert lists to numpy arrays
balanced_clip_ids = np.array(balanced_clip_ids)
balanced_assembly_coactivations = np.array(balanced_assembly_coactivations)

# Shuffle to ensure random distribution
shuffled_indices = np.random.default_rng(seed=747).permutation(len(balanced_clip_ids))
balanced_clip_ids = balanced_clip_ids[shuffled_indices]
balanced_assembly_coactivations = balanced_assembly_coactivations[shuffled_indices]

# Output the balanced arrays
print("Balanced clip_ids:", balanced_clip_ids)
print("Balanced assembly_coactivations:", balanced_assembly_coactivations)

# %%
balanced_clip_ids.shape, balanced_assembly_coactivations.shape

# %%
plt.hist(balanced_clip_ids)

# %%
# # Stefan Breadcrumb: Run from here

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(balanced_assembly_coactivations, balanced_clip_ids.ravel(), test_size=0.2, random_state=747)

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (50, 50, 100), (100, 100)],
    'activation': ['relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive', 'invscaling'],
    'batch_size': [64, 128, 256, 512, 1024],
    'max_iter': [300]
}

from sklearn.model_selection import RandomizedSearchCV


clf = MLPClassifier(random_state=747, early_stopping=True, n_iter_no_change=10)

# grid_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1, verbose=1)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)
best_clf = grid_search.best_estimator_

# predict the clip_ids
y_pred = best_clf.predict(X_test_scaled)

# print out the accuracy
print("Accuracy Score of", accuracy_score(y_test, y_pred))

# plot the confusion matrix
assembly_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(assembly_cm, annot=True, vmax = 100, cmap=greymap, annot_kws={"size": 55 / np.sqrt(len(assembly_cm)), "color": 'white'})
plt.title("Assembly Clip ID Classification Count", fontsize=22)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Truth', fontsize=20)
plt.yticks(rotation=0, fontsize=18) 
plt.xticks(fontsize=18)
plt.savefig('./figure4_plots/assembly_balanced_clip_id_decoder_MLPClassifier.png', dpi = 1200)
plt.savefig('./figure4_plots/assembly_balanced_clip_id_decoder_MLPClassifier.pdf')
plt.show()

# %%
# print('\n Best estimator:')
# print(grid_search.best_estimator_)
# print('\n Best hyperparameters:')
# print(grid_search.best_params_)

# %%
#### Develop normalized Heatmap
assembly_cm_norm = np.round((assembly_cm.astype('float') / assembly_cm.sum(axis=1)[:, np.newaxis])*100)
plt.figure(figsize=(12,8))
sns.heatmap(assembly_cm_norm, annot=True, vmax = 100, cmap=greymap, annot_kws={"size": 55 / np.sqrt(len(assembly_cm)), "color": 'white'})
ax.figure.axes[-1].tick_params(labelsize=18)
plt.title("Assembly Clip ID Classification Percent", fontsize=24)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Truth', fontsize=20)
plt.yticks(rotation=0, fontsize=18)
plt.xticks(fontsize=18) 
plt.savefig('./figure4_plots/assembly_balanced_clip_id_percentage_decoder_MLPClassifier.png', dpi = 1200)
plt.savefig('./figure4_plots/assembly_balanced_clip_id_percentage_decoder_MLPClassifier.pdf')
plt.show()

# %%
print(clip_ids.shape, random_ensemble_coactivity_time_traces.shape)

# Get unique clip_ids and their counts
unique_clip_ids, counts = np.unique(clip_ids, return_counts=True)

# Determine the minimum count of any clip_id
min_count = np.min(counts)

# Create new lists for balanced clip_ids and corresponding assembly_coactivations
balanced_clip_ids = []
balanced_random_ensemble_coactivations = []

# Sample min_count indices for each clip_id
for clip_id in unique_clip_ids:
    # Get indices of the current clip_id
    indices = np.where(clip_ids == clip_id)[0]
    # Randomly sample min_count indices
    np.random.seed(747)
    sampled_indices = np.random.choice(indices, min_count, replace=False)
    # Append the sampled indices' values to the new lists
    balanced_clip_ids.extend(clip_ids[sampled_indices])
    balanced_random_ensemble_coactivations.extend(random_ensemble_coactivity_time_traces[sampled_indices])

# Convert lists to numpy arrays
balanced_clip_ids = np.array(balanced_clip_ids)
balanced_random_ensemble_coactivations = np.array(balanced_random_ensemble_coactivations)

# Shuffle to ensure random distribution
shuffled_indices = np.random.default_rng(seed=747).permutation(len(balanced_clip_ids))
balanced_clip_ids = balanced_clip_ids[shuffled_indices]
balanced_random_ensemble_coactivations = balanced_random_ensemble_coactivations[shuffled_indices]

# Output the balanced arrays
print("Balanced clip_ids:", balanced_clip_ids)
print("Balanced random_ensemble_coactivations:", balanced_random_ensemble_coactivations)

# %%
balanced_clip_ids.shape, balanced_random_ensemble_coactivations.shape

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(balanced_random_ensemble_coactivations, balanced_clip_ids.ravel(), test_size=0.2, random_state=747)

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define the model
clf = MLPClassifier(random_state=747)

param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (50, 50, 100), (100, 100)],
    'activation': ['relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive', 'invscaling'],
    'batch_size': [64, 128, 256, 512, 1024],
    'max_iter': [300]
}


clf = MLPClassifier(random_state=747, early_stopping=True, n_iter_no_change=10)

# grid_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=300, cv=5, n_jobs=-1, verbose=1)
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)
best_clf = grid_search.best_estimator_

# predict the clip_ids
y_pred = best_clf.predict(X_test_scaled)

# print out the accuracy
print("Accuracy Score of", accuracy_score(y_test, y_pred))

# plot the confusion matrix
rand_ensemble_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(rand_ensemble_cm, annot=True, vmax = 100, cmap=greymap, annot_kws={"size": 55 / np.sqrt(len(rand_ensemble_cm)), "color": 'white'})
plt.title("Random Ensemble Clip ID Classification Count", fontsize=20)
ax.figure.axes[-1].tick_params(labelsize=18)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Truth', fontsize=20)
plt.yticks(rotation=0, fontsize=18)
plt.xticks(fontsize=18)
plt.savefig('./figure4_plots/random_ensemble_balanced_clip_id_decoder_MLPClassifier.png', dpi = 1200)
plt.savefig('./figure4_plots/random_ensemble_balanced_clip_id_decoder_MLPClassifier.pdf')
plt.close()

# %%

# plot the confusion matrix
rand_ensemble_cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(rand_ensemble_cm, annot=True, vmax = 100, cmap=greymap, annot_kws={"size": 50 / np.sqrt(len(rand_ensemble_cm))})
plt.title("Random Ensemble Balanced Clip ID Classification", fontsize=20)
ax.figure.axes[-1].tick_params(labelsize=18)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Truth', fontsize=20)
plt.yticks(rotation=0, fontsize=18)
plt.xticks(fontsize=18)
plt.savefig('./figure4_plots/asdf_random_ensemble_balanced_clip_id_decoder_MLPClassifier.png', dpi = 1200)
plt.savefig('./figure4_plots/asdf_random_ensemble_balanced_clip_id_decoder_MLPClassifier.pdf')
plt.close()

# %%
print('\n Best estimator:')
print(grid_search.best_estimator_)
print('\n Best hyperparameters:')
print(grid_search.best_params_)

# %%
#### Develop normalized Heatmap
rand_ensemble_cm_norm = np.round((rand_ensemble_cm.astype('float') / rand_ensemble_cm.sum(axis=1)[:, np.newaxis])*100)
plt.figure(figsize=(10,7))
sns.heatmap(rand_ensemble_cm_norm, annot=True, vmax = 100, cmap=greymap, annot_kws={"size": 55 / np.sqrt(len(rand_ensemble_cm)), "color": 'white'})
plt.title("Random Ensemble Clip ID Classification Percent", fontsize=24)
ax.figure.axes[-1].tick_params(labelsize=18)
plt.xlabel('Predicted', fontsize=20)
plt.ylabel('Truth', fontsize=20)
plt.yticks(rotation=0, fontsize=18)
plt.xticks(fontsize=18)
plt.savefig('./figure4_plots/random_ensemble_balanced_clip_id_percentage_decoder_MLPClassifier.png', dpi = 1200)
plt.savefig('./figure4_plots/random_ensemble_balanced_clip_id_percentage_decoder_MLPClassifier.pdf')
plt.close()

# %%
### Produce Statistical Test Between Two Heatmaps
assembly_cm_norm_mean = np.mean(assembly_cm_norm.flatten())
assembly_cm_norm_std = np.std(assembly_cm_norm.flatten())
rand_ensemble_cm_mean = np.mean(rand_ensemble_cm_norm.flatten())
rand_ensemble_cm_std = np.std(rand_ensemble_cm_norm.flatten())
# print(f"Assembly Heatmap: Mean = {assembly_cm_norm_mean}, Std = {assembly_cm_norm_std}")
# print(f"Random Ensemble Heatmap: Mean = {rand_ensemble_cm_mean}, Std = {rand_ensemble_cm_std}")

# Mann-Whitney U test
mw_u_stat, mw_p_value = stats.mannwhitneyu(assembly_cm_norm.flatten(), rand_ensemble_cm_norm.flatten())
print(f"Mann-Whitney U test: u_stat = {mw_u_stat}, p_value = {mw_p_value}")

# Rank-biserial correlation
rank_biserial = 1 - 2 * mw_u_stat / (len(assembly_cm_norm.flatten()) * len(rand_ensemble_cm_norm.flatten()))
print(f"Mann-Whitney U Test Rank-biserial correlation = {rank_biserial}")

# Wilcoxon rank-sum
wrs_stat, wrs_p_value = stats.ranksums(assembly_cm_norm.flatten(), rand_ensemble_cm_norm.flatten())
print(f"Wilcoxon Rank-Sum U test: u_stat = {wrs_stat}, p_value = {wrs_p_value}")

# %%
### Statistical test of just the Diagonal, Implying Accuracy
assembly_accuracy = np.diag(assembly_cm_norm)
rand_ensemble_accuracy = np.diag(rand_ensemble_cm_norm)

assembly_accuracy_mean = np.mean(assembly_accuracy)
assembly_accuracy_std = np.std(assembly_accuracy)
rand_ensemble_accuracy_mean = np.mean(rand_ensemble_accuracy)
rand_ensemble_accuracy_std = np.std(rand_ensemble_accuracy)
# print(f"Assembly Heatmap: Mean = {assembly_accuracy_mean}, Std = {assembly_accuracy_std}")
# print(f"Random Ensemble Heatmap: Mean = {rand_ensemble_accuracy_mean}, Std = {rand_ensemble_cm_std}")

# Mann-Whitney U test
mw_u_stat, mw_p_value = stats.mannwhitneyu(assembly_accuracy, rand_ensemble_accuracy, alternative = 'greater')
print(f"Mann-Whitney U test: u_stat = {mw_u_stat}, p_value = {mw_p_value}")

# Rank-biserial correlation
rank_biserial = 1 - 2 * mw_u_stat / (len(assembly_accuracy) * len(rand_ensemble_accuracy))
print(f"Mann-Whitney U Test Rank-biserial correlation = {rank_biserial}")

# Wilcoxon rank-sum
wrs_stat, wrs_p_value = stats.ranksums(assembly_accuracy, rand_ensemble_accuracy, alternative = 'greater')
print(f"Wilcoxon Rank-Sum U test: u_stat = {wrs_stat}, p_value = {wrs_p_value}")

# # %% [markdown]
# ### Reproduce Oracle Score Analysis with Random Ensembles

#%%
pika_rois_all_dict = {}
pika_rois_in_assembly_dict = {}
pika_rois_no_assembly_dict = {}
for plane_n, roi_ns in rois_dict.items():
    pika_rois = []
    pika_rois_in_assembly = []
    pika_rois_no_assembly = []
    # print(plane_n)
    for roi_n in roi_ns:
        score = daf.get_pika_classifier_score(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        if score > 0.5:  # Using the threshold from team PIKA, per https://github.com/zhuangjun1981/v1dd_physiology/blob/main/v1dd_physiology/example_notebooks/2022-06-27-data-fetching-basic.ipynb
            pika_rois.append(roi_n)
            if int(roi_n[4:]) in assembly_neurons:
                pika_rois_in_assembly.append(roi_n)
            else:
                pika_rois_no_assembly.append(roi_n)
    pika_rois_all_dict[plane_n] = pika_rois
    pika_rois_in_assembly_dict[plane_n] = pika_rois_in_assembly
    pika_rois_no_assembly_dict[plane_n] = pika_rois_no_assembly

total_rois = np.sum([len(val) for val in pika_rois_all_dict.values()])
neuron_all_movie_oracle_r_values = np.zeros((total_rois, 9))

total_rois = np.sum([len(val) for val in pika_rois_in_assembly_dict.values()])
neuron_in_assembly_movie_oracle_r_values = np.zeros((total_rois, 9))

total_rois = np.sum([len(val) for val in pika_rois_no_assembly_dict.values()])
neuron_no_assembly_movie_oracle_r_values = np.zeros((total_rois, 9))

count_n_all = -1
count_n_in_assembly = -1
count_n_no_assembly = -1
for curr_dict, oracle_array, c in zip([pika_rois_all_dict, pika_rois_in_assembly_dict, pika_rois_no_assembly_dict], 
        [neuron_all_movie_oracle_r_values, neuron_in_assembly_movie_oracle_r_values, neuron_no_assembly_movie_oracle_r_values],
        [1,2,3]):
    for plane_n, pika_roi_ns in curr_dict.items():
        for roi_n in pika_roi_ns:
            if c == 1:
                count_n_all += 1
                current_count = count_n_all
            elif c == 2:
                count_n_in_assembly += 1
                current_count = count_n_in_assembly
            elif c == 3:
                count_n_no_assembly += 1
                current_count = count_n_no_assembly
            ### Get Time Trace
            dff, dff_ts = daf.get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type='dff')
            f_binary_raster = activity_raster[:, int(roi_n[4:])-1]
            
            # Get Repeated Natural Movies
            trial_fluorescence = []
            presentation = nwb_f['stimulus']['presentation']
            nm_timestamps = np.array(
                presentation['natural_movie'].get('timestamps'))
            nm_data = np.array(presentation['natural_movie'].get('data'))
            new_clips = np.where(nm_data[:, 2] == 0)[0]
            clip_duration = 300  # new_clips[1]-1
            for repeat_id in range(new_clips.shape[0]):
                frames_to_capture = np.where(dff_ts >= nm_timestamps[new_clips[repeat_id]])[
                    0][0:clip_duration]
                trial_fluorescence.append(f_binary_raster[frames_to_capture])
            trial_fluorescence_np = np.array(trial_fluorescence)
            for trial_idx in range(trial_fluorescence_np.shape[0]):
                removed_trial = trial_fluorescence_np[trial_idx]
                remaining_trials = np.delete(
                    trial_fluorescence_np, trial_idx, 0)
                r, p = scipy.stats.pearsonr(
                    removed_trial, np.mean(remaining_trials, 0))
                oracle_array[current_count, trial_idx] = r

assembly_movie_oracle_r_values = np.zeros((passing_roi_count, 9))
f = assembly_coactivity_trace
# print(coactivity_trace)
f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
                ['Fluorescence']['f_raw_subtracted'].get('timestamps'))
for roi_n in range(passing_roi_count):

    # Get Repeated Natural Movies
    trial_fluorescence = []
    presentation = nwb_f['stimulus']['presentation']
    nm_timestamps = np.array(
        presentation['natural_movie'].get('timestamps'))
    nm_data = np.array(presentation['natural_movie'].get('data'))
    new_clips = np.where(nm_data[:, 2] == 0)[0]
    clip_duration = 300  # new_clips[1]-1
    for repeat_id in range(new_clips.shape[0]):
        frames_to_capture = np.where(f_ts >= nm_timestamps[new_clips[repeat_id]])[
            0][0:clip_duration]
        trial_fluorescence.append(f[frames_to_capture, roi_n])
    trial_fluorescence_np = np.array(trial_fluorescence)
    for trial_idx in range(trial_fluorescence_np.shape[0]):
        removed_trial = trial_fluorescence_np[trial_idx]
        remaining_trials = np.delete(
            trial_fluorescence_np, trial_idx, 0)
        r, p = scipy.stats.pearsonr(
            removed_trial, np.mean(remaining_trials, 0))
        assembly_movie_oracle_r_values[roi_n, trial_idx] = r

# Plot Movie Oracles
mean_over_holdouts = np.mean(assembly_movie_oracle_r_values, 1)
fig = plt.figure()
plt.title('Assembly natural movie oracle score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.hist(mean_over_holdouts[:], bins=50)
#plt.savefig('./figure4_plots/oracle_dists2/assemblies_esteps_150000__affinity_04_session'+str(13)+'_movies.png')
plt.show()

random_ensemble_movie_oracle_r_values = np.zeros((passing_roi_count, 9))
f = random_ensembles_coactivity_trace
# print(coactivity_trace)
f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
                ['Fluorescence']['f_raw_subtracted'].get('timestamps'))
for roi_n in range(passing_roi_count):

    # Get Repeated Natural Movies
    trial_fluorescence = []
    presentation = nwb_f['stimulus']['presentation']
    nm_timestamps = np.array(
        presentation['natural_movie'].get('timestamps'))
    nm_data = np.array(presentation['natural_movie'].get('data'))
    new_clips = np.where(nm_data[:, 2] == 0)[0]
    clip_duration = 300  # new_clips[1]-1
    for repeat_id in range(new_clips.shape[0]):
        frames_to_capture = np.where(f_ts >= nm_timestamps[new_clips[repeat_id]])[
            0][0:clip_duration]
        trial_fluorescence.append(f[frames_to_capture, roi_n])
    trial_fluorescence_np = np.array(trial_fluorescence)
    for trial_idx in range(trial_fluorescence_np.shape[0]):
        removed_trial = trial_fluorescence_np[trial_idx]
        remaining_trials = np.delete(
            trial_fluorescence_np, trial_idx, 0)
        r, p = scipy.stats.pearsonr(
            removed_trial, np.mean(remaining_trials, 0))
        random_ensemble_movie_oracle_r_values[roi_n, trial_idx] = r

import ptitprince as pt
# Plot Movie Oracles
mean_over_holdouts = np.mean(random_ensemble_movie_oracle_r_values, 1)
fig = plt.figure()
plt.title('Random Ensemble natural movie oracle score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.hist(mean_over_holdouts[:], bins=50)
#plt.savefig('./figure4_plots/oracle_dists2/assemblies_esteps_150000__affinity_04_session'+str(13)+'_movies.png')
plt.show()

# Create a figure
plt.figure(figsize=(10,7))
sns.set_theme(style="whitegrid")

all_arr = [np.array(assembly_movie_oracle_r_values).flatten(),
            np.array(random_ensemble_movie_oracle_r_values).flatten(),
            np.array(neuron_all_movie_oracle_r_values).flatten(),
            np.array(neuron_in_assembly_movie_oracle_r_values).flatten(),
            np.array(neuron_no_assembly_movie_oracle_r_values).flatten()]

# Prepare data for raincloud plot
data = pd.DataFrame({
    "Values": np.concatenate(all_arr),
    "Group": [f"Assemblies"] * len(assembly_movie_oracle_r_values.flatten()) + \
                [f"Random\nEnsembles"] * len(random_ensemble_movie_oracle_r_values.flatten()) + \
                [f"All\nCells"] * len(neuron_all_movie_oracle_r_values.flatten()) + \
                [f"Assembly\nCells"] * len(neuron_in_assembly_movie_oracle_r_values.flatten()) + \
                [f"Non-Assembly\nCells"] * len(neuron_no_assembly_movie_oracle_r_values.flatten())
})

# Create the raincloud plot
ax = pt.RainCloud(
    y="Values",
    x="Group",
    data=data,
    palette=[(.4, .6, .8, .5), 'grey'],
    width_viol=0.6,  # Adjust violin width
    alpha=0.8,  # Transparency of the cloud
    move=0.1,  # Adjust position of violins
    point_size = 4,
    orient="v"  # Horizontal orientation
)

plt.ylim(-0.2, 1.2)

# Calculate p-values for Wilcoxon rank-sum tests
groups = ["Random\nEnsembles", "All\nCells", "Assembly\nCells", "Non-Assembly\nCells"]
p_values = []
for group in groups:
    group_values = data[data["Group"] == group]["Values"]
    p_value = stats.ranksums(data[data["Group"] == "Assemblies"]["Values"], group_values, alternative='two-sided').pvalue
    print(f"p-value for Assemblies greater than {group}: {p_value:.3g}")
    p_values.append(p_value)

# Add significance markers as number of asterisks
for i, p_value in enumerate(p_values):
    if p_value < 0.001:
        significance = 'p < 0.001\n***'
    elif p_value < 0.01:
        significance = 'p < 0.01\n**'
    elif p_value < 0.05:
        significance = 'p < 0.05\n*'
    else:
        significance = 'ns'
    ax.annotate(significance, xy=(i + 1,  1.1), #data[data["Group"] == groups[i]]["Values"].max()
                horizontalalignment='center', size=16, color='black', weight='bold')

# Add a multiline title to include the p-value, add y_label
title = f'Natural Movie Oracle Scores'
plt.title(title, size=24)
# plt.xlabel('Cell Grouping', size=20)
plt.xticks(fontsize=20)  # Adjust size of xticks
plt.yticks(fontsize=20)  # Adjust size of yticks
plt.ylabel("Oracle Score", size=20)

plt.tight_layout()
plt.savefig('./figure4_plots/oracle_scores_dff_all_sets_raincloud.png', dpi = 1200)
plt.savefig('./figure4_plots/oracle_scores_dff_all_sets_raincloud.pdf')
plt.savefig('./figure4_plots/oracle_scores_dff_all_sets_raincloud.svg')
plt.show()


# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(data=all_arr,
#                 notch=True, showcaps=True,
#                 flierprops={"marker": "x"},
#                 boxprops={"facecolor": (.4, .6, .8, .5)},
#                 medianprops={"color": "coral"},
#             )
# ax.set_xticklabels(["Assemblies", "Random Ensembles", "All Cells", "Assembly Cells", "Non-Assembly Cells"], size = 14)
# ax.set_title('Natural Movie Oracle Score', size = 20)
# ax.set_ylabel('Oracle Score', size = 15)

# medians = np.array(
#     [np.median(np.array(assembly_movie_oracle_r_values).flatten()),
#      np.median(np.array(random_ensemble_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_all_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_in_assembly_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_no_assembly_movie_oracle_r_values).flatten())]
# )

# vertical_offset = (medians * 0.0)+1.05  # offset from median for display
# p_values = [np.nan,
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(random_ensemble_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(neuron_all_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(neuron_in_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5)]

# for xtick in ax.get_xticks():
#     if xtick != 0:
#         ax.text(xtick, medians[xtick] + vertical_offset[xtick], p_values[xtick], 
#                 horizontalalignment='center', size='small', color='black', weight='semibold')

# plt.savefig('./figure4_plots/oracle_scores_histogram_dff_all_sets.png', dpi = 1200)
# # plt.show()
# plt.close()

# # %%
# # Plot Movie Oracles
# plt.figure(figsize=(14, 10))

# all_arr = [np.array(assembly_movie_oracle_r_values).flatten(),
#             np.array(random_ensemble_movie_oracle_r_values).flatten(),
#             np.array(neuron_all_movie_oracle_r_values).flatten(),
#             np.array(neuron_in_assembly_movie_oracle_r_values).flatten(),
#             np.array(neuron_no_assembly_movie_oracle_r_values).flatten()]
# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(data=all_arr,
#                 notch=True, showcaps=True,
#                 flierprops={"marker": "x"},
#                 boxprops={"facecolor": (.4, .6, .8, .5)},
#                 medianprops={"color": "coral"},
#             )
# ax.set_xticklabels(["Assemblies", "Random Ensembles", "All Cells", "Assembly Cells", "Non-Assembly Cells"], size = 15)
# ax.set_title('Natural Movie Oracle Score', size = 20)
# ax.set_ylabel('Oracle Score', size = 16)
# plt.yticks(fontsize=15)

# medians = np.array(
#     [np.median(np.array(assembly_movie_oracle_r_values).flatten()),
#      np.median(np.array(random_ensemble_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_all_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_in_assembly_movie_oracle_r_values).flatten()),
#      np.median(np.array(neuron_no_assembly_movie_oracle_r_values).flatten())]
# )

# vertical_offset = medians * 0.15 # offset from median for display
# p_values = [np.nan,
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(random_ensemble_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(neuron_all_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(neuron_in_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5),
#             'P-Val: {:.3g}'.format(stats.ranksums(np.array(assembly_movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten(), alternative = 'greater').pvalue, 5)]

# for xtick in ax.get_xticks():
#     if xtick != 0:
#         ax.text(xtick, medians[xtick] + vertical_offset[xtick], p_values[xtick], 
#                 horizontalalignment='center', size= 12, color='black', weight='semibold')

# plt.savefig('./figure4_plots/oracle_scores_histogram_dff_all_sets.png', dpi = 1200)
# plt.show()

# wrs_p_value = stats.ranksums(np.array(neuron_all_movie_oracle_r_values).flatten(), np.array(neuron_in_assembly_movie_oracle_r_values).flatten())
# print(f"Rank-Sum (All Cells vs Assembly Cells): p_value = {wrs_p_value}")
# wrs_p_value = stats.ranksums(np.array(neuron_all_movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten())
# print(f"Rank-Sum (All Cells vs Non-Assembly Cells): p_value = {wrs_p_value}")
# wrs_p_value = stats.ranksums(np.array(neuron_in_assembly_movie_oracle_r_values).flatten(), np.array(neuron_no_assembly_movie_oracle_r_values).flatten())
# print(f"Rank-Sum (Assembly Cells vs Non-Assembly Cells): p_value = {wrs_p_value}")

# %% [markdown]
# ### Reproduce Sparsity Plot with Random Ensembles

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_ginis2(coactivity_trace, null_trace=None, all_trace=None, a_trace=None, no_a_trace=None):
    num_assemblies = coactivity_trace.shape[1]
    gini_values = [gini(coactivity_trace[:,i]) for i in range(num_assemblies)]
    labels = [f'A {i+1}' for i in range(num_assemblies)]
    
    if no_a_trace is not None:
        gini_values_no_a = [gini(no_a_trace[:,i]) for i in range(no_a_trace.shape[1])]
        mean_gini_no_a = np.nanmean(gini_values_no_a)
        # labels.insert(0, 'Nonassembly\nCells')
    
    if a_trace is not None:
        gini_values_a = [gini(a_trace[:,i]) for i in range(a_trace.shape[1])]
        mean_gini_a = np.nanmean(gini_values_a)
        # labels.insert(0, 'Assembly\nCells')
    
    # if all_trace is not None:
    #     gini_values_all = [gini(all_trace[:,i]) for i in range(all_trace.shape[1])]
    #     mean_gini_all = np.nanmean(gini_values_all)
    #     labels.insert(0, 'All\nCells')
    
    if null_trace is not None:
        gini_values_null = [gini(null_trace[:,i]) for i in range(num_assemblies)]
        mean_gini_null = np.nanmean(gini_values_null)
        # labels.insert(0, 'Null')
        
    print(gini_values)
    print(gini_values_null)
    print(labels)

    # Create a DataFrame for the grouped bar plot
    gini_df = pd.DataFrame({
        'Assembly Gini Values': gini_values,
        'Random Ensemble Gini Values': gini_values_null,
        'Labels': labels
    })

    # print(gini_df)

    # Melt the DataFrame to long format
    gini_df_melted = gini_df.melt(id_vars='Labels', value_vars=['Assembly Gini Values', 'Random Ensemble Gini Values'], 
                                  var_name='Type', value_name='Gini Coefficient')

    p_value = stats.ranksums(np.array(gini_values_null).flatten(), np.array(gini_values).flatten(), alternative='two-sided').pvalue

    # Create a grouped bar plot
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x='Labels', y='Gini Coefficient', hue='Type', data=gini_df_melted, palette='Paired')
    ax.set_title(f'Sparsity of Activity\np={p_value:0.4g}', size=24)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20, rotation=45)    
    ax.set_ylabel('Gini Coefficient', size=24)
    ax.set_xlabel('Assembly or Matched Random Ensemble', size=24)
    plt.ylim((0, 1.0))

    # Add horizontal lines for mean_gini_a and mean_gini_no_a
    if a_trace is not None:
        ax.axhline(mean_gini_a, color='cornflowerblue', linestyle='--', linewidth=2)
        ax.text(len(labels)-5.75, mean_gini_a-0.04, f'Assembly Cells Mean: {mean_gini_a:.3f}', 
                color='cornflowerblue', ha='left', va='bottom', fontsize=16, weight='bold')

    if no_a_trace is not None:
        ax.axhline(mean_gini_no_a, color='slategray', linestyle='--', linewidth=2)
        ax.text(len(labels)-5.75, mean_gini_no_a, f'Nonassembly Cells Mean: {mean_gini_no_a:.3f}', 
                color='slategray', ha='left', va='bottom', fontsize=16, weight='bold')

    # Add annotations for clarity (optional)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 30.5), textcoords='offset points', size=18, weight='bold', rotation=90)

    plt.legend(bbox_to_anchor=(0, 0.95), loc='upper left', frameon=True, fontsize=16)
    # plt.legend(, loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('./figure4_plots/sparsity_with_Gini_coefficient_by_assembly_and_random_ensembles.png')
    plt.savefig('./figure4_plots/sparsity_with_Gini_coefficient_by_assembly_and_random_ensembles.pdf')
    plt.savefig('./figure4_plots/sparsity_with_Gini_coefficient_by_assembly_and_random_ensembles.svg')

    # gini_values_all = gini_values_a + gini_values_no_a
    
    # # Create a figure
    # plt.figure(figsize=(24,8))
    # sns.set_theme(style="whitegrid")

    # all_arr = [np.array(gini_values).flatten(),
    #         np.array(gini_values_null).flatten(),
    #         np.array(gini_values_all).flatten(),
    #         np.array(gini_values_a).flatten(),
    #         np.array(gini_values_no_a).flatten()]

    # print(np.min(gini_values))
    # print(np.max(gini_values))
    # print(np.min(gini_values_all))
    # print(np.max(gini_values_all))
    # print(np.min(gini_values_a))
    # print(np.max(gini_values_a))
    # print(np.min(gini_values_no_a))
    # print(np.max(gini_values_no_a))

    # # Prepare data for raincloud plot
    # data = pd.DataFrame({
    #     "Gini Coefficients": np.concatenate(all_arr),
    #     "Cell Grouping": [f"Assemblies\nn={len(gini_values)}"] * len(gini_values) + \
    #                 [f"Random\nEnsembles\nn={len(gini_values_null)}"] * len(gini_values_null) + \
    #                 [f"All\nCells\nn={len(gini_values_all)}"] * len(gini_values_all) + \
    #                 [f"Assembly\nCells\nn={len(gini_values_a)}"] * len(gini_values_a) + \
    #                 [f"Non-Assembly\nCells\nn={len(gini_values_no_a)}"] * len(gini_values_no_a)
    # })

    # # Create the raincloud plot
    # ax = pt.RainCloud(
    #     y="Gini Coefficients",
    #     x="Cell Grouping",
    #     data=data,
    #     palette=[(.4, .6, .8, .5), 'grey'],
    #     width_viol=0.6,  # Adjust violin width
    #     alpha=0.8,  # Transparency of the cloud
    #     move=0.1,  # Adjust position of violins
    #     point_size = 6,
    #     orient="v",  # Horizontal orientation
    #     linewidth=2
    # )

    
    #  # # Calculate p-value
    # p_value = stats.ranksums(np.array(gini_values_null).flatten(), np.array(gini_values).flatten(), alternative='two-sided').pvalue

    # # # Add significance stars
    # if p_value < 0.001:
    #     significance = 'P < 0.001\n***'
    # elif p_value < 0.01:
    #     significance = 'P < 0.01\n**'
    # elif p_value < 0.05:
    #     significance = 'P < 0.05\n*'
    # else:
    #     significance = 'ns'

    # # Add p-value annotation

    # ax.text(0.5, 0.9, f'{significance}', 
    #             color='black', ha='center', va='bottom', fontsize=20, weight='bold')
    
    # ax.annotate(f'P-Val: {p_value:.3g}', xy=(1, medians[1] + vertical_offset[1]), xytext=(1, medians[1] + vertical_offset[1] + 0.05),
    #             arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='center', size=18, color='black', weight='semibold')

    

    # plt.ylabel('Gini Coefficient', size=20, weight='bold')
    # plt.xticks(fontsize=20, weight='bold')
    # plt.yticks(fontsize=20, weight='bold')
    # # plt.xlabel("Groups", size=20, weight='bold')

    # plt.tight_layout()
    # plt.savefig('./figure4_plots/gini_coefficients_raincloud_plot.png', dpi=1200)
    # plt.savefig('./figure4_plots/gini_coefficients_raincloud_plot.pdf', dpi=1200)
    # plt.savefig('./figure4_plots/gini_coefficients_raincloud_plot.svg', dpi=1200)
    # plt.show()

# %%
print(activity_raster.shape)
# print(np.min(no_assembly_neurons))
# print(np.min(assembly_neurons))
plot_ginis2(coactivity_trace=assembly_coactivity_trace, null_trace=random_ensembles_coactivity_trace, all_trace=activity_raster, a_trace=activity_raster[:,assembly_neurons], no_a_trace=activity_raster[:,no_assembly_neurons])


