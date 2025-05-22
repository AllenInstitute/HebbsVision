# importing packages
# import v1dd_physiology.data_fetching as daf
# from caveclient import CAVEclient
# import nglui.statebuilder as sb
import os.path
import numpy as np
import pandas as pd

import random
import scipy.io
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import csv
import pickle
import math
import h5py

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# matplotlib.use('TkAgg')


def make_url(client, df):
    img_source = client.info.image_source()
    img_layer = sb.ImageLayerConfig(name='img',
                                    source=img_source,
                                    )

    seg_source = client.info.segmentation_source()
    seg_layer = sb.SegmentationLayerConfig(name='seg',
                                           source=seg_source,
                                           selected_ids_column='pre_pt_root_id')

    points = sb.PointMapper(
        'ctr_pt_position', linked_segmentation_column='pre_pt_root_id')
    anno_layer = sb.AnnotationLayerConfig(
        name='annos', mapping_rules=points, linked_segmentation_layer='seg')

    # anno_layer = sb.AnnotationLayerConfig(name='annos')
    anno_layer._array_data
    my_state_builder = sb.StateBuilder(
        layers=[img_layer, seg_layer, anno_layer], client=client)
    link = sb.helpers.make_url_robust(
        df, my_state_builder, client, shorten='always')
    print(link)
    return link


# Use a color-blind friendly color list
COLOR_SAFE_LIST = ['#E69F00', '#56B4E9', '#009E73',
                   '#F0E442', '#0072B2', '#D55E00', '#000000', '#CC79A7']

for i in range(10):
    COLOR_SAFE_LIST = COLOR_SAFE_LIST + COLOR_SAFE_LIST


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
    gini_values = [gini(coactivity_trace[i]) for i in range(num_assemblies)]
    labels = [f'{i+1}' for i in range(num_assemblies)]

    # Create a base bar plot
    plt.figure()
    plt.bar(x=np.arange(len(gini_values)), height=gini_values, tick_label=labels)

    # Apply Labels:
    plt.title("Assembly Sparsity")            # Add a title
    plt.xlabel("Assemblies")                 # Label the x-axis
    plt.ylabel("Gini Coefficient")                     # Label the y-axis

    # Remove chartjunk (redundant elements)
    plt.tick_params(axis='both', which='both', length=0)  # Hide ticks
    plt.gca().spines['top'].set_visible(False)            # Hide top spine
    plt.gca().spines['right'].set_visible(False)          # Hide right spine

    # Add annotations for clarity (optional)
    for i, value in enumerate(gini_values):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')
    plt.savefig('sparsity_with_Gini_coefficient_by_assembly.png')




ACTIVITY_RASTER = scipy.io.loadmat(
    "/home/julian/scan13/esteps_150000_affinity_04_sessionM409828_13_ACTIVITY-RASTER.mat", struct_as_record=True, squeeze_me=True)
SGC_ASSEMBLIES = scipy.io.loadmat(
    "esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)

print(ACTIVITY_RASTER.keys())

activity_raster = ACTIVITY_RASTER['activity_raster']
activity_raster_peaks = ACTIVITY_RASTER['activity_raster_peaks']

coactivity_trace = activity_raster.mean(axis=1)

assemblies = SGC_ASSEMBLIES['assemblies']
print(assemblies)
assembly_coactivity_trace = np.vstack(
    [activity_raster[:, A-1].mean(axis=1) for A in assemblies]).T
assembly_coactivity_trace.shape

get_assembly_time_trace(assembly_coactivity_trace)
plot_ginis(assembly_coactivity_trace)

# # Orientation Tuning


# # And finally the dff drifting grating response
# nwb_f = h5py.File(
#     '/home/berteau/v1DD/nwbs/processed/M409828_13_20181213.nwb', 'r')


# rm_path = daf.get_rm_path(nwb_f=nwb_f)
# rm_f = h5py.File(rm_path, 'r')
# dgcrm = daf.get_dgcrm(rm_f=rm_f, plane_n='plane0',
#                       roi_n='102', trace_type='dff', dgc_type='windowed')

# # dgcrm.get_df_response_table()
# # dgcrt, _, _, _ = for_plot.get_df_response_table(baseline_win=baseline_win, response_win=response_win)
# dgcrm_plot = dgcrm.remove_blank_cond()
# dg_f = plt.figure(figsize=(20, 6))
# _ = dgcrm_plot.plot_all_traces(
#     baseline_win=(-0.5, 0.), response_win=(0., 2.), f=dg_f, color='k', lw=0.5)
# plt.savefig('./scan13/three_hypotheses2/session13_nonassembly_roi' +
#             rois[neuron]+'_dg_traces.png')

# dgcrt, _, _, _ = dgcrm.get_zscore_response_table(
#     baseline_win=(-0.5, 0.), response_win=(0., 2.))
# dg_ori_f, dof_ax = plt.subplots(subplot_kw={'projection': 'polar'})
# _ = dgcrt.plot_dire_tuning(axis=dof_ax, response_dir='pos', is_collapse_sf=False, is_collapse_tf=False,
#                            trace_color='#ff0000', postprocess='elevate', is_plot_errbar=True, is_normalize=False, is_arc=False)
# plt.savefig('./scan13/three_hypotheses2/session13_nonassembly_roi' +
#             rois[neuron]+'_dg_orienation.png')



# def get_dff_response_table(self, baseline_win=(-0.5, 0.), response_win=(0., 1.), bias=0, warning_level=0.1):
#     """
#     this is suppose to give the most robust measurement of df/f response table.

#     for each condition:
#     1. mean_baseline is calculated by averaging across all trials and all data points in the baseline_win
#     2. mean_response is calculated by averaging across all trials and all data points in the response_win
#     3. df/f for each condition is defined by
#         (mean_response - mean_baseline) / mean_baseline and response table is generated

#     # separate operation
#     4. for each trial of each condition, df is calculated by (mean_response - mean_baseline) / mean_baseline
#     5. one-way anova is performed from these trial responses
#     6. peak positive condition and peak negative condition is selected from previously generated response table
#     7. ttest is performed for these two conditions against blank trial responses


# v1DD_session13_SGC_ASSEMBLIES = SGC_ASSEMBLIES

# # print(v1DD_session13_SGC_ASSEMBLIES["assemblies"])
# # print(v1DD_session13_SGC_ASSEMBLIES["assemblies"][0].shape)
# # print(v1DD_session13_SGC_ASSEMBLIES["assemblies"][1].shape)
# # print(v1DD_session13_SGC_ASSEMBLIES["assemblies"][2].shape)

# patterns = v1DD_session13_SGC_ASSEMBLIES['assembly_pattern_detection']['activityPatterns'].item()
# print(patterns)

# ### Get the indexes of the activity patterns that correspond to the assembly activations
# activity_pattern_indexes = v1DD_session13_SGC_ASSEMBLIES['assembly_pattern_detection']['assemblyIActivityPatterns'].item()
# print(activity_pattern_indexes)

# pattern_neuron_lists = []

# for assembly_idx in range(len(activity_pattern_indexes)):
#     pattern_neuron_lists.append([])
#     for pattern_index in activity_pattern_indexes[assembly_idx]:
#         pattern = v1DD_session13_SGC_ASSEMBLIES['assembly_pattern_detection']['activityPatterns'].item()[pattern_index]
#         pattern_neuron_lists[assembly_idx].append(np.nonzero(pattern))


# # SNR Analysis of assembly neurons
# rf_data = np.load(savepath+'rf_data.npz')
# snr = rf_data["snr"]
# peaks_ons = rf_data["peaks_ons"]
# peaks_offs = rf_data["peaks_offs"]
# dff_low_qual_cells = rf_data["dff_low_qual_cells"]
# unresponsive_cells = rf_data["unresponsive_cells"]

# snr_f = snr[np.logical_not(dff_low_qual_cells)]
# snr_f = snr_f[np.isfinite(snr_f)]

# print np.sum(dff_low_qual_cells), 'cells with low dff correlation to original trace'
# print np.sum(unresponsive_cells), 'cells with no significant pixels in chi square test'

# good_cells = np.logical_not(np.logical_or(np.squeeze(dff_low_qual_cells), np.squeeze(unresponsive_cells)))
# print np.sum(good_cells), 'responsive cells out of',good_cells.shape[0]
