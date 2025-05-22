from difflib import diff_bytes
import h5py
import v1dd_physiology.data_fetching as daf
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import pandas as pd
import csv
import pickle
import math
# import cv2


def distance(x, y):
    return math.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

# def process(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_canny = cv2.Canny(img_gray, 0, 50)
#     img_dilate = cv2.dilate(img_canny, None, iterations=1)
#     img_erode = cv2.erode(img_dilate, None, iterations=1)
#     return img_erode


ACTIVITY_RASTER = scipy.io.loadmat(
    "/home/julian/scan13/esteps_150000_affinity_04_sessionM409828_13_ACTIVITY-RASTER.mat", struct_as_record=True, squeeze_me=True)
SGC_ASSEMBLIES = scipy.io.loadmat(
    "/home/julian/scan13/esteps_150000_affinity_04_sessionM409828_13_SGC-ASSEMBLIES.mat", struct_as_record=True, squeeze_me=True)

print(ACTIVITY_RASTER.keys())

activity_raster = ACTIVITY_RASTER['activity_raster']
activity_raster_peaks = ACTIVITY_RASTER['activity_raster_peaks']

coactivity_trace = activity_raster.mean(axis=1)

# nwb_f = h5py.File('nwbs/processed/M416296_12_20181210.nwb', 'r')
nwb_f = h5py.File('nwbs/processed/M409828_13_20181213.nwb', 'r')
# nwb_f = h5py.File('nwbs/processed/M409828_14_20181214.nwb', 'r')

sess_id = daf.get_session_id(nwb_f=nwb_f)
print(sess_id)

plane_ns = daf.get_plane_names(nwb_f=nwb_f)
print("Planes: ", plane_ns)

for plane_n in plane_ns:
    depth = daf.get_plane_depth(nwb_f=nwb_f, plane_n=plane_n)
    print(f'depth of {plane_n}: {depth} um')

fs = []
dffs = []
events = []
locomotions = []
rois = []
pika_rois = []
coords = []
# f = np.array(nwb_f['processing']['rois_and_traces_plane0']
#              ['Fluorescence']['f_raw_subtracted'].get('data'))
f = coactivity_trace
f_ts = np.array(nwb_f['processing']['rois_and_traces_plane0']
                ['Fluorescence']['f_raw_subtracted'].get('timestamps'))
# events
clip_ts = f.shape[1]


locomotion_raw = np.array(
    nwb_f['processing']['locomotion']['Position']['distance'].get('data'))
locomotion_delta = np.array([locomotion_raw[i+1] - locomotion_raw[i]
                            for i in range(locomotion_raw.shape[0]-1)])
locomotion_timestamps = np.array(
    nwb_f['processing']['locomotion']['Position']['distance'].get('timestamps'))
locomotion_reference_frames = np.array(
    nwb_f['processing']['locomotion']['Position']['distance'].get('reference_frame'))

time_matched_locomotion = []
for f_t in range(f_ts.shape[0]):
    time_window_start = f_ts[f_t]
    if (f_t >= f_ts.shape[0]-1) or (f_ts[f_t+1] > max(locomotion_timestamps)):
        time_window_end = max(locomotion_timestamps)
    else:
        time_window_end = f_ts[f_t+1]
    if time_window_end >= min(locomotion_timestamps):
        locomotion_to_sum = np.where(
            (locomotion_timestamps > time_window_start) * (locomotion_timestamps < time_window_end))[0]
        time_matched_locomotion.append(
            np.sum(locomotion_delta[locomotion_to_sum]))
    else:
        time_matched_locomotion.append(0)

# where locomotion_timestamps > time and locomotion_timestamps < time+1

total_roi_count = 0
for plane_n in plane_ns:
    roi_ns = daf.get_roi_ns(nwb_f=nwb_f, plane_n=plane_n)
    total_roi_count += len(roi_ns)

total_movie_oracle_r_values = np.zeros((0, 9))

for plane_n in plane_ns:
    plane_number = int(plane_n[-1])
    roi_ns = daf.get_roi_ns(nwb_f=nwb_f, plane_n=plane_n)
    pika_roi_ids = daf.get_pika_roi_ids(nwb_f=nwb_f, plane_n=plane_n)
    depth = daf.get_plane_depth(nwb_f=nwb_f, plane_n=plane_n)
    # plane_projection = daf.get_plane_projections(nwb_f=nwb_f, plane_n=plane_n)
    # daf.get_greedy_rf
    # daf.get_plane_traces()

    # Get count of passing ROIs
    passing_roi_count = 0
    for roi_n in roi_ns:
        score = daf.get_pika_classifier_score(
            nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        if score > 0.5:
            passing_roi_count += 1

    print(f'there are {len(roi_ns)} rois in {plane_n} of session: {sess_id}:')
    roi_count = 0
    oracle_r_values = np.zeros((passing_roi_count, 25, 8))
    movie_oracle_r_values = np.zeros((passing_roi_count, 9))
    # print(roi_ns[0:50])
    for roi_n in roi_ns:
        print('\n\t', roi_n)
        score = daf.get_pika_classifier_score(
            nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n)
        if score > 0.5:  # Using the threshold from team PIKA, per https://github.com/zhuangjun1981/v1dd_physiology/blob/main/v1dd_physiology/example_notebooks/2022-06-27-data-fetching-basic.ipynb
            nonzero_midpoint = int(
                len(daf.get_roi_mask(nwb_f, plane_n, roi_n).nonzero()[1])/2)
            x = daf.get_roi_mask(nwb_f, plane_n, roi_n).nonzero()[
                0][nonzero_midpoint]
            y = daf.get_roi_mask(nwb_f, plane_n, roi_n).nonzero()[
                1][nonzero_midpoint]
            coords.append((x, y, depth))

            # daf.get_strf(plan_n=plane_n, roi_n=roi_n, trace_type='fluorescence')

            # print(daf.get_roi_mask(nwb_f, plane_n, roi_n))
            # mask = daf.get_roi_mask(nwb_f, plane_n, roi_n) == 1
            # roi_mask_locations = plane_projection[plane_number][mask]
            # mask = daf.get_roi_mask(nwb_f, plane_n, roi_n)
            # h, w, _ = mask.shape
            # contours, _ = cv2.findContours(process(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # contours[0]
            # mask[mask > 0][0]
            # print(len(pika_roi_ids))
            rois.append(plane_n+'_'+roi_n)
            # pika_rois.append(pika_roi_ids[roi_count])
            # f, ts = daf.get_single_trace(nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type='subtracted')
            # f = list(np.array(nwb_f['processing']['rois_and_traces_'+str(plane_n)]['Fluorescence']['f_raw_subtracted'].get('data'))[roi_number, :])
            f, f_ts = daf.get_single_trace(
                nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type='subtracted')
            # f_timestamps[]
            fs.append(f[0:clip_ts])
            # locomotions.append([])
            # for f_t in range(f_ts.shape[0]):
            #     time_window_start = f_ts[f_t]
            #     if (f_t >= f_ts.shape[0]-1) or (f_ts[f_t+1] > max(locomotion_timestamps)):
            #         time_window_end = max(locomotion_timestamps)
            #     else:
            #         time_window_end = f_ts[f_t+1]
            #     if time_window_end >= min(locomotion_timestamps):
            #         locomotion_to_sum = np.where((locomotion_timestamps > time_window_start) * (locomotion_timestamps < time_window_end))[0]
            #         locomotions[-1].append(np.sum(locomotion_delta[locomotion_to_sum]))
            #     else:
            #         locomotions[-1].append(0)
            dff, dff_ts = daf.get_single_trace(
                nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type='dff')
            dffs.append(dff[0:clip_ts])
            event, event_ts = daf.get_single_trace(
                nwb_f=nwb_f, plane_n=plane_n, roi_n=roi_n, trace_type='events')
            events.append(event[0:clip_ts])
            # print(len(f), len(dff), len(event))

            # Get Repeated Natural Movies
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
                trial_fluorescence.append(f[frames_to_capture])
            trial_fluorescence_np = np.array(trial_fluorescence)
            for trial_idx in range(trial_fluorescence_np.shape[0]):
                removed_trial = trial_fluorescence_np[trial_idx]
                remaining_trials = np.delete(
                    trial_fluorescence_np, trial_idx, 0)
                r, p = scipy.stats.pearsonr(
                    removed_trial, np.mean(remaining_trials, 0))
                movie_oracle_r_values[roi_count, trial_idx] = r

            # Get repeated drifting gratings
            stims = daf.get_stim_list(nwb_f=nwb_f)
            dgc_onsets = daf.get_dgc_onset_times(nwb_f, dgc_type='windowed')
            presentation = nwb_f['stimulus']['presentation']
            num_samples = np.array(
                presentation['drifting_gratings_windowed'].get('num_samples'))
            grating_number = 0

            for dgc in dgc_onsets.keys():
                onsets = dgc_onsets[dgc]
                trial_fluorescence = []
                for onset_id in range(onsets.shape[0]):
                    frames_to_capture = np.where(f_ts >= onsets[onset_id])[
                        0][0:num_samples]
                    trial_fluorescence.append(f[frames_to_capture])
                trial_fluorescence_np = np.array(trial_fluorescence)
                for trial_idx in range(trial_fluorescence_np.shape[0]):
                    removed_trial = trial_fluorescence_np[trial_idx]
                    remaining_trials = np.delete(
                        trial_fluorescence_np, trial_idx, 0)
                    r, p = scipy.stats.pearsonr(
                        removed_trial, np.mean(remaining_trials, 0))
                    oracle_r_values[roi_count, grating_number, trial_idx] = r

                grating_number += 1
            roi_count += 1
    total_movie_oracle_r_values = np.append(
        total_movie_oracle_r_values, movie_oracle_r_values, 0)

    # Plot Grating Oracles
    mean_over_holdouts = np.mean(oracle_r_values, 2)
    fig, axes = plt.subplots(2, 3)
    fig.suptitle
    for clip_index in range(oracle_r_values.shape[1]):
        fig = plt.figure()
        plt.title(plane_n + ' grating'+str(clip_index))
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.hist(mean_over_holdouts[:, clip_index], bins=50)
        plt.savefig('oracle_dists/session'+str(sess_id)+'_' +
                    plane_n+'_grating'+str(clip_index)+'.png')
        plt.close()
    np.save('oracle_dists/session'+str(sess_id)+'_'+plane_n +
            '_gratings_oracle_r_values.npy', oracle_r_values)
    np.save('oracle_dists/session'+str(sess_id)+'_'+plane_n +
            '_gratings_oracle_scores.npy', mean_over_holdouts)

    # Plot Movie Oracles
    mean_over_holdouts = np.mean(movie_oracle_r_values, 1)
    fig = plt.figure()
    plt.title(plane_n + ' natural movie oracle score')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.hist(mean_over_holdouts[:], bins=50)
    plt.savefig('oracle_dists/session'+str(sess_id)+'_'+plane_n+'_movies.png')
    plt.close()
    np.save('oracle_dists/session'+str(sess_id)+'_'+plane_n +
            '_natural_movie_oracle_r_values.npy', oracle_r_values)
    np.save('oracle_dists/session'+str(sess_id)+'_'+plane_n +
            '_natural_movie_oracle_scores.npy', mean_over_holdouts)

# Plot total Movie Oracles
total_mean_over_holdouts = np.mean(total_movie_oracle_r_values, 1)
fig = plt.figure()
plt.title(plane_n + ' natural movie oracle score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.hist(total_mean_over_holdouts[:], bins=50)
plt.savefig('oracle_dists/session'+str(sess_id)+'_total_movies.png')
plt.close()
np.save('oracle_dists/session'+str(sess_id) +
        'total_natural_movie_oracle_r_values.npy', total_movie_oracle_r_values)
np.save('oracle_dists/session'+str(sess_id) +
        'total_natural_movie_oracle_scores.npy', total_mean_over_holdouts)

movie_oracle_r_values

# for f in fs:
#     f = f[:21530]

# for dff in dffs:
#     dff = dff[:21530]

# for event in events:
#     event = event[:21530]

fs_n = np.array(fs).T
dffs_n = np.array(dffs).T
events_n = np.array(events).T
coords_n = np.array(coords)
rois_n = np.array(rois)
locomotion_n = np.array(time_matched_locomotion)

np.save('session'+str(sess_id)+'_f.npy', fs_n)
np.save('session'+str(sess_id)+'_dff.npy', dffs_n)
np.save('session'+str(sess_id)+'_events.npy', events_n)
np.save('session'+str(sess_id)+'_locations.npy', coords_n)
np.save('session'+str(sess_id)+'_roi_id_by_index.npy', rois_n)
np.save('session'+str(sess_id)+'_locomotion.npy', locomotion_n)
with open('session'+str(sess_id)+'roi_id_by_index_list.pkl', 'wb') as out_file:
    pickle.dump(rois, out_file)

dummy_file_dict = {}
dummy_file_dict2 = {}
dummy_file_dict2['Name'] = []
dummy_file_dict2['x'] = []
dummy_file_dict2['y'] = []
dummy_file_dict2['z'] = []
high_oracle_roi_idxs = np.where(total_mean_over_holdouts > 0.3)[0]
for roi_number in high_oracle_roi_idxs:
    dummy_file_dict2['Name'].append(str(rois[roi_number]))
    dummy_file_dict2['x'].append(coords[roi_number][0])
    dummy_file_dict2['y'].append(coords[roi_number][1])
    dummy_file_dict2['z'].append(coords[roi_number][2])
    dummy_file_dict[str(rois[roi_number])] = (
        coords[roi_number][0], coords[roi_number][1], coords[roi_number][2])

ranks13 = np.load('extract_from_cluster/ranks13.npy')
# high_ranks_file_dict = {}
high_ranks_file_dict2 = {}
high_ranks_file_dict2['Name'] = []
high_ranks_file_dict2['x'] = []
high_ranks_file_dict2['y'] = []
high_ranks_file_dict2['z'] = []
high_ranks_file_dict2['rank'] = []
high_ranks_file_dict2['Distance'] = []
high_oracle_roi_idxs = np.where(ranks13 > 0.9)[0]
for roi_number in high_oracle_roi_idxs:
    high_ranks_file_dict2['Name'].append(str(rois[roi_number]))
    high_ranks_file_dict2['x'].append(coords[roi_number][0])
    high_ranks_file_dict2['y'].append(coords[roi_number][1])
    high_ranks_file_dict2['z'].append(coords[roi_number][2])
    high_ranks_file_dict2['rank'].append(ranks13[roi_number])
    high_ranks_file_dict2['Distance'].append(
        distance((coords[roi_number][0], coords[roi_number][1]), (256, 256)))
    # high_ranks_file_dict[str(rois[roi_number])] = (coords[roi_number][0], coords[roi_number][1], coords[roi_number][2])


high_ranks_dataframe = pd.DataFrame.from_dict(
    high_ranks_file_dict2).sort_values('Distance')
with open('session'+str(sess_id)+'_high_rank_selected_neurons.pkl', 'wb') as out_file:
    pickle.dump(high_ranks_dataframe, out_file)

low_ranks_file_dict2 = {}
low_ranks_file_dict2['Name'] = []
low_ranks_file_dict2['x'] = []
low_ranks_file_dict2['y'] = []
low_ranks_file_dict2['z'] = []
low_ranks_file_dict2['rank'] = []
low_ranks_file_dict2['Distance'] = []
high_oracle_roi_idxs = np.where(ranks13 < 0.1)[0]
for roi_number in high_oracle_roi_idxs:
    low_ranks_file_dict2['Name'].append(str(rois[roi_number]))
    low_ranks_file_dict2['x'].append(coords[roi_number][0])
    low_ranks_file_dict2['y'].append(coords[roi_number][1])
    low_ranks_file_dict2['z'].append(coords[roi_number][2])
    low_ranks_file_dict2['rank'].append(ranks13[roi_number])
    low_ranks_file_dict2['Distance'].append(
        distance((coords[roi_number][0], coords[roi_number][1]), (256, 256)))
    # low_ranks_file_dict[str(rois[roi_number])] = (coords[roi_number][0], coords[roi_number][1], coords[roi_number][2])


low_ranks_dataframe = pd.DataFrame.from_dict(
    low_ranks_file_dict2).sort_values('Distance')
with open('session'+str(sess_id)+'_low_rank_selected_neurons.pkl', 'wb') as out_file:
    pickle.dump(low_ranks_dataframe, out_file)

w = csv.writer(open('session'+str(sess_id) +
               '_selected_neurons_dummy_file.csv', 'w'))
# loop over dictionary keys and values
for key, val in dummy_file_dict.items():
    w.writerow([key, val])

dataframe = pd.DataFrame.from_dict(dummy_file_dict2)
print(dataframe)

with open('session'+str(sess_id)+'_selected_neurons_dummy_file.pkl', 'wb') as out_file:
    pickle.dump(dataframe, out_file)
