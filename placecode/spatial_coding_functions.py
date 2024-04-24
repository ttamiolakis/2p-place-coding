from scipy.signal import find_peaks
import numpy as np
from skimage.feature import peak_local_max
import random
from scipy.stats import zscore, kstest
import h5py
import pandas as pd
import scipy.sparse


def event_numbers(data, threshold, max_distance):
    peaks, _ = find_peaks(data, height=threshold, distance=max_distance)

    return data.iloc[peaks]


def make_firing_rate_maps(data, num_rounds, num_units, num_bins):
    # Initialize a 3-dimensional array to store firing rate maps for each cell and round
    firing_rate_maps = np.zeros((int(num_units), num_rounds, num_bins))

    for cell in range(num_units):
        for round_num in range(1, num_rounds + 1):
            # Filter data for the current round and cell
            round_data = data[data['Rounds'] == round_num]

            # Calculate histogram for the current cell and round with specified number of bins
            hist, _ = np.histogram(
                round_data['Distance'], bins=num_bins, weights=round_data[cell])

            # Calculate histogram of count of activations in each bin
            count_hist, _ = np.histogram(round_data['Distance'], bins=num_bins)

            # Calculate average firing rate map for the current cell and round
            avg_firing_rate_map = np.divide(
                hist, count_hist, out=np.zeros_like(hist), where=(count_hist != 0))
            avg_firing_rate_map[np.isnan(avg_firing_rate_map)] = 0

            # Store the average firing rate map in the firing_rate_maps array
            firing_rate_maps[cell, round_num - 1] = avg_firing_rate_map

    return firing_rate_maps

# taking the zscore flurorescence. finding the peaks in it and making a binary panda frame out of it.
# meaning with 0 and 1. zero is events and 1 is events

# def make_binary(data,peak_threshold=3,peak_distance=10):
#         #the data file will be the the firing rate map of every cell
#         data_binarized=np.zeros_like(data) #making a copy of the original file
#         for unit in range(data.shape[0]):
#             peaks_all_data=peak_local_max(data[unit],threshold_abs=peak_threshold,min_distance=peak_distance,exclude_border=False)
#             data_binarized[unit][peaks_all_data[:, 0], peaks_all_data[:, 1]] = 1
#         return data_binarized


def make_binary(data, peak_threshold=3, peak_distance=10):
    # the data file will be the the firing rate map of every cell
    data_binarized = np.zeros_like(data)  # making a copy of the original file
    # iterating through cells
    for unit in range(data.shape[0]):
        for round in range(data.shape[1]):
            data_per_round = data[unit][round]
            peaks, _ = find_peaks(
                data_per_round, height=peak_threshold, distance=peak_distance)
            data_binarized[unit][round][peaks] = 1
    return data_binarized


def adding_parameters(zscore_fluo_pd, raw_fluo_pd, param_file):

    # panda frame for time
    time_hdf = h5py.File(param_file)['inferred']['belt_dict']['tsscn']
    time_hdf = pd.DataFrame(time_hdf)
    time_hdf.columns = ['Time (ms)']
    # panda frame for distance
    distance_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['distance']
    distance_hdf = pd.DataFrame(distance_hdf)
    distance_hdf.columns = ['Distance']
    # panda frame for speed
    speed_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['speed']
    speed_hdf = pd.DataFrame(speed_hdf)
    speed_hdf.columns = ['Speed']
    # panda frame for number of rounds
    rounds_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['rounds']
    rounds_hdf = pd.DataFrame(rounds_hdf)
    rounds_hdf.columns = ['Rounds']
    rounds_hdf = rounds_hdf.astype(int)
    # panda frame for running(yes or no running)
    running_hdf = h5py.File(param_file)['inferred']['belt_scn_df']['running']
    running_hdf = pd.DataFrame(running_hdf)
    running_hdf.columns = ['Running']
    running_hdf = running_hdf.astype(int)

    #####################################################################################################################

    # adding all the parameters in one panda frame for z score and raw data
    zscore_fluo_pd = pd.concat([zscore_fluo_pd, time_hdf, distance_hdf,
                               speed_hdf, rounds_hdf, running_hdf], axis=1, ignore_index=True)
    raw_fluo_pd = pd.concat([raw_fluo_pd, time_hdf, distance_hdf,
                            speed_hdf, rounds_hdf, running_hdf], axis=1, ignore_index=True)
    # Create a mapping dictionary for column renaming
    rename_mapping = {old_col: new_col for old_col, new_col in zip(
        zscore_fluo_pd.columns[-5:], ['Time (ms)', 'Distance', 'Speed', 'Rounds', 'Running'])}
    # Rename the columns
    zscore_fluo_pd = zscore_fluo_pd.rename(columns=rename_mapping)
    raw_fluo_pd = raw_fluo_pd.rename(columns=rename_mapping)

    return zscore_fluo_pd, raw_fluo_pd


def cell_morphology(dataset):
    # opening the hpf5 file
    hdf = h5py.File(dataset)
    # defining the spatial parameters
    A_data = hdf['estimates']['A']['data']
    A_indices = hdf['estimates']['A']['indices']
    A_indptr = hdf['estimates']['A']['indptr']
    A_shape = hdf['estimates']['A']['shape']
    # number of neurons
    n_neurons = len(hdf['estimates']['C'])
    spatial = scipy.sparse.csc.csc_matrix(
        (A_data, A_indices, A_indptr), shape=A_shape).todense()
    spatial = np.array(spatial)  # change type to allow np.reshape (?)
    # spatial = np.reshape(spatial[:,cell_number], (512, 512)) # (262144 -> 512x512, i.e. "unflatten")

    return spatial


# def ks_test_analysis(data,data_avg=None,n_shuffles=None,num_rounds=None,num_bins=None):
#     shuffled_ks=[] #array where I will put the ks distances where I will compare the shuffled data with my first shuffling
#     baseline=data.copy()

#     #shuffle 1 for the baseline
#     for i in range(num_rounds):
#         shuf=random.randint(1,150)
#         baseline[i]=np.roll(baseline[i],shuf)
#     baseline_avg=np.mean(baseline,axis=0)

#     baseline_ks,_=kstest(data_avg,baseline_avg)


#     # now I will shuffle many times and then compare

#     for n in range(1,shuffling_times):
#         data_shuffle=data.copy()
#         for i in range(num_rounds):
#             shuf=random.randint(1,150)
#             data_shuffle[i]=np.roll(data_shuffle[i],shuf)


#         data_shuffle=np.mean(data_shuffle,axis=0)
#         ks_shuffle,p_value_=kstest(baseline_avg,data_shuffle)
#         shuffled_ks.append(ks_shuffle)
