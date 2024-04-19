from scipy.signal import find_peaks
import numpy as np
from skimage.feature import peak_local_max

def event_numbers(data,threshold,max_distance):    
    peaks, _=find_peaks(data, height=threshold,distance=max_distance)

    return data.iloc[peaks]


def make_firing_rate_maps(data,num_rounds,num_units,num_bins):    
    # Initialize a 3-dimensional array to store firing rate maps for each cell and round
    firing_rate_maps = np.zeros((int(num_units), num_rounds, num_bins))

    for cell in range(num_units):
        for round_num in range(1,num_rounds + 1):
            round_data = data[data['Rounds'] == round_num]  # Filter data for the current round and cell
            
            # Calculate histogram for the current cell and round with specified number of bins
            hist, _ = np.histogram(round_data['Distance'], bins=num_bins, weights=round_data[cell])
            
            # Calculate histogram of count of activations in each bin
            count_hist, _ = np.histogram(round_data['Distance'], bins=num_bins)
            
            # Calculate average firing rate map for the current cell and round
            avg_firing_rate_map = np.divide(hist, count_hist, out=np.zeros_like(hist), where=(count_hist != 0))
            avg_firing_rate_map[np.isnan(avg_firing_rate_map)] = 0
            
            # Store the average firing rate map in the firing_rate_maps array
            firing_rate_maps[cell, round_num - 1] = avg_firing_rate_map

    return firing_rate_maps

#taking the zscore flurorescence. finding the peaks in it and making a binary panda frame out of it.
#meaning with 0 and 1. zero is events and 1 is events

# def make_binary(data,peak_threshold=3,peak_distance=10):
#         #the data file will be the the firing rate map of every cell
#         data_binarized=np.zeros_like(data) #making a copy of the original file
#         for unit in range(data.shape[0]):
#             peaks_all_data=peak_local_max(data[unit],threshold_abs=peak_threshold,min_distance=peak_distance,exclude_border=False)
#             data_binarized[unit][peaks_all_data[:, 0], peaks_all_data[:, 1]] = 1
#         return data_binarized

def make_binary(data,peak_threshold=3,peak_distance=10):
        #the data file will be the the firing rate map of every cell
        data_binarized=np.zeros_like(data) #making a copy of the original file
        #iterating through cells
        for unit in range(data.shape[0]):
            for round in range(data.shape[1]):
                 data_per_round=data[unit][round]
                 peaks, _=find_peaks(data_per_round, height=peak_threshold,distance=peak_distance)
                 data_binarized[unit][round][peaks]=1
        return data_binarized
    
#def calcium_trace(data,cell_number):