from scipy.signal import find_peaks
import numpy as np
from skimage.feature import peak_local_max
import random
from scipy.stats import zscore,kstest

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
