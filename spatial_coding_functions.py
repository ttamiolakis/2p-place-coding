def event_numbers(data,threshold,max_distance):
    from scipy.signal import find_peaks
    peaks, _=find_peaks(data, height=threshold,distance=max_distance)

    return data.iloc[peaks]

def make_firing_rate_maps(data,num_rounds,num_units,num_bins):
    import numpy as np
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
