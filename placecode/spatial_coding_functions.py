import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy.sparse
from typing import Tuple
from numpy.typing import ArrayLike


def vector_sum(radii: ArrayLike, angles: ArrayLike) -> Tuple[float, float]:
    """Given an 1D array of vector lengths and an 1D array of corresponding radian angles,
    return the vectorial sum as a tuple (radius, radian angle).

    Parameters
    ----------
    angles : np.array
        An 1D array of angles (radian).
    radii : np.array
        An 1D array of vector lengths.
    Returns
    -------
    tuple(float, float)
        The radius and angle of the sum vector.
    Raises
    -------
    ValueError: if input parameters are invalid
    """
    if len(angles) != len(radii):  # passed arrays of different lenghts, or both empty
        raise ValueError(f"Shape of angles and radii do not match.")
    elif len(angles) == 0:
        raise ValueError(f"Empty input arrays detected.")
    x = np.sum(radii*np.cos(angles))  # sum up x components
    y = np.sum(radii*np.sin(angles))
    angle = np.arctan2(y, x)
    radius = np.sqrt(x**2 + y**2)
    return (radius, angle)


def event_numbers(data, threshold, max_distance):
    peaks, _ = find_peaks(data, height=threshold, distance=max_distance)
    return data.iloc[peaks]


def make_firing_rate_maps(data,rounds, num_units, num_bins):
    # Initialize a 3-dimensional array to store firing rate maps for each cell and round
    firing_rate_maps = np.zeros((int(num_units), len(rounds), num_bins))

    for cell in range(num_units):        
        for round_num in rounds:
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


def make_binary(data, peak_threshold, peak_distance):
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
    distance_hdf = h5py.File(param_file)['inferred']['belt_scn_dict']['distance']
    distance_hdf = pd.DataFrame(distance_hdf)
    distance_hdf.columns = ['Distance']
    # panda frame for speed
    speed_hdf = h5py.File(param_file)['inferred']['belt_scn_dict']['speed']
    speed_hdf = pd.DataFrame(speed_hdf)
    speed_hdf.columns = ['Speed']
    # panda frame for number of rounds
    rounds_hdf = h5py.File(param_file)['inferred']['belt_scn_dict']['rounds']
    rounds_hdf = pd.DataFrame(rounds_hdf)
    rounds_hdf.columns = ['Rounds']
    rounds_hdf = rounds_hdf.astype(int)
    # panda frame for running(yes or no running)
    running_hdf = h5py.File(param_file)['inferred']['belt_scn_dict']['running']
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


def read_spatial(A_data, A_indices, A_indptr, A_shape, n_components, resolution, unflatten: bool = False) -> np.array:
    """Given the numpy arrays data, indices, indptr, shape, read the sparse encoded spatial component data and
    reshape it into (n_components, resolution_x, resolution_y)

    Parameters
    ----------
    A_data : np.array
        The data field of the sparse encoding
    A_indices : np.array
        The indices field of the sparse encoding
    A_indptr : np.array
        The indptr field of the sparse encoding
    A_shape : np.array
        The shape field of the sparse encoding
    n_components : int
        the number of components in the CaImAn data
    resolution : tuple(int, int), or [int, int], or np.array(shape=(2,), dtype=dtype("int32"))
        the resolution of the 2p recording. It should be read out from CaImAn dims.
    unflatten : bool
        default: False. If True, the individual spatial components will be converted into 2d arrays. If False,
        left as 1d/flat numpy arrays.
    Returns
    -------
    np.array of shape (n_components, resolution_x * resolution_y ) if unflatten=False, else (n_components, *resolution)
        The dense matrix form of the spatial components.
    """
    spatial = scipy.sparse.csc.csc_matrix(
        (A_data, A_indices, A_indptr), shape=A_shape).todense()  # returns array with dimensions (flat resolution, n_components)
    spatial = np.array(spatial)  # change type to numpy array
    # (262144 -> 512x512, i.e. "unflatten" along imaging resolution)
    spatial = np.swapaxes(spatial, 0, 1)
    if unflatten:
        # TODO: need to test if x and y are in correct order (for asymmetric resolution).
        spatial = np.reshape(spatial, (n_components, *resolution))
    return spatial


def filter_event_count(binary_events: np.array, n_events_threshold: int) -> np.array:
    """Get the indices of units in binary_events that showed events > n_events_threshold. 
    Parameters
    ----------
    binary_events : np.array(shape=(n_units, n_rounds_used, n_bins))
        a 3D numpy array containing binary firing count per cell per round per spatial bin
    n_events_threshold : int
        The number of firing events over the whole data one cell must strictly exceed to be accepted. 

    Returns
    -------
    np.array(shape=(variable, ))
    a 1D numpy array containing the indices of binary_events first axis that pass the threshold. 

    """
    # sum binary events over all rounds and all bins for each cell, compare to threshold
    # boolean array of shape (n_units, )
    cells_with_many_events = np.sum(
        binary_events, axis=(1, 2)) > n_events_threshold
    return np.where(cells_with_many_events)


def cell_morphology(dataset):
    # opening the hpf5 file
    with h5py.File(dataset, "r") as hdf:
        # defining the spatial parameters
        A_data = hdf['estimates']['A']['data']
        A_indices = hdf['estimates']['A']['indices']
        A_indptr = hdf['estimates']['A']['indptr']
        A_shape = hdf['estimates']['A']['shape']
        # number of neurons
        n_neurons = len(hdf['estimates']['C'])
        spatial = scipy.sparse.csc.csc_matrix(
            (A_data, A_indices, A_indptr), shape=A_shape).todense()
        spatial = np.array(spatial)  # change type to numpy array
        # spatial = np.reshape(spatial[:,cell_number], (512, 512)) # (262144 -> 512x512, i.e. "unflatten")
        spatial=np.reshape(spatial, (512, 512, n_neurons))
        return spatial


