import h5py
import numpy as np
import pandas as pd

class CellTrackingSingleAnimal:
    def __init__(self):
        pass

    def filter_cross_registered_place_cells(df, sessions, place_cell_arrays):

        ''''
        df: panda frame representing cross registered cells
        sessions: represent behavioral condition eg pre or post stim
        place_cell_arrays: defines which cells are place cells
        '''

        '''
        the functions finds the cross registered cells between sessions which I provide that
        are also place cells
        
        '''
        if not sessions:
            return df  # Return the original DataFrame if no sessions are specified
        
        mask = df[sessions[0]].isin(place_cell_arrays[sessions[0]])
        for session in sessions[1:]:
            mask &= df[session].isin(place_cell_arrays[sessions[0]])
        
        filtered_df = df[mask]
        return filtered_df[sessions]

    def cell_movement(df,source_condition,targ_condition,source_type,target_type):

        ''''
        df: panda frame representing cross registered cells
        source_condition: represents behavioral condition where the experiments took place eg pre stim
        targ_condition: represents behavioral condition where the experiments took place eg post stim

        source_type: which type of cell am I looking for in the initial/source condition eg place cell, silent cell
        target_type: which type of cell am I looking for in the final/target condition eg place cell, silent cell
        '''

        '''
        function finds specific transition between cross registered cell in different sessions eg place cell to non place cell
        non place cell to place cell etc
        
        '''
        
        mask = df[source_condition].isin(source_type[source_condition])
        mask &= df[targ_condition].isin(target_type[targ_condition])

        filtered_df = df[mask]
        return filtered_df[[source_condition,targ_condition]]
    
    def find_angle_between_sessions(theta,max_indices):
        '''
        theta: array of evenly spaced values representing the 150 bins of activation so I can plot them in a circle
        max_indices are the indices of maximum activation 
        '''
        '''
        find the angle between lines that connect the center of  a circle to the place of maximum activation of a plce cell
        it is used as a metric for representational drift
        '''
        theta1 = theta[max_indices[0]]
        theta2 = theta[max_indices[1]]
        angle_between_lines = np.abs(np.rad2deg(theta2 - theta1))
        if angle_between_lines > 180:
            angle_between_lines = 360 - angle_between_lines

        return angle_between_lines
    
    def isolating_cell_type(file,cell_type):

        '''
        isolating specific cell type from a specfic file
         Parameters:
        file (str): The path to the HDF5 file.
        cell_type (str): The dataset name corresponding to the cell type.

        Returns:
        np.ndarray: An array of the specified cell type per day.

        '''  

        with h5py.File(file, 'r') as f:
            cell_type_per_day = f[cell_type][:]
        return np.array(cell_type_per_day)
        # cell_type_per_day=h5py.File(file[cell_type])
        # cells_per_day=np.array(cell_type_per_day)
        # return cells_per_day





class CellTrackingMultipleAnimals:
    def _init_(self):
        pass

    def cell_movement(df,animal,source_condition,targ_condition,source_type,target_type):

        ''''
        df: panda frame representing cross registered cells
        animal: which animal I am cross registering from
        source_condition: represents behavioral condition where the experiments took place eg pre stim
        targ_condition: represents behavioral condition where the experiments took place eg post stim

        source_type: which type of cell am I looking for in the initial/source condition eg place cell, silent cell
        target_type: which type of cell am I looking for in the final/target condition eg place cell, silent cell
        '''

        '''
        function finds specific transition between cross registered cell in one specific animal in different sessions eg place cell to non place cell
        non place cell to place cell etc
        
        '''

        mask = df[source_condition].isin(source_type[f'{animal}_{source_condition}'][0])
        mask &= df[targ_condition].isin(target_type[f'{animal}_{targ_condition}'][0])

        filtered_df = df[mask]
        return filtered_df[[source_condition,targ_condition]]