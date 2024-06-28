import h5py
import numpy as np
import pandas as pd

class CellTrackingSingleAnimal:
    def __init__(self):
        pass

    def filter_cross_registered_place_cells(df, sessions, place_cell_arrays):
        if not sessions:
            return df  # Return the original DataFrame if no sessions are specified
        
        mask = df[sessions[0]].isin(place_cell_arrays[sessions[0]][0])
        for session in sessions[1:]:
            mask &= df[session].isin(place_cell_arrays[session][0])
        
        filtered_df = df[mask]
        return filtered_df[sessions]

    def cell_movement(df,source_condition,targ_condition,source_type,target_type):
        
        mask = df[source_condition].isin(source_type[source_condition][0])
        mask &= df[targ_condition].isin(target_type[targ_condition][0])

        filtered_df = df[mask]
        return filtered_df[[source_condition,targ_condition]]
    

class CellTrackingMultipleAnimals:
    def _init_(self):
        pass

    def cell_movement(df,animal,source_condition,targ_condition,source_type,target_type):
        mask = df[source_condition].isin(source_type[f'{animal}_{source_condition}'][0])
        mask &= df[targ_condition].isin(target_type[f'{animal}_{targ_condition}'][0])

        filtered_df = df[mask]
        return filtered_df[[source_condition,targ_condition]]