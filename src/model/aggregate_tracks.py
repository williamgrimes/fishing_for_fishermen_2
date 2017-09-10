"""
functions to aggregate tracks into single sample of features
"""
import os
import numpy as np
import pandas as pd 

import time

from utils.parallel import *
from src.features.distance_calculations import *


def aggregate_track(track):
    """
    data aggregation methods for each vessel track to combine
    to a single row.
    """
    aggregations = {
        'TrackNumber': {
    	'count': 'count'
        },
        'Time(seconds)': {
    	'time_duration': lambda x: max(x) - min(x)
        },
        'time_diff': {
    	'time_diff_median': 'median',
    	'time_diff_std': 'std',
    	'time_diff_mad': 'mad'
        },
        'distance': {
    	'distance_median': 'median',
    	'distance_max': 'max',
    	'distance_std': 'std',
    	'distance_mad': 'mad'
        },
        'distance_to_shore': {
    	'distance_to_shore_median': 'median',
    	'distance_to_shore_max': 'max',
    	'distance_to_shore_std': 'std',
    	'distance_to_shore_mad': 'mad'
        },
        'distance_to_port': {
    	'distance_to_port_median': 'median',
    	'distance_to_port_max': 'max',
    	'distance_to_port_std': 'std',
    	'distance_to_port_mad': 'mad'
        },
        'in_eez': {
    	'ratio_in_eez': lambda x: x.sum() / x.count()
        },
        'course': {
    	'course_std': 'std',
    	'course_mad': 'mad'
        },
        'course_diff': {
    	'course_diff_std': 'std',
    	'course_diff_mad': 'mad'
        },
        'speed': {
    	'speed_median': 'median',
    	'speed_std': 'std',
    	'speed_mad': 'mad'
        },
        'SOG': {
    	'sog_median': 'median',
    	'sog_std': 'std',
    	'sog_mad': 'mad'
        },
        'Oceanic Depth': {
    	'oceanic_depth_median': 'median',
    	'oceanic_depth_std': 'std',
    	'oceanic_depth_mad': 'mad'
        },
        'Chlorophyll Concentration': {
    	'choloro_conc_median': 'median',
    	'choloro_conc_std': 'std',
    	'choloro_conc_mad': 'mad'
        },
        'Salinity': {
    	'salinity_median': 'median',
    	'salinity_std': 'std',
    	'salinity_mad': 'mad'
        },
        'Water Surface Elevation': {
    	'water_surface_elevation_median': 'median',
    	'water_surface_elevation_std': 'std',
    	'water_surface_elevation_mad': 'mad'
        },
        'Sea Temperature': {
    	'sea_temperature_median': 'median',
    	'sea_temperature_std': 'std',
    	'sea_temperature_mad': 'mad'
        },
        'Thermocline Depth': {
    	'thermo_depth_median': 'median',
    	'thermo_depth_std': 'std',
    	'thermo_depth_mad': 'mad'
        },
        'Eastward Water Velocity': {
    	'east_water_velocity_median': 'median',
    	'east_water_velocity_std': 'std',
  	'east_water_velocity_mad': 'mad'
        },
        'Northward Water Velocity': {
    	'north_water_velocity_median': 'median',
    	'north_water_velocity_std': 'std',
    	'north_water_velocity_mad': 'mad'
        }
    }
    track = track.groupby('TrackNumber').agg(aggregations).reset_index()
    track = track.drop('TrackNumber', axis=1)
    track.columns = track.columns.get_level_values(1)
    return track

def aggregate_training_data():
    """
    apply aggregations to training data and generate a csv of
    training.csv for results to be used in modelling. this iterates
    over all the vessel trajectories aggregates and appends to csv.
    A try catch is implemented since some of the vessel tracks
    contain anomalous data inputs and are ignored.
    """
    training = pd.read_csv(os.environ['DATA_FOLDER'] + "training.txt",
                           names = ['vessel_number', 'vessel_type'])

    for index, row in training.iterrows():
        vessel_number = row.vessel_number
        print('processing training vessel ' + str(vessel_number))
        vessel_type = row.vessel_type
        track = pd.read_csv((os.environ['PROJECT_FOLDER'] +
                            "src/data/vessel_tracks/" +
                            str(vessel_number) + ".csv"))

        df = aggregate_track(track)

        vessel_number = pd.Series(vessel_number, name = 'vessel_number')
        vessel_type = pd.Series(vessel_type, name = 'vessel_type')
        df = pd.concat([vessel_type, vessel_number, df], axis=1)
        if os.path.isfile((os.environ['PROJECT_FOLDER'] +
                          'src/data/training.csv')) == False:
            df.to_csv((os.environ['PROJECT_FOLDER'] + 'src/data/training.csv'),
                      index=False, mode='w')
        else:
            df.to_csv((os.environ['PROJECT_FOLDER'] + 'src/data/training.csv'),
                      index=False, mode='a', header=False)


def aggregate_testing_data():
    """
    apply aggregations to testing data and generate a csv of
    testing.csv for results to be used in modelling. this iterates
    over all the vessel trajectories aggregates and appends to csv.
    A try catch is implemented since some of the vessel tracks
    contain anomalous data inputs and are ignored.
    """
    testing = pd.read_csv(os.environ['DATA_FOLDER'] + "testing.txt",
                           names = ['vessel_number'])

    for index, row in testing.iterrows():
        vessel_number = row.vessel_number
        print('processing testing vessel ' + str(vessel_number))
        track = pd.read_csv((os.environ['PROJECT_FOLDER'] +
                             "src/data/vessel_tracks/" +
                             str(vessel_number) + ".csv"))

        df = aggregate_track(track)

        vessel_number = pd.Series(vessel_number, name = 'vessel_number')
        df = pd.concat([vessel_number, df], axis=1)

        if os.path.isfile((os.environ['PROJECT_FOLDER'] +
                          'src/data/testing.csv')) == False:
            df.to_csv((os.environ['PROJECT_FOLDER'] + 'src/data/testing.csv'),
                      index=False, mode='w')
        else:
            df.to_csv((os.environ['PROJECT_FOLDER'] + 'src/data/testing.csv'),
                      index=False, mode='a', header=False)

def fill_missing_values():
    """
    where data missing fill with mean of means for row
    """
    training = pd.read_csv(os.environ['PROJECT_FOLDER'] +
                            "src/data/training.csv")
    training = training.fillna(training.mean())
    training.to_csv((os.environ['PROJECT_FOLDER'] + 'src/data/training.csv'),
              index=False)

    testing = pd.read_csv(os.environ['PROJECT_FOLDER'] +
                           "src/data/testing.csv")
    testing = testing.fillna(testing.mean())
    testing.to_csv((os.environ['PROJECT_FOLDER'] + 'src/data/testing.csv'),
              index=False)


if __name__ == '__main__':
    aggregate_training_data()
    aggregate_testing_data()
    fill_missing_values()
