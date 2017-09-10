"""
Functions to perform data cleaning and transformation on training and
test data for modelling.

Notes:
    EEZ is defined as 200 nautical miles or 370 km
"""

import os


import numpy as np
import pandas as pd
import time

from utils.parallel import *
from src.features.distance_calculations import *
from src.features.bearing_calculations import *


def clean_data(df):
    """
    Clean data for modelling and generate extra features
    including distance to shore and distance to port.
    """
    # removes first row time = 0
    df.dropna(how='any', inplace=True)
    df.drop_duplicates(inplace=True)

    # remove points where speed is greater than > 100 mph
    df = df[df['speed'] < 100]

    # replace unknowns with mean of column
    df = df.replace(-99999, np.nan)
    df = df.fillna(df.mean())
    return df

def generate_features(df):
    """
    Clean data for modelling and generate extra features
    including distance to shore and distance to port.
    """

    # time since last message
    df['time_diff'] = df['Time(seconds)'].diff()

    # generate course/bearing
    df = generate_course(df)
    df['course_diff'] = df['course'].diff()

    df = df[((df['Longitude'] > -180) & (df['Longitude'] < 180)) &
            ((df['Latitude'] > -90) & (df['Latitude'] < 90))]

    # generate distance travelled
    df = generate_distance(df)

    # calculated speed since last point in miles per hour
    df['speed'] = df['distance'] / (df['time_diff']/3600)

    # generate distance_to_shore, country
    coastline = load_coastline_coordinates()
    shore_dist = distance_to_shore(df['Longitude'], df['Latitude'], coastline)
    shore_dist.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, shore_dist], axis=1)

    # generate distance_to_port
    ports = load_ports_coordinates()
    df['distance_to_port'] = distance_to_port(df['Longitude'], df['Latitude'],
                                              ports)
    # generate is in_eez
    df['in_eez'] = np.where(df['distance_to_shore']<=370, 1, 0)

    return df

def preprocess_data(vessel_number):
    """
    perform data cleaning and feature generation
    """
    df = pd.read_csv(os.environ['DATA_FOLDER'] + "vessel_tracks/" +
                     str(vessel_number))
    try:
        print("processing " + str(vessel_number))
        df = generate_features(df)
        df = clean_data(df)
        df.to_csv((os.environ['PROJECT_FOLDER'] + "src/data/vessel_tracks/"
                         + str(vessel_number)), index=False)
    except:
        print("failed to process " + str(vessel_number))


if __name__ == '__main__':
    #111688, 112630 null vessels
    input_dir = str(os.environ['DATA_FOLDER'] + "vessel_tracks/")
    parallelise_files(input_dir, preprocess_data)
