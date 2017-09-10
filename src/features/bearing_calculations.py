import numpy as np
import pandas as pd
import math

from os import environ

def calculate_bearing(lat_1, lon_1, lat_2, lon_2):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - lat_1: latitude of first point must be in decimal degrees
      - lon_1: longitude of first point must be in decimal degrees
      - lat_2: latitude of second point must be in decimal degrees
      - lon_2: longitude of second point must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    lat_1 = math.radians(lat_1)
    lon_1 = math.radians(lon_1)
    lat_2 = math.radians(lat_2)
    lon_2 = math.radians(lon_2)

    diff_lon = lon_2 - lon_1

    x = math.sin(diff_lon) * math.cos(lat_2)
    y = math.cos(lat_1) * math.sin(lat_2) - (math.sin(lat_1)
            * math.cos(lat_2) * math.cos(diff_lon))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

def generate_course(df):
    """
    Calculates the bearing between consecutive points in pandas df.
    :Inputs:
      two lat, lon pairs
    :Returns:
      dataframe with course column
    """
    df = df.reset_index(drop=True)
    df['course'] = np.nan
    df_shift = df.shift()
    for index, row in df.iterrows():
        lat_1 = df.iloc[index].Latitude
        lon_1 = df.iloc[index].Longitude
        lat_2 = df_shift.iloc[index].Latitude
        lon_2 = df_shift.iloc[index].Longitude
        df.loc[index, 'course'] = calculate_bearing(lat_1, lon_1, lat_2, lon_2)
    return df
