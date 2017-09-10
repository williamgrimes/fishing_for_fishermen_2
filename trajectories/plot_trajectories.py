import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def plot_track(df, vessel_number, vessel_type):
    """
    plots vessel track, after doing data cleaning excluding vessels
    with a max speed beyond 100 mph based on distance and time since
    last signal. This reduces spurious AIS signals and should improve
    the modelling.

    this function will plot to a png file, with a tight bounding box
    in merc projection, using Nasa bluemarble colours.

    Arguments
    df: data frame of vessel tracks
    vessel_number: the vessel track number
    vessel_type: if known the vessel type, this determines the output folder
    """
    lons = df['Longitude'].tolist()
    lats = df['Latitude'].tolist()

    lat_min = np.min(lats, axis=0)
    lat_max = np.max(lats, axis=0)
    lon_min = np.min(lons, axis=0)
    lon_max = np.max(lons, axis=0)
    lat_buffer = max(abs(lat_max - lat_min)/10, 0.05)
    lon_buffer = max(abs(lon_max - lon_min)/10, 0.05)

    m = Basemap(llcrnrlon = lon_min - lon_buffer,
                llcrnrlat = lat_min - lat_buffer,
                urcrnrlon = lon_max + lat_buffer,
                urcrnrlat = lat_max + lat_buffer,
                lat_0 = (lat_max - lat_min)/2,
                lon_0 = (lon_max-lon_min)/2,
                projection = 'merc',
                resolution = 'l'
                )
    m.bluemarble()
    m.drawcoastlines()

    x, y = m(lons, lats)
    m.plot(x, y, linewidth = 0.3, color = 'w')

    plt.savefig(vessel_type + "/" + str(vessel_number) +'_traj.png',
                transparent = True,
                bbox_inches='tight'
               )
    plt.clf()

def plot_training_trajectories():
    """
    plot tracks of training data vessels where type is known
    """
    df = pd.read_csv(os.environ['DATA_FOLDER'] + "training.txt", header=None)
    for index, row in df.iterrows():
        vessel_number = row[0]
        vessel_type = row[1]
        df = pd.read_csv(os.environ['PROJECT_FOLDER'] +
                         "src/data/vessel_tracks/" +
                         str(vessel_number) + ".csv")
        try:
            plot_track(df, vessel_number, vessel_type)
            print("plotting vessel number " + str(vessel_number))
        except:
            print("failed to plot vessel number " + str(vessel_number))

def plot_testing_trajectories():
    """
    plot tracks of testing data vessels where type is unknown
    """
    df = pd.read_csv(os.environ['DATA_FOLDER'] + "testing.txt", header=None)
    for index, row in df.iterrows():
        vessel_number = row[0]
        vessel_type = 'unknown'
        df = pd.read_csv(os.environ['PROJECT_FOLDER'] +
                         "src/data/vessel_tracks/" +
                         str(vessel_number) + ".csv")
        try:
            plot_track(df, vessel_number, vessel_type)
            print("plotting vessel number " + str(vessel_number))
        except:
            print("failed to plot vessel number " + str(vessel_number))

if __name__ == "__main__":
    plot_training_trajectories()
    plot_testing_trajectories()
