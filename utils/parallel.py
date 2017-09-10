"""
Functions to manipulate data in parallel.
"""
import os
import numpy as np
import pandas as pd
import time

from multiprocessing import Pool, cpu_count
from os import listdir

def parallelise_dataframe(df, func, num_cores=None):
    '''
    Perform function in parallel on pandas data frame where if the
    num_cores is not specified then use the number of available
    cores -1.
    Arguments:
    df (dataframe to manipulate)
    func (function to apply)
    num_cores (number of cores to parallelise)
    Returns:
    The data frame processed by function in parallel.
    '''
    if num_cores==None:
        num_cores = cpu_count() - 1
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def parallelise_files(input_dir, func, num_cores=None):
    '''
    Perform function in parallel over files where if the
    num_cores is not specified then use the number of available
    cores -1.
    Arguments:
    input_dir: directory of input files
    func (function to apply)
    num_cores (number of cores to parallelise)
    '''
    if num_cores==None:
        num_cores = cpu_count() - 1
    p = Pool(num_cores)
    p.map(func, sorted(listdir(input_dir)))
