# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 28.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_reader import download_csv, ENCODING

# TODO: Not yet calculated. Updated soon.
REMOTE_COLLABORATIVE_DISTANCES_URL = 'https://www.dropbox.com/s/wnojcqmpdojcanq/collaborative_distances.csv?dl=1'
REMOTE_COMPETITIVE_DISTANCES_URL = 'https://www.dropbox.com/s/af6xtcti5dxy0la/competitive_distances.csv?dl=1'

LOCAL_COLLABORATIVE_DISTANCES_DIR = 'stats/collaborative_distances.csv'
LOCAL_COMPETITIVE_DISTANCES_DIR = 'stats/competitive_distances.csv'


def get_collaborative_distances():
    """

    :return collaborative_distances: (ndarray[float]) 690 * 690 matrix of collaborative distances.
    """
    # Check COLLABORATIVE_DISTANCES_PATH.
    if Path(LOCAL_COLLABORATIVE_DISTANCES_DIR).exists():
        # If COLLABORATIVE_DISTANCES_PATH exists, read csv.
        collaborative_distances = pd.read_csv(LOCAL_COLLABORATIVE_DISTANCES_DIR, low_memory=False, encoding=ENCODING,
                                              header=None)
    else:
        # Else, download collaborative_distances from REMOTE_COLLABORATIVE_DISTANCES_URL, and save it to
        # COLLABORATIVE_DISTANCES_PATH.
        collaborative_distances = download_csv(REMOTE_COLLABORATIVE_DISTANCES_URL, LOCAL_COLLABORATIVE_DISTANCES_DIR)

    # Turn it to a CHAMPION_SIZE * CHAMPION_SIZE ndarray.
    collaborative_distances = collaborative_distances.as_matrix()

    # Return collaborative_distances
    return collaborative_distances


def get_competitive_distances():
    """

    :return competitive_distances: 690 * 690 matrix of competitive distances.
    """
    # Check COMPETITIVE_DISTANCES_PATH.
    if Path(LOCAL_COMPETITIVE_DISTANCES_DIR).exists():
        # If COMPETITIVE_DISTANCES_PATH exists, read csv.
        competitive_distances = pd.read_csv(LOCAL_COMPETITIVE_DISTANCES_DIR, low_memory=False, encoding=ENCODING,
                                            header=None)
    else:
        # Else, download competitive_distances from REMOTE_COMPETITIVE_DISTANCES_URL, and save it to
        # COMPETITIVE_DISTANCES_PATH.
        competitive_distances = download_csv(REMOTE_COMPETITIVE_DISTANCES_URL, LOCAL_COMPETITIVE_DISTANCES_DIR)

    # Turn it to a CHAMPION_SIZE * CHAMPION_SIZE ndarray.
    competitive_distances = competitive_distances.as_matrix()

    # Return competitive_distances
    return competitive_distances


if __name__ == '__main__':
    print(get_collaborative_distances())
    print(get_competitive_distances())
