# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 28.
"""
import pandas as pd
import numpy as np

from data.data_reader import download_csv
from data.data_processor import CHAMPION_SIZE
from stats.distance_calculator import COLLABORATIVE_DISTANCES_PATH, COMPETITIVE_DISTANCES_PATH

# TODO: Not yet calculated. Updated soon.
REMOTE_COLLABORATIVE_DISTANCES_URL = ''
REMOTE_COMPETITIVE_DISTANCES_URL = ''


def get_collaborative_distances():
    """

    :return collaborative_distances: (ndarray[float])
    """
    # Check COLLABORATIVE_DISTANCES_PATH.

    # If COLLABORATIVE_DISTANCES_PATH exists, read csv.

    # Else, download collaborative_distances from REMOTE_COLLABORATIVE_DISTANCES_URL, and save it to
    # COLLABORATIVE_DISTANCES_PATH.

    # Turn it to a CHAMPION_SIZE * CHAMPION_SIZE ndarray.

    # Return collaborative_distances


def get_competitive_distances():
    """

    :return competitive_distances: (ndarray[float])
    """
    # Check COMPETITIVE_DISTANCES_PATH.

    # If COMPETITIVE_DISTANCES_PATH exists, read csv.

    # Else, download competitive_distances from REMOTE_COMPETITIVE_DISTANCES_URL, and save it to
    # COMPETITIVE_DISTANCES_PATH.

    # Turn it to a CHAMPION_SIZE * CHAMPION_SIZE ndarray.

    # Return competitive_distances


if __name__ == '__main__':
    print(get_collaborative_distances())
    print(get_competitive_distances())
