# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 28.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data.data_processor import OTHER_TEAM_WINS_CSV, OTHER_TEAM_MATCHES_CSV, SAME_TEAM_WINS_CSV, \
    SAME_TEAM_MATCHES_CSV, CHAMPION_SIZE
from data.data_reader import ENCODING

# Remote and local file paths
COLLABORATIVE_DISTANCE_PATH = "stats/collaborative_distances.csv"
COMPETITIVE_DISTANCE_PATH = "stats/competitive_distances.csv"


def calculate_collaborative_distances():
    """

    :return distances: (ndarray[float]) 690 * 690 matrix of collaborative distances.
    """
    # Check stats/collaborative_distance.csv.
    if Path(COLLABORATIVE_DISTANCE_PATH).exists():
        # If stats/collaborative_distance.csv is exists,
        # return stats/collaborative_distance.csv as a 690 * 690 numpy.ndarray.
        distances = pd.read_csv(COLLABORATIVE_DISTANCE_PATH, low_memory=False, encoding=ENCODING)
    else:
        # If there is no stats/collaborative_distance.csv, do below codes.
        # Get matrix of matches and wins.
        team_matches = pd.read_csv(SAME_TEAM_MATCHES_CSV).as_matrix()
        team_wins = pd.read_csv(SAME_TEAM_WINS_CSV).as_matrix()

        # If there are few games, set the losing probability almost 100%,
        # because there are some reasons they didn't choose that combination.
        team_matches[np.where(team_matches <= 10)] = 10000

        losing_probability = 1 - np.divide(team_wins, team_matches)

        # Calculate the distances between collaborative players.
        distances = 4 * np.abs(losing_probability - 0.5) * (losing_probability - 0.5) + 1
        distances = pd.DataFrame(np.array(distances).reshape(CHAMPION_SIZE, CHAMPION_SIZE))

        # Save the result to stats/collaborative_distance.csv.
        distances.to_csv(COLLABORATIVE_DISTANCE_PATH, header=False, index=False)

    # Return stats/collaborative_distance.csv as a 690 * 690 numpy.ndarray.
    distances = distances.as_matrix()
    return distances


def calculate_competitive_distances():
    """

    :return distances: (ndarray[float]) 690 * 690 matrix of competitive distances.
    """
    # Check stats/competitive_distance.csv.
    if Path(COMPETITIVE_DISTANCE_PATH).exists():
        # If stats/competitive_distance.csv is exists,
        # return stats/competitive_distance.csv as a 690 * 690 numpy.ndarray.
        distances = pd.read_csv(COMPETITIVE_DISTANCE_PATH, low_memory=False, encoding=ENCODING)
    else:
        # If there is no stats/competitive_distance.csv, do below codes.
        # Get matrix of matches and wins.
        team_matches = pd.read_csv(OTHER_TEAM_MATCHES_CSV).as_matrix()
        team_wins = pd.read_csv(OTHER_TEAM_WINS_CSV).as_matrix()

        # If there are few games, set the losing probability almost 100%,
        # because there are some reasons they didn't choose that combination.
        team_matches[np.where(team_matches <= 10)] = 10000

        losing_probability = 1 - np.divide(team_wins, team_matches)

        # Calculate the distances between competitive players.
        distances = 4 * np.abs(losing_probability - 0.5) * (losing_probability - 0.5) + 1
        distances = pd.DataFrame(np.array(distances).reshape(CHAMPION_SIZE, CHAMPION_SIZE))

        # Save the result to stats/competitive_distance.csv.
        distances.to_csv(COMPETITIVE_DISTANCE_PATH, header=False, index=False)

    # Return stats/competitive_distance.csv as a 690 * 690 numpy.ndarray.
    distances = distances.as_matrix()
    return distances


if __name__ == '__main__':
    print(calculate_collaborative_distances())
    print(calculate_competitive_distances())
