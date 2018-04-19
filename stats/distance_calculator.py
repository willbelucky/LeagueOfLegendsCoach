# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 28.
"""

import numpy as np
import pandas as pd

from data.data_processor import OTHER_TEAM_WINS_CSV, OTHER_TEAM_MATCHES_CSV, SAME_TEAM_WINS_CSV, \
    SAME_TEAM_MATCHES_CSV, CHAMPION_SIZE

# Remote and local file paths
COLLABORATIVE_DISTANCE_PATH = "stats/collaborative_distances.csv"
COMPETITIVE_DISTANCE_PATH = "stats/competitive_distances.csv"


def calculate_distances(team_matches_csv, team_wins_csv):
    team_matches = pd.read_csv(team_matches_csv).as_matrix()
    team_wins = pd.read_csv(team_wins_csv).as_matrix()
    team_wins += 1e-10
    # If there is no game, set the winning probability 50%.
    winning_probability = np.divide(team_wins, team_matches)
    winning_probability[np.where(np.isnan(winning_probability))] = 0.5
    winning_probability[np.where(np.isinf(winning_probability))] = 0.5
    losing_probability = 1 - winning_probability
    # Calculate the distances between collaborative players.
    distances = get_distances(losing_probability)
    distances = pd.DataFrame(np.array(distances).reshape(CHAMPION_SIZE, CHAMPION_SIZE))
    # Return stats/collaborative_distance.csv as a 690 * 690 numpy.ndarray.
    distances = distances.as_matrix()
    return distances


def get_distances(losing_probability):
    return 4 * np.abs(losing_probability - 0.5) * (losing_probability - 0.5) + 1


def calculate_collaborative_distances():
    """

    :return distances: (ndarray[float]) 690 * 690 matrix of collaborative distances.
    """
    distances = calculate_distances(SAME_TEAM_MATCHES_CSV, SAME_TEAM_WINS_CSV)
    return distances


def calculate_competitive_distances():
    """

    :return distances: (ndarray[float]) 690 * 690 matrix of competitive distances.
    """
    distances = calculate_distances(OTHER_TEAM_MATCHES_CSV, OTHER_TEAM_WINS_CSV)
    return distances


if __name__ == '__main__':
    pd.DataFrame(calculate_collaborative_distances()).to_csv('stats/collaborative_distances.csv', header=False, index=False)
    pd.DataFrame(calculate_competitive_distances()).to_csv('stats/competitive_distances.csv', header=False, index=False)

