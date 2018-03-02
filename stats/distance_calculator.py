# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 28.
"""
import pandas as pd
from numpy import genfromtxt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from data.data_processor import OTHER_TEAM_LOSES_CSV,OTHER_TEAM_MATCHES_CSV,OTHER_TEAM_WINS_CSV,SAME_TEAM_LOSES_CSV,SAME_TEAM_MATCHES_CSV,SAME_TEAM_WINS_CSV

from data.data_processor import get_same_team_matches, get_same_team_wins, get_same_team_loses

# Remote and local file paths
COLLABORATIVE_DISTANCE_PATH = "stats/collaborative_distance.csv"
COMPETITIVE_DISTANCE_PATH = "stats/competitive_distance.csv"

# CSV encoding type
ENCODING = 'utf-8'

CHAMPION_SIZE = 690


def calculate_collaborative_distance():
    """

    :return:
    """
    # Check /stats/distance.csv.

    # If /stats/distance.csv is exists, return /stats/distance.csv as a 690 * 690 numpy.ndarray.

    # If there is no /stats/distance.csv, do below codes.
    # Get matrix of matches, wins ans loses.

    # Calculate the distance between collaborative players.

    # Save the result to /stats/distance.csv.

    # Return /stats/distance.csv as a 680 * 680 numpy.ndarray.
    if Path(COLLABORATIVE_DISTANCE_PATH).exists():
        distance = pd.read_csv(COLLABORATIVE_DISTANCE_PATH, low_memory=False, encoding=ENCODING)
    else:
        same_team_matches = pd.read_csv(SAME_TEAM_MATCHES_CSV).as_matrix()
        same_team_wins = pd.read_csv(SAME_TEAM_WINS_CSV).as_matrix()

        same_team_matches[np.where(same_team_matches <= 10)]=10000


        df=np.divide(same_team_wins, same_team_matches)
        df[np.where(df <= 0.7)] = 0
        df[np.where(df > 0.7)] = 1


        df2=[]
        G=nx.from_numpy_matrix(df)
        for i in range(CHAMPION_SIZE):
            for j in range(CHAMPION_SIZE):
                try:
                    df2.append(nx.shortest_path_length(G, source=i, target=j))
                except nx.NetworkXNoPath:
                    df2.append(0)


        df2=pd.DataFrame(np.array(df2).reshape(CHAMPION_SIZE,CHAMPION_SIZE))
        distance = df2.to_csv(COLLABORATIVE_DISTANCE_PATH, header=False, index=False)



    return distance

def calculate_competitive_distance():
    """

    :return:
    """
    # Check /stats/distance.csv.

    # If /stats/distance.csv is exists, return /stats/distance.csv as a 680 * 680 numpy.ndarray.

    # If there is no /stats/distance.csv, do below codes.
    # Get matrix of matches, wins ans loses.

    # Calculate the distance between competitive players.

    # Save the result to /stats/distance.csv.

    # Return /stats/distance.csv as a 680 * 680 numpy.ndarray.
    if Path(COMPETITIVE_DISTANCE_PATH).exists():
        distance = pd.read_csv(COMPETITIVE_DISTANCE_PATH, low_memory=False, encoding=ENCODING)
    else:
        other_team_matches = pd.read_csv(OTHER_TEAM_MATCHES_CSV).as_matrix()
        other_team_wins = pd.read_csv(OTHER_TEAM_WINS_CSV).as_matrix()

        other_team_matches[np.where(other_team_matches <= 10)]=10000


        df=np.divide(other_team_wins,other_team_matches)
        df[np.where(df <= 0.7)] = 0
        df[np.where(df > 0.7)] = 1

        G = nx.from_numpy_matrix(df)
        print(nx.shortest_path_length(G, source=1, target=3))
        print(nx.shortest_path_length(G, source=5, target=14))


        """
        df2 = []
        G = nx.from_numpy_matrix(df)
        for i in range(CHAMPION_SIZE):
            for j in range(CHAMPION_SIZE):
                try:
                    df2.append(nx.shortest_path_length(G, source=i, target=j))
                except nx.NetworkXNoPath:
                    df2.append(0)

        
        df2 = pd.DataFrame(np.array(df2).reshape(CHAMPION_SIZE, CHAMPION_SIZE))
        
        distance = df2.to_csv(COMPETITIVE_DISTANCE_PATH, header=False, index=False)
        """
    return 0





if __name__ == '__main__':
    #calculate_collaborative_distance()
    calculate_competitive_distance()
