# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 25.
"""
import numpy as np
import pandas as pd

from data.data_reader import get_champs, get_participants

# Result file paths
OTHER_TEAM_LOSES_CSV = "/data/other_team_loses.csv"
OTHER_TEAM_WINS_CSV = "/data/other_team_wins.csv"
OTHER_TEAM_MATCHES_CSV = "/data/other_team_matches.csv"
SAME_TEAM_LOSES_CSV = "/data/same_team_loses.csv"
SAME_TEAM_WINS_CSV = "/data/same_team_wins.csv"
SAME_TEAM_MATCHES_CSV = "/data/same_team_matches.csv"

# Columns name
ROLE = 'role'
CHAMPION_ID = 'champion_id'
WIN = 'win'
TEAM = 'team'
MATCH_ID = 'match_id'

CHAMPION_SIZE = 690


def get_same_team_matches():
    """

    :return same_team_matches: (numpy.array[int]) 690 * 690 matrix with number of matches between champions
        in same team.
    """

    df = get_participants()
    df2 = get_champs()

    arraymatrix = np.zeros(690 * 690)
    for i in range(len(df)):
        for j in range(min(i + 1, len(df)), min(i + 6, len(df)), 1):
            if (df[MATCH_ID][i] == df[MATCH_ID][j] and df['team'][i] == df['team'][j]):
                A = df2.index[df2['champion_id'] == df['champion_id'][i]].values
                B = df2.index[df2['role'] == df['role'][i]].values
                A = np.ndarray.tolist(A)
                B = np.ndarray.tolist(B)
                idxrow = set(A).intersection(B)
                idxrow = list(idxrow)[0]

                C = df2.index[df2['champion_id'] == df['champion_id'][j]].values
                D = df2.index[df2['role'] == df['role'][j]].values
                C = np.ndarray.tolist(C)
                D = np.ndarray.tolist(D)
                idxcol = set(C).intersection(D)
                idxcol = list(idxcol)[0]
                print(i)

                arraymatrix[idxrow * 690 + idxcol] += 1
                arraymatrix[idxcol * 690 + idxrow] += 1

    arraymatrix = np.reshape(arraymatrix, (690, 690))
    return arraymatrix


def get_same_team_wins():
    """

    :return same_team_wins: (numpy.array[int]) 690 * 690 matrix with number of matches between champions
        in same team.
    """

    df = get_participants()
    df2 = get_champs()

    arraymatrix = np.zeros(690 * 690)
    for i in range(len(df)):
        if (df['win'][i] == True):
            for j in range(min(i + 1, len(df)), min(i + 6, len(df)), 1):
                if (df[MATCH_ID][i] == df[MATCH_ID][j] and df['team'][i] == df['team'][j] and (df['win'][j] == True)):
                    A = df2.index[df2['champion_id'] == df['champion_id'][i]].values
                    B = df2.index[df2['role'] == df['role'][i]].values
                    A = np.ndarray.tolist(A)
                    B = np.ndarray.tolist(B)
                    idxrow = set(A).intersection(B)
                    idxrow = list(idxrow)[0]

                    C = df2.index[df2['champion_id'] == df['champion_id'][j]].values
                    D = df2.index[df2['role'] == df['role'][j]].values
                    C = np.ndarray.tolist(C)
                    D = np.ndarray.tolist(D)
                    idxcol = set(C).intersection(D)
                    idxcol = list(idxcol)[0]

                    arraymatrix[idxrow * 690 + idxcol] += 1
                    arraymatrix[idxcol * 690 + idxrow] += 1
        else:
            continue
    arraymatrix = np.reshape(arraymatrix, (690, 690))
    return arraymatrix


def get_same_team_loses():
    """

    :return same_team_loses: (numpy.array[int]) 690 * 690 matrix with number of matches between champions
        in same team.
    """
    df = get_participants()
    df2 = get_champs()

    arraymatrix = np.zeros(690 * 690)
    for i in range(len(df)):
        if (df['win'][i] == False):
            print(i)
            for j in range(min(i + 1, len(df)), min(i + 6, len(df)), 1):
                if (df[MATCH_ID][i] == df[MATCH_ID][j] and df['team'][i] == df['team'][j] and (df['win'][j] == False)):
                    A = df2.index[df2['champion_id'] == df['champion_id'][i]].values
                    B = df2.index[df2['role'] == df['role'][i]].values
                    A = np.ndarray.tolist(A)
                    B = np.ndarray.tolist(B)
                    idxrow = set(A).intersection(B)
                    idxrow = list(idxrow)[0]

                    C = df2.index[df2['champion_id'] == df['champion_id'][j]].values
                    D = df2.index[df2['role'] == df['role'][j]].values
                    C = np.ndarray.tolist(C)
                    D = np.ndarray.tolist(D)
                    idxcol = set(C).intersection(D)
                    idxcol = list(idxcol)[0]

                    arraymatrix[idxrow * 690 + idxcol] += 1
                    arraymatrix[idxcol * 690 + idxrow] += 1
        else:
            continue
    arraymatrix = np.reshape(arraymatrix, (690, 690))
    return arraymatrix


def get_other_team_matches():
    """

    :return other_team_matches: (numpy.array[int]) 690 * 690 matrix with number of matches between champions
        in other teams.
    """
    df = get_participants()
    df2 = get_champs()

    arraymatrix = np.zeros(690 * 690)
    for i in range(len(df)):
        for j in range(min(i + 1, len(df)), min(i + 11, len(df)), 1):
            if (df[MATCH_ID][i] == df[MATCH_ID][j] and df['team'][i] != df['team'][j]):
                A = df2.index[df2['champion_id'] == df['champion_id'][i]].values
                B = df2.index[df2['role'] == df['role'][i]].values
                A = np.ndarray.tolist(A)
                B = np.ndarray.tolist(B)
                idxrow = set(A).intersection(B)
                idxrow = list(idxrow)[0]

                C = df2.index[df2['champion_id'] == df['champion_id'][j]].values
                D = df2.index[df2['role'] == df['role'][j]].values
                C = np.ndarray.tolist(C)
                D = np.ndarray.tolist(D)
                idxcol = set(C).intersection(D)
                idxcol = list(idxcol)[0]

                arraymatrix[idxrow * 690 + idxcol] += 1
                arraymatrix[idxcol * 690 + idxrow] += 1

    arraymatrix = np.reshape(arraymatrix, (690, 690))
    return arraymatrix


def get_other_team_wins():
    """

    :return other_team_wins: (numpy.array[int]) 690 * 690 matrix with number of matches between champions
        in other teams.
        A(win) VS B(lose)
    """
    df = get_participants()
    df2 = get_champs()

    arraymatrix = np.zeros(690 * 690)
    for i in range(len(df)):
        if (df['win'][i] == True):
            for j in range(min(max(i - 11, 0), len(df)), min(i + 11, len(df)), 1):
                if (df[MATCH_ID][i] == df[MATCH_ID][j] and df['team'][i] != df['team'][j] and (df['win'][j] == False)):
                    A = df2.index[df2['champion_id'] == df['champion_id'][i]].values
                    B = df2.index[df2['role'] == df['role'][i]].values
                    A = np.ndarray.tolist(A)
                    B = np.ndarray.tolist(B)
                    idxrow = set(A).intersection(B)
                    idxrow = list(idxrow)[0]

                    C = df2.index[df2['champion_id'] == df['champion_id'][j]].values
                    D = df2.index[df2['role'] == df['role'][j]].values
                    C = np.ndarray.tolist(C)
                    D = np.ndarray.tolist(D)
                    idxcol = set(C).intersection(D)
                    idxcol = list(idxcol)[0]

                    arraymatrix[idxrow * 690 + idxcol] += 1

        else:
            continue
    arraymatrix = np.reshape(arraymatrix, (690, 690))
    return arraymatrix


def get_other_team_loses():
    """

    :return other_team_loses: (numpy.array[int]) 690 * 690 matrix with number of matches between champions
        in other teams.
        A(lose) VS B(win)
    """
    df = get_participants()
    df2 = get_champs()

    arraymatrix = np.zeros(690 * 690)
    for i in range(len(df)):
        if (df['win'][i] == False):
            for j in range(min(max(i - 11, 0), len(df)), min(i + 11, len(df)), 1):
                if (df[MATCH_ID][i] == df[MATCH_ID][j] and df['team'][i] != df['team'][j] and (df['win'][j] == True)):
                    A = df2.index[df2['champion_id'] == df['champion_id'][i]].values
                    B = df2.index[df2['role'] == df['role'][i]].values
                    A = np.ndarray.tolist(A)
                    B = np.ndarray.tolist(B)
                    idxrow = set(A).intersection(B)
                    idxrow = list(idxrow)[0]

                    C = df2.index[df2['champion_id'] == df['champion_id'][j]].values
                    D = df2.index[df2['role'] == df['role'][j]].values
                    C = np.ndarray.tolist(C)
                    D = np.ndarray.tolist(D)
                    idxcol = set(C).intersection(D)
                    idxcol = list(idxcol)[0]

                    arraymatrix[idxrow * 690 + idxcol] += 1
        else:
            continue
    arraymatrix = np.reshape(arraymatrix, (690, 690))
    return arraymatrix


if __name__ == '__main__':
    from multiprocessing.dummy import Pool as ThreadPool
    import multiprocessing


    def save_processed_data(*args):
        print('Start {}'.format(multiprocessing.Process()))
        func, path = args[0]
        same_team_matches = pd.DataFrame(func())
        same_team_matches.to_csv(path)
        print('Done {}'.format(multiprocessing.Process()))
        return same_team_matches


    pool = ThreadPool(6)
    args = [
        (get_same_team_matches, SAME_TEAM_MATCHES_CSV,),
        (get_same_team_wins, SAME_TEAM_WINS_CSV,),
        (get_same_team_loses, SAME_TEAM_LOSES_CSV,),
        (get_other_team_matches, OTHER_TEAM_MATCHES_CSV,),
        (get_other_team_wins, OTHER_TEAM_WINS_CSV,),
        (get_other_team_loses, OTHER_TEAM_LOSES_CSV,)
    ]
    results = pool.map_async(save_processed_data, args)

    pool.close()
    pool.join()
    print('Finally done!')
