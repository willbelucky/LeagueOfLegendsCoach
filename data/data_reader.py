# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 25.
"""
import pandas as pd
from pathlib import Path

REMOTE_PARTICIPANTS_URL = "https://www.dropbox.com/s/q8eqzthipnbe1i1/participants.csv?dl=1"
LOCAL_PARTICIPANTS_DIR = "/data/participants.csv"
REMOTE_CHAMPS_URL = "https://www.dropbox.com/s/v4hblg5umeie2uy/champs.csv?dl=1"
LOCAL_CHAMPS_DIR = "/data/champs.csv"
REMOTE_STATS1_URL = "https://www.dropbox.com/s/fozr2zbglqxdg45/stats1.csv?dl=1"
LOCAL_STATS1_DIR = "/data/stats1.csv"

def __download_csv(url):
    if url is not None:
        table = pd.read_csv(url, low_memory=False,
                            encoding='utf-8')
        print('Download a csv file from {}.'.format(url))
    else:
        raise FileNotFoundError('A csv file is not exist in {}.'.format(url))

    return table

def get_stats1():
    """
    :return champs: (DataFrame)

    """
    pass
    # check local csv file exists or not.
    if Path(LOCAL_STATS1_DIR).exists():
        # if it exists, load csv file.
        table = pd.read_csv(LOCAL_STATS1_DIR, parse_dates=parse_dates, low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        table = __download_csv(REMOTE_STATS1_URL)
    return table


def get_champs():
    """


    :return champs: (DataFrame)
        columns champion_id | (int) The id of champions
                role        | (int) 0: Top, 1: Jungle, 2: Mid, 3: Carry, 4: Support
                name        | (str) The name of champions
    """
    pass
    # check local csv file exists or not.
    if Path(LOCAL_CHAMPS_DIR).exists():
        # if it exists, load csv file.
        table = pd.read_csv(LOCAL_CHAMPS_DIR, parse_dates=parse_dates, low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        table = __download_csv(REMOTE_CHAMPS_URL)

    return table


def get_participants():
    """

    :return participants: (DataFrame)
        columns participant_id  | (int) The id of participants
                team            | (bool) True: Blue, False: Red
                champion_id     | (int) The id of champions
                role            | (int) 0: Top, 1: Jungle, 2: Mid, 3: Carry, 4: Support
                win             | (bool) True: win, False: lose


    """
    # check local csv file exists or not.
    if Path(LOCAL_PARTICIPANTS_DIR).exists():
        # if it exists, load csv file.
        table = pd.read_csv(LOCAL_PARTICIPANTS_DIR, parse_dates=parse_dates, low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        table = __download_csv(REMOTE_PARTICIPANTS_URL)

    # do data processing
    df_stats1 = get_stats1()
    df_champs = get_champs()
    df_participants = table
    df_participants_2 = pd.merge(df_participants, df_champs, left_on='championid', right_on='id', how='left')
    df_participants_3 = pd.merge(df_participants_2, df_stats1, left_on='id_x', right_on='id', how='left')
    dataframe = df_participants_3[['id_x', 'matchid', 'player', 'championid', 'name', 'role', 'position', 'win']]

    # return champs dataframe.

    return dataframe

df=get_participants()