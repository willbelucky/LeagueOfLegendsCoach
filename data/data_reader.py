# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 25.
"""
import pandas as pd
from pathlib import Path
import numpy as np

REMOTE_PARTICIPANTS_URL = "https://www.dropbox.com/s/q8eqzthipnbe1i1/participants.csv?dl=1"
LOCAL_PARTICIPANTS_DIR = "data/participants.csv"
REMOTE_CHAMPS_URL = "https://www.dropbox.com/s/v4hblg5umeie2uy/champs.csv?dl=1"
LOCAL_CHAMPS_DIR = "data/champs.csv"
REMOTE_STATS1_URL = "https://www.dropbox.com/s/fozr2zbglqxdg45/stats1.csv?dl=1"
LOCAL_STATS1_DIR = "data/stats1.csv"
REMOTE_STATS2_URL = "https://www.dropbox.com/s/2moeacflbn59qba/stats2.csv?dl=1"
LOCAL_STATS2_DIR = "data/stats2.csv"


def __download_csv(url):
    if url is not None:
        table = pd.read_csv(url, low_memory=False,
                            encoding='utf-8')
        print('Download a csv file from {}.'.format(url))
    else:
        raise FileNotFoundError('A csv file is not exist in {}.'.format(url))

    return table

def get_stats():
    """
    :return champs: (DataFrame)

    """
    pass
    # check local csv file exists or not.
    if Path(LOCAL_STATS1_DIR).exists():
        # if it exists, load csv file.
        table1 = pd.read_csv(LOCAL_STATS1_DIR, low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        table1 = __download_csv(REMOTE_STATS1_URL)

    if Path(LOCAL_STATS2_DIR).exists():
        # if it exists, load csv file.
        table2 = pd.read_csv(LOCAL_STATS2_DIR,low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        table2 = __download_csv(REMOTE_STATS2_URL)

    table=pd.concat([table1, table2])

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
        table = pd.read_csv(LOCAL_CHAMPS_DIR, low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        table = __download_csv(REMOTE_CHAMPS_URL)

    return table


def get_participants():
    """

    :return participants: (DataFrame)
        columns participant_id  | (int) The id of participants
                match_id        | (int) The id of matches
                team            | (bool) True: Blue, False: Red
                champion_id     | (int) The id of champions
                role            | (int) 0: Top, 1: Jungle, 2: Mid, 3: Carry, 4: Support
                win             | (bool) True: win, False: lose


    """
    # check local csv file exists or not.
    if Path(LOCAL_PARTICIPANTS_DIR).exists():
        # if it exists, load csv file.
        df_participants = pd.read_csv(LOCAL_PARTICIPANTS_DIR, low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        df_participants = __download_csv(REMOTE_PARTICIPANTS_URL)

    # do data processing
    df_stats = get_stats()
    df_participants = df_participants.rename(columns={"id": "participant_id", "matchid": "match_id", "championid": "champion_id",'role':'role1'})
    df_participants_3 = pd.merge(df_participants, df_stats, left_on='participant_id', right_on='id', how='left')

    # player -> team
    df_participants_3['team'] = np.where((df_participants_3.player == 1)or (df_participants_3.player == 2) or (df_participants_3.player == 3) or (df_participants_3.player == 4) or (df_participants_3.player == 5 ),True)
    df_participants_3['team'] = np.where((df_participants_3.player == 6) or (df_participants_3.player == 7) or (df_participants_3.player == 8) or (df_participants_3.player == 9) or (df_participants_3.player == 10), False)

    # role, position -> role
    df_participants_3 = df_participants_3.rename(columns={'role':'role1'})
    df_participants_3['role'] = np.nan
    df_participants_3.loc[df_participants_3['position'] == 'MID','role']= 2
    df_participants_3.loc[df_participants_3['position'] == 'TOP', 'role'] = 0
    df_participants_3.loc[df_participants_3['position'] == 'JUNGLE', 'role'] =1
    df_participants_3['role'] = np.where((df_participants_3.position=='BOT')&(df_participants_3.role1=='DUO_SUPPORT'),4)
    df_participants_3['role'] = np.where((df_participants_3.position == 'BOT') & (df_participants_3.role1 == 'DUO_CARRY'), 3)

    # select columns we use.
    dataframe = df_participants_3[['participant_id', 'match_id', 'team', 'champion_id',  'role', 'position', 'win']]


    return dataframe

df=get_participants()