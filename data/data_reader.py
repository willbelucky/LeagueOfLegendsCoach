# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 25.
"""
import pandas as pd
from pathlib import Path
import numpy as np

# Remote and local file paths
REMOTE_PARTICIPANTS_URL = "https://www.dropbox.com/s/q8eqzthipnbe1i1/participants.csv?dl=1"
LOCAL_PARTICIPANTS_DIR = "data/participants.csv"
REMOTE_CHAMPS_URL = "https://www.dropbox.com/s/v4hblg5umeie2uy/champs.csv?dl=1"
LOCAL_CHAMPS_DIR = "data/champs.csv"
REMOTE_STATS1_URL = "https://www.dropbox.com/s/fozr2zbglqxdg45/stats1.csv?dl=1"
LOCAL_STATS1_DIR = "data/stats1.csv"
REMOTE_STATS2_URL = "https://www.dropbox.com/s/2moeacflbn59qba/stats2.csv?dl=1"
LOCAL_STATS2_DIR = "data/stats2.csv"
REMOTE_BANS_URL = "https://www.dropbox.com/s/bcg30zlx7fbnzye/teambans.csv?dl=1"
LOCAL_BANS_DIR = "data/teambans.csv"
LOCAL_PURE_PARTICIPANTS_URL = "data/pure_participants.csv"

# CSV encoding type
ENCODING = 'utf-8'

# Column names
WIN = 'win'
POSITION = 'position'
TEAM = 'team'
CHAMPION_ID = "champion_id"
MATCH_ID = "match_id"
PARTICIPANT_ID = "participant_id"
NAME = 'name'
ROLE = 'role'

ROLES = ['Top', 'Jungle', 'Mid', 'Carry', 'Support']


def download_csv(remote_url, local_dir, header=False):
    """
    Download csv file from remote repository and save it to local folder.

    :param remote_url: (string) An URL of a remote file.
    :param local_dir: (string) A local file path.
    :param header: (bool) If header is True, read header and index of csv file, else don't read and save header and
        index.

    :return table: (DataFrame)
    """
    read_header = 'infer'
    if not header:
        read_header = None
    if remote_url is not None:
        table = pd.read_csv(remote_url, low_memory=False, encoding=ENCODING, header=read_header)
        table.to_csv(local_dir, header=header, index=False, encoding=ENCODING)
        print('Download a csv file from {}.'.format(remote_url))
    else:
        raise FileNotFoundError('A csv file is not exist in {}.'.format(remote_url))

    return table


def _get_stats():
    """
    Concatenate stats1 and stats2 and return it.

    :return stats: (DataFrame)
        columns participant_id  | (int) The id of participants
                win             | (int) 1 for win, 0 for lose.
    """
    # check local csv file exists or not.
    if Path(LOCAL_STATS1_DIR).exists():
        # if it exists, load csv file.
        stats1 = pd.read_csv(LOCAL_STATS1_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        stats1 = download_csv(REMOTE_STATS1_URL, LOCAL_STATS1_DIR, True)

    if Path(LOCAL_STATS2_DIR).exists():
        # if it exists, load csv file.
        stats2 = pd.read_csv(LOCAL_STATS2_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        stats2 = download_csv(REMOTE_STATS2_URL, LOCAL_STATS2_DIR, True)

    stats = pd.concat([stats1, stats2])

    stats = stats.rename(columns={"id": "participant_id"})

    return stats


def get_champs():
    """


    :return champs: (DataFrame)
        columns champion_id | (int) The id of champions
                role        | (int) 0: Top, 1: Jungle, 2: Mid, 3: Carry, 4: Support
                name        | (str) The name of champions
    """
    # check local csv file exists or not.
    if Path(LOCAL_CHAMPS_DIR).exists():
        # if it exists, load csv file.
        champs = pd.read_csv(LOCAL_CHAMPS_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        champs = download_csv(REMOTE_CHAMPS_URL, LOCAL_CHAMPS_DIR, True)

    role_assigned_champs = []
    for index, role in enumerate(ROLES):
        role_assigned_champ = champs.copy()
        role_assigned_champ[ROLE] = index
        role_assigned_champ[NAME] = role + '_' + role_assigned_champ[NAME]
        role_assigned_champs.append(role_assigned_champ)

    champs = pd.concat(role_assigned_champs, ignore_index=True)
    champs = champs.sort_values(by=['id', 'role'])
    champs = champs.reset_index(drop=True)
    champs = champs.rename(columns={"id": CHAMPION_ID})
    champs = champs[[CHAMPION_ID, ROLE, NAME]]
    return champs


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
        participants = pd.read_csv(LOCAL_PARTICIPANTS_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        participants = download_csv(REMOTE_PARTICIPANTS_URL, LOCAL_PARTICIPANTS_DIR, True)

    # do data processing
    stats = _get_stats()
    # noinspection SpellCheckingInspection
    participants = participants.rename(
        columns={"id": PARTICIPANT_ID, "matchid": MATCH_ID, "championid": CHAMPION_ID, ROLE: 'role1'}
    )
    participants = pd.merge(participants, stats, on=PARTICIPANT_ID, how='left')

    # player -> team
    participants[TEAM] = False
    participants.loc[participants['player'].isin([1, 2, 3, 4, 5]), TEAM] = True

    # role, position -> role
    participants[ROLE] = 3
    participants.loc[participants[POSITION] == 'TOP', ROLE] = 0
    participants.loc[participants[POSITION] == 'JUNGLE', ROLE] = 1
    participants.loc[participants[POSITION] == 'MID', ROLE] = 2
    participants.loc[np.logical_and(participants[POSITION] == 'BOT', participants['role1'] == 'DUO_SUPPORT'), ROLE] = 4

    # type of win: float -> bool
    participants[WIN] = participants[WIN].astype(bool)

    # select columns we use.
    participants = participants[[PARTICIPANT_ID, MATCH_ID, TEAM, CHAMPION_ID, ROLE, WIN]]

    return participants


def get_bans():
    """

    :return stats: (DataFrame)
        columns       match_id        | (int) The id of matches
                      banned_id       | (array of int) The array of ids of banned champions in each match

    """
    pass
    # check local csv file exists or not.
    if Path(LOCAL_BANS_DIR).exists():
        # if it exists, load csv file.
        bans = pd.read_csv(LOCAL_BANS_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        bans = download_csv(REMOTE_BANS_URL, LOCAL_BANS_DIR)

    bans = bans.rename(columns={"matchid": MATCH_ID, "championid": CHAMPION_ID})
    bans = bans[[MATCH_ID, CHAMPION_ID]]

    return bans


def get_pure_participants():
    """
    1. The number of players of a match should be 0.
    2. Both team should have all 5 role, Top solo, Jungle, Mid solo, Bottom carry, Bottom support.

    :return pure_participants: (DataFrame)
        columns participant_id  | (int) The id of participants
                match_id        | (int) The id of matches
                team            | (bool) True: Blue, False: Red
                champion_id     | (int) The id of champions
                role            | (int) 0: Top, 1: Jungle, 2: Mid, 3: Carry, 4: Support
                win             | (bool) True: win, False: lose
    """
    # check local csv file exists or not.
    if Path(LOCAL_PURE_PARTICIPANTS_URL).exists():
        # if it exists, load csv file.
        pure_participants = pd.read_csv(LOCAL_PURE_PARTICIPANTS_URL, low_memory=False, encoding=ENCODING)
    else:
        # else, purify participants and return them after saving.
        participants = get_participants()

        df = participants
        df['sum'] = 1
        df['role_sum'] = 1
        for i in range(len(df)):
            if (df['role'][i] == 0):
                df.at[i, 'role_sum'] = 1
            elif (df['role'][i] == 1):
                df.at[i, 'role_sum'] = 10
            elif (df['role'][i] == 2):
                df.at[i, 'role_sum'] = 100
            elif (df['role'][i] == 3):
                df.at[i, 'role_sum'] = 1000
            elif (df['role'][i] == 4):
                df.at[i, 'role_sum'] = 10000
            else:
                df.at[i, 'role_sum'] = 10000000

        df2 = df.groupby('match_id').sum()

        FIND_TEAMMATES_10 = df2.iloc[np.where(df2['sum'] == 10)]
        FIND_ROLE_20 = FIND_TEAMMATES_10.iloc[np.where(FIND_TEAMMATES_10['role_sum'] == 22222)]
        FIND_ROLE_20 = FIND_ROLE_20.index
        array_FIND_TEAMMATES_10 = np.array(FIND_ROLE_20)

        df3 = pd.DataFrame()
        df3['match_id'] = array_FIND_TEAMMATES_10

        participants = participants.merge(df3, on='match_id')

        pure_participants = participants
        pure_participants.to_csv(LOCAL_PURE_PARTICIPANTS_URL, index=False, encoding=ENCODING)

    return pure_participants


if __name__ == '__main__':
    print(get_bans())
    print(get_pure_participants())
    print(get_champs())
