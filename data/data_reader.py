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


def _download_csv(remote_url, local_dir):
    """
    Download csv file from remote repository and save it to local folder.

    :param remote_url: (string) An URL of a remote file.
    :param local_dir: (string) A local file path.

    :return table: (DataFrame)
    """
    if remote_url is not None:
        table = pd.read_csv(remote_url, low_memory=False, encoding=ENCODING)
        table.to_csv(local_dir, encoding=ENCODING)
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
    pass
    # check local csv file exists or not.
    if Path(LOCAL_STATS1_DIR).exists():
        # if it exists, load csv file.
        stats1 = pd.read_csv(LOCAL_STATS1_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        stats1 = _download_csv(REMOTE_STATS1_URL, LOCAL_STATS1_DIR)

    if Path(LOCAL_STATS2_DIR).exists():
        # if it exists, load csv file.
        stats2 = pd.read_csv(LOCAL_STATS2_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        stats2 = _download_csv(REMOTE_STATS2_URL, LOCAL_STATS2_DIR)

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
    pass
    # check local csv file exists or not.
    if Path(LOCAL_CHAMPS_DIR).exists():
        # if it exists, load csv file.
        champs = pd.read_csv(LOCAL_CHAMPS_DIR, low_memory=False, encoding=ENCODING)
    else:
        # else, download csv file from remote repository and save it to local folder.
        champs = _download_csv(REMOTE_CHAMPS_URL, LOCAL_CHAMPS_DIR)

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
        participants = _download_csv(REMOTE_PARTICIPANTS_URL, LOCAL_PARTICIPANTS_DIR)

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


if __name__ == '__main__':
    print(get_participants())
    print(get_champs())
