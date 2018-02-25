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


def get_champs():
    """


    :return champs: (DataFrame)
        columns champion_id | (int) The id of champions
                role        | (int) 0: Top, 1: Jungle, 2: Mid, 3: Carry, 4: Support
                name        | (str) The name of champions
    """
    pass
    # check local csv file exists or not.
    # if it exists, load csv file.
    # else, download csv file from remote repository and save it to local folder.

    # do data processing

    # return champs dataframe.


def __download_csv(url):
    if url is not None:
        table = pd.read_csv(url, low_memory=False,
                            encoding='utf-8')
        print('Download a csv file from {}.'.format(url))
    else:
        raise FileNotFoundError('A csv file is not exist in {}.'.format(url))

    return table


def get_participants(cls, index=None, parse_dates=None):
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
        table = pd.read_csv(cls.__csv_file_dir, parse_dates=parse_dates, low_memory=False, encoding='utf-8')
    else:
        # else, download csv file from remote repository and save it to local folder.
        table = cls.__download_csv(REMOTE_PARTICIPANTS_URL)

    # do data processing

    # return champs dataframe.

    return table
