# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 5. 8.
"""
import numpy as np
import pandas as pd
from pandas import Series

from data.data_reader import get_pure_participants, get_bans, WIN, CHAMPION_ID, MATCH_ID, PARTICIPANT_ID, Path

# CSV encoding type
ENCODING = 'utf-8'

# File name
LOCAL_DRAFT_STATUSES_URL = 'data/draft_statuses.csv'

# Column names
POSITION = 'position'
TEAM = 'team'
NAME = 'name'
ROLE = 'role'
ROLES = ['Top', 'Jungle', 'Mid', 'Carry', 'Support']

BANNED_CHAMPS = 'banned_champs'
OUR_CHAMPS = 'our_champs'
OTHER_CHAMPS = 'other_champs'
GB_9 = 'gb9'
GB_8 = 'gb8'
GB_7 = 'gb7'
GB_6 = 'gb6'
GB_5 = 'gb5'
GB_4 = 'gb4'
GB_3 = 'gb3'
GB_2 = 'gb2'
GB_1 = 'gb1'
CAL_CHAMP = 'cal_champs'
PICK_1 = 'pick_1'
PICK_123 = 'pick_123'
PICK_12345 = 'pick_12345'
PICK_67 = 'pick_67'
PICK_6789 = 'pick_6789'
CHANCE_ID = "chance_id"
FIRST_PARTICIPANT_ID = 9


def get_draft_statuses():
    """

    :return: (DataFrame)
        match_id        | (int)
        participant_id  | (int)
        chance_id       | (int)
        champion_id     | (int)
        role            | (int)
        banned_champs   | (int) champion_id
        our_champs      | (int) champion_id * 5 + role
        other_champs    | (int) champion_id * 5 + role
        win             | (bool)
    """
    # check local csv file exists or not.
    if Path(LOCAL_DRAFT_STATUSES_URL).exists():
        # if it exists, load csv file.
        draft_statuses = pd.read_csv(LOCAL_DRAFT_STATUSES_URL, low_memory=False, encoding=ENCODING)
    else:
        participants = get_pure_participants()
        participants[CAL_CHAMP] = (participants[CHAMPION_ID] - 1) * 5 + participants['role']

        player_1 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(0).reset_index()
        player_2 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(1).reset_index()
        player_3 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(2).reset_index()
        player_4 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(3).reset_index()
        player_5 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(4).reset_index()

        player_1 = player_1.rename(columns={CAL_CHAMP: GB_1})
        player_2 = player_2.rename(columns={CAL_CHAMP: GB_2})
        player_3 = player_3.rename(columns={CAL_CHAMP: GB_3})
        player_4 = player_4.rename(columns={CAL_CHAMP: GB_4})
        player_5 = player_5.rename(columns={CAL_CHAMP: GB_5})

        player_6 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(5).reset_index()
        player_7 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(6).reset_index()
        player_8 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(7).reset_index()
        player_9 = participants.groupby(MATCH_ID)[CAL_CHAMP].nth(8).reset_index()

        player_6 = player_6.rename(columns={CAL_CHAMP: GB_6})
        player_7 = player_7.rename(columns={CAL_CHAMP: GB_7})
        player_8 = player_8.rename(columns={CAL_CHAMP: GB_8})
        player_9 = player_9.rename(columns={CAL_CHAMP: GB_9})

        player_12 = pd.merge(player_1, player_2, on=MATCH_ID, how='inner')
        player_123 = pd.merge(player_12, player_3, on=MATCH_ID, how='inner')
        player_1234 = pd.merge(player_123, player_4, on=MATCH_ID, how='inner')

        player_67 = pd.merge(player_6, player_7, on=MATCH_ID, how='inner')
        player_678 = pd.merge(player_67, player_8, on=MATCH_ID, how='inner')

        # player 1 (chance1 other_champs & chance2 our_champs)
        pick_1 = participants
        gb1 = pick_1.groupby(MATCH_ID).nth(0)[CAL_CHAMP]

        def getvalue(x):
            return tuple([gb1[x]])

        pick_1[PICK_1] = pick_1[MATCH_ID].map(getvalue)

        # player 6 & 7
        pick_67 = pd.merge(player_6, player_7, on=MATCH_ID, how='inner')
        pick_67.loc[:, PICK_67] = pd.Series(list(zip(pick_67.loc[:, GB_6], pick_67.loc[:, GB_7])))
        pick_67 = pick_67.drop([GB_6, GB_7], 1)
        pick_67 = pd.DataFrame(pick_67.values.repeat(10, axis=0), columns=pick_67.columns)

        # player 1 & 2 & 3 (chance3 other_champs & chance4 our_champs)
        pick_123 = pd.merge(player_12, player_3, on=MATCH_ID, how='inner')
        pick_123.loc[:, PICK_123] = pd.Series(list(zip(pick_123.loc[:, GB_1], pick_123.loc[:, GB_2], pick_123.loc[:, GB_3])))
        pick_123 = pick_123.drop([GB_1, GB_2, GB_3], 1)
        pick_123 = pd.DataFrame(pick_123.values.repeat(10, axis=0), columns=pick_123.columns)

        # player 6 & 7 & 8 & 9 (chance4 other_champs & chance5 our_champs)
        pick_6789 = pd.merge(player_678, player_9, on=MATCH_ID, how='inner')
        pick_6789.loc[:, PICK_6789] = pd.Series(list(zip(pick_6789.loc[:, GB_6], pick_6789.loc[:, GB_7], pick_6789.loc[:, GB_8], pick_6789.loc[:, GB_9])))
        pick_6789 = pick_6789.drop([GB_6, GB_7, GB_8, GB_9], 1)
        pick_6789 = pd.DataFrame(pick_6789.values.repeat(10, axis=0), columns=pick_6789.columns)

        # player 1 & 2 & 3 & 4 & 5 (chance5 other_champs)
        pick_12345 = pd.merge(player_1234, player_5, on=MATCH_ID, how='inner')
        pick_12345.loc[:, PICK_12345] = pd.Series(list(zip(pick_12345.loc[:, GB_1], pick_12345.loc[:, GB_2], pick_12345.loc[:, GB_3], pick_12345.loc[:, GB_4], pick_12345.loc[:, GB_5])))
        pick_12345 = pick_12345.drop([GB_1, GB_2, GB_3, GB_4, GB_5], 1)
        pick_12345 = pd.DataFrame(pick_12345.values.repeat(10, axis=0), columns=pick_12345.columns)

        # for now delete the duplicates
        pick_67 = pick_67.drop(MATCH_ID, 1)
        pick_123 = pick_123.drop(MATCH_ID, 1)
        pick_6789 = pick_6789.drop(MATCH_ID, 1)
        pick_12345 = pick_12345.drop(MATCH_ID, 1)

        final_df = pd.concat([pick_1, pick_67, pick_123, pick_6789, pick_12345], axis=1)
        final_df[OUR_CHAMPS] = ''
        final_df[OTHER_CHAMPS] = ''

        # player 6 & 7
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 5) % 10, final_df[PICK_1], final_df[OTHER_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 6) % 10, final_df[PICK_1], final_df[OTHER_CHAMPS])

        # player 2 & 3
        final_df[OUR_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 1) % 10, final_df[PICK_1], final_df[OUR_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 1) % 10, final_df[PICK_67], final_df[OTHER_CHAMPS])
        final_df[OUR_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 2) % 10, final_df[PICK_1], final_df[OUR_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 2) % 10, final_df[PICK_67], final_df[OTHER_CHAMPS])

        # player 8 & 9
        final_df[OUR_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 7) % 10, final_df[PICK_67], final_df[OUR_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 7) % 10, final_df[PICK_123], final_df[OTHER_CHAMPS])
        final_df[OUR_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 8) % 10, final_df[PICK_67], final_df[OUR_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 8) % 10, final_df[PICK_123], final_df[OTHER_CHAMPS])

        # player 4 & 5
        final_df[OUR_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 3) % 10, final_df[PICK_123], final_df[OUR_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 3) % 10, final_df[PICK_6789], final_df[OTHER_CHAMPS])
        final_df[OUR_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 4) % 10, final_df[PICK_123], final_df[OUR_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 4) % 10, final_df[PICK_6789], final_df[OTHER_CHAMPS])

        # player 10
        final_df[OUR_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 9) % 10, final_df[PICK_6789], final_df[OUR_CHAMPS])
        final_df[OTHER_CHAMPS] = np.where(final_df[PARTICIPANT_ID] % 10 == (FIRST_PARTICIPANT_ID + 9) % 10, final_df[PICK_12345], final_df[OTHER_CHAMPS])

        final_df = final_df.dropna(subset=[WIN])
        final_df.win = final_df.win.astype(int)
        final_df = final_df.drop([PICK_1, PICK_67, PICK_123, PICK_6789, PICK_12345], 1)

        df_bans = get_bans()

        aggregated = df_bans.groupby(MATCH_ID)[CHAMPION_ID].apply(list)
        aggregated.name = BANNED_CHAMPS
        final_df = final_df.join(aggregated, on=MATCH_ID)

        chance_id = Series([0, 2, 2, 4, 4, 1, 1, 3, 3, 5] * int(len(final_df) / 10), name=CHANCE_ID)

        final_df = pd.concat([final_df, chance_id], axis=1)
        draft_statuses = final_df[[MATCH_ID, PARTICIPANT_ID, CHANCE_ID, CHAMPION_ID, ROLE, BANNED_CHAMPS, OUR_CHAMPS, OTHER_CHAMPS, WIN]]
        draft_statuses.to_csv(LOCAL_DRAFT_STATUSES_URL, index=False, encoding=ENCODING)
    return draft_statuses


if __name__ == '__main__':
    draft_statuses = get_draft_statuses()
    print(draft_statuses)
