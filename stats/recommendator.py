# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 5. 8.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

from data.data_reader import get_champs, get_pure_participants, MATCH_ID, PARTICIPANT_ID, CHAMPION_ID, ROLE, WIN
from stats.distance_calculator import calculate_collaborative_distances, calculate_competitive_distances
from data.data_preprocessor import get_draft_statuses, CHANCE_ID, BANNED_CHAMPS, OUR_CHAMPS, OTHER_CHAMPS


def recommend_best_champions(banned_champs, our_champs, other_champs, coefficients):
    """

    :param banned_champs: (list[int]) 6 banned champions. champion_id.
    :param our_champs: (list[int]) champion_id * 5 + role
    :param other_champs: (list[int]) champion_id * 5 + role
    :param coefficients: (list) intercept and 45 betas.

    :return best_champions: (dict)
        ex) our_champs = [8, 14]
            {
                0(role): 19(champion_id),
                1: 192,
                2: 123
            }
            There are 3 and 4 role in our champs. So, we will not recommend 3 and 4.
    """

    #### top 0, jungle 1 , mid 2, carry 3, support 4
    ## x1 0,1 -> 1 x2 0,2 -> 2 x3 0,3 -> 3 x4 0,4 -> 4 x5 1,2 -> 5 x6 1,3 x7 1,4 x8 2,3 4+3+1 x9 2,4 4+3+2 x10 3,4

    NUMBER_OF_OTHER_CHAMPS = len(other_champs)
    NUMBER_OF_OUR_CHAMPS = len(our_champs)
    ROLE = [0, 1, 2, 3, 4]

    champs = get_champs()
    champs['cal_champs'] = 5 * champs['champion_id'] + champs['role']

    champs_list = np.array(champs['champion_id'].drop_duplicates()).tolist()

    # except champs_id: banned_champs + our_champs_id + other_champs_id
    other_champs_id = []
    other_champs_role = []
    other_champs_index_list = []

    for i in range(NUMBER_OF_OTHER_CHAMPS):
        other_champs_id.append(other_champs[i] // 5)
        other_champs_role.append(other_champs[i] % 5)

        other_champs_index = champs.iloc[np.where(champs['cal_champs'] == other_champs[i])]
        other_champs_index = other_champs_index.index

        other_champs_index_list.append(other_champs_index)

    our_champs_id = []
    our_champs_role = []
    our_champs_index_list = []

    for j in range(NUMBER_OF_OUR_CHAMPS):
        our_champs_id.append(our_champs[j] // 5)
        our_champs_role.append(our_champs[j] % 5)
        our_champs_index = champs.iloc[np.where(champs['cal_champs'] == our_champs[j])]
        our_champs_index = our_champs_index.index

        our_champs_index_list.append(our_champs_index)

    except_champs_id = other_champs_id + banned_champs + our_champs_id
    champs_list_except_selected_ones = list(set(champs_list) - set(except_champs_id))
    champs_list_except_selected_ones.sort()
    champs_list_except_selected_ones_clean = [x for x in champs_list_except_selected_ones if x is not None]

    remain_roles = list(set(ROLE) - set(our_champs_role))
    collaborative_distances = calculate_collaborative_distances()
    competitive_distances = calculate_competitive_distances()

    best_champions = {}
    for role in remain_roles:
        best_champions_list = []
        best_xb = -2
        x_array = np.array(np.ones(46))
        b_array = np.array(coefficients)
        for champ_id in champs_list_except_selected_ones_clean:
            champs_index = champs.iloc[np.where(champs['champion_id'] == champ_id)]
            champs_index = champs_index.iloc[np.where(champs_index['role'] == role)]
            champs_index = champs_index.index
            champs_index = np.array(champs_index)[0]

            ### our champ - new our champ##
            for m in our_champs_index_list:
                m_role = champs['role'][m[0]]
                m_champs_roles = champs['role'][champs_index]

                coeff_m_num = 0
                for m1 in range(min(m_role, m_champs_roles)):
                    coeff_m_num = coeff_m_num + (4 - m1)
                coeff_m_num = abs(m_role - m_champs_roles) + coeff_m_num
                x_array[coeff_m_num] = collaborative_distances[champs_index, m[0]]

            ### other champ - new our champ ##
            for n in other_champs_index_list:
                n_role = champs['role'][n[0]]
                n_champs_roles = champs['role'][champs_index]
                coeff_n_num = 0
                coeff_n_num = coeff_n_num + 21 + 5 * n_champs_roles + n_role
                x_array[coeff_n_num] = competitive_distances[champs_index, n[0]]

            ### our champ - our champ##
            for m1 in our_champs_index_list:
                for m2 in our_champs_index_list:
                    if m1 == m2:
                        break
                    else:
                        m1_role = champs['role'][m1[0]]
                        m2_role = champs['role'][m2[0]]

                        coeff_m1_num = 0
                        for min_m1 in range(min(m1_role, m2_role)):
                            coeff_m1_num = coeff_m1_num + (4 - min_m1)
                        coeff_m1_num = abs(m1_role - m2_role) + coeff_m1_num
                        x_array[coeff_m1_num] = collaborative_distances[m2[0], m1[0]]

            ### other champ - other champ##
            for n1 in other_champs_index_list:
                for n2 in other_champs_index_list:
                    if n1 == n2:
                        break
                    else:
                        n1_role = champs['role'][n1[0]]
                        n2_role = champs['role'][n2[0]]

                        coeff_n1_num = 0
                        for min_n1 in range(min(n1_role, n2_role)):
                            coeff_n1_num = coeff_n1_num + (4 - min_n1) + 10
                        coeff_n1_num = abs(n1_role - n2_role) + coeff_n1_num
                        x_array[coeff_n1_num] = collaborative_distances[n2[0], n1[0]]

            ### our champ _ other_champ##
            for o1 in our_champs_index_list:
                for o2 in other_champs_index_list:
                    o1_role = champs['role'][o1[0]]
                    o2_role = champs['role'][o2[0]]

                    coeff_o1_num = 0
                    coeff_o1_num = coeff_o1_num + 21 + 5 * o1_role + o2_role
                    x_array[coeff_o1_num] = competitive_distances[o1[0], o2[0]]

            xb_sum = np.inner(x_array, b_array)

            if best_xb < xb_sum:
                best_xb = xb_sum

                if champs_index != None:
                    best_champions_list.append(champs_index)
                else:
                    continue
                if champ_id == champs_list_except_selected_ones_clean[-1]:
                    print(champs['name'][best_champions_list[-1]])
                    best_champions_role = champs['role'][best_champions_list[-1]]
                    best_champions_id = champs['champion_id'][best_champions_list[-1]]
                    best_champions_winning_rate = np.exp(best_xb) / (1 + np.exp(best_xb))
                    best_champions[best_champions_role] = (best_champions_id, best_champions_winning_rate)


            else:
                if champ_id == champs_list_except_selected_ones_clean[-1]:

                    print(champs['name'][best_champions_list[-1]])
                    best_champions_role = champs['role'][best_champions_list[-1]]
                    best_champions_id = champs['champion_id'][best_champions_list[-1]]
                    best_champions_winning_rate = np.exp(best_xb) / (1 + np.exp(best_xb))
                    best_champions[best_champions_role] = (best_champions_id, best_champions_winning_rate)

                else:
                    continue

    return best_champions


def simulate_original_pick(draft_statuses, coefficients):
    """

    :return original_pick_winning_probabilities: (Series[flot])
    """
    champs = get_champs()
    champs['cal_champs'] = 5 * champs['champion_id'] + champs['role']

    collaborative_distances = calculate_collaborative_distances()
    competitive_distances = calculate_competitive_distances()

    original_pick_winning_probabilities = []
    for (index, row) in tqdm(draft_statuses.iterrows()):
        champion_id = row[CHAMPION_ID]
        role = row[ROLE]
        our_champs = row[OUR_CHAMPS]
        other_champs = row[OTHER_CHAMPS]

        x_array = np.array(np.ones(46))
        b_array = np.array(coefficients)

        our_champs_index_list = []
        other_champs_index_list = []

        for our_champ_1 in our_champs:
            our_champs_index = champs.iloc[np.where(champs['cal_champs'] == our_champ_1)]
            our_champs_index = our_champs_index.index

            our_champs_index_list.append(our_champs_index)

        for other_champ in other_champs:
            other_champs_index = champs.iloc[np.where(champs['cal_champs'] == other_champ)]
            other_champs_index = other_champs_index.index

            other_champs_index_list.append(other_champs_index)

        champs_index = champs.iloc[np.where(champs['champion_id'] == champion_id)]
        champs_index = champs_index.iloc[np.where(champs_index['role'] == role)]
        champs_index = champs_index.index
        champs_index = np.array(champs_index)[0]

        ### our champ - new our champ##
        for our_champ_1 in our_champs_index_list:
            our_champ_role = champs['role'][our_champ_1]
            new_pick_role = champs['role'][champs_index]

            coeff_m_num = 0
            for our_champ_1 in range(min(our_champ_role, new_pick_role)):
                coeff_m_num = coeff_m_num + (4 - our_champ_1)
            coeff_m_num = abs(our_champ_role - new_pick_role) + coeff_m_num
            x_array[coeff_m_num] = collaborative_distances[champs_index, our_champ_1]

        ### other champ - new our champ ##
        for other_champ in other_champs_index_list:
            other_champ_role = champs['role'][other_champ]
            new_pick_role = champs['role'][champs_index]
            coeff_n_num = 0
            coeff_n_num = coeff_n_num + 21 + 5 * new_pick_role + other_champ_role
            x_array[coeff_n_num] = competitive_distances[champs_index, other_champ]

        ### our champ - our champ##
        for our_champ_1 in our_champs_index_list:
            for our_champ_2 in our_champs_index_list:
                if our_champ_1 == our_champ_2:
                    break
                else:
                    m1_role = champs['role'][our_champ_1]
                    m2_role = champs['role'][our_champ_2]

                    coeff_m1_num = 0
                    for min_m1 in range(min(m1_role, m2_role)):
                        coeff_m1_num = coeff_m1_num + (4 - min_m1)
                    coeff_m1_num = abs(m1_role - m2_role) + coeff_m1_num
                    x_array[coeff_m1_num] = collaborative_distances[our_champ_2, our_champ_1]

        ### other champ - other champ##
        for other_champ_1 in other_champs_index_list:
            for other_champ_2 in other_champs_index_list:
                if other_champ_1 == other_champ_2:
                    break
                else:
                    n1_role = champs['role'][other_champ_1]
                    n2_role = champs['role'][other_champ_2]

                    coeff_n1_num = 0
                    for min_n1 in range(min(n1_role, n2_role)):
                        coeff_n1_num = coeff_n1_num + (4 - min_n1) + 10
                    coeff_n1_num = abs(n1_role - n2_role) + coeff_n1_num
                    x_array[coeff_n1_num] = collaborative_distances[other_champ_2, other_champ_1]

        ### our champ _ other_champ##
        for our_champ in our_champs_index_list:
            for other_champ in other_champs_index_list:
                o1_role = champs['role'][our_champ]
                o2_role = champs['role'][other_champ]

                coeff_o1_num = 0
                coeff_o1_num = coeff_o1_num + 21 + 5 * o1_role + o2_role
                x_array[coeff_o1_num] = competitive_distances[our_champ, other_champ]

        xb_sum = np.inner(x_array, b_array)
        original_pick_winning_probability = np.exp(xb_sum) / (1 + np.exp(xb_sum))
        original_pick_winning_probabilities.append(original_pick_winning_probability)

    original_pick_winning_probabilities = pd.Series(original_pick_winning_probabilities)
    return original_pick_winning_probabilities


def simulate_recommended_pick(draft_statuses, coefficients):
    champs = get_champs()
    champs['cal_champs'] = 5 * champs['champion_id'] + champs['role']

    recommended_pick_winning_probabilities = []
    for (index, row) in tqdm(draft_statuses.iterrows()):
        banned_champs = row[BANNED_CHAMPS]
        our_champs = row[OUR_CHAMPS]
        other_champs = row[OTHER_CHAMPS]

        best_champions = recommend_best_champions(banned_champs, our_champs, other_champs, coefficients)
        best_winning_probability = 0
        for role, (champion_id, winning_probability) in best_champions.items():
            if winning_probability > best_winning_probability:
                best_winning_probability = winning_probability

        recommended_pick_winning_probabilities.append(best_winning_probability)

    recommended_pick_winning_probabilities = pd.Series(recommended_pick_winning_probabilities)
    return recommended_pick_winning_probabilities


if __name__ == '__main__':
    draft_statuses = get_draft_statuses()[:100]
    coefficients = pd.read_csv('data/mcmc_betas.csv', header=None)[1]

    original_pick_winning_probabilities = simulate_original_pick(draft_statuses, coefficients)
    recommended_pick_winning_probabilities = simulate_recommended_pick(draft_statuses, coefficients)

    print('Original average winning probability: {}'.format(np.mean(original_pick_winning_probabilities)))
    print('Recommendation average winning probability: {}'.format(np.mean(recommended_pick_winning_probabilities)))

