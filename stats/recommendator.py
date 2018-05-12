# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 5. 8.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.data_reader import get_champs, get_pure_participants
from stats.distance_calculator import calculate_collaborative_distances, calculate_competitive_distances


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
    NUMBER_OF_OTHER_CHAMPS = len(other_champs)
    NUMBER_OF_OUR_CHAMPS = len(our_champs)
    ROLE = [0,1,2,3,4]

    champs = get_champs()
    champs_list = np.array(champs['champion_id'].drop_duplicates()).tolist()



    # except champs_id: banned_champs + our_champs_id + other_champs_id
    other_champs_id = []
    other_champs_role = []
    other_champs_index_list = []

    for i in range(NUMBER_OF_OTHER_CHAMPS):
        other_champs_id = other_champs_id.append(other_champs[i] // 5)
        other_champs_role = other_champs_role.append(other_champs[i] % 5)

        other_champs_index = champs.iloc[np.where(champs['champion_id'] == (other_champs[i] // 5))]
        other_champs_index = other_champs_index.iloc[np.where(champs_index['role'] == (other_champs[i] % 5))]
        other_champs_index = other_champs_index.index

        other_champs_index_list = other_champs_index_list.append(other_champs_index)

    our_champs_id = []
    our_champs_role = []
    our_champs_index_list = []


    for j in range(NUMBER_OF_OUR_CHAMPS):
        our_champs_id = our_champs_id.append(our_champs[j] // 5)
        our_champs_role = our_champs_role.append(our_champs[j] % 5)

        our_champs_index = champs.iloc[np.where(champs['champion_id'] == (our_champs[j] // 5))]
        our_champs_index = our_champs_index.iloc[np.where(champs_index['role'] == (our_champs[j] % 5))]
        our_champs_index = our_champs_index.index

        our_champs_index_list = our_champs_index_list.append(our_champs_index)

    except_champs_id = other_champs_id + banned_champs + our_champs
    champs_list_except_selected_ones = list(set(champs_list) - set(except_champs_id))

    remain_roles = list(set(ROLE) - set(our_champs_role))
    collaborative_distances = calculate_collaborative_distances()
    competitive_distances = calculate_competitive_distances()

    Y_value = 0
    sum = 0
    best_champions_list = []
    for k in remain_roles:
        for l in champs_list_except_selected_ones:
            champs_index = champs.iloc[np.where(champs['champion_id'] == l)]
            champs_index = champs.iloc[np.where(champs_index['role'] == k)]
            champs_index = champs_index.index
            for m in our_champs_index_list:
                sum = sum + collaborative_distances[champs_index, m]*coefficients[len(our_champs_index_list)*(len(our_champs_index_list)-1)/2+our_champs_index_list.index(m)+1]
            for n in other_champs_index_list:
                sum = sum + competitive_distances[champs_index, n]*coefficients[len(our_champs_index_list)*len(other_champs_index_list)/2+other_champs_index_list.index(n)+1]

            if Y_value < sum:
                Y_value = sum
                best_champions_list = best_champions_list.append(champs_index)
                if l == champs_list_except_selected_ones[-1]:
                    best_champions_role = best_champions_list[-1] % 5
                    best_champions_id = best_champions_list[-1] // 5 + 1
                    best_champions[best_champions_role] = best_champions_id

                    remain_roles = remain_roles.append(best_champions_role)
                    champs_list_except_selected_ones = champs_list_except_selected_ones.append(best_champions_id)
            else:
                if l == champs_list_except_selected_ones[-1]:
                    best_champions_role = best_champions_list[-1] % 5
                    best_champions_id = best_champions_list[-1] // 5 + 1
                    best_champions[best_champions_role] = best_champions_id

                    remain_roles = remain_roles.append(best_champions_role)
                    champs_list_except_selected_ones = champs_list_except_selected_ones.append(best_champions_id)
                else:
                    continue



    best_champions = {}
    return best_champions


