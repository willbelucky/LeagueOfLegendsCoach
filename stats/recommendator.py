# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 5. 8.
"""


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
    best_champions = {}
    return best_champions


if __name__ == '__main__':
    banned_champs
    our_champs
    other_champs
    coefficients
    recommendation = recommend_best_champions(banned_champs, our_champs, other_champs, coefficients)
    print(recommendation)
