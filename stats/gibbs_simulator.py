# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 5. 9.
"""
import numpy as np
from stats.regression_simulator import evaluate_predictions, get_data_set
from stats.gibbs_sampling import get_mcmc_betas


def build_simulation_model(train_Xs, test_Xs, train_Ys, test_Ys):
    """
    :param train_Xs: (ndarray[float]) An array of collaborative distances and competitive distances,
        size of 46 * (N * (1 - test_size)).
    :param test_Xs: (ndarray[float]) An array of collaborative distances and competitive distances,
        size of 46 * (N * test_size).
    :param train_Ys: (ndarray[float]) A series whether win or not. 1.0: win, 0.0: lose,
        size of (N * (1 - test_size)).
    :param test_Ys: (ndarray[float]) A series whether win or not. 1.0: win, 0.0: lose,
        size of (N * test_size).
    :param n_init: (int) The number of iterations of initializer
    :param tune: (int) The number of iterations to tune.

    :return betas: (ndarray[float]) Betas of logistic linear regression.
    """
    mcmc_betas = get_mcmc_betas(train_Ys, train_Xs)
    beta_X = np.dot(test_Xs, mcmc_betas)
    predicted_Ys = np.exp(beta_X) / (1 + np.exp(beta_X))

    accuracy, f1_score, AUC = evaluate_predictions(test_Ys, predicted_Ys, 'Win probability(%)', True, True,
                                                   postfix='gibbs_sampling')

    return mcmc_betas, accuracy, f1_score, AUC


if __name__ == '__main__':
    test_size = 0.2
    train_Xs, test_Xs, train_Ys, test_Ys = get_data_set(test_size=test_size)
    mcmc_betas, accuracy, f1_score, AUC = build_simulation_model(train_Xs, test_Xs, train_Ys, test_Ys)
    mcmc_betas.to_csv('stats/mcmc_betas.csv')
    print("accuracy: {}, f1_score: {}, AUC: {}".format(accuracy, f1_score, AUC))
