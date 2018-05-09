# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 5. 8.
"""
import matplotlib.pyplot as plt
import pandas
import pymc3 as pm
from pymc3.glm import GLM
from pymc3.glm.families import Binomial

from stats.regression_simulator import get_data_set

plt.style.use('seaborn-darkgrid')


def get_mcmc_betas(train_Ys, train_Xs, n_init=200000, tune=500):
    """

    :return mcmc_betas: (Series) Coefficients of intercept and betas.
    """
    print('train_size:', len(train_Xs))

    train_data = pandas.DataFrame({
        'Y': train_Ys,
        'X_1': train_Xs[:, 1],
        'X_2': train_Xs[:, 2],
        'X_3': train_Xs[:, 3],
        'X_4': train_Xs[:, 4],
        'X_5': train_Xs[:, 5],
        'X_6': train_Xs[:, 6],
        'X_7': train_Xs[:, 7],
        'X_8': train_Xs[:, 8],
        'X_9': train_Xs[:, 9],
        'X_10': train_Xs[:, 10],
        'X_11': train_Xs[:, 11],
        'X_12': train_Xs[:, 12],
        'X_13': train_Xs[:, 13],
        'X_14': train_Xs[:, 14],
        'X_15': train_Xs[:, 15],
        'X_16': train_Xs[:, 16],
        'X_17': train_Xs[:, 17],
        'X_18': train_Xs[:, 18],
        'X_19': train_Xs[:, 19],
        'X_20': train_Xs[:, 20],
        'X_21': train_Xs[:, 21],
        'X_22': train_Xs[:, 22],
        'X_23': train_Xs[:, 23],
        'X_24': train_Xs[:, 24],
        'X_25': train_Xs[:, 25],
        'X_26': train_Xs[:, 26],
        'X_27': train_Xs[:, 27],
        'X_28': train_Xs[:, 28],
        'X_29': train_Xs[:, 29],
        'X_30': train_Xs[:, 30],
        'X_31': train_Xs[:, 31],
        'X_32': train_Xs[:, 32],
        'X_33': train_Xs[:, 33],
        'X_34': train_Xs[:, 34],
        'X_35': train_Xs[:, 35],
        'X_36': train_Xs[:, 36],
        'X_37': train_Xs[:, 37],
        'X_38': train_Xs[:, 38],
        'X_39': train_Xs[:, 39],
        'X_40': train_Xs[:, 40],
        'X_41': train_Xs[:, 41],
        'X_42': train_Xs[:, 42],
        'X_43': train_Xs[:, 43],
        'X_44': train_Xs[:, 44],
        'X_45': train_Xs[:, 45]
    })

    with pm.Model():
        GLM.from_formula('Y ~ X_1 + X_2 + X_3 + X_4 + X_5 + X_6 + X_7 + X_8 + X_9 + X_10'
                         ' + X_11 + X_12 + X_13 + X_14 + X_15 + X_16 + X_17 + X_18 + X_19 + X_20'
                         ' + X_21 + X_22 + X_23 + X_24 + X_25 + X_26 + X_27 + X_28 + X_29 + X_30'
                         ' + X_31 + X_32 + X_33 + X_34 + X_35 + X_36 + X_37 + X_38 + X_39 + X_40'
                         ' + X_41 + X_42 + X_43 + X_44 + X_45',
                         train_data, family=Binomial())
        trace = pm.sample()
        summary = pm.summary(trace)
        pm.traceplot(trace)
        plt.show()

    mcmc_betas = summary['mean']
    return mcmc_betas


if __name__ == '__main__':
    test_size = 0.995
    train_Xs, test_Xs, train_Ys, test_Ys = get_data_set(test_size=test_size)
    mcmc_betas = get_mcmc_betas(train_Ys, train_Xs)
    print(mcmc_betas)
