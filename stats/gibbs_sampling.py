# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 5. 8.
"""
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3.glm import GLM
from pymc3.glm.families import Binomial
import theano.tensor as tt

plt.style.use('seaborn-darkgrid')


if __name__ == '__main__':

    df_logistic = pandas.DataFrame({'x1': X1, 'y': Y > np.median(Y)})

    with pm.Model() as model_glm_logistic:
        GLM.from_formula('Y ~ X_1 + X_2 + X_3 + X_4 + X_5 + X_6 + X_7 + X_8 + X_9 + X_10'
                         ' + X_11 + X_12 + X_13 + X_14 + X_15 + X_16 + X_17 + X_18 + X_19 + X_20'
                         ' + X_21 + X_22 + X_23 + X_24 + X_25 + X_26 + X_27 + X_28 + X_29 + X_30'
                         ' + X_31 + X_32 + X_33 + X_34 + X_35 + X_36 + X_37 + X_38 + X_39 + X_40'
                         ' + X_41 + X_42 + X_43 + X_44 + X_45', df_logistic, family=Binomial())
        trace = pm.sample()
        pm.traceplot(trace)
        plt.show()
