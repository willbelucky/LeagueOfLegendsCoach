# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 28.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scikitplot.metrics import plot_confusion_matrix
from scipy import stats
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data.data_reader import get_champs, get_pure_participants, WIN, CHAMPION_ID, ROLE, PARTICIPANT_ID, MATCH_ID, TEAM
from stats.distance_calculator import calculate_collaborative_distances, calculate_competitive_distances

MODEL_PATH = 'stats/logistic_regression.pkl'


def get_data_set(test_size=0.2):
    """
    :param test_size: (float) The size of test. test_size should be bigger than 0, and smaller than 1.

    :return train_Xs: (ndarray[float]) An array of collaborative distances and competitive distances, size of 45 * N.
    :return test_Xs: (ndarray[float]) An array of collaborative distances and competitive distances, size of 45 * N.
    :return train_Ys: (Series[Bool]) A series whether win or not. True: win, False: lose, size of N.
    :return test_Ys: (Series[Bool]) A series whether win or not. True: win, False: lose, size of N.
    """
    assert 0 < test_size < 1

    # Get Ys from participants.
    participants = get_pure_participants()
    wins = participants[WIN]
    Ys = wins.loc[[i for j, i in enumerate(wins.index) if j % 10 == 0]].astype(float).values

    # Get Xs from get_collaborative_distances and get_competitive_distances.
    champs = get_champs()
    champs = champs.reset_index()
    participants = pd.merge(participants, champs, on=(CHAMPION_ID, ROLE))
    participants = participants.sort_values(by=[MATCH_ID, TEAM, ROLE], ascending=[True, False, True])
    champion_indexes = participants['index'].values

    # First 5 champions were in blue team, and next 5 champions were in red team.
    match_number = len(Ys)
    champion_indexes = champion_indexes.reshape((match_number, 10))
    Xs = np.zeros((match_number, 45))

    # Get distances from get_collaborative_distances and get_competitive_distances
    collaborative_distances = calculate_collaborative_distances()
    competitive_distances = calculate_competitive_distances()

    for champion_index, X in tqdm(zip(champion_indexes, Xs)):
        # X_0 ~ X_9: Collaborative distances between blue team members.
        x_index = 0
        for i in range(0, 5):
            for j in range(i + 1, 5):
                X[x_index] = collaborative_distances[champion_index[i]][champion_index[j]]
                x_index += 1

        # X_10 ~ X_19: Collaborative distances between red team members.
        for i in range(5, 10):
            for j in range(i + 1, 10):
                X[x_index] = collaborative_distances[champion_index[i]][champion_index[j]]
                x_index += 1

        # X_20 ~ X_44: Competitive distances between blue team member and red team member.
        for i in range(0, 5):
            for j in range(5, 10):
                X[x_index] = competitive_distances[champion_index[i]][champion_index[j]]
                x_index += 1

    # Add intercept term to X.
    Xs = np.concatenate((np.array([[1.0]] * Xs.shape[0]), Xs), axis=1)
    train_Xs, test_Xs, train_Ys, test_Ys = train_test_split(Xs, Ys, test_size=test_size)

    return train_Xs, test_Xs, train_Ys, test_Ys


# noinspection PyPep8Naming
def plot_roc_curve(fpr, tpr, AUC, title=None, label=None, color='darkorange'):
    """

    :param fpr: (ndarray) The false-positive rate.
    :param tpr: (ndarray) The true-positive rate.
    :param AUC: (float) The AUC(Area Under the Curve) of the ROC curve.
    :param title: (str or None) If a title is not None, print the title on graphs.
    :param label: (str or None) If a label is not None, set the label as a label of a line.
    :param color: ()
    :return:
    """
    if title is None:
        title = 'ROC curve'

    if label is None:
        label = 'Regression'

    plt.plot(fpr, tpr, color=color, label=label + ' (area = %0.2f)' % AUC)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")


# noinspection PyPep8Naming
def evaluate_predictions(y_actual: np.ndarray, y_prediction: np.ndarray,
                         title=None, confusion_matrix_plotting=False, roc_curve_plotting=False, postfix=''):
    """

    :param y_actual: (ndarray[float]) The actual y values.
    :param y_prediction: (ndarray[float]) The predicted y values.
    :param title: (str or None) If a title is not None, print the title on graphs.
    :param confusion_matrix_plotting: (bool) If confusion_matrix_plotting is True, plot the confusion matrix.
    :param roc_curve_plotting: (bool) If roc_curve_plotting is True,
        plot the ROC(Receiver Operating Characteristic) curve.
    :param postfix: (string) The postfix of saved files.

    :return accuracy: (float) The portion of correct predictions.
    :return f1_score: (float) The harmonic mean of precision and recall
    :return AUC: (float) The AUC(Area Under the Curve) of the ROC(Receiver Operating Characteristic) curve.
    """
    assert len(y_actual) == len(y_prediction)
    y_prediction = np.where(y_prediction > 0.5, 1.0, 0.0)

    confusion_matrix = cm(y_actual.astype(bool), y_prediction.astype(bool))
    TN, FP, FN, TP = confusion_matrix.ravel()
    Precision = TP / (TP + FP + 1e-20)  # The portion of actual 1 of prediction 1.
    Recall = TP / (TP + FN + 1e-20)  # The portion of prediction 1 of actual 1.
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # The portion of correct predictions.
    f1_score = 2 / (1 / Precision + 1 / Recall + 1e-20)  # The harmonic mean of precision and recall.

    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_actual, y_prediction)
    AUC = auc(fpr, tpr)

    if confusion_matrix_plotting:
        plot_confusion_matrix(y_actual, y_prediction, title=title)
        plt.savefig('stats/confusion_matrix_{}.png'.format(postfix))
        plt.show()

    if roc_curve_plotting:
        plt.figure()
        plot_roc_curve(fpr, tpr, AUC, title=title)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.savefig('stats/roc_curve_{}.png'.format(postfix))
        plt.show()

    return accuracy, f1_score, AUC


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

    :return betas: (ndarray[float]) Betas of logistic linear regression.
    """
    logistic = LogisticRegression(fit_intercept=False, max_iter=1e5, tol=1e-5)
    logistic.fit(train_Xs, train_Ys)
    predicted_Ys = logistic.predict(test_Xs)

    accuracy, f1_score, AUC = evaluate_predictions(test_Ys, predicted_Ys, 'Blue team win probability(%)', True, True,
                                                   postfix='regression')

    return logistic, accuracy, f1_score, AUC


def save_model(model):
    """

    :param model:
    """
    joblib.dump(model, MODEL_PATH)


def load_model():
    """

    :return model:
    """
    model = joblib.load(MODEL_PATH)
    return model


def calculate_p_value(model, train_Xs, train_Ys):
    sse = np.sum((model.predict(train_Xs) - train_Ys) ** 2, axis=0) / float(train_Xs.shape[0] - train_Xs.shape[1])
    se = np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(train_Xs.T, train_Xs))))
    t = model.coef_ / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), train_Ys.shape[0] - train_Xs.shape[1]))

    return p


if __name__ == '__main__':
    test_size = 0.2
    train_Xs, test_Xs, train_Ys, test_Ys = get_data_set(test_size=test_size)
    model, accuracy, f1_score, AUC = build_simulation_model(train_Xs, test_Xs, train_Ys, test_Ys)
    save_model(model)
    p = calculate_p_value(model, train_Xs, train_Ys)
    result = pd.DataFrame(data={'coefficient': model.coef_[0], 'p-value': p[0]})
    result.to_csv('stats/regression_result.csv')
