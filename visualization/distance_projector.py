# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 28.
"""
import os

from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA

from stats.distance_calculator import calculate_collaborative_distances

IMAGE_PATH = 'visualization/collaborative_distances.png'


def project_collaborative_distances(dimension=10):
    """

    """
    # Get collaborative_distances by get_collaborative_distances.
    collaborative_distances = calculate_collaborative_distances()

    # Draw collaborative_distances using MDS.
    mds = manifold.MDS(n_components=dimension, metric=True, n_init=dimension, max_iter=10000, eps=1e-4,
                       n_jobs=os.cpu_count())
    pos = mds.fit(collaborative_distances).embedding_
    clf = PCA(n_components=10)
    pca_pos = clf.fit_transform(pos)
    pos_x = [x[0] for x in pca_pos]
    pos_y = [x[1] for x in pca_pos]
    plt.figure(figsize=(15, 15))
    plt.scatter(pos_x, pos_y)

    # Save the image in IMAGE_PATH.
    plt.savefig(fname=IMAGE_PATH)
    plt.close()


if __name__ == '__main__':
    project_collaborative_distances(20)
