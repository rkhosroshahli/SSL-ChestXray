import numpy as np
import pandas as pd
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
import skfuzzy as fuzz


def get_data(train_ssl_df):

    return 0


def Clustering_model(features, labels, batch_size):

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        features.T, 15, 2, error=0.005, maxiter=10, init=None)

    u = u.T

    nmis = 0
    for i in range(batch_size):
        i_lbl = labels[i]
        if(np.sum(i_lbl) == 0):
            i_lbl = np.append(i_lbl, [1])
        else:
            i_lbl = np.append(i_lbl, [0])

        nmi = 1 - normalized_mutual_info_score(i_lbl/np.sum(i_lbl), u[i])
        nmis += nmi
    nmis_mean = nmis/batch_size

    return nmis_mean
