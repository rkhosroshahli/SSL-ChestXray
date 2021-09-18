import numpy as np
import pandas as pd
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
import skfuzzy as fuzz


def get_data(train_ssl_df):

    return 0


def Clustering_model(train_data, model, optimizer, device):

    #imgs, labels = get_data(train_ssl_df)

    for i, data in enumerate(train_data):

        imgs, labels, image_idx = data

        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        print(imgs.size())
        labels = labels.to(device)

        optimizer.zero_grad()
        model.train()
        outputs = model(imgs)

        featured_out = outputs.cpu().detach().numpy()
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            featured_out.T, 15, 2, error=0.005, maxiter=10, init=None)

        true_lbl = labels.cpu().detach().numpy()
        u = u.T

        nmis = 0
        for i in range(4000):
            i_lbl = true_lbl[i]
            if(np.sum(i_lbl) == 0):
                i_lbl = np.append(i_lbl, [1])
            else:
                i_lbl = np.append(i_lbl, [0])

            nmi = 1 - normalized_mutual_info_score(i_lbl/np.sum(i_lbl), u[i])
            nmis += nmi
        nmis_mean = nmis/4000
