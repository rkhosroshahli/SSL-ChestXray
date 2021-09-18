import torch
from torch import nn
from utils import *
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics.cluster import normalized_mutual_info_score
# fcmeans import FCM
import skfuzzy as fuzz
#from evaluation import *


def BatchIterator(
    model,
    phase,
    Data_loader,
    criterion,
    optimizer,
    device
):

    # --------------------  Initial paprameterd
    grad_clip = 0.5  # clip gradients at an absolute value of

    print_freq = 1000
    running_loss = 0.0

    PRED_LABEL = ['Atelectasis',
                  'Cardiomegaly',
                  'Effusion',
                  'Infiltration',
                  'Mass',
                  'Nodule',
                  'Pneumonia',
                  'Pneumothorax',
                  'Consolidation',
                  'Edema',
                  'Emphysema',
                  'Fibrosis',
                  'Pleural_Thickening',
                  'Hernia']

    pseudo_labels = []
    pseudo_indexes = []

    for i, data in enumerate(Data_loader):

        imgs, labels, image_idx = data

        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(imgs)

        #preds = outputs

        seq = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 14), nn.Sigmoid()).to(device)

        model.classifier = seq
        model.train()
        outputs_sgd = model(imgs)

        #m = torch.nn.Linear(1024, 14).to(device)
        # print(outputs_sgd.size())
        #out = seq(outputs)
        #output = torch.sigmoid(out)

        loss = criterion(outputs_sgd, labels)
        print(loss)

        # featured_out = outputs.cpu().detach().numpy()
        # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        #     featured_out.T, 15, 2, error=0.005, maxiter=10, init=None)

        # #print(np.sum(u, axis=0))
        # print("***************")

        # # fcm = FCM(n_clusters=14)
        # # imgs_np=outputs.cpu().detach().numpy()
        # # print(imgs_np.shape)
        # # fcm.fit(imgs_np)
        # # print(fcm.centers)
        # # print(fcm.predict(imgs_np))

        # # clustering = SpectralCoclustering(n_clusters=14, random_state=0).fit(preds.cpu().detach().numpy())
        # # rows = clustering.rows_.T
        # # rows= 1 * rows
        # true_lbl = labels.cpu().detach().numpy()
        # u = u.T

        # nmis = 0
        # for i in range(batch_size):
        #     i_lbl = true_lbl[i]
        #     if(np.sum(i_lbl) == 0):
        #         i_lbl = np.append(i_lbl, [1])
        #     else:
        #         i_lbl = np.append(i_lbl, [0])

        #     nmi = 1 - normalized_mutual_info_score(i_lbl/np.sum(i_lbl), u[i])
        #     nmis += nmi
        # nmis_mean = nmis/batch_size

        # final_loss = (loss + nmis_mean) / 2
        # print(type(final_loss))
        # print("Classification loss: ", loss)
        # print("Clustering loss: ", nmis_mean)

        # print("Final loss: ", final_loss)
        # break

        if phase == 'train':

            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights

        running_loss += loss * batch_size
        # if (i % 200 == 0):
        #print(str(i * batch_size))

        if phase == 'pseudo_label':
            Eval = pd.read_csv("../results/Threshold.csv")
            thrs = [Eval["bestthr"][Eval[Eval["label"] == "Atelectasis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Cardiomegaly"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Effusion"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Infiltration"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Mass"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Nodule"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumonia"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pneumothorax"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"]
                                         == "Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] ==
                                         "Pleural_Thickening"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Hernia"].index[0]]]

            probs = outputs.cpu().data.numpy()

            # get predictions and true values for each item in batch
            for j in range(0, batch_size):
                pseudos = []

                for k in range(labels.size()[1]):
                    if probs[j, k] >= thrs[k]:
                        pseudos.append(PRED_LABEL[k])

                if len(pseudos) == 0:
                    pseudos = 'No Finding'
                else:
                    pseudos = "|".join(pseudos)

                pseudo_labels.append(pseudos)
                pseudo_indexes.append(image_idx[j])

    if phase == 'pseudo_label':
        return pseudo_indexes, pseudo_labels, running_loss

    return running_loss
