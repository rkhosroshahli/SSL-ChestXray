import torch
from utils import *
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics.cluster import normalized_mutual_info_score
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

        preds = outputs

        m = torch.nn.Linear(100, 14).to(device)
        out = m(outputs)
        outputs = torch.sigmoid(out)

        loss = criterion(outputs, labels)

        clustering = SpectralCoclustering(n_clusters=14, random_state=0).fit(preds.cpu().detach().numpy())
        rows = clustering.rows_.T
        rows= 1*rows

        nmis = 0
        for i in range(rows.shape[0]):
            nmi = normalized_mutual_info_score(labels.cpu().detach().numpy()[i], rows[i])
            nmis += nmi
        nmis_mean = nmis/rows.shape[0]

        final_loss = (loss +nmis_mean) / 2
        print("Classification loss: ", loss)
        print("Clustering loss: ", nmis_mean)

        print("Final loss: ", final_loss)

        if phase == 'train':

            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights

        running_loss += final_loss * batch_size
        #if (i % 200 == 0):
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
                    Eval["bestthr"][Eval[Eval["label"] == "Consolidation"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Edema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Emphysema"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Fibrosis"].index[0]],
                    Eval["bestthr"][Eval[Eval["label"] == "Pleural_Thickening"].index[0]],
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