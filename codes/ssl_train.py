import numpy as np
import random
from tqdm import tqdm
from batch_iterator import BatchIterator
from utils import *
from dataset import NIH
import pandas as pd
import warnings
import torchvision.transforms as transforms
import torch
from torch import nn
from torchvision import models
import datetime
import time
import csv
import os
from clustering import Clustering_model
from network import DenseNet121
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")
#from ResNetModel import *


def ModelTrain(train_lbl_df, train_unlbl_df, val_df, path_image, ModelType, CriterionType, device, LR):

    # Training parameters
    batch_size = 4

    workers = 1  # mean: how many subprocesses to use for data loading.
    N_LABELS = 14
    start_epoch = 0
    # number of epochs to train for (if early stopping is not triggered)
    num_epochs = 10

    train_lbl_df = train_lbl_df.filter(
        ['Image Index', 'Finding Labels'], axis=1)
    train_unlbl_df = train_unlbl_df.filter(
        ['Image Index', 'Finding Labels'], axis=1)

    train_df_size = len(train_lbl_df)
    val_df_size = len(val_df)

    random_seed = 33  # random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #image, label, item["Image Index"]

    val_loader = torch.utils.data.DataLoader(
        NIH(val_df[:800], path_image=path_image, transform=transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)

    if ModelType == 'densenet':
        # model = models.densenet121(pretrained=True)
        # num_ftrs = model.classifier.in_features

        # model_2 = model
        # model.classifier = nn.Sequential(
        #     nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, 128), nn.ReLU(),
        #     nn.Linear(128, 14), nn.Sigmoid())
        model = DenseNet121(out_size=N_LABELS, drop_rate=0.2)

    if ModelType == 'ResNet50':
        model = ResNet50NN()

    if ModelType == 'ResNet34':
        model = ResNet34NN()

    if ModelType == 'ResNet18':
        model = ResNet18NN()

    if ModelType == 'Resume':
        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)

    model = model.to(device)

    if CriterionType == 'BCELoss':
        criterion = nn.BCELoss().to(device)

    epoch_losses_train = []
    epoch_losses_val = []
    epoch_losses_pl = []

    since = time.time()

    best_loss = 999999
    best_epoch = -1
    pseudo_image_idx = torch.tensor(0)
    pseudo_labels = torch.tensor(0)
    train_ssl_df = {}
    # --------------------------Start of epoch loop
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        if epoch == 0:
            print("first training phase")
            train_ssl_df = train_lbl_df

        else:
            print("pseudo labeling phases")
            pseudo_df = pd.DataFrame({'Image Index': pseudo_image_idx,
                                      'Finding Labels': pseudo_labels
                                      })
            train_ssl_df = pd.concat(
                [train_ssl_df, pseudo_df], ignore_index=True)

            print("train set size: ", len(train_ssl_df))

        unlabeled_df = train_unlbl_df[(epoch) * 800:(epoch+1) * 800]
        unlabeled_size = len(unlabeled_df)

        train_loader = torch.utils.data.DataLoader(
            NIH(train_ssl_df, path_image=path_image, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Scale(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)

        unlabeled_loader = torch.utils.data.DataLoader(
            NIH(unlabeled_df, path_image=path_image, transform=transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                normalize
            ])),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)

    # -------------------------- Start of phase
        #model = model_2
        phase = 'train'
        # Classification loss calculation
        optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, model.parameters()), lr=LR)
        running_loss, train_features, train_labels = BatchIterator(model=model, phase=phase, Data_loader=train_loader,
                                                                   criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train.item())
        print("Train_losses:", epoch_losses_train)

        # Clustering loss calculation
        nmi_loss = Clustering_model(
            np.array(train_features), np.array(train_labels), batch_size)

        print(print("Clustering loss: ", nmi_loss))

        coloss = (epoch_loss_train + nmi_loss)/2

        print(print("Final loss: ", coloss))

        # dont need pseudo labeling

        # a function to find pseudo labels for trained images

        phase = 'val'
        optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, model.parameters()), lr=LR)
        running_loss, _, _ = BatchIterator(model=model, phase=phase, Data_loader=val_loader,
                                           criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val.item())
        print("Validation_losses:", epoch_losses_val)

        # phase = 'pseudo_label'
        # optimizer = torch.optim.Adam(params=filter(
        #     lambda p: p.requires_grad, model.parameters()), lr=LR)
        # pseudo_image_idx, pseudo_labels, running_loss = BatchIterator(
        #     model=model, phase=phase, Data_loader=unlabeled_loader, criterion=criterion, optimizer=optimizer, device=device)
        # epoch_loss_pl = running_loss / unlabeled_size
        # epoch_losses_pl.append(epoch_loss_pl.item())
        # print("Pseudo Labels_losses:", epoch_losses_pl)

        # checkpoint model if has best val loss yet
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            checkpoint(model, best_loss, best_epoch, LR)

            # log training and validation loss over each epoch
        with open("results/log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(
                    ["epoch", "train_loss", "val_loss", "Seed", "LR"])
            logwriter.writerow(
                [epoch, epoch_loss_train, epoch_loss_val, random_seed, LR])
# -------------------------- End of phase

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 2) + " as not seeing improvement in val loss")
                LR = LR / 2
                print("created new optimizer with LR " + str(LR))
                if ((epoch - best_epoch) >= 10):
                    print("no improvement in 10 epochs, break")
                    break
        #old_epoch = epoch
    # ------------------------- End of epoch loop
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    Saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, batch_size)
    #
    checkpoint_best = torch.load('../results/checkpoint')
    model = checkpoint_best['model']

    best_epoch = checkpoint_best['best_epoch']
    print(best_epoch)

    return model, best_epoch
