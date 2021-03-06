import torch
from train import ModelTrain
from learning_curve import PlotLearnignCurve
from test import make_pred_multilabel
from nih import plot_sort_median
#from Actual_TPR import Actual_TPR

import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# ---------------------- on q
path_image = "../images/"

train_df_path = "../datas/train.csv"
test_df_path = "../datas/test.csv"
val_df_path = "../datas/val.csv"


diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
            'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
# age_decile = ['0-20', '20-40', '40-60', '60-80', '80-']
age_decile = ['40-60', '60-80', '20-40', '80-', '0-20']
gender = ['M', 'F']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODE = "train"  # Select "train" or "test", "Resume", "plot", "Threshold", "plot15"

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df path", train_df_size)

    test_df = pd.read_csv(test_df_path)
    test_df_size = len(test_df)
    print("test_df path", test_df_size)

    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("val_df path", val_df_size)

    if MODE == "train":
        ModelType = "densenet"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(
            train_df, val_df, path_image, ModelType, CriterionType, device, LR)

        PlotLearnignCurve()

    if MODE == "test":
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)

        CheckPointData = torch.load('../results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, device)

    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(
            train_df, val_df, path_image, ModelType, CriterionType, device, LR)

        PlotLearnignCurve()

    if MODE == "plot":
        gt = pd.read_csv("../results/True.csv")
        pred = pd.read_csv("../results/bipred.csv")
        factor = [gender, age_decile]
        factor_str = ['Patient Gender', 'Patient Age']
        for i in range(len(factor)):
            # plot_frequency(gt, diseases, factor[i], factor_str[i])
            # plot_TPR_NIH(pred, diseases, factor[i], factor_str[i])
            plot_sort_median(pred, diseases, factor[i], factor_str[i])
           # Actual_TPR(pred, diseases, factor[i], factor_str[i])

    #         plot_Median(pred, diseases, factor[i], factor_str[i])


if __name__ == "__main__":
    main()
