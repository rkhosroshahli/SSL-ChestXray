from torchvision.models.densenet import densenet121

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size, drop_rate=0):
        super(DenseNet121, self).__init__()
        self.densenet121 = densenet121(
            pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, out_size), nn.Sigmoid()
        )

        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        # if self.drop_rate > 0:
        #     out = self.drop_layer(out)
        self.activations = out
        out = self.densenet121.classifier(out)

        return self.activations, out
