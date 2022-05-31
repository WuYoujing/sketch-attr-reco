import torch.nn as nn
from torchvision import transforms, models


class FeatureExtraction(nn.Module):
    def __init__(self, pretrained):
        super(FeatureExtraction, self).__init__()

        self.model = models.resnet18(pretrained=pretrained)

        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, img):
        return self.model(img)


class FullConnect(nn.Module):
    def __init__(self, input_dim=512, output_dim=2):
        super(FullConnect, self).__init__()

        output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Dropout(p=0.7),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.Dropout(p=0.7),
            nn.Linear(16,4),
            nn.ReLU(True),
            nn.Dropout(p=0.7),
            nn.Linear(4, output_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        return self.sigmoid(self.fc(inputs))


class Sum_model(nn.Module):
    def __init__(self, pretrained):
        super(Sum_model, self).__init__()

        self.featureExtractor = FeatureExtraction(pretrained)

        self.feature_dim = 512
        self.cla = FullConnect(input_dim=self.feature_dim)

    def forward(self, img):
        features = self.featureExtractor(img)
        haircla = self.cla(features)
        return haircla
