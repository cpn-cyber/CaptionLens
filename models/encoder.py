import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        modules = list(self.resnet.children())[:-2]  # keep up to last conv layer
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # output: (B, 2048, 14, 14)
        self.conv1x1 = nn.Conv2d(2048, embed_size, kernel_size=1)  # reduce channel dim

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # (B, 2048, H, W)
        features = self.adaptive_pool(features)  # (B, 2048, 14, 14)
        features = self.conv1x1(features)  # (B, embed_size, 14, 14)
        features = features.flatten(2).permute(0, 2, 1)  # (B, 196, embed_size)
        return features  # each image as sequence of 196 vectors
