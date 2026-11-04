import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn
import torch

class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.vgg = self.__get_vgg_model()
        self.target_layers = {
            1: "relu1_1",
            6: "relu2_1",
            11: "relu3_1",
            20: "relu4_1",
        }

    def forward(self, x):
        features = {}

        for idx, layer in enumerate(self.vgg):
            x = layer(x)

            if idx in self.target_layers:
                features[self.target_layers[idx]] = x

            if idx == 20:
                break

        return features

    def __get_vgg_model(self):
        vgg_19_model = vgg19(
            weights=VGG19_Weights.IMAGENET1K_V1
        ).features
        vgg_19_model = vgg_19_model.to(device = self.device)
        vgg_19_model.eval()
                
        # Freeze all parameters
        for param in vgg_19_model.parameters():
            param.requires_grad = False
            
        return vgg_19_model

def normalize_vgg(image):
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=image.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=image.device
    ).view(1, 3, 1, 1)

    return (image - mean) / std