import sys, re
from torch import nn
from segmentation_models_pytorch import Unet


def load_segment(backbone_name, num_class = 2, pretrained = False):
    if pretrained:
        weight = None
    else:
        weight = 'imagenet'
    print(f'Loading Unet: {backbone_name}. Using pretrained: {weight}')
    model = Unet(encoder_name=backbone_name,    # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=weight,         # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=num_class,
                activation=None)
    return model


class Unet_Segmentation(nn.Module):
    def __init__(self, backbone, num_class, pretrained, **kwargs):
        super(Unet_Segmentation, self).__init__()
        self.model = load_segment(backbone, num_class, pretrained)

    def forward(self, data):
        return self.model(data)
    
