import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from lib.augmentations import MacenkoNormalizer
from lib.vision_transformer import vit_small

CAMELYON_NORMALIZATION_MEAN = [0.6684, 0.5115, 0.6791]
CAMELYON_NORMALIZATION_STD = [0.2521, 0.2875, 0.2100]


def get_image_encoder_CAMELYON16(checkpoint="./checkpoint_c16.pth"):
    model = vit_small()
    data = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(data)
    model.eval()
    return model

def get_transforms():
    transform = nn.Sequential(*[
        MacenkoNormalizer("cpu"),
        transforms.Normalize(mean=CAMELYON_NORMALIZATION_MEAN,
                             std=CAMELYON_NORMALIZATION_STD),
    ])

    return transform

def get_protossdml_encoder(checkpoint="./checkpoint_c16.pth"):
    return get_image_encoder_CAMELYON16(checkpoint), get_transforms()
