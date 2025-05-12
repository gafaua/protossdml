import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
from torchstain.torch.normalizers import TorchMacenkoNormalizer

CAMELYON_NORMALIZATION_MEAN = [0.6684, 0.5115, 0.6791]
CAMELYON_NORMALIZATION_STD = [0.2521, 0.2875, 0.2100]

class TwoCropsTransform(nn.Module):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, device):
        super().__init__()
        self.base_transform = base_transform
        self.device = device

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

    def forward(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class RandomSharpenAugmentation(nn.Module):
    def __init__(self, min_kernel, max_kernel, sigma=2, p=0.5):
        super().__init__()
        self.kernel_choices = list(range(min_kernel, max_kernel, 2))
        self.p = p
        self.sigma=sigma

    def forward(self, img: torch.Tensor):
        if np.random.random() < self.p:
            kernel = int(np.random.choice(self.kernel_choices))
            gaussian = F.gaussian_blur(img, kernel, self.sigma)
            return ((img - gaussian)+img).clip(0,1)
        else:
            return img


class MacenkoNormalizer(nn.Module):
    def __init__(self, device="cpu") -> None:
        super().__init__()
        self.norm = TorchMacenkoNormalizer()
        self.device = device
        self.norm.HERef = self.norm.HERef.to(device)
        self.norm.maxCRef = self.norm.maxCRef.to(device)

    def forward(self, x):
        try:
            with torch.no_grad():
                y, _, _ = self.norm.normalize(x*255, stains=False)
                y = y / 255.
                x = y.permute(2, 0, 1)
        except IndexError:
            #print("WARN: kth Error in Macenko Normalizer, the patch probably doesn't contain any tissue.")
            #functional.to_pil_image(x).save("kth.png")
            pass
        except torch._C._LinAlgError:
            #print("WARN: linalg Error in Macenko Normalizer, the patch probably doesn't contain any tissue.")
            #functional.to_pil_image(x).save("linalg.png")
            pass

        return x


def get_augmentations(args):
    crop=224

    augmentations = []
    if args.fixed_scale:
        print("USING FIXED SCALE")
        augmentations.append(transforms.RandomCrop(crop))
    else:
        print("NO FIXED SCALE")
        augmentations.append(transforms.RandomResizedCrop(crop, scale=(0.2, 1.), antialias=True))

    augmentations.append(MacenkoNormalizer(device=args.device))
    augmentations.append(transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8))

    if args.random_sharpen:
        print("USING RANDOM SHARPEN")
        augmentations.append(RandomSharpenAugmentation(1, 33))
    else:
        print("NO RANDOM SHARPEN")

    augmentations.extend([
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, [.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=CAMELYON_NORMALIZATION_MEAN,
                                  std=CAMELYON_NORMALIZATION_STD)
    ])

    augmentations = nn.Sequential(
        transforms.RandomCrop(384 if args.fixed_scale else crop),
        TwoCropsTransform(nn.Sequential(*augmentations).to(args.device), args.device)
    )

    return augmentations

