import sys
from typing import List

import numpy as np
import torch.nn as nn
from PIL import Image
from shapely import Point, Polygon, STRtree
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from lib.patch_extraction import get_patches

sys.path.append("/home/travail/ASAP/ASAP-ASAP-2.1/install/usr/local/bin")
import multiresolutionimageinterface as mir

from lib.wsi_analysis.utils import read_polygon_annotations_from_xml

CAMELYON_DATASET_PATH = "/store2/travail/data/CAMELYON16/"


class WSIDataset(Dataset):
    def __init__(self,
                 path: str,
                 patch_size: int,
                 resolution: float,
                 transforms=None,
                 annotations_files: List[str] = None,
                 annotations_type: str = "camelyon_xml",
                 patho_only=False,
                 consensus_only=False,
                 verbose=False) -> None:
        """WSI Wrapper for analysis as dataset of patches

        Args:
            path (str): path to WSI file (.ndpi)
            patch_size (int): size of patches to extract from the WSI
            resolution (float): resolution in power units. Typically 10, 20, 40
            norm_type (str): Normalization procedure to apply to patches
        """
        self.wsi_path = path
        self.patch_size = patch_size
        self.resolution = resolution
        self.patho_only = patho_only

        if annotations_type == "camelyon_xml":
            # Use the CAMELYON provided tissue masks
            # These lines are necessary to enable the use of ASAP and multiresolutionimageinterface
            name = self.wsi_path.split("/")[-1].split(".")[0]
            mask_path = f"{CAMELYON_DATASET_PATH}/masks/{name}_mask.tif"
            mask = mir.MultiResolutionImageReader().open(mask_path)
            level_width, level_height = mask.getLevelDimensions(level=5)
            mask = mask.getUCharPatch(startX=0, startY=0, width=level_width, height=level_height, level=5)
            self.patches = get_patches(wsi_path=self.wsi_path, patch_size=self.patch_size, resolution=self.resolution, mask=mask)
        else:
            self.patches = get_patches(wsi_path=self.wsi_path, patch_size=self.patch_size, resolution=self.resolution, mask_otsu_factor=0.4)

        # Get all Top-left patch coords
        self.coords = []
        for i in range(len(self.patches)):
            coord = (self.patches.locations_df["x"][i], self.patches.locations_df["y"][i])
            self.coords.append(coord)

        if annotations_files is None:
            self.anomalies = [-1] * len(self.patches)
            self.annotations = None
        else:
            if annotations_type == "camelyon_xml":
                self.annotations = read_polygon_annotations_from_xml(annotations_files[0]) # only 1 annotation file per WSI

                self.__handle_camelyon_annotations()
            else:
                raise NotImplementedError(f"No support for {annotations_type} annotations type")

        if patho_only:
            self.indices = np.array(list(range(len(self.patches))))[(np.array(self.anomalies)==1).nonzero()]
        elif consensus_only:
            self.indices = np.array(list(range(len(self.patches))))[(np.array(self.anomalies)!=0).nonzero()]

        if transforms is None:
            self.transforms = None
        else:
            self.transforms = transforms

        if verbose:
            print(f"{len(self.patches):,} patches of size {patch_size}x{patch_size} generated from {path} at resolution power {resolution}")
            print(f"{len(self.patches) - np.count_nonzero(np.array(self.anomalies)==-1):,} patches within anomalous regions")
            if patho_only:
                print("ONLY INDEXING PATCHES WITHIN PATHOLOGICAL REGIONS")


    def __handle_camelyon_annotations(self):
        self.anomalies = -np.ones(len(self.patches)) # [-1] for True negatives

        # We use here the Polygon annotations to precisely get tumor boundaries
        tumors = [Polygon(p) for p in self.annotations["Tumor"]]
        exclusions = [Polygon(p) for p in self.annotations["Exclusion"]]

        mult = self.patches.wsi.info.objective_power/self.resolution
        points = [Point((x+self.patch_size/2)*mult, (y+self.patch_size/2)*mult) for x,y in self.coords]
        tumors = STRtree(tumors)
        exclusions = STRtree(exclusions)
        in_tumors = tumors.query(points, predicate="intersects")
        self.anomalies[in_tumors[0]] = 1

        if len(exclusions) > 0 :
            in_exclusions = exclusions.query(points, predicate="intersects")
            self.anomalies[in_exclusions[0]] = -1


    def __len__(self):
        if self.patho_only:
            return len(self.indices)
        else:
            return len(self.patches)


    def __getitem__(self, idx):
        if self.patho_only:
            patch = self.patches[int(self.indices[idx])]
        else:
            patch = self.patches[idx]

        img = Image.fromarray(patch)

        img = to_tensor(img)

        if self.transforms is not None:
            device = "cpu"
            if type(self.transforms) == nn.Sequential:
                if hasattr(self.transforms, "device"):
                    device = self.transforms.device
                elif hasattr(self.transforms[0], "device"):
                    device = self.transforms[0].device

            img = self.transforms(img.to(device))

        return img, self.anomalies[idx], self.coords[idx]
