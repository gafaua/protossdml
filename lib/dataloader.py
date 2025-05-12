from typing import List

import numpy as np
from wsiloader import WSIDataloader

from lib.patch_extraction import get_patches


def make_bunch_wsi_loader(wsi_paths: List[str],
                          wsi_resolution: List[int],
                          patch_size: int,
                          transforms,
                          batch_size: int,
                          num_workers: int,
                          shuffle=False,
                          overlap=0.0):

    def patch_generator(wsi_path):
        wsi_res = np.random.choice(wsi_resolution)

        return get_patches(wsi_path, patch_size, wsi_res, overlap)

    sampler = WSIDataloader(
        wsi_paths=wsi_paths,
        patch_generator=patch_generator,
        transforms=transforms,
        transforms_device="cuda",
        collate_fn=None,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
    )

    return sampler
