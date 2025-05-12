import numpy as np
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader

from lib.patch_extraction.masker import get_mask
from lib.patch_extraction.patch_extractor import CustiomSlidingWindowPatchExtractor


def get_patches(wsi_path: str,
                patch_size: int,
                resolution: float,
                overlap: float=0.0,
                mask_otsu_factor: float=0.4,
                mask:np.ndarray=None):
    """_summary_

    Args:
        wsi_path (str): Path to wsi
        patch_size (int): Size of square patch to extract from WSI.
        resolution (float): Magnification objective to use to extract patches (2.5, 5.0, 10.0, 20.0)
        overlap (float, optional): Ratio of overlap between neighboring patches. Defaults to 0.0.
        mask_otsu_factor (float, optional): Importance given to Otsu factor in the masking process, as
                                            opposed to the dominant color method. We advise a value in [0.4, 0.6],
                                            but it will depend on the stain and tissue. Defaults to 0.4.
        mask (np.ndarray, optional): Preprocessed mask to use to override the automatic tissue detection. Defaults to None.

    Returns:
        CustiomSlidingWindowPatchExtractor: Tiatoolbox based sliding window patch extractor enabling iteration and indexing
                                            of the patches.
    """
    wsi = WSIReader.open(input_img=wsi_path)

    if mask is None:
        mask = get_mask(wsi, min_region_size=100, kernel_size=3, otsu_factor=mask_otsu_factor) #These values were found empirically
    else:
        mask = VirtualWSIReader(mask.squeeze().astype(np.uint8), info=wsi.info, mode="bool")

    patches = CustiomSlidingWindowPatchExtractor(
        input_img=wsi,
        patch_size=(patch_size,)*2,
        stride=(patch_size - int(overlap*patch_size),)*2,
        resolution=resolution,
        units="power",
        input_mask=mask,
        min_mask_ratio=0.3,
        within_bound=True,
    )

    return patches

