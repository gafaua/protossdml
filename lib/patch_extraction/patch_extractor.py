from pathlib import Path
from typing import Tuple

from numpy import ndarray
from tiatoolbox.tools.patchextraction import (
    SlidingWindowPatchExtractor,
)
from tiatoolbox.wsicore.wsireader import WSIReader


class CustiomSlidingWindowPatchExtractor(SlidingWindowPatchExtractor):
    def __init__(self,
                 input_img: str | Path | ndarray,
                 patch_size: int | Tuple[int, int],
                 input_mask: str | Path | ndarray | WSIReader = None,
                 resolution: int | float | Tuple[float, float] = 0,
                 units: str = "level",
                 stride: int | Tuple[int, int] = None,
                 pad_mode: str = "constant",
                 pad_constant_values: int | Tuple[int, int] = 0,
                 within_bound: bool = False,
                 min_mask_ratio: float = 0):
        super().__init__(input_img, patch_size, input_mask, resolution, units, stride, pad_mode, pad_constant_values, within_bound, min_mask_ratio)

    def __len__(self):
        return self.locations_df.shape[0] if self.locations_df is not None else 0


