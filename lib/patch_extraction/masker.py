import cv2
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from tiatoolbox.tools.tissuemask import TissueMasker
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, WSIReader


class BetterMorphologicalMasker(TissueMasker):
    """Tissue Masker based on Tiatoolbox's MorphologicalMasker.

    It combines Otsu's method and a dominant color detection based on the 
    K-Means algorithm in RGB space to establish a cutoff threshold.

    """
    def __init__(
        self, *, kernel_size=None, min_region_size=None, otsu_factor=0.4,
    ) -> None:
        """Initialise a morphological masker.

        Args:
            mpp (float or tuple(float)):
                The microns per-pixel of the image to be masked. Used to
                calculate kernel_size a 64/mpp, optional.
            power (float or tuple(float)):
                The objective power of the image to be masked. Used to
                calculate kernel_size as 64/objective_power2mpp(power),
                optional.
            kernel_size (int or tuple(int)):
                Size of elliptical kernel in x and y, optional.
            min_region_size (int):
                Minimum region size in pixels to consider as foreground.
                Defaults to area of the kernel.
            otsu_factor (float):
                Importance to give to the threshold found by Otsu's method compared
                with the dominant color method. Must be in the range [0, 1].

        """
        self.min_region_size = min_region_size
        self.otsu_factor = np.clip(otsu_factor, 0, 1)
        self.threshold = None

        def make_kernel(size):
            # Ensure kernel_size is a length 2 numpy array
            kernel_size = np.array(size)
            if kernel_size.size != 2:
                kernel_size = kernel_size.repeat(2)

            # Convert to an integer double/ pair
            kernel_size = tuple(np.round(kernel_size).astype(int))

            # Create structuring element for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            return kernel

        self.kernel_size = make_kernel(kernel_size)

        # Set min region size to kernel area if None
        if self.min_region_size is None:
            self.min_region_size = np.sum(self.kernel)


    def fit(self, images: np.ndarray, masks=None) -> None:
        """Find a binary threshold using KMeans dominant colors.

        Args:
            images (:class:`numpy.ndarray`):
                List of images with a length 4 shape (N, height, width,
                channels).
            masks (:class:`numpy.ndarray`):
                Unused here, for API consistency.

        """
        images_shape = np.shape(images)
        if len(images_shape) != 4:
            raise ValueError(
                "Expected 4 dimensional input shape (N, height, width, 3)"
                f" but received shape of {images_shape}."
            )

        # Convert RGB images to greyscale
        grey_images = [x[..., 0] for x in images]
        if images_shape[-1] == 3:
            grey_images = np.zeros(images_shape[:-1], dtype=np.uint8)
            for n, image in enumerate(images):
                grey_images[n] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        pixels = np.concatenate([np.array(grey).flatten() for grey in grey_images])
        self.threshold_otsu = threshold_otsu(pixels)

        # Find Threshold using KMeans on all pixels (find dominant color)
        kmeans = KMeans(2, n_init="auto")
        kmeans.fit(pixels.reshape(-1, 1))
        self.threshold = np.max(kmeans.cluster_centers_)

        self.fitted = True


    def transform(self, images: np.ndarray):
        """Create masks using the found threshold followed by morphological operations.


        Args:
            images (:class:`numpy.ndarray`):
                List of images with a length 4 shape (N, height, width,
                channels).

        Returns:
            :class:`numpy.ndarray`:
                List of images with a length 4 shape (N, height, width,
                channels).

        """

        results = []
        for image in images:
            if len(image.shape) == 3 and image.shape[-1] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            mask = (gray < (self.threshold_otsu*self.otsu_factor+self.threshold*(1-self.otsu_factor))).astype(np.uint8)
            mask &= (gray > 70).astype(np.uint8)

            # Remove black/grey edges
            if len(image.shape) == 3 and image.shape[-1] == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                mask_edges = cv2.inRange(hsv, np.array([0,0,0]), np.array([180, 40, 150]))
                mask &= ~mask_edges

            _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            sizes = stats[1:, -1]
            for i, size in enumerate(sizes):
                if size < self.min_region_size:
                    mask[output == i + 1] = 0

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_size)

            results.append(mask.astype(bool))

        return results

def get_mask(wsi: WSIReader,
             thumbnail_resolution=0.15625,
             units="power",
             min_region_size=100,
             kernel_size=3,
             otsu_factor=0.4):
    thumbnail = wsi.slide_thumbnail(resolution=thumbnail_resolution, units=units)

    masker = BetterMorphologicalMasker(
                       min_region_size=min_region_size,
                       kernel_size=kernel_size,
                       otsu_factor=otsu_factor)
    mask_img = masker.fit_transform([thumbnail])[0]

    return VirtualWSIReader(mask_img.astype(np.uint8), info=wsi.info, mode="bool")

