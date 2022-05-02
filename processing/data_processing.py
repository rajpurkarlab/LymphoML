import cv2 as cv
import matplotlib
import numpy as np

# Taken from DLBCL-Morph: extracting_roi_tma.ipynb.
# Takes in a patch (PIL image) and returns whether or not the patch is sufficiently non-white.
def is_patch_nonwhite_first(patch):
    PERCENT_WHITE_PIXELS_THRESHOLD = 0.95
    SAT_THRESHOLD = 0.05

    hsv_patch = matplotlib.colors.rgb_to_hsv(patch)
    saturation = hsv_patch[:,:,1]
    percent = np.mean(saturation < SAT_THRESHOLD)
    return percent <= PERCENT_WHITE_PIXELS_THRESHOLD

# Taken from DLBCL-Morph: extracting_roi_tma.ipynb.
# Takes in a patch (PIL image) and returns whether or not the patch is sufficiently non-white.
def is_patch_nonwhite_second(patch):
    GRAD_ZERO_THRESHOLD = 500
    gray = cv.cvtColor(np.array(patch), cv.COLOR_RGB2GRAY)
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    mag = np.abs(sobelx) + np.abs(sobely)
    return np.sum(mag == 0) <= GRAD_ZERO_THRESHOLD

# Extracts (default: 224 x 224) patches from a single core in the passed-in TMA.
# The tissue core is defined by: top-left: (xs, ys), bottom-right: (xe, ye).
# Only patches that are sufficiently non-white are returned.
# Returns an np.array of dimension: n x 224 x 224 x 3 (where n = # of returned patches)
def get_patches_from_core(tma, xs, ys, xe, ye, patch_size=224):
    patches = []
    for y in range(ys, ye, patch_size):
        for x in range(xs, xe, patch_size):
            patch = tma.read_region([x, y], 0, [patch_size, patch_size]).convert('RGB')
            if is_patch_nonwhite_first(patch) and is_patch_nonwhite_second(patch):
                patches.append(np.array(patch))
                
    if len(patches) == 0:
        return np.array([])
    return np.stack(patches)