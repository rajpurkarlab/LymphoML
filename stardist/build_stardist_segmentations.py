import glob
import h5py
import os

# Import the StarDist 2D segmentation models.
from stardist.models import StarDist2D
# Import the recommended normalization technique for stardist.
from csbdeep.utils import normalize

# Import squidpy and additional packages needed for this tutorial.
import squidpy as sq
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# Read Config
config_path = '../config/config.json'
with open(config_path, "r") as cnf:
    config = json.load(cnf)

PATH_TO_PROCESSED_DATA = config["processed"]
PATH_TO_TMA_PATCHES = os.path.join(PATH_TO_PROCESSED_DATA, "tma_patches")
PATH_TO_TMA_CORES = os.path.join(PATH_TO_PROCESSED_DATA, "cores")
PATH_TO_OUTPUT_DATA = os.path.join(PATH_TO_PROCESSED_DATA, "stardist/stardist_core_segmentations")

tma_core_filenames = glob.glob(PATH_TO_TMA_CORES + "/*.tif")
model = StarDist2D.from_pretrained('2D_versatile_he')

def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None):
    axis_norm = (0,1,2) # normalize channels jointly
    # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
    # this is the default normalizer noted in StarDist examples.
    img = normalize(img, 1, 99.8, axis=axis_norm)
    labels, _ = model.predict_instances(img, nms_thresh=nms_thresh, prob_thresh=prob_thresh)
    return labels

def get_leaf_filename(tma_core_filename):
    return tma_core_filename.split("/")[-1]

# Run segmentations for each TMA core.
def main():
    num_cores_processed = 0
    for filename in tma_core_filenames:
        img = sq.im.ImageContainer(np.array(Image.open(filename)), layer="image")
        sq.im.segment(
            img=img,
            layer="image",
            channel=None,
            method=stardist_2D_versatile_he,
            layer_added='segmented_stardist',
            prob_thresh=0.3,
            nms_thresh=None
        )
        segmented_img = np.squeeze(img['segmented_stardist'].data)
        segmented_img[segmented_img > 0] = 255
        
        name = get_leaf_filename(filename)
        num_cores_processed += 1
        Image.fromarray(segmented_img).convert("L").save(os.path.join(PATH_TO_OUTPUT_DATA, name))

if __name__ == "__main__":
	main()
