# Imports

import numpy as np
import openslide
import pandas as pd
from PIL import Image

from tqdm import tqdm

import os
import glob

# Constants

# Paths to raw TMA files.
PATH_TO_RAW_DATA = "/deep/group/aihc-bootcamp-fall2021/lymphoma/raw"
PATH_TO_IMAGES = os.path.join(PATH_TO_RAW_DATA, "svs")

# Path to processed data (extracted TMA patches)
PATH_TO_PROCESSED_DATA = "/deep/group/aihc-bootcamp-fall2021/lymphoma/processed"
PATH_TO_TMA_CORES = os.path.join(PATH_TO_PROCESSED_DATA, "cores")

# Path to annotations and labels (diagnoses)
PATH_TO_RAW_DATA = "/deep/group/aihc-bootcamp-fall2021/lymphoma/raw"
PATH_TO_ANNOTATIONS_CSV = os.path.join(PATH_TO_RAW_DATA, "cores")

# Get the list of paths to TMA svs files and TMA annotations csv files.
# We want the tma_slides_paths[i] to correspond to tma_annotations_paths[i] (hence, the calls to "sort").
tma_slides_paths = sorted(glob.glob(PATH_TO_IMAGES + "/*_TMA*.svs"), key= lambda s : s.split("_")[1])
tma_annotations_paths = sorted(glob.glob(PATH_TO_ANNOTATIONS_CSV + "/TMA*_annotations.csv"))

tma_names = ["TMA1", "TMA2", "TMA3", "TMA4", "TMA5", "TMA6A", "TMA6B", "TMA8"]
tma_ids = [1, 2, 3, 4, 5, 6, 6, 8]
tma_slides = [openslide.OpenSlide(tma_slide) for tma_slide in tma_slides_paths]
tma_annotations = [pd.read_csv(filename) for filename in tma_annotations_paths]

def tma_to_tiffs(tma_id, tma_annotations, tma_slide):
    patient_ids = set()
    patient_id_repeats = {}
    for index, row in tqdm(tma_annotations.iterrows(),total=tma_annotations.shape[0]):
        patient_id = row["Name"]
        name = f"tma_{tma_id}_{patient_id}"
        
        # Deal with duplicate patients
        if (patient_id not in patient_ids):
            patient_id_repeats[patient_id] = 0
        patient_id_repeats[patient_id] += 1
        name += f"_v{patient_id_repeats[patient_id]}"
        
        name.replace(" ", "")
        
        xs, ys, width, height = int(row["X"]), int(row["Y"]), int(row["Width"]), int(row["Height"])
        core = np.array(tma_slide.read_region([xs, ys], 0, [width, height]).convert('RGB'))
        
        if core.size == 0:
            print(f"No core found for TMA: {tma_id}, Patient: {patient_id}")
            continue
            
        core_path = os.path.join(PATH_TO_TMA_CORES, name + ".tif")
        Image.fromarray(core).save(core_path)
        
        patient_ids.add(patient_id)
    
def all_tmas_to_tiffs(tma_ids, tma_slides, tma_annotations):
    assert(len(tma_slides) == len(tma_annotations))
    for i in range(len(tma_slides)):
        tma_id = tma_ids[i]
        print(f"Cores for TMA {tma_id}...\n")
        tma_to_tiffs(tma_id, tma_annotations[i], tma_slides[i])
        print()

if __name__ == "__main__":
    all_tmas_to_tiffs(tma_ids, tma_slides, tma_annotations)