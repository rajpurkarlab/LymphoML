import glob
import itertools
import json
import math
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

# Read config
config_path = '../../config/config.json'
with open(config_path, "r") as cnf:
    config = json.load(cnf)
    
PATH_TO_RAW_DATA = config["raw"] 
PATH_TO_PROCESSED_DATA = config["processed"]
PATH_TO_INPUT = os.path.join(PATH_TO_PROCESSED_DATA, "cores")
PATH_TO_CELLPROFILER_OUT = os.path.join(PATH_TO_PROCESSED_DATA, "cellprofiler_out")
PATH_TO_CELLPROFILER_FEATURES = os.path.join(PATH_TO_CELLPROFILER_OUT, "stardist")

# Path to annotations and labels (diagnoses)
PATH_TO_ANNOTATIONS_CSV = os.path.join(PATH_TO_RAW_DATA, "cores")

# Constants
CORE_LIST = sorted(map(os.path.basename, glob.glob(os.path.join(PATH_TO_INPUT, "*.tif"))))
TMA_GROUPED_LIST = {i: list(ele) for i, ele in itertools.groupby(CORE_LIST, lambda s: s[:5])}

def get_tma_name_from_filename(filename):
    return filename.split("/")[-1].split("_")[0].lower()

def add_patient_id_to_annotations_df(tma_annotations_df):
    patient_ids = set()
    patient_id_repeats = {}
    patient_id_column = []
    for index, row in tma_annotations_df.iterrows():
        patient_id = row["Name"]
        name = patient_id
        # Deal with duplicate patients
        if (patient_id not in patient_ids):
            patient_id_repeats[patient_id] = 0
        patient_id_repeats[patient_id] += 1
        name += f"_v{patient_id_repeats[patient_id]}"
        patient_id_column.append(name)
        patient_ids.add(patient_id)
    tma_annotations_df["patient_id"] = patient_id_column
    return tma_annotations_df

def extract_patient_name_from_image(image):
    # image: tma_1_E0002 B_v1.tif
    # returns: E0002 B_v1.tif
    patient_with_extension = "_".join(image.split("_")[2:])[:-4]
    return os.path.splitext(patient_with_extension)[0]

def add_patch_ids(df, tma_annotations_df, patch_size, patch_num):
    patient_to_width = tma_annotations_df.set_index('patient_id')['Width'].to_dict()
    patient_to_height = tma_annotations_df.set_index('patient_id')['Height'].to_dict()
    
    patch_id_list = []
    patient_name_list = []
    group_id_list = []
    for image, x, y in zip(df["Image"], df["Location_Center_X"], df["Location_Center_Y"]):
        patient_name = extract_patient_name_from_image(image)
        image_without_extension, extension = image[:-4], image[-4:]
        width, height = patient_to_width[patient_name], patient_to_height[patient_name]
        if (np.isnan(x) or np.isnan(y)):
            patch_id = x
            patch_id_list.append(patch_id)
            patient_name_list.append(patient_name)
            group_id_list.append(f"{image_without_extension}-{patch_id}{extension}")
            continue
        # Note: this case is for an issue with TMA6A vs TMA6B -> we don't know whether the placenta/tonsil patient is 
        # from TMA6A or TMA6B (because we grouped TMA6A and TMA6B together into "TMA6" when running CellProfiler).
        # We could get an assertion error if we map the patient to the wrong placenta/tonsil core.
        # This is not an issue since we don't consider placenta/tonsil cores when training models.
        if (x > width or y > height) and (patient_name.startswith("placenta") or patient_name.startswith("tonsil")):
            print(patient_name, x, y, width, height)
        else:
            assert(x <= width and y <= height)

        patch_width, patch_height = patch_size, patch_size
        if patch_num != None:
            patch_num_sqroot = int(math.sqrt(patch_num))
            patch_width = math.ceil(width / patch_num_sqroot)
            patch_height = math.ceil(height / patch_num_sqroot)

        grid_len_x, grid_len_y = math.ceil(width / patch_width), math.ceil(height / patch_height)
        patch_x, patch_y = int(x / patch_width), int(y / patch_height)
        patch_id = grid_len_x * patch_y + patch_x
        patch_id_list.append(patch_id)
        patient_name_list.append(patient_name)
        group_id_list.append(f"{image_without_extension}-{patch_id}{extension}")
    df["patch_id"] = patch_id_list
    df["patient_id"] = patient_name_list
    df["group_id"] = group_id_list
    return df

def add_patch_ids_to_cellprofiler_features(tma_id, patch_size, patch_num):
    fnames = sorted(glob.glob(os.path.join(PATH_TO_CELLPROFILER_FEATURES, f"tma_{tma_id}/*.csv")))
    tma_annotations_paths = sorted(glob.glob(PATH_TO_ANNOTATIONS_CSV + "/TMA*_annotations.csv"))
    tma_annotations_map = {get_tma_name_from_filename(filename): pd.read_csv(filename) for filename in tma_annotations_paths}

    for fname in fnames:

        directory = fname.split("/")[-2]
        tma_name = directory.replace("_", "")
        base = os.path.basename(fname)[:-4].lower()

        print(f"Processing {fname}...")

        df = pd.read_csv(fname, usecols=["ImageNumber", "ObjectNumber", "Location_Center_X", "Location_Center_Y"])

        # Set image column and remove useless columns
        if directory[:-1] == "tma_":
            df["Image"] = df["ImageNumber"].apply(lambda i: TMA_GROUPED_LIST[directory][i-1])
        else:
            df["Image"] = df["ImageNumber"].apply(lambda i: CORE_LIST[i-1])
        
        if tma_name == "tma6":
            tma_annotations_df = pd.concat([tma_annotations_map["tma6a"], tma_annotations_map["tma6b"]], ignore_index=True)
        else:
            tma_annotations_df = tma_annotations_map[tma_name]
        tma_annotations_df = add_patient_id_to_annotations_df(tma_annotations_df)

        df = add_patch_ids(df, tma_annotations_df, patch_size, patch_num)
        if patch_size != None:
            sub_dir = f"patch_size={patch_size}"
        else:
            sub_dir = f"patch_num={patch_num}"
        outdir = os.path.join(PATH_TO_CELLPROFILER_FEATURES, f"{directory}/patches/{sub_dir}")
        full_path = os.path.join(outdir, f"{base}_with_patch_ids.csv")
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        print(f"Saving to {full_path}...")
        df.to_csv(full_path)
    print("Done!")

def is_perfect_square(n):
    return math.sqrt(n) ** 2 == n

def main():
    parser = ArgumentParser()
    parser.add_argument("--patch_size", "-p", type=int, help="Size of patches to group extracted cell/cytoplasm features into.")
    parser.add_argument("--patch_num", "-n", type=int, help="Number of patches per core to group extracted cell/cytoplasm features into.")
    args = parser.parse_args()
    assert(args.patch_size == None or args.patch_num == None)
    if args.patch_num != None:
        # Make sure number of patches is a perfect square.
        assert(is_perfect_square(args.patch_num))
    tma_ids = [1,2,3,4,5,6,8]
    for i in tma_ids:
        add_patch_ids_to_cellprofiler_features(i, args.patch_size, args.patch_num)

if __name__ == "__main__":
    main()
    # python patch_identifiers.py -n 1
