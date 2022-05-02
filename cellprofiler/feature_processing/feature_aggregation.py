# Imports

import glob
import itertools
import json
import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

# Read config
config_path = '../../config/config.json'
with open(config_path, "r") as cnf:
    config = json.load(cnf)
    
# Paths and constants

# Paths for Aggregation
PATH_TO_RAW_DATA = config["raw"] 
PATH_TO_PROCESSED_DATA = config["processed"]
PATH_TO_INPUT = os.path.join(PATH_TO_PROCESSED_DATA, "cores")
PATH_TO_CELLPROFILER_OUT = os.path.join(PATH_TO_PROCESSED_DATA, "cellprofiler_out")

PATH_TO_DATA_SPLITS = config["data_splits"]

PATH_TO_TRAIN_TEST_SPLIT = os.path.join(PATH_TO_RAW_DATA, "custom_train_test_split.csv")
PATH_TO_DIAGNOSES = os.path.join(PATH_TO_RAW_DATA, "core_labels.csv")

PATH_TO_SPLITS_OUTPUT = os.path.join(PATH_TO_DATA_SPLITS, "custom_splits/cellprofiler")

# Constants
CORE_LIST = sorted(map(os.path.basename, glob.glob(os.path.join(PATH_TO_INPUT, "*.tif"))))
TMA_GROUPED_LIST = {i: list(ele) for i, ele in itertools.groupby(CORE_LIST, lambda s: s[:5])}

COLUMNS_TO_REMOVE = ["ImageNumber", "Parent_Cells", "Parent_Nuclei", "Object_Number"]
TMA_ID = "TMA ID"
CASE = "CASE"
LABEL = "label"

NUM_PERCENTILE_BUCKETS = 4

# Helper functions

def agg_df_features(fname: str, path_to_agg_cellprofiler_features: str, agg_type: str):

    print(f"Processing {fname}...")

    # Extract info
    path_to_dir = os.path.dirname(fname)
    directory = fname.split("/")[-2]
    base = os.path.basename(fname)[:-4].lower()

    df = pd.DataFrame(pd.read_csv(fname))

    # Set image column and remove useless columns
    if directory[:-1] == "tma_":
        df["Image"] = df["ImageNumber"].apply(lambda i: TMA_GROUPED_LIST[directory][i-1])
    else:
        df["Image"] = df["ImageNumber"].apply(lambda i: CORE_LIST[i-1])

    for col in COLUMNS_TO_REMOVE:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    def mean(df):
        return df.mean()

    def std(df):
        return df.std()

    def skew(df):
        return df.skew()

    def kurt(df):
        return df.apply(pd.Series.kurt)

    def percentile(q: float):
        func = lambda s: s.quantile(q)
        return func

    agg_funcs = dict(**{'mean': mean, 'std': std, 'skew': skew, 'kurt': kurt},
                 **{f"{ 100 / NUM_PERCENTILE_BUCKETS * (i+1)}%": percentile((i + 1) / NUM_PERCENTILE_BUCKETS) for i in range(NUM_PERCENTILE_BUCKETS - 1)})

    # Load patches
    patch_path = os.path.join(path_to_dir, "patches/nuclei_with_patch_ids.csv")
    if os.path.exists(patch_path) and agg_type == "patch":
        print("Patch Path:", patch_path)
        patches = pd.DataFrame(pd.read_csv(patch_path))
        df["group_id"] = patches["group_id"]
    else:
        df["group_id"] = df["Image"].copy()

    # Aggregate features for a single TMA
    group_df = df.groupby('group_id')
    aggs = {i: agg_funcs[i](group_df) for i in agg_funcs}
    for i in aggs:
        aggs[i].rename({col : col + "_" + i for col in aggs[i].columns}, axis=1, inplace=True)
        aggs[i] = aggs[i].reset_index()

    all_aggs_df = list(aggs.values())[0]
    for agg in list(aggs.values())[1:]:
        all_aggs_df = pd.merge(all_aggs_df, agg, how='inner', left_on="group_id", right_on="group_id")

    all_aggs_df["tma_id"] = all_aggs_df["group_id"].apply(lambda s: "tma" + s.split("_")[1])
    all_aggs_df["Image"] = all_aggs_df["group_id"].apply(lambda s: s.split("-")[0])

    print(f"Number of rows in aggregated df: {all_aggs_df.shape[0]}")

    outpath = os.path.join(path_to_agg_cellprofiler_features, f"cellprofiler_{directory}_{base}_features.csv")
    print(f"Saving to {outpath}...")
    all_aggs_df.to_csv(outpath)
    print("Done!")

def agg_features_all(path_to_celllprofiler_features: str, path_to_agg_cellprofiler_features: str, agg_type: str):
    fnames = sorted(glob.glob(os.path.join(path_to_celllprofiler_features, "*_[0-9]/*.csv")))

    for fname in fnames:
        agg_df_features(fname, path_to_agg_cellprofiler_features, agg_type)


def make_splits(path_to_agg_cellprofiler_features: str, path_to_splits_output: str):

    # Load datasplits
    data_split_df = pd.DataFrame(pd.read_csv(PATH_TO_TRAIN_TEST_SPLIT, delimiter=','))

    if 'patient_id' in data_split_df.columns:
        data_split_df['case'] = data_split_df['patient_id'].apply(lambda s: s.split()[0])

    data_split_map = data_split_df.set_index('case')['split'].to_dict()

    # Feature files
    nucl_features_files = sorted(glob.glob(os.path.join(path_to_agg_cellprofiler_features, "cellprofiler_*_nuclei_features.csv")))
    cyt_features_files = sorted(glob.glob(os.path.join(path_to_agg_cellprofiler_features, "cellprofiler_*_cytoplasm_features.csv")))

    # Merge features for each group
    cell_features_dfs = [
        pd.merge(pd.read_csv(nucl), pd.read_csv(cyt), on="Image", how="inner") for nucl, cyt in zip(nucl_features_files, cyt_features_files)
    ]

    # Concat all groups
    cell_features_df = pd.concat(df for df in cell_features_dfs).reset_index()
    cell_features_df.drop(cell_features_df.columns[[0, 1]], axis=1, inplace=True)

    cell_features_df["tma_id"] = cell_features_df["tma_id_y"].copy()
    cell_features_df.drop(["tma_id_x", "tma_id_y"], axis=1, inplace=True)

    tma_case_to_diagnosis = pd.read_csv(PATH_TO_DIAGNOSES, delimiter=',')
    tma_case_to_diagnosis[CASE] = tma_case_to_diagnosis[CASE].apply(lambda patient_id : patient_id.replace(" ", ""))

    labels = []
    for (tma_id, patient_id) in tqdm(zip(cell_features_df["tma_id"], cell_features_df["Image"]), total=cell_features_df["tma_id"].size):
        tma_id_key = int(tma_id[3]) # tma1 -> 1, tma6a -> 6, tma6b -> 6
        patient_id_key = patient_id.split("_")[2].replace(" ","")
        condition = (tma_case_to_diagnosis[CASE] == patient_id_key) & (tma_case_to_diagnosis[TMA_ID] == tma_id_key)
        tma_case_to_diagnosis_row = tma_case_to_diagnosis[condition]
        if len(tma_case_to_diagnosis_row[LABEL].values) == 0:
            print(f"Could not find diagnosis for: {patient_id_key}")
            label = -1
        elif len(tma_case_to_diagnosis_row[LABEL].values) > 1:
            print(f"ERROR: There should only be one entry for a specific TMA ID X patient ID")
            label = -1
        else:
            label = tma_case_to_diagnosis_row[LABEL].values[0]
        labels.append(label)
    cell_features_df[LABEL] = labels
    # Exclude rows with label < 0
    cell_features_df = cell_features_df[cell_features_df[LABEL] >= 0]

    # Set split and keep track of which patients not in split
    included_patient_ids = set()
    excluded_patient_ids = set()
    splits = []
    for patient_id in cell_features_df["Image"]:
        patient_id_key = patient_id.split("_")[2].replace(" ", "")[:5]
        if (patient_id_key in data_split_map):
            split = data_split_map[patient_id_key]
            included_patient_ids.add(patient_id_key)
        else:
            split = "None"
            excluded_patient_ids.add(patient_id_key)
        splits.append(split)
    cell_features_df["split"] = splits

    # Split df's and save
    train_df = cell_features_df[cell_features_df["split"] == "train"]
    val_df = cell_features_df[cell_features_df["split"] == "val"]
    test_df = cell_features_df[cell_features_df["split"] == "test"]

    train_df.to_csv(os.path.join(path_to_splits_output, "train.csv"), index=False)
    val_df.to_csv(os.path.join(path_to_splits_output, "val.csv"), index=False)
    test_df.to_csv(os.path.join(path_to_splits_output, "test.csv"), index=False)

def main():
    parser = ArgumentParser()

    parser.add_argument("--feature_type", "-f", type=str,
                    help="type of features to be processed (improc|stardist)")
    parser.add_argument("--agg_type", "-a", type=str, default="core",
                    help="type of features to be processed (improc|stardist)")

    args = parser.parse_args()

    path_to_cellprofiler_features = os.path.join(PATH_TO_CELLPROFILER_OUT, args.feature_type)
    path_to_agg_cellprofiler_features = os.path.join(path_to_cellprofiler_features, f"{args.agg_type}_aggregated")
    path_to_splits_output = os.path.join(PATH_TO_DATA_SPLITS, "custom_splits/cellprofiler", f"{args.feature_type}_{args.agg_type}")

    for d in (path_to_cellprofiler_features, path_to_agg_cellprofiler_features, path_to_splits_output):
        if not os.path.isdir(d):
            os.makedirs(d)

    # agg_features_all(path_to_cellprofiler_features, path_to_agg_cellprofiler_features, args.agg_type)
    make_splits(path_to_agg_cellprofiler_features, path_to_splits_output)


if __name__ == "__main__":
    main()

