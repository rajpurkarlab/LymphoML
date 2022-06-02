# An Interpretable Computational Method Identifies Morphologic Features that Correlate with Lymphoma Subtype

## Prerequisites

1. Clone the [lymphoma-ml](https://github.com/stanfordmlgroup/lymphoma-ml) repository:

```Shell
git clone https://github.com/stanfordmlgroup/lymphoma-ml.git
```

2. Make the virtual environment:

```Shell
conda env create -f environment.yml
```

3. Activate the virtual environment:

```Shell
conda activate lymphoma
```

4. Download the latest version of [CellProfiler](https://cellprofiler.org/releases/). We recommend running CellProfiler in headless mode (using the command-line) for running over a large number of images. 

## Code Organization

The code in our study is organized in the following major components, which roughly corresponds to the directory structure in this repo.

1. Data Processing
2. StarDist
3. Deep-Learning Models
4. CellProfiler
5. Spatial Feature Extraction
6. Interpretable Models
7. Statistical Analysis

## Code Usage

We walk through the steps to reproduce the results in our study.

Note: to reproduce the deep-learning results, you only need to run the notebooks/scripts specified in sections 1-3.

### Data Processing

The [processing](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/processing) directory contains files used to process the raw data for ingestion in deep-learning models or CellProfiler.

- First, run [process_raw_data_to_hdf5.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/process_raw_data_to_hdf5.ipynb) to extract patches from each TMA SVS file and save the results in HDF5 file format.
- Next, run [cores_to_tiff.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/cores_to_tiff.py) to save each TMA core as a TIFF file. 
- Finally, run [process_hdf5_to_data_splits.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/process_hdf5_to_data_splits.ipynb), which splits the data into train/val/test splits.

### Stardist

The [Stardist](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/stardist) directory contains files used to run the StarDist algorithm for nuclei segmentation.

- Run [build_stardist_segmentations.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/stardist/build_stardist_segmentations.py) to run a pre-trained StarDist model checkpoint over each TMA core.
- The [stardist_tutorial.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/stardist/stardist_tutorial.ipynb) notebook displays the output of StarDist on sample patches/cores.

### Deep-Learning

**TODO**

### CellProfiler

The [CellProfiler](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler) directory contains files used to run the CellProfiler pipeline on each TMA core and train/evaluate models for lymphoma subtype classification.

#### Pipelines

The [pipelines](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler/pipelines) subdirectory contains the CellProfiler project and pipeline files. Run the CellProfiler pipeline using the following command (e.g. for TMA 1):

```Shell
cellprofiler -c -r -p stardist.cppipe -o ~/processed/cellprofiler_out/stardist/tma_1 -i ~/processed/cellprofiler_in/tma_1
```

#### Feature Processing

The [feature_processing](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler/feature_processing) subdirectory contains files used to process the output CellProfiler spreadsheets.

- Run [patch_identifiers.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/cellprofiler/feature_processing/patch_identifiers.py) to assign a `patch_id` for each cell. The flags `-p` and `-n` can be used to specify the number of pixels per patch or the number of patches extracted per core respectively.

Run the following command to extract nine (approximately) equally-sized patches from each core.
```Shell
python patch_identifiers.py -n 9
```
- Run [feature_aggregation.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/cellprofiler/feature_processing/feature_aggregation.py) to aggregate features across all cells with the same `patch_id`. 

#### Models

The [models](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler/models) subdirectory contains files used to train/evaluate gradient boosting models for lymphoma subtype classification.

- Run [lgb_model.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/cellprofiler/models/lgb_model.ipynb) to train and evaluate a gradient boosting model on the CellProfiler features. 

By default, this notebook runs eight-way lymphoma subtype classification using only nuclear morphological features. This notebook also contains options for performing different modifications of this base task: 
- Set `ENABLE_DLBCL_CLASSIFICATION` to perform DLBCL vs non-DLBCL classification 
- Set `ENABLE_LABEL_GROUPING` to grouping lymphoma subtypes into clinically relevant categories
- Set `FEATURES` to experiment using other features (e.g. nuclear intensity/texture features, cytoplasmic features, or all features).

TODO: 
- Add support for using IHC stains (either to this base notebook or to another new notebook for running IHC experiments)

### Spatial

The [spatial](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/spatial) subdirectory contains files used to extract Ripley K function values and concatenate this information to the rest of H&E featurs. 

- Section "Save Centroid Location info" in [spatial_features_processing.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/spatial/spatial_features_processing.ipynb) saves centroids location information from stardist output into csv files.
- Run [ripleyK.r](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/spatial/ripleyK.r) to compute the spatial relationships between centroids on each saved patch.
- Run "Concatenating R Output to Other H&E Features" section in [spatial_features_processing.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/spatial/spatial_features_processing.ipynb) to combine the spatial features and your current feature dataframe.

### Stats

The [stats](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/stats) subdirectory contains code that we use to compute confidence intervals for all our experiment results. 

- Set `num_replicates` to change number of bootstrapped samples to generate.
- Set `per_class` to compute confidence intervals for a specific class/label.
