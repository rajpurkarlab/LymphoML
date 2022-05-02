# An Interpretable Computational Method Identifies Morphologic Features that Correlate with Lymphoma Subtype

### Abstract

Lymphomas are malignancies that are derived from lymphocytes. There are more than 50 different types of lymphoma that vary in terms of clinical behavior, morphology, and response to therapies and thus accurate classification is essential for appropriate management of patients. Computer vision methods have been applied to the task of lymphoma diagnosis by several groups with excellent performance when distinguishing between two to four categories. In this paper, using a set of 670 cases of adult lymphoma obtained from a center in Guatemala City, we propose a novel, interpretable computer vision method that achieves a diagnostic accuracy of 64.06% among 8 lymphoma subtypes using only tissue microarray core sections stained with hematoxylin and eosin (H&E), which is equivalent to experienced hematopathologists. The features that provided the highest diagnostic yield were nuclear shape features. Nuclear texture, cytoplasmic features, and architectural features provided smaller gains in accuracy that were not statistically significant. Though architectural features provide diagnostically-relevant clues for lymphoma diagnosis, the spatial relationships between cells provided only a small increase in accuracy. We use Shapley additive explanations analysis to identify specific features that correlate with particular diagnoses. Finally, we find that the H&E-based model can reduce the number of immunostains necessary to achieve a similar diagnostic accuracy. This study represents the largest pixel-level interpretable analysis of lymphomas to date and suggests that computer vision tools that maximize the diagnostic yield from H&E-stained tissue can reduce the number of immunostains necessary to obtain an accurate diagnosis. 

## Code Usage

We walk through the steps to reproduce the results in our study.

The code in our study is organized in the following major components, which roughly corresponds to the directory structure in this repo.

1. Data Processing
2. StarDist
3. Deep-Learning Models
4. CellProfiler
5. Spatial Feature Extraction
6. Interpretable Models
7. Statistical Analysis

To reproduce the deep-learning results, you only need to run the notebooks and scripts specified in sections 1-3.

### Data Processing

The [processing](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/processing) directory contains files used to process the raw data into a format suitable for ingestion into deep-learning models or a CellProfiler pipeline.

- First, run [process_raw_data_to_hdf5.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/process_raw_data_to_hdf5.ipynb) to transforms each input tissue microarray (TMA) SVS file and QuPath annotation file into an HDF5 file. This notebook performs patch extraction on each TMA core and saves the output HDF5 files.

- Next, run [cores_to_tiff.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/cores_to_tiff.py) to save each TMA core as a TIFF file (used as input for the CellProfiler pipeline). 

- Finally, run [process_hdf5_to_data_splits.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/process_hdf5_to_data_splits.ipynb), which splits the dataset into a train, validation, and test splits, according to a pre-determined train-val-test splits of the patients in the dataset.

### Stardist

The [Stardist](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/stardist) directory contains files used to run the StarDist algorithm on TMA cores for nuclei segmentation.

- Run [build_stardist_segmentations.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/stardist/build_stardist_segmentations.py), which runs a pre-trained StarDist model checkpoint used for brightfield H&E images over each TMA core.
- The [stardist_tutorial.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/stardist/stardist_tutorial.ipynb) notebook displays the output of the StarDist algorithm on sample patches and cores on our dataset.

### Deep-Learning

**TODO**

### CellProfiler

The [CellProfiler](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler) directory contains the files used to run the CellProfiler pipeline on each TMA core and train and evaluate interpretable models for lymphoma subtype classification.

#### Pipelines

The [pipelines](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler/pipelines) subdirectory contains the CellProfiler project and pipeline files. Run the CellProfiler pipeline using the following command (e.g. for TMA 1):

`cellprofiler -c -r -p stardist.cppipe -o ~/processed/cellprofiler_out/stardist/tma_1 -i ~/processed/cellprofiler_in/tma_1`

#### Feature Processing

The [feature_processing](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler/feature_processing) subdirectory contains files used to process the output CellProfiler spreadsheets.

- Run [patch_identifiers.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/cellprofiler/feature_processing/patch_identifiers.py) to assign a `patch_id` for each cell. Use the provided flags `-p` and `-n` can be used to specify the number of pixels per patch or the number of patches extracted per core respectively. For example, run the following command: `python patch_identifiers.py -n 9` to extract nine (approximately) equally-sized patches from each core.

- Run [feature_aggregation.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/cellprofiler/feature_processing/feature_aggregation.py) to aggregate features across all cells with the same `patch_id`. 

#### Models

The [models](https://github.com/stanfordmlgroup/lymphoma-ml/tree/main/cellprofiler/models) subdirectory contains files used to train and evaluate gradient boosting models on the processed CellProfiler features to perform the lymphoma subtype classification task.

- Run [lgb_model.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/cellprofiler/models/lgb_model.ipynb) to train and evaluate a gradient boosting model on the CellProfiler features. By default, this notebook runs eight-way lymphoma subtype classification using only nuclear morphological features. This notebook also contains options for performing tasks: DLBCL vs non-DLBCL classification or grouping lymphoma subtypes into clinically relevant categories, or using different sets of features (e.g. nuclear intensity/texture features, cytoplasmic features).

TODO: 
- Add support for label grouping
- Add support for using IHC stains
- Specify how users can update certain constants to run experiments with different settings (e.g. change the task (8-way classification, 5-way classification (with label grouping), DLBCL vs non-DLBCL), use different features, or use IHC stains).

### Spatial

**TODO**

### Stats

**TODO**

### Overall TODOs

- Fix all file references (use a config file?) 
