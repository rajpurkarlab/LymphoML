# An Interpretable Computational Method identifies Morphologic Features that Correlate with Lymphoma Subtype

### Abstract

Lymphomas are malignancies that are derived from lymphocytes. There are more than 50 different types of lymphoma that vary in terms of clinical behavior, morphology, and response to therapies and thus accurate classification is essential for appropriate management of patients. Computer vision methods have been applied to the task of lymphoma diagnosis by several groups with excellent performance when distinguishing between two to four categories. In this paper, using a set of 670 cases of adult lymphoma obtained from a center in Guatemala City, we propose a novel, interpretable computer vision method that achieves a diagnostic accuracy of 64.06% among 8 lymphoma subtypes using only tissue microarray core sections stained with hematoxylin and eosin (H&E), which is equivalent to experienced hematopathologists. The features that provided the highest diagnostic yield were nuclear shape features. Nuclear texture, cytoplasmic features, and architectural features provided smaller gains in accuracy that were not statistically significant. Though architectural features provide diagnostically-relevant clues for lymphoma diagnosis, the spatial relationships between cells provided only a small increase in accuracy. We use Shapley additive explanations analysis to identify specific features that correlate with particular diagnoses. Finally, we find that the H&E-based model can reduce the number of immunostains necessary to achieve a similar diagnostic accuracy. This study represents the largest pixel-level interpretable analysis of lymphomas to date and suggests that computer vision tools that maximize the diagnostic yield from H&E-stained tissue can reduce the number of immunostains necessary to obtain an accurate diagnosis. 

## Code Usage
Below we walk through the code files in this repo. 

### Data Processing

This directory contains files used to process the raw data into a format suitable for ingestion into deep-learning models or a CellProfiler pipeline.

First, run [process_raw_data_to_hdf5.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/process_raw_data_to_hdf5.ipynb) to transforms each input tissue microarray (TMA) SVS file and QuPath annotation file into an HDF5 file. This notebook performs patch extraction on each TMA core and saves the output HDF5 files.

Next, run [cores_to_tiff.py](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/cores_to_tiff.py) to save each TMA core as a TIFF file (used as input for the CellProfiler pipeline). 

Finally, run [process_hdf5_to_data_splits.ipynb](https://github.com/stanfordmlgroup/lymphoma-ml/blob/main/processing/process_hdf5_to_data_splits.ipynb), which splits the dataset into a train, validation, and test splits, according to a pre-determined train-val-test splits of the patients in the dataset.

### Stardist

### Deep-Learning

### CellProfiler

#### Feature Processing

#### Models
