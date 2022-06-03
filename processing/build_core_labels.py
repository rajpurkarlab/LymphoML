import pandas as pd
import os
from pandas import read_excel

PATH_TO_RAW_DATA = "/deep/group/aihc-bootcamp-fall2021/lymphoma/raw"
PATH_TO_MASTER_KEY = os.path.join(PATH_TO_RAW_DATA,
	"Guatemala Project Data vFINAL.xlsx")
PATH_TO_CLPA_FILE = os.path.join(PATH_TO_RAW_DATA,
	"CLPA Diagnostic Bin.xlsx")
CASE = "CASE"
WHO_DIAGNOSIS = "2017 WHO DIAGNOSIS"
CLPA_DIAGNOSIS = "CLPA Diagnostic Bin"
LABEL = "label"
TMA_ID = "TMA ID"
OUTPUT_FILE = "core_labels.csv"

def standardize_key(key):
	# Make the key lowercase and replace all hyphens with spaces
	return key.lower().replace("-", " ")

def get_df_from_set(set_index, set_name):
	# Read the diagnoses from sheet 'set_name' in the Master Key file and store
	# them in the dataframe.
	df = read_excel(PATH_TO_MASTER_KEY, sheet_name=set_name,
		engine='openpyxl')
	df[TMA_ID] = set_index
	return df

def get_clpa_from_who_diagnosis(who_diagnosis, clpa_df):
	# Get the CLPA Diagnostic bin from the corresponding WHO diagnosis.
    key = who_diagnosis.lower().replace("-", " ")
    return clpa_df[clpa_df[WHO_DIAGNOSIS] == key][CLPA_DIAGNOSIS].values[0]

def main():
	# Mapping from CLPA Diagnostic Bin to label.
	clpa_to_label = {"DLBCL": 0, "HL": 1, "Agg BCL": 2, "FL": 3, "MCL": 4,
	                 "MZL": 5, "NKTCL": 6, "TCL": 7, "Nonmalignant": 8,
	                 "Excluded": -1}
	all_sheet_names = ["Set(1)", "Set(2)", "Set(3)", "Set (4)", "Set (5)",
	                   "Set (6)", "Set (7)", "Set (8)"]
	# Create a dataframe containing the diagnoses for each patient ID in all
	# the TMAs inside the Master Key.
	output_df = pd.concat([get_df_from_set(set_index + 1, set_name) for
		set_index, set_name in enumerate(all_sheet_names)])
	# Only include rows in the output_df that contain a valid CASE id (i.e.,
	# rows that start with 'E0') (https://stackoverflow.com/questions/27975069)
	output_df = output_df[output_df[CASE].str.contains("E0", na = False)]
	# Remove extraneous whitespace from the CASE names.
	output_df[CASE] = output_df[CASE].map(lambda s: s.strip())
	# Read in the CLPA Diagnostic Bin file, which maps each WHO Diagnosis to
	# the corresponding CLPA diagnosis.
	clpa_df = read_excel(PATH_TO_CLPA_FILE, sheet_name="Sheet1",
		engine='openpyxl')
	# Standardize the WHO diagnoses in the CLPA dataframe so we can use them as
	# a key to lookup the CLPA diagnosis.
	clpa_df[WHO_DIAGNOSIS] = clpa_df[WHO_DIAGNOSIS].apply(standardize_key)
	# Build a dictionary that maps from the WHO diagnosis to the corresponding
	# CLPA diagnosis.
	who_to_clpa_map = {key : get_clpa_from_who_diagnosis(key, clpa_df) for key
	 in set(output_df[WHO_DIAGNOSIS].values)}
	# Add the CLPA Diagnosis and label as a new columns to the dataframe.
	output_df[CLPA_DIAGNOSIS] = output_df[WHO_DIAGNOSIS].map(who_to_clpa_map)
	output_df[LABEL] = output_df[CLPA_DIAGNOSIS].map(clpa_to_label)
	# Save output_df to a CSV file, extracting the relevant columns.
	output_df.to_csv(OUTPUT_FILE, sep=",", columns=[TMA_ID, CASE, WHO_DIAGNOSIS,
		CLPA_DIAGNOSIS, LABEL])

if __name__ == "__main__":
	main()
