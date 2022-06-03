import pandas as pd
import os
from pandas import read_excel

PATH_TO_RAW_DATA = "/deep/group/aihc-bootcamp-fall2021/lymphoma/raw"
PATIENT_TO_SPLIT_FILENAME = os.path.join(PATH_TO_RAW_DATA, "List_for_Oscar_10.15.2021.xlsx")
ALL_CASE_COL = "case"
DIAG_COL = "diag"
SET_COL = "set"
VAL_CASE_COL = "input_dx$case[-part.index]"
TRAIN_COL = "train"
TEST_COL = "test"
SPLIT_COL = "split"
OUTPUT_FILE = "train_test_split.csv"

df = read_excel(PATIENT_TO_SPLIT_FILENAME, sheet_name="Sheet1",
		engine='openpyxl')

validation_patient_ids = set(df[df[VAL_CASE_COL].str.contains("E0", na = False)][VAL_CASE_COL])
output_df = df[[ALL_CASE_COL, DIAG_COL]].copy()
data_split = []
for index, row in output_df.iterrows():
	patient_id = row[ALL_CASE_COL]
	if patient_id in validation_patient_ids:
		data_split.append(TEST_COL)
	else:
		data_split.append(TRAIN_COL)
output_df["split"] = data_split
output_df.to_csv(OUTPUT_FILE, sep=",")
