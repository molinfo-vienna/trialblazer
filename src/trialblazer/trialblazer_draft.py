# Preprocess compounds
import os
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
import pandas as pd
import pickle
import ast
from Model import Trialblazer

# these three functions are in folder Dataset_preprocess/Preprocess_compound
from Preprocess_FoodDB import preprocess, CheckOutputResult
from separate_multicomponents_test import separate_multicomponents_test
from remove_duplicate import remove_duplicate_splitted_files

# these two functions are in folder Descriptor_calculation
from process_similarity_results import (
    remove_tested_inactive_targets,
    remove_invariant_target,
)
from process_target_features import process_target_features, add_morgan_fingerprints


"""The following 5 steps are used to preprocess the compounds in dataset. """

"""Step 1, preprocess compounds"""
refinedInputFile = "/data/local/Druglikness_prediction/external_test_set/approved_testset_final_withname.csv"
refinedOutputFolder = Path(
    "/data/local/Druglikness_prediction/external_test_set/approved_drug_testset_final/"
)
preprocess(
    refinedInputFile, refinedOutputFolder
)  # input: a dataframe with SMILES and ID, outpyut: a folder with the preprocessed files
CheckOutputResult(
    refinedOutputFolder
)  # generate a csv file from log to check the output result

"""Step 2, separate multicomponents"""
DB_processedDir = Path(
    "/data/local/Druglikness_prediction/external_test_set/approved_drug_testset_final/preprocessedSmiles"
)  # the folder generated from step 1
separate_multicomponents_test(
    DB_processedDir
)  # input: a folder with the preprocessed files, output: a folder with the separated multicomponents, for this step, I modified the script from anya

"""Step 3, remove duplicates"""
refinedInputFolder = Path(
    "/data/local/Druglikness_prediction/external_test_set/approved_drug_testset_final/separate_multicom_GetMolFrags"
)  # the folder generated from step 2
refinedOutputFolder = refinedInputFolder.parent / "uniqueSmiles"
if not refinedOutputFolder.exists():
    refinedOutputFolder.mkdir()
remove_duplicate_splitted_files(
    refinedInputFolder, refinedOutputFolder, "approved_testset_final"
)  # input: a folder with the separated multicomponents, output: a folder with the unique smiles files

"""Step 4, combine the unique smiles files, this step I haven't made it into a function"""
unique_smiles_path = Path(
    "/data/local/Druglikness_prediction/external_test_set/approved_drug_testset_final/uniqueSmiles/"
)
df_list = []
for filenames in os.listdir(unique_smiles_path):
    print(filenames)
    file = unique_smiles_path / filenames
    df = pd.read_csv(file, sep="\t", index_col=None, header=0)
    df_list.append(df)
preprocessed_df = pd.concat(df_list, axis=0, ignore_index=True)

"""Step 5, remove stereochemical information from the preprocessed SMILES and filter the compounds to retain only small molecules"""
preprocessed_df["Molecule"] = preprocessed_df["preprocessedSmiles"].apply(
    Chem.MolFromSmiles
)
preprocessed_df["SmilesForDropDu"] = preprocessed_df["Molecule"].apply(
    Chem.MolToSmiles, isomericSmiles=False
)
preprocessed_df["Molecule"] = preprocessed_df["SmilesForDropDu"].apply(
    Chem.MolFromSmiles
)
preprocessed_df["mw"] = preprocessed_df.Molecule.apply(
    lambda mol: round(Descriptors.MolWt(mol), 3)
)
preprocessed_df_mw = preprocessed_df[preprocessed_df["mw"].between(150, 850)]

"""The following steps are used to calculate the descriptors for the compounds in dataset."""
"""Step 6, load the processed ChEMBL data and preprocessed training target features"""
# load the preprocessed active and inactive targets from the ChEMBL database,
# these targets are preprocessed through Step 1-5, but in the application it is not necessary to calculate it from the scratch
h5file_active = (
    "/data/local/Druglikness_prediction/similarity_chembl/WholeInOne/WholeInOne.h5"
)
h5file_inactive = "/data/local/Druglikness_prediction/similarity_chembl/chembl/InactivateTargets/temp_save_inactive/inactive_targets.h5"
output_path_temp_save_testdata_active = (
    "/data/local/Druglikness_prediction/external_test_set/temp_save"
)
output_path_temp_save_testdata_inactive = "/data/local/Druglikness_prediction/external_test_set/approved_drugs_inactive_target/temp_save"

inactive_preprocessed_target_unique_smiles = pd.read_csv(
    "/data/local/Druglikness_prediction/external_test_set/Eudra_target_features/inactive_target_preprocessed.csv",
    sep="|",
)
active_preprocessed_target_unique_smiles = pd.read_csv(
    "/data/local/Druglikness_prediction/external_test_set/Eudra_target_features/active_target_preprocessed.csv",
    sep="|",
)
active_preprocessed_target_unique_smiles[
    "target_id"
] = active_preprocessed_target_unique_smiles["target_id"].apply(ast.literal_eval)
inactive_preprocessed_target_unique_smiles[
    "target_id"
] = inactive_preprocessed_target_unique_smiles["target_id"].apply(ast.literal_eval)

# load the preprocessed training target features
training_target_features = pd.read_csv(
    "/data/local/Druglikness_prediction/similarity_chembl/chembl/InactivateTargets/checked_targets_results/checked_ori.csv"
)

# Load the generated active and inactive fpe (fingerprints engine)
with open(
    "/data/local/Druglikness_prediction/external_test_set/Eudra_target_features/active_fpe.pkl",
    "rb",
) as f:
    active_fpe = pickle.load(f)
with open(
    "/data/local/Druglikness_prediction/external_test_set/Eudra_target_features/inactive_fpe.pkl",
    "rb",
) as f:
    inactive_fpe = pickle.load(f)

# load the preprocessed target features list
file_path = "/home/hzhang/HuanniZ/Data/training_target_list.csv"
with open(file_path, "r") as file:
    reader = csv.reader(file)
    training_target_list = [
        row[0] for row in reader
    ]  # get the list of traing_targets_list, number of targets: 777

"""Step 7, calculate and process the Tanimoto similarity results, the query data is the preprocessed data from step 1-5, the output of this step is the target feature"""
# Example:
# if "preprocessed_df_mw" is test data (this is the application of the model):
(
    testset_active_binarize_list,
    testset_active_binarized_target_remain,
) = process_target_features(
    output_path_temp_save=output_path_temp_save_testdata_active,
    training_target_list=training_target_list,  # only need to used the same targets as the training data
    fpe=active_fpe,
    preprocessed_target_unique_smiles=active_preprocessed_target_unique_smiles,
    query_data=preprocessed_df_mw,
)

(
    testset_inactive_binarize_list,
    testset_inactive_binarized_target_remain,
) = process_target_features(
    output_path_temp_save=output_path_temp_save_testdata_inactive,
    training_target_list=training_target_list,  # only need to used the same targets as the training data
    fpe=inactive_fpe,
    preprocessed_target_unique_smiles=inactive_preprocessed_target_unique_smiles,
    query_data=preprocessed_df_mw,
)
testset_filtered_targets, testset_target_list = remove_tested_inactive_targets(
    testset_inactive_binarized_target_remain, testset_active_binarized_target_remain
)  # testset_filtered_targets is the target features I need for testset compounds

"""Step 8, calculate Morgan2 fingerprints for the training and test data"""
n_bits = 2048
morgan_cols = [f"morgan2_b{i}" for i in range(n_bits)]
training_target_features = add_morgan_fingerprints(
    training_target_features, morgan_cols
)
testset_filtered_targets = add_morgan_fingerprints(
    testset_filtered_targets, morgan_cols
)

"""Final step, employ the model"""
# The input of Trialblazer is a dataframe of training featrues and the binary label of each compound, and the test set,
# the output is a dataframe with the PrOCTOR socre and prediction results for each compound in test set.
M2FPs_PBFPs = morgan_cols + training_target_list
X = training_target_features[M2FPs_PBFPs]
y = training_target_features.Mark
test_set = testset_filtered_targets
result_with_score = Trialblazer(
    X, y, test_set, threshold=0.06264154114736771, k=900, unsure_if_toxic=False
)
# if the user sure about the compounds is safe, e.g. compounds in AD-ES dataset (approved drugs), the parameter unsure_if_toxic should be set to False, otherwise True (default)

"""The ideal way of the model function can be something like this (this "Trialblazer_compeleted" function doesn't exist now):"""
result_with_score = Trialblazer_compeleted(test_set)
print(
    "the PrOCTOR score and prediction results for each compound in the test set are: XXX"
)

# example files:
test_set = "/home/hzhang/HuanniZ/Data/example/trialblazer_draft/approved_testset_final_withname.csv"
result_with_score = (
    "/home/hzhang/HuanniZ/Data/example/trialblazer_draft/approved_drug_with_score.csv"
)
