# Preprocess compounds
import os
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
import pandas as pd
import pickle
import ast
from .models import Trialblazer
import tempfile

# these three functions are in folder Dataset_preprocess/Preprocess_compound
from .Dataset_preprocess.Preprocess_compound.Preprocess_FoodDB import (
    preprocess,
    CheckOutputResult,
)
from .Dataset_preprocess.Preprocess_compound.separate_multicomponents_test import (
    separate_multicomponents_test,
)
from .Dataset_preprocess.Preprocess_compound.remove_duplicate import (
    remove_duplicate_splitted_files,
)

# these two functions are in folder Descriptor_calculation
from .Descriptor_calculation.process_similarity_results import (
    remove_tested_inactive_targets,
    remove_invariant_target,
)
from .Descriptor_calculation.process_target_features import (
    process_target_features,
    add_morgan_fingerprints,
)


"""The following 5 steps are used to preprocess the compounds in dataset. """

# module_folder = os.path.join(os.path.dirname(trialblazer.__file__))
module_folder = os.path.join(os.path.dirname(__file__), "..")

test_folder_data = os.path.join(module_folder, "..", "..", "tests", "data")

base_model_folder = Path(module_folder) / "data" / "base_model"


def run(model_folder=base_model_folder, out_folder=None, data_folder=test_folder_data):
    if out_folder is None:
        out_folder_obj = (
            tempfile.TemporaryDirectory()
        )  # os.path.join(test_folder_data, "..", "temp")
        tout_folder = out_folder_obj.name

    temp_folder = tempfile.TemporaryDirectory()
    temp_folder2 = tempfile.TemporaryDirectory()
    """Step 1, preprocess compounds"""
    # refinedInputFile = "/data/local/Druglikness_prediction/external_test_set/approved_testset_final_withname.csv"
    refinedInputFile = os.path.join(test_folder_data, "test_input.csv")

    refinedOutputFolder = Path(out_folder)
    preprocess(
        refinedInputFile, refinedOutputFolder
    )  # input: a dataframe with SMILES and ID, outpyut: a folder with the preprocessed files
    CheckOutputResult(
        refinedOutputFolder
    )  # generate a csv file from log to check the output result

    """Step 2, separate multicomponents"""
    DB_processedDir = Path(out_folder) / "preprocessedSmiles"
    # the folder generated from step 1
    separate_multicomponents_test(
        DB_processedDir
    )  # input: a folder with the preprocessed files, output: a folder with the separated multicomponents, for this step, I modified the script from anya

    """Step 3, remove duplicates"""
    refinedInputFolder = Path(out_folder) / "separate_multicom_GetMolFrags"
    # the folder generated from step 2
    refinedOutputFolder = refinedInputFolder.parent / "uniqueSmiles"
    if not refinedOutputFolder.exists():
        refinedOutputFolder.mkdir()
    remove_duplicate_splitted_files(
        refinedInputFolder, refinedOutputFolder, "approved_testset_final"
    )  # input: a folder with the separated multicomponents, output: a folder with the unique smiles files

    """Step 4, combine the unique smiles files, this step I haven't made it into a function"""
    unique_smiles_path = Path(out_folder) / "uniqueSmiles"

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
    output_path_temp_save_testdata_active = (
        temp_folder  # create a folder to save temperory files for the test data
    )
    output_path_temp_save_testdata_inactive = (
        temp_folder2  # create a folder to save temperory files for the test data
    )

    inactive_preprocessed_target_unique_smiles = pd.read_csv(
        Path(model_folder) / "inactive_target_preprocessed.csv",
        sep="|",
    )  # this preprocessed target smiles is precalculated by previous preprocessed steps base on chembl data, it doesn't need to be re-calculated.
    active_preprocessed_target_unique_smiles = pd.read_csv(
        Path(model_folder) / "active_target_preprocessed.csv",
        sep="|",
    )  # this preprocessed target smiles is precalculated by previous preprocessed steps base on chembl data, it doesn't need to be re-calculated.
    active_preprocessed_target_unique_smiles[
        "target_id"
    ] = active_preprocessed_target_unique_smiles["target_id"].apply(
        ast.literal_eval
    )  # this step is for converting the string type when I read the file from csv
    inactive_preprocessed_target_unique_smiles[
        "target_id"
    ] = inactive_preprocessed_target_unique_smiles["target_id"].apply(
        ast.literal_eval
    )  # this step is for converting the string type when I read the file from csv

    # load the preprocessed training target features
    training_target_features = pd.read_csv(
        Path(model_folder) / "training_target_features.csv"
    )  # this training_target_features is calculated previously and don't need to be re-calculated

    # Load the generated active and inactive fpe (fingerprints engine)
    # These two fpe files are generated previously by using process_target_features function based on trainig data, and don't need to regenerated, I removed h5 files because if we have the fpe, then basically don't need the h5 file
    with open(
        Path(model_folder) / "active_fpe.pkl",
        "rb",
    ) as f:
        active_fpe = pickle.load(f)
    with open(
        Path(model_folder) / "inactive_fpe.pkl",
        "rb",
    ) as f:
        inactive_fpe = pickle.load(f)

    # load the preprocessed target features list, this is generated by the result of process_target_features function based on training data, and don't need to regenerated
    file_path = Path(model_folder) / "training_target_list.csv"
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
    
    testset_filtered_targets_id = testset_filtered_targets.merge(preprocessed_df_mw[['SmilesForDropDu','id']], how ='left', on='SmilesForDropDu')
    

    """Step 8, calculate Morgan2 fingerprints for the training and test data"""
    n_bits = 2048
    morgan_cols = [f"morgan2_b{i}" for i in range(n_bits)]
    training_target_features = add_morgan_fingerprints(
        training_target_features, morgan_cols
    )
    testset_filtered_targets_id = add_morgan_fingerprints(
        testset_filtered_targets_id, morgan_cols
    )

    """Final step, employ the model"""
    # The input of Trialblazer is a dataframe of training featrues and the binary label of each compound, and the test set,
    # the output including a dataframe with the PrOCTOR socre and prediction results for each compound in test set, and the cloestest similairty between test compounds and training compounds
    M2FPs_PBFPs = morgan_cols + training_target_list
    y = training_target_features.Mark
    test_set = testset_filtered_targets_id
    with open(
        "/data/local/Druglikness_prediction/dataset_characteristic_check/training_data_fpe.pkl",
        "rb",
    ) as f:
        trainingdata_fpe = pickle.load(f)  # this fpe don't need to be generated again

    prediction = Trialblazer(
        training_target_features,
        y,
        M2FPs_PBFPs,
        test_set,
        0.06264154114736771,
        900,
        trainingdata_fpe,
        unsure_if_toxic=False,
    )

    # if the user sure about the compounds is safe, e.g. compounds in AD-ES dataset (approved drugs), the parameter unsure_if_toxic should be set to False, otherwise True (default)

    """The ideal way of the model function can be something like this (this "Trialblazer_compeleted" function doesn't exist now):"""
    # result_with_score, cloest_distance = Trialblazer_compeleted(test_set)
    # print(
    #     "the PrOCTOR score and prediction results for each compound in the test set are: XXX"
    # )

    # example files:
    test_set = "/home/hzhang/HuanniZ/Data/example/trialblazer_draft/approved_testset_final_withname.csv"
    prediction = "/home/hzhang/HuanniZ/Data/example/trialblazer_draft/approved_drug_with_score.csv"


if __name__ == "__main__":
    run()
