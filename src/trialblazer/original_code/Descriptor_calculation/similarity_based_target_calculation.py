import numpy as np
import pandas as pd
from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file
from rdkit import Chem


def target_features_preprocess(
    chembl_data,
    preprocessed_chembl_data,
    outputFolder,
    remove_multi_components=False,
):
    if remove_multi_components:
        removed_multicom_chembl = preprocessed_chembl_data[
            ~preprocessed_chembl_data.id.str.contains("x")
        ]
        print(
            f"Number of multi-component compounds removed: {len(preprocessed_chembl_data) - len(removed_multicom_chembl)}",
        )
    else:
        removed_multicom_chembl = preprocessed_chembl_data
    chembl_data.rename(columns={"chembl_id": "id"}, inplace=True)
    chembl_data["id"] = chembl_data["id"].astype(str)
    merged_data = removed_multicom_chembl.merge(
        chembl_data[["id", "target_id"]],
        how="left",
        on="id",
    )
    merged_data["MolWithoutStereo"] = merged_data["preprocessedSmiles"].apply(
        Chem.MolFromSmiles,
    )
    merged_data["SmilesWithoutStereo"] = merged_data["MolWithoutStereo"].apply(
        lambda x: Chem.MolToSmiles(x, isomericSmiles=False),
    )
    target_label_smpl = merged_data[
        ["target_id", "SmilesWithoutStereo"]
    ].dropna(subset=["target_id"])
    target_id_list, preprocessed_target_unique_smiles, fpe = generate_h5file(
        target_label_smpl,
        outputFolder,
    )
    return target_id_list, preprocessed_target_unique_smiles, fpe


def tanimoto_similarity_calculation(
    fpe,
    query_dataset,
    split_data=False,
    num_splits=None,
):
    results_append = []
    my_smi_append = []
    if split_data:
        if not num_splits:
            raise ValueError(
                "num_splits must be provided when split_data is True.",
            )
        gen = np.array_split(query_dataset, num_splits)
        for index, batch in generator_yield_gen(gen):
            for my_smi in generator_yield_smiles(batch):
                results = fpe.similarity(my_smi, 0, n_workers=20)
                results_append.append(results)
                my_smi_append.append(my_smi)
    else:
        for my_smi in generator_yield_smiles(query_dataset):
            results = fpe.similarity(my_smi, 0, n_workers=31)
            results_append.append(results)
            my_smi_append.append(my_smi)
    results_whole = pd.DataFrame(
        {
            "my_smi": my_smi_append,
            "similarity_results": results_append,
        },
    )
    return results_whole


def generate_h5file(preprocessed_target, outputFolder):
    target_id_list = list(preprocessed_target["target_id"].unique())
    target_id_list = [x for x in target_id_list if str(x) != "nan"]
    print(f"numbers of unique target:{len(target_id_list)}")
    preprocessed_target_unique_smiles = (
        preprocessed_target.groupby("SmilesWithoutStereo")["target_id"]
        .apply(list)
        .reset_index()
    )
    list_smi = preprocessed_target_unique_smiles[
        "SmilesWithoutStereo"
    ].tolist()
    print(f"numbers of unique SMILES:{len(list_smi)}")
    
    fpe = FPSim2Engine(outputFolder)
    print("h5 file generated!")
    return target_id_list, preprocessed_target_unique_smiles, fpe


def generator_yield_gen(gen):
    for i, array in enumerate(gen):
        yield i, array


def generator_yield_smiles(dataset):
    for smi in dataset["SmilesForDropDu"]:
        yield smi
