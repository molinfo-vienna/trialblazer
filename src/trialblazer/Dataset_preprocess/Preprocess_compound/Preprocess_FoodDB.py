import base64
import hashlib
import multiprocessing as mp
import os
import pandas as pd
from .preprocess_molecules import preprocess_database
from .split_files import split_large_files


# def get_data_from_DB(inputFile, reformatedFile):
# smilesDict = dict()
# with open(inputFile, encoding="utf-8") as FoodDBFile:
#     moleculeCsv = pd.read_csv(FoodDBFile, delimiter=None)
def get_data_from_DB(moleculeCsv, reformatedFile, smiles_col="SMILES", id_col=None):
    smilesDict = {}
    smiles = moleculeCsv[smiles_col]
    moleculeCsv["hash_id"] = moleculeCsv[smiles_col].apply(
        lambda x: base64.urlsafe_b64encode(hashlib.md5(x.encode("utf8")).digest())
        .decode()
        .strip()
        .strip("="),
    )
    if id_col is not None:
        ID = moleculeCsv[id_col]
    elif "chembl_id" in moleculeCsv:
        ID = moleculeCsv["chembl_id"]
    else:
        moleculeCsv["chembl_id"] = None
        ID = moleculeCsv["hash_id"]
    smilesDict = dict(zip(smiles, ID))
    with open(reformatedFile, "w", encoding="utf-8") as f:
        pd.DataFrame.from_dict(smilesDict, orient="index").to_csv(f)
    return smilesDict


def get_molecule_dict_as_smi_output(smilesDict, outputFile) -> None:
    with open(outputFile, "w", encoding="utf-8") as of:
        of.write("smiles ID\n")
        for smiles, ID in smilesDict.items():
            if smiles != "":
                of.write(f"{smiles} {ID}\n")


def preprocess(moleculeCsv, outputFolder, smiles_col="SMILES", id_col=None) -> None:
    # def preprocess(inputFile, outputFolder):
    if not outputFolder.exists():
        outputFolder.mkdir()

    reformatedFolder = outputFolder / "smiles_reformat"
    if not reformatedFolder.exists():
        reformatedFolder.mkdir()
    reformatedName = "trialblazer_smiles.smi"
    # reformatedName = inputFile.split("/")[-1].split(".")[0] + ".smi"
    reformatedFile = reformatedFolder / reformatedName

    smilesDict = get_data_from_DB(
        moleculeCsv, reformatedFile, smiles_col=smiles_col, id_col=id_col,
    )
    # smilesDict = get_data_from_DB(inputFile, reformatedFile)
    get_molecule_dict_as_smi_output(smilesDict, reformatedFile)

    # Split it into multiple files
    splitFolder = outputFolder / "smiles_input"
    if not splitFolder.exists():
        splitFolder.mkdir()

    split_large_files(reformatedFile, 2000)

    # structure preprocess
    preprocessedDir = outputFolder / "preprocessedSmiles"
    if not preprocessedDir.exists():
        preprocessedDir.mkdir()

    databaseFilesJobList = []
    for subfile in os.listdir(splitFolder):
        databaseFilesJobList.append(os.path.join(splitFolder, subfile))
    pool = mp.Pool(processes=max(1, mp.cpu_count() - 1))
    pool.map(preprocess_database, databaseFilesJobList)
    pool.close()
    pool.join()


def CheckOutputResult(outputFolder) -> None:
    logdir = outputFolder / "log"
    if not logdir.exists():
        logdir.mkdir()
    CheckErrorFile = outputFolder / "error.csv"
    for filename in os.listdir(logdir):
        if filename.endswith(".log"):
            log_files = os.path.join(logdir, filename)
        with open(log_files) as f:
            lines = f.readlines()
            CheckFile = {}
            for line in lines:
                line = line.strip()
                Smiles = line.split(" ", 1)[0]
                error = line.split(" ", 1)[1]
                CheckFile[Smiles] = error
        with open(CheckErrorFile, "a") as of:
            for key, value in CheckFile.items():
                if value != "[]":
                    of.write(f"{key}:\t{value}\n")
