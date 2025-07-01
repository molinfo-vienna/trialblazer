import re
import cirpy
import numpy as np
import pandas as pd
import pubchempy as pcp
import spacy
from drug_named_entity_recognition import find_drugs


def ExtractingDrugsNamesAndSmilesMatching(dataset, Name_column):
    FailToExtract, GoToSmilesMatching = Extract_drug_name(dataset, Name_column)
    cirpy_result, GoToPubchempy = cirpy_SmilesMatching(
        FailToExtract,
        GoToSmilesMatching,
    )
    pubchempy_result, GoTo_NextRound = pubchempy_SmilesMatching(
        GoToPubchempy,
        "name_to_use",
    )
    SmilesMatched = pd.concat(
        [cirpy_result, pubchempy_result],
        ignore_index=True,
    ).reset_index()
    return SmilesMatched, GoTo_NextRound


def replace_text(ori_name):
    if pd.isna(ori_name):
        return ori_name
    pattern = [r"(p|P)lus", r"(C|c)ombin", r"(A|a)nd", r"( or )", r"(\+)"]
    ori_name = ori_name.replace("18F", "")
    if ori_name.endswith("mab"):
        return "antibody"
    if any(re.findall(p, ori_name) for p in pattern):
        return "multi-drugs"
    return ori_name


def DNER_with_spacy(DataFrame):
    test_with_spacy = []
    doc_append = []
    nlp = spacy.blank("en")
    for row in DataFrame["replace_Name"].values():
        if pd.notna(row):
            doc = nlp(row)
            doc_append.append(doc)
            with_spacy = find_drugs([t.text for t in doc], is_ignore_case=True)
            test_with_spacy.append(with_spacy)
        else:
            test_with_spacy.append([])
    return test_with_spacy


def extract_results(LM_output):
    name_append = []
    synonyms_append = []
    drugbank_id_append = []
    for row in LM_output:
        if row:
            first_tuple = row[0]
            if isinstance(first_tuple[0], dict):
                name_append.append(first_tuple[0].get("name", np.nan))
                synonyms_append.append(first_tuple[0].get("synonyms", np.nan))
                drugbank_id_append.append(
                    first_tuple[0].get("drugbank_id", np.nan),
                )
            else:
                name_append.append(np.nan)
                synonyms_append.append(np.nan)
                drugbank_id_append.append(np.nan)
        else:
            name_append.append(np.nan)
            synonyms_append.append(np.nan)
            drugbank_id_append.append(np.nan)


    return name_append, synonyms_append, drugbank_id_append


def select_drugs_GoToSecondRound(processed_dataframe):
    s = []
    c = []
    for i, item in enumerate(processed_dataframe["DNER_with_spacy"]):
        if len(item) == 0:
            s.append(i)
        else:
            c.append(i)
    GoToSecondRound = processed_dataframe.iloc[s]
    with_InFo = processed_dataframe.iloc[c]
    return GoToSecondRound, with_InFo


def cirpy_SmilesMatching(FailToExtract, GoToSmilesMatching):
    smiles_append = []
    name = "replace_Name"
    for df in [FailToExtract, GoToSmilesMatching]:
        for row in df[name].values():
            try:
                if row:
                    smiles = cirpy.resolve(row, "smiles")
                    smiles_append.append(smiles)
                else:
                    smiles_append.append(np.nan)
            except Exception:
                smiles_append.append(np.nan)
        name = "name_to_use"
    GoToSmileMatching = pd.concat(
        [FailToExtract, GoToSmilesMatching],
        ignore_index=True,
    )
    GoToSmileMatching["SMILES"] = smiles_append
    cirpy_result = GoToSmileMatching.loc[~GoToSmileMatching["SMILES"].isna()]

    GoToPubchempy = GoToSmileMatching.loc[GoToSmileMatching["SMILES"].isna()]
    if "SMILES_ORI" in GoToPubchempy.columns:
        SMILES_ORI = GoToPubchempy["SMILES_ORI"].to_list()
        GoToPubchempy["SMILES"] = SMILES_ORI

        cirpy_result_2 = GoToPubchempy[GoToPubchempy["SMILES"] != " "]
        GoToPubchempy = GoToPubchempy[GoToPubchempy["SMILES"] == " "]

        cirpy_result = pd.concat(
            [cirpy_result, cirpy_result_2],
            ignore_index=True,
        )
    return cirpy_result, GoToPubchempy


def get_smiles_and_compounds(df, name):
    smiles_append = []
    cs_append = []
    for row in df[name].values():
        if pd.notna(row):
            cs = pcp.get_compounds(row, "name")
            if len(cs) == 1:
                cs_append.append(cs[0])
                smiles_append.append(cs[0].isomeric_smiles)
            elif len(cs) > 1:
                cs_append.append(cs)
                smiles_append.append([item.isomeric_smiles for item in cs])
            else:
                cs_append.append(cs)
                smiles_append.append(np.nan)
        else:
            cs_append.append([])
            smiles_append.append(np.nan)
    return cs_append, smiles_append


def pubchempy_SmilesMatching(GoToPubchempy, name):
    cs_append, smiles_append = get_smiles_and_compounds(GoToPubchempy, name)
    GoToPubchempy["matched_compound"] = cs_append
    GoToPubchempy["SMILES"] = smiles_append

    pubchempy_result_1 = GoToPubchempy.loc[~GoToPubchempy["SMILES"].isna()]
    GoTo_NextRound = GoToPubchempy.loc[GoToPubchempy["SMILES"].isna()]

    name = "replace_Name"
    cs_append, smiles_append = get_smiles_and_compounds(GoTo_NextRound, name)
    GoTo_NextRound["matched_compound"] = cs_append
    GoTo_NextRound["SMILES"] = smiles_append

    pubchempy_result_2 = GoTo_NextRound.loc[~GoTo_NextRound["SMILES"].isna()]
    pubchempy_result = pd.concat([pubchempy_result_1, pubchempy_result_2])
    GoTo_NextRound = GoTo_NextRound.loc[GoTo_NextRound["SMILES"].isna()]
    return pubchempy_result, GoTo_NextRound


def Extract_drug_name(input_df, input_name):
    input_df["replace_Name"] = input_df[input_name].apply(replace_text)
    input_df["DNER_with_spacy"] = DNER_with_spacy(input_df)
    name, synonyms, drugbank_id = extract_results(input_df["DNER_with_spacy"])
    input_df["name_to_use"] = name
    input_df["synonyms"] = synonyms
    input_df["drugbank_id"] = drugbank_id
    name_NeedToExcluded = ["antibody", "multi-drugs", "-"]
    input_df = input_df[
        (input_df["replace_Name"].str.startswith(tuple(name_NeedToExcluded)) is False)
    ]
    FailToExtract, GoToSmilesMatching = select_drugs_GoToSecondRound(input_df)
    return FailToExtract, GoToSmilesMatching
