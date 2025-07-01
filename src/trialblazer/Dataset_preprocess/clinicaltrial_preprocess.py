import os
import re
import zipfile
import pandas as pd

substring_1 = ["(T|t)oxic"]
substring_2 = [
    "(U|u)nrelated",
    "(N|n)ot (R|r)elated",
    "unknown",
    "(N|n)ot (S|s)tudy (R|r)elated",
    "non-related",
    "not due to",
    "other than",
    "(E|e)valuable",
    "Unable to complete",
    "no human safety concern",
    "concern for high systemic drug levels",
    "no participant had a dose limiting toxicity",
    "Enrol",
    "evalua",
    "Ongoing",
    "minimiz",
    "monitoring",
    "(R|r)eview",
    "rare",
    "potential",
    "non clinical toxicology",
    "gastrointestinal toxicity",
    "Toxicity assessment",
    "re-doing",
    "no excess of",
    "Unable to safely escalate",
    "check",
    "FDA",
    "concerns",
    "lack of response activity",
    "Investigators analyze the toxicity data",
]


def extract_filesFromZip(input_Path, dir_out) -> None:
    for filename in os.listdir(input_Path):
        if filename.endswith(".zip"):
            input_file = os.path.join(input_Path, filename)
            folder_name = input_file.split("/")[-1].split(".")[0]
            unzipFolder = dir_out / folder_name
            if not dir_out.exists():
                dir_out.mkdir()
            if not unzipFolder.exists():
                unzipFolder.mkdir()
            with zipfile.ZipFile(input_file) as zf:
                zf.extract("studies.txt", unzipFolder)
                zf.extract("interventions.txt", unzipFolder)
                zf.extract("drop_withdrawals.txt", unzipFolder)


def map_names_intervention(
    Agg_name,
    toxicity_negative,
    toxicity_negative_excluded,
):
    # Use "interventions set" to map the name of drugs through nct_id
    lst = [";", "and"]
    OneDrug = Agg_name[~Agg_name["name"].str.contains("|".join(lst))]
    OneDrug = OneDrug.rename(columns={"name": "Name"})
    toxicity_negative_sim = toxicity_negative[
        ["nct_id", "reason", "why_stopped", "phase", "overall_status"]
    ]
    toxicity_negative_excluded_sim = toxicity_negative_excluded[
        ["nct_id", "reason", "why_stopped", "phase", "overall_status"]
    ]
    # match the name using nct_id
    toxic_onedrug = OneDrug.merge(
        toxicity_negative_sim,
        how="right",
        on="nct_id",
    )
    benign_onedrug = OneDrug.merge(
        toxicity_negative_excluded_sim,
        how="right",
        on="nct_id",
    )

    return toxic_onedrug, benign_onedrug


def keywords_filtering(df, category):
    contains_sub1 = df[category].str.contains("|".join(substring_1))
    contains_sub2 = df[category].str.contains("|".join(substring_2))
    selected_toxicity = df[contains_sub1 & ~contains_sub2]
    non_toxicity = df[~contains_sub1 | contains_sub2]
    return selected_toxicity, non_toxicity


def toxicity_set(studies_set, drop_withdrawals_set):
    # dropna in the "why_stopped" and "reason" categories
    drop_withdrawals_dropna = drop_withdrawals_set.dropna(subset=["reason"])
    studies_simplify_dropna = studies_set.dropna(subset=["why_stopped"])

    studies_simplify_dropna_sim = studies_simplify_dropna[
        ["nct_id", "why_stopped", "phase", "overall_status"]
    ]
    drop_withdrawals_dropna_sim = drop_withdrawals_dropna[["nct_id", "reason"]]

    # nan_set, prepare for the benign set
    drop_withdrawals_set_nan = drop_withdrawals_set[
        drop_withdrawals_set["reason"].isna()
    ]
    studies_set_nan = studies_set[studies_set["why_stopped"].isna()]
    merged_two_nan_set = drop_withdrawals_set_nan.merge(
        studies_set_nan,
        how="outer",
        on="nct_id",
    )
    merged_two_nan_set_sim = merged_two_nan_set[
        ["nct_id", "reason", "why_stopped", "phase", "overall_status"]
    ]

    # filter toxicity description in "why_stopped" and "reason" categories
    selected_toxicity_why_stopped, non_toxicity_why_stopped = keywords_filtering(
        studies_simplify_dropna_sim, "why_stopped",
    )
    selected_toxicity_reason, non_toxicity_reason = keywords_filtering(
        drop_withdrawals_dropna_sim,
        "reason",
    )
    toxicity_negative = selected_toxicity_reason.merge(
        selected_toxicity_why_stopped,
        how="outer",
        on="nct_id",
    )

    # filter the non-toxic set (benign_ori)
    toxicity_negative_excluded = non_toxicity_reason.merge(
        non_toxicity_why_stopped,
        how="outer",
        on="nct_id",
    )
    toxicity_negative_excluded_inclnan = merged_two_nan_set_sim.append(
        toxicity_negative_excluded,
        ignore_index=True,
    )
    return toxicity_negative, toxicity_negative_excluded_inclnan


def check_name(interventions_set):
    interventions_Drug = interventions_set.loc[
        interventions_set.intervention_type == "Drug"
    ]
    Name_exclude_placebo = interventions_Drug[
        interventions_Drug.name.str.contains("(P|p)lacebo") is False
    ]
    Name_exclude_placebo_du = Name_exclude_placebo.drop_duplicates(
        subset=["nct_id", "name"],
        keep="last",
    )  # deduplicates the drugs in the same trial but using different doses
    Agg_name = (
        Name_exclude_placebo_du.groupby(["nct_id"])["name"]
        .agg(lambda x: "; ".join(x.astype(str)))
        .reset_index()
    )
    return Agg_name, Name_exclude_placebo_du


def preprocess_aact(unzipFolder_path):
    toxicity_negative_append = []
    Agg_name_append = []
    toxicity_negative_excluded_append = []
    intervention_set_ori_append = []
    for filename in os.listdir(unzipFolder_path):
        match = re.search(r"\d{6}", filename)
        if not filename.endswith(".zip") and match:
            input_file = os.path.join(unzipFolder_path, filename)

            studies_str_path = input_file + "/" + "studies" + ".txt"
            interventions_str_path = input_file + "/" + "interventions" + ".txt"
            drop_withdrawals_str_path = input_file + "/" + "drop_withdrawals" + ".txt"

            studies_set = pd.read_csv(studies_str_path, sep="|")
            interventions_set = pd.read_csv(interventions_str_path, sep="|")
            drop_withdrawals_set = pd.read_csv(
                drop_withdrawals_str_path,
                sep="|",
            )

            Agg_name, intervention_set_ori = check_name(interventions_set)
            toxicity_negative, toxicity_negative_excluded_inclnan = toxicity_set(
                studies_set, drop_withdrawals_set,
            )

            # Append the results in different files
            toxicity_negative_append.append(toxicity_negative)
            toxicity_negative_excluded_append.append(
                toxicity_negative_excluded_inclnan,
            )
            Agg_name_append.append(Agg_name)
            intervention_set_ori_append.append(intervention_set_ori)

    toxicity_negative_set_total = pd.concat(toxicity_negative_append, axis=0)
    toxicity_negative_excluded_append_total = pd.concat(
        toxicity_negative_excluded_append,
        axis=0,
    )
    Agg_name_append_total = pd.concat(Agg_name_append, axis=0)
    intervention_set_ori_total = pd.concat(intervention_set_ori_append, axis=0)
    (
        toxicity_negative_onedrug,
        toxicity_negative_exclude_onedrug,
    ) = map_names_intervention(
        Agg_name_append_total,
        intervention_set_ori_total,
        toxicity_negative_set_total,
        toxicity_negative_excluded_append_total,
    )
    return (
        Agg_name_append_total,
        toxicity_negative_onedrug,
        toxicity_negative_exclude_onedrug,
        intervention_set_ori_total,
    )


def prepocess_wholeset(count_path) -> None:
    toxic_append = []
    Agg_name_append = []
    benign_onedrug_append = []
    intervention_set_ori_append = []
    for filename in os.listdir(count_path):
        match = re.search(r"\d{4}", filename)
        if not filename.endswith(".zip") and match:
            input_file = os.path.join(count_path, filename)
            (
                Agg_name,
                toxic_onedrug,
                benign_onedrug,
                intervention_set_ori,
            ) = preprocess_aact(input_file)

            toxic_append.append(toxic_onedrug)
            benign_onedrug_append.append(benign_onedrug)
            Agg_name_append.append(Agg_name)
            intervention_set_ori_append.append(intervention_set_ori)

    toxic_append_total = pd.concat(toxic_append, axis=0)
    Agg_name_append_total = pd.concat(Agg_name_append, axis=0)
    benign_append_total = pd.concat(benign_onedrug_append, axis=0)
    intervention_set_ori_append_total = pd.concat(
        intervention_set_ori_append,
        axis=0,
    )

    toxic_append_total.nct_id.unique().size
    Agg_name_append_total.nct_id.unique().size
    intervention_set_ori_append_total.nct_id.unique().size

    check_other_phase = count_path + "/" + "check_other_phase_1"
    if not os.path.exists(check_other_phase):
        os.makedirs(check_other_phase)

    Merge_name_path = check_other_phase + "/" + "toxicity_multi_drugs" + ".csv"
    toxicity_negative_excluded_path = (
        check_other_phase + "/" + "toxicity_multidrugs_excluded" + ".csv"
    )
    Agg_name_append_total_append_total_path = (
        check_other_phase + "/" + "Agg_name" + ".csv"
    )
    intervention_set_ori_append_total_path = (
        check_other_phase + "/" + "intervention_set_ori" + ".csv"
    )


    toxic_append_total.to_csv(Merge_name_path, sep="|")
    benign_append_total.to_csv(toxicity_negative_excluded_path, sep="|")
    Agg_name_append_total.to_csv(
        Agg_name_append_total_append_total_path,
        sep="|",
    )
    intervention_set_ori_append_total.to_csv(
        intervention_set_ori_append_total_path,
        sep="|",
    )
