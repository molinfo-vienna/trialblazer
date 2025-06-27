import tempfile
import os
import requests
import time
import tarfile

import sqlite3
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from .trialblazer import Trialblazer


def label_str_to_int(x):
    if x == "inactive":
        return 0
    elif x == "active":
        return 1
    else:
        return np.nan


class TrialTrainer(object):
    """
    Used to train a model from scratch.
    Needed:
     - A version of chembl
     - A training set file (curated)
     - Optional: an extra test set (typically all benign)
    """

    chembl_query = """
WITH valid_data AS (
    SELECT *
    FROM activities
    WHERE standard_type IN ('Kd', 'Potency', 'AC50', 'IC50', 'Ki', 'EC50')
    AND standard_relation == '='
    AND standard_units == 'nM'
    AND potential_duplicate == 0
    AND (data_validity_comment IS NULL OR data_validity_comment == 'Manually validated')
)
SELECT assays.chembl_id AS assay_id, 
       assay_type, 
       target_dictionary.pref_name, 
       standard_type,
       molecule_dictionary.chembl_id AS molecule_id, 
       target_dictionary.chembl_id AS target_id,
       canonical_smiles,
       standard_value, 
       pchembl_value 
FROM valid_data
LEFT JOIN compound_structures USING (molregno) 
LEFT JOIN assays USING (assay_id) 
LEFT JOIN target_dictionary USING (tid)
LEFT JOIN molecule_dictionary USING (molregno)
INNER JOIN (
    SELECT assay_id, 
           standard_type
    FROM valid_data 
    GROUP BY assay_id, standard_type 
    ) USING (
        assay_id, 
        standard_type
        )
    WHERE confidence_score IN (7, 8, 9)
"""

    def __init__(
        self,
        chembl_version=34,
        training_set=None,
        model_folder=None,
        extra_test_set=None,
        chembl_folder=None,
        archive_folder=None,
        inactive_threshold=20000,
        active_threshold=10000,
        size_limit=None,
    ):
        self.chembl_version = chembl_version
        if chembl_folder is None:
            self.chembl_folder = os.path.join(
                os.environ["HOME"], ".trialblazer", "chembl"
            )
        else:
            self.chembl_folder = chembl_folder
        if model_folder is None:
            self.model_folder = os.path.join(
                os.path.dirname(__file__),
                "data",
                "base_model",
            )
        else:
            self.model_folder = model_folder
        if training_set is None:
            self.training_set = os.path.join(
                self.model_folder,
                "training_target_features.csv",
            )
        else:
            self.training_set = training_set
        self.extra_test_set = extra_test_set
        self.archive_folder = archive_folder
        self.inactive_threshold = inactive_threshold
        self.active_threshold = active_threshold
        self.size_limit = size_limit

    def chembl_download(self, version=None):
        """
        Downloading and decompressing the archive if the database file does not exist
        """
        if not os.path.exists(self.chembl_folder):
            os.makedirs(self.chembl_folder)
        filepath = os.path.join(
            self.chembl_folder, f"chembl_{self.chembl_version}.sqlite"
        )
        if not os.path.exists(filepath):
            filename = f"chembl_{self.chembl_version}_sqlite.tar.gz"
            url = (
                f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{self.chembl_version}/{filename}",
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                if self.archive_folder is None:
                    output_path = os.path.join(tmpdir, filename)
                else:
                    output_path = os.path.join(
                        self.archive_folder,
                        filename,
                    )
                if not os.path.exists(output_path):
                    with requests.get(url, stream=True) as response:
                        response.raise_for_status()  # Raise an error for bad status codes (e.g., 404, 500)
                        print(
                            f"Downloading Chembl SQLite {self.chembl_version} to {output_path}"
                        )
                        with open(output_path, "wb") as file:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    file.write(chunk)
                print(f"Decompressing Chembl SQLite {self.chembl_version} archive")
                target_file = f"chembl_{self.chembl_version}/chembl_{self.chembl_version}_sqlite/chembl_{self.chembl_version}.db"
                with tarfile.open(output_path, "r:gz") as tar:
                    # Check if the target file exists in the archive
                    if target_file in tar.getnames():
                        # Extract the specific file to the desired location
                        with tar.extractfile(target_file) as file:
                            buffer_size = 50 * 1024 * 1024
                            with open(filepath, "wb") as output_file:
                                while True:
                                    chunk = file.read(buffer_size)
                                    if not chunk:  # Stop when no more data is available
                                        break
                                    output_file.write(chunk)
                        print(f"Extracted '{target_file}' to '{filepath}'.")
                    else:
                        print(
                            f"File '{target_file}' not found in the archive. Existing files in archive:"
                        )
                        for f in tar.getnames():
                            print(f)

    def process_activity(self):
        with sqlite3.connect(
            os.path.join(self.chembl_folder, f"chembl_{self.chembl_version}.sqlite")
        ) as con:
            df = pd.read_sql(self.chembl_query, con=con)
        df = df.dropna()
        # remove stereochemistry information and using median activity value as representative activity value for the compounds
        df["mol"] = df["canonical_smiles"].apply(Chem.MolFromSmiles)
        df["SMILES_withoutStereoChem"] = df.mol.apply(
            Chem.MolToSmiles, isomericSmiles=False
        )
        df_grouped_median = (
            df.groupby(["SMILES_withoutStereoChem", "target_id"])["standard_value"]
            .median()
            .reset_index()
        )

        # get the median number of activity value for the same target, same compounds that were tested in different assays
        df_grouped_median = (
            df.groupby(["SMILES_withoutStereoChem", "target_id"])["standard_value"]
            .median()
            .reset_index()
        )

        df_grouped_median["LABEL"] = df_grouped_median["standard_value"].apply(
            self.activity_filter
        )
        df_grouped_median.rename(
            columns={"standard_value": "standard_value_median"}, inplace=True
        )
        df_grouped_median.drop_duplicates(inplace=True)

        df_grouped_median["LABEL"] = df_grouped_median["LABEL"].map(label_str_to_int)
        df_grouped_median_active = df_grouped_median[df_grouped_median.LABEL == 1]
        df_grouped_median_inactive = df_grouped_median[df_grouped_median.LABEL == 0]

        # here the preprocess means the steps 1-5 in model Trialblazer (before calculate the Tanimoto similarity)
        self.active_target_preprocessed = Trialblazer.preprocess(
            moleculeCsv=df_grouped_median_active
        )
        self.inactive_target_preprocessed = Trialblazer.preprocess(
            moleculeCsv=df_grouped_median_inactive
        )
        self.active_target_preprocessed.to_csv("active_blah.csv")
        self.inactive_target_preprocessed.to_csv("inactive_blah.csv")

    def activity_filter(self, x):
        if x >= self.inactive_threshold:
            return "inactive"
        elif x < self.active_threshold:
            return "active"
        else:
            return np.nan
