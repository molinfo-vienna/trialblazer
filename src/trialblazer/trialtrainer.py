import os
import sqlite3
import tarfile
import tempfile
import numpy as np
import pandas as pd
import requests
from FPSim2.io import create_db_file
from rdkit import Chem
from .trialblazer import Trialblazer


def label_str_to_int(x):
    if x == "inactive":
        return 0
    if x == "active":
        return 1
    return np.nan


class TrialTrainer:
    """Used to train a model from scratch.
    Needed:
     - A version of chembl
     - A training set file (curated)
     - Optional: an extra test set (typically all benign).
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
        morgan_nbits=2048,
    ):
        self.chembl_version = chembl_version
        if chembl_folder is None:
            self.chembl_folder = os.path.join(
                os.environ["HOME"], ".trialblazer", "chembl",
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
        self.morgan_nbits = morgan_nbits

    def chembl_download(self, version=None) -> None:
        """Downloading and decompressing the archive if the database file does not exist."""
        if not os.path.exists(self.chembl_folder):
            os.makedirs(self.chembl_folder)
        filepath = os.path.join(
            self.chembl_folder, f"chembl_{self.chembl_version}.sqlite",
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
                        with open(output_path, "wb") as file:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    file.write(chunk)
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
                    else:
                        for _f in tar.getnames():
                            pass

    def process_activity(self, con=None) -> None:
        if con is None:
            with sqlite3.connect(
                os.path.join(self.chembl_folder, f"chembl_{self.chembl_version}.sqlite"),
            ) as con:
                df = pd.read_sql(self.chembl_query, con=con, chunksize=self.size_limit)
                if self.size_limit is not None:
                    df = next(df)
        else:
            df = pd.read_sql(self.chembl_query, con=con, chunksize=self.size_limit)
            if self.size_limit is not None:
                df = next(df)
        df = df.dropna()
        # remove stereochemistry information and using median activity value as representative activity value for the compounds
        df["mol"] = df["canonical_smiles"].apply(Chem.MolFromSmiles)
        df["SMILES_withoutStereoChem"] = df.mol.apply(
            Chem.MolToSmiles, isomericSmiles=False,
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
            self.activity_filter,
        )
        df_grouped_median = df_grouped_median.rename(
            columns={"standard_value": "standard_value_median"},
        )
        df_grouped_median = df_grouped_median.drop_duplicates()

        df_grouped_median["LABEL"] = df_grouped_median["LABEL"].map(label_str_to_int)
        df_grouped_median_active = df_grouped_median[df_grouped_median.LABEL == 1]
        df_grouped_median_inactive = df_grouped_median[df_grouped_median.LABEL == 0]

        # here the preprocess means the steps 1-5 in model Trialblazer
        self.active_target_preprocessed = self.preprocess(df_grouped_median_active)
        self.inactive_target_preprocessed = self.preprocess(df_grouped_median_inactive)

    def write_target_preprocessed(self, output_folder=None, force=False) -> None:
        if output_folder is None:
            output_folder = os.path.join(
                self.model_folder, "generated", "target_preprocessed",
            )
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filepath = os.path.join(output_folder, "active_target_preprocessed.csv")
        if force or not os.path.exists(filepath):
            self.active_target_preprocessed.to_csv(
                filepath,
                index=False,
                sep="|",
            )
        filepath = os.path.join(output_folder, "inactive_target_preprocessed.csv")
        if force or not os.path.exists(filepath):
            self.inactive_target_preprocessed.to_csv(
                filepath,
                index=False,
                sep="|",
            )

    def preprocess(self, moleculeCsv):
        df = Trialblazer.preprocess(
            moleculeCsv=moleculeCsv,
            smiles_col="SMILES_withoutStereoChem",
            id_col="target_id",
        )
        df = df[["SmilesForDropDu", "id"]].rename(
            columns={"SmilesForDropDu": "SmilesWithoutStereo", "id": "target_id"},
        )
        df["target_id"] = df["target_id"].apply(lambda x: [x])
        return df

    def activity_filter(self, x):
        if x >= self.inactive_threshold:
            return "inactive"
        if x < self.active_threshold:
            return "active"
        return np.nan

    def write_h5(self, smiles, filename, output_folder=None, force=False) -> None:
        """Writing h5 fingerprints database from FPSim2."""
        if output_folder is None:
            output_folder = os.path.join(self.model_folder, "generated", "fingerprints")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, filename)
        if force or not os.path.exists(output_file):
            smiles_gen = ((s, i) for i, s in enumerate(smiles))
            create_db_file(
                mols_source=smiles_gen,
                filename=output_file,
                mol_format="smiles",
                fp_type="Morgan",
                fp_params={"radius": 2, "fpSize": self.morgan_nbits},
            )
        else:
            pass

    def build_model_data(self, con=None, cleanup=True) -> None:
        preprocessed_folder = os.path.join(
            self.model_folder, "generated", "preprocessed",
        )
        fpe_folder = os.path.join(self.model_folder, "generated", "fingerprints")
        if not os.path.exists(
            os.path.join(fpe_folder, "active_fpe.h5"),
        ) or not os.path.exists(os.path.join(fpe_folder, "inactive_fpe.h5")):
            if os.path.exists(
                os.path.join(preprocessed_folder, "active_target_preprocessed.csv"),
            ):
                self.active_target_preprocessed = pd.read_csv(
                    os.path.join(preprocessed_folder, "active_target_preprocessed.csv"),
                    sep="|",
                )
            if os.path.exists(
                os.path.join(preprocessed_folder, "inactive_target_preprocessed.csv"),
            ):
                self.inactive_target_preprocessed = pd.read_csv(
                    os.path.join(
                        preprocessed_folder, "inactive_target_preprocessed.csv",
                    ),
                    sep="|",
                )
            if not hasattr(self, "active_target_preprocessed") or not hasattr(
                self, "inactive_target_preprocessed",
            ):
                if con is None:
                    self.chembl_download()
                self.process_activity(con=con)
                self.write_target_preprocessed()
            else:
                pass
            self.write_h5(
                smiles=self.active_target_preprocessed["SmilesWithoutStereo"],
                filename="active_fpe.h5",
            )
            self.write_h5(
                smiles=self.inactive_target_preprocessed["SmilesWithoutStereo"],
                filename="inactive_fpe.h5",
            )
        else:
            pass
        if not os.path.exists(os.path.join(fpe_folder, "training_data_fpe.h5")):
            self.load_training_data()
            self.write_h5(
                smiles=self.training_data["SmilesForDropDu"],
                filename="training_data_fpe.h5",
            )

        else:
            pass
        if cleanup:
            self.cleanup()

    def load_training_data(self, sep=",") -> None:
        if not hasattr(self, "training_data"):
            self.training_data = pd.read_csv(self.training_set, sep=sep)

    def cleanup(self) -> None:
        for a in (
            "inactive_target_preprocessed",
            "active_target_preprocessed",
            "training_data",
        ):
            if hasattr(self, a):
                delattr(self, a)
