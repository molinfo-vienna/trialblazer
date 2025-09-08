from __future__ import annotations
import ast
import logging
import os
import pickle
import re
import tempfile
import zipfile
from pathlib import Path
import pandas as pd
import requests
from FPSim2 import FPSim2Engine
import tarfile

# Preprocess compounds
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from .Dataset_preprocess.Preprocess_compound.Preprocess_FoodDB import (
    CheckOutputResult,
)

# these three functions are in folder Dataset_preprocess/Preprocess_compound
from .Dataset_preprocess.Preprocess_compound.Preprocess_FoodDB import (
    preprocess,
)
from .Dataset_preprocess.Preprocess_compound.remove_duplicate import (
    remove_duplicate_splitted_files,
)
from .Dataset_preprocess.Preprocess_compound.separate_multicomponents_test import (
    separate_multicomponents_test,
)

# these two functions are in folder Descriptor_calculation
from .Descriptor_calculation.process_similarity_results import (
    remove_tested_inactive_targets,
)
from .Descriptor_calculation.process_target_features import (
    add_morgan_fingerprints,
)
from .Descriptor_calculation.process_target_features import (
    process_target_features,
)
from .models import trialblazer_func
from .models import trialblazer_train

logger = logging.getLogger(__name__).addHandler(logging.NullHandler())


class Trialblazer:
    """Wrapper to load the model, the input smiles, set the different parameters and the methods, and then store the results."""

    def __init__(
        self,
        input_file: None | str = None,
        model_folder: None | str = None,
        M2FP_only: bool = False,
        threshold: float | None = None,
        k: int | None = None,
        remove_MultiComponent_cpd: bool = True,
        # features: None | list[str] = None,
        morgan_n_bits: int = 2048,
        model_url: str | None = "https://zenodo.org/records/15783346/files/precalculated_data_for_trialblazer_model.tar.gz",
        archive_type:str="tar.gz",
        top_folder:bool=True,
    ) -> None:
        """Create the triablazer object."""
        self.input_file = input_file
        self.k = k
        self.threshold = threshold
        self.remove_MultiComponent_cpd = remove_MultiComponent_cpd
        self.morgan_n_bits = morgan_n_bits
        if model_folder is None:
            self.model_folder = os.path.join(
                os.environ["HOME"], ".trialblazer", "models", "base_model",
            )
        else:
            self.model_folder = model_folder
        self.M2FP_only = M2FP_only
        if threshold is not None:
            self.threshold = threshold
        elif M2FP_only:
            self.threshold = 0.10338700489734302
        else:
            self.threshold = 0.06264154114736771
        if k is not None:
            self.k = k
        elif M2FP_only:
            self.k = 800
        else:
            self.k = 900
        self.model_url = model_url
        self.archive_type = archive_type
        self.top_folder=top_folder

    def import_smiles(self, smiles: list[str] | None = None) -> None:
        """Importing smiles either from the input file or from a list of smiles."""
        # if not hasattr(self, "smiles"):
        #     init_smiles = []
        # else:
        #     init_smiles = self.smiles
        # set_smiles = set(init_smiles)
        # self.smiles = init_smiles + [s for s in smiles if s not in set_smiles]
        if smiles is None:
            smiles = []
        smiles_df = pd.DataFrame([{"SMILES": s} for s in smiles])
        if not hasattr(self, "smiles"):
            self.smiles = smiles_df
        else:
            self.smiles = pd.concat(self.smiles, smiles_df)

    def import_smiles_file(
        self, smiles_file: str | None = None, force: bool = False,
    ) -> None:
        """Importing smiles either from the input file or from a list of smiles."""
        if not hasattr(self, "smiles") or force:
            if smiles_file is None:
                smiles_file = self.input_file
            if smiles_file is not None:
                smiles_df = pd.read_csv(smiles_file)

                if not hasattr(self, "smiles"):
                    self.smiles = smiles_df
                else:
                    self.smiles = pd.concat(self.smiles, smiles_df)
                # read_smiles = smiles_df["SMILES"].to_list()
                # self.import_smiles(read_smiles)

    def download_model(self,archive_type:str|None=None,top_folder:bool|None=None) -> None:
        if archive_type is None:
            archive_type=self.archive_type
        if top_folder is None:
            top_folder=self.top_folder
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        if not os.path.exists(
            os.path.join(self.model_folder, "training_target_features.csv"),
        ):
            if self.model_url is not None:
                model_url = self.model_url
            elif "TRIALBLAZER_URL" in os.environ:
                model_url = os.environ["TRIALBLAZER_URL"]
            else:
                msg = "No specified value for model_url, aborting download."
                raise ValueError(msg)

            with tempfile.TemporaryDirectory() as tempdir:
                with requests.get(model_url, stream=True) as response:
                    response.raise_for_status()  # Raise an error for HTTP issues

                    # Save the archive file to a temporary location
                    archive_path = os.path.join(tempdir, f"temp_model.{archive_type}")
                    with open(archive_path, "wb") as temp_archive:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:  # Filter out keep-alive chunks
                                temp_archive.write(chunk)

                if archive_type == 'zip':
                    with zipfile.ZipFile(archive_path, "r") as zip_file:
                        if top_folder:
                            all_members = zip_file.namelist()
                            members = [m for m in all_members if not m.startswith('.')]
                            top_level_folder = os.path.commonpath(members)

                            for member in members:
                                if member.endswith("/"):
                                    continue
                                member_path = os.path.relpath(member, top_level_folder)
                                target_path = os.path.join(self.model_folder, member_path)

                                print(f'Installing {target_path}')
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                                with zip_file.open(member) as source, open(target_path, "wb") as target:
                                    target.write(source.read())
                        else:
                            zip_file.extractall(path=self.model_folder)

                elif archive_type == 'tar.gz':
                    with tarfile.open(archive_path, "r:gz") as tar:
                        if top_folder:
                            all_members = tar.getmembers()
                            members = [m for m in all_members if not m.name.startswith('.')]
                            top_level_folder = os.path.commonpath([m.name for m in members])

                            for member in members:
                                member_path = os.path.relpath(member.name, top_level_folder)
                                target_path = os.path.join(self.model_folder, member_path)
                                print(f'Installing {target_path}')

                                if member.isdir():
                                    os.makedirs(target_path, exist_ok=True)
                                else:
                                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                    with tar.extractfile(member) as source, open(target_path, "wb") as target:
                                        target.write(source.read())
                        else:
                            tar.extractall(path=self.model_folder)

                else:
                    raise ValueError(f'Archive type not supported: {archive_type}')

    def run(self, force: bool = False) -> None:
        """Running model and storing results in self.result.
        If self.result already exists, recalculating only if "force" is specified.
        """
        if not hasattr(self, "result") or force:
            self.import_smiles_file(force=force)
            self.load_model()
            self.prepare_testset(force=force)
            self.run_model()

    def get_dataframe(self) -> pd.DataFrame:
        """Returns result as a dataframe."""
        if not hasattr(self, "result"):
            msg = "No result in Trialblazer object: Run the model first with the run() method"
            raise OSError(
                msg,
            )
        df = self.result.copy()

        PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smi", molCol="mol")
        # df["ROMol"] = df["SMILES"].apply(Chem.MolFromSmiles)
        return df

    def write(
        self, output_file: str = "trialblazer_output.csv", sep: str = "|",
    ) -> None:
        """Write to file."""
        if not hasattr(self, "result"):
            msg = "No result in Trialblazer object: Run the model first with the run() method"
            raise OSError(
                msg,
            )
        self.result.set_index("id").sort_index().to_csv(
            output_file, index=True, sep=sep,
        )

    def run_model(self) -> None:
        """Once the model is loaded and the input data preprocessed, run the prediction."""
        if not hasattr(self, "model_data"):
            msg = "No model data in Trialblazer object: Load the model first with the load_model() method"
            raise OSError(
                msg,
            )
        self.result = trialblazer_func(
            classifier=self.classifier,
            selector=self.selector,
            test_set=self.test_set,
            threshold=self.threshold,
            training_fpe=self.model_data["training_data_fpe"],
            remove_MultiComponent_cpd=self.remove_MultiComponent_cpd,
            features=self.model_data["features"],
            training_set=self.model_data["training_target_features"],
        )

    @classmethod
    def preprocess(cls, moleculeCsv, out_folder=None, smiles_col="SMILES", id_col=None):
        """Preprocess the input data."""
        if out_folder is None:
            out_folder_obj = (
                tempfile.TemporaryDirectory()
            )  # os.path.join(test_folder_data, "..", "temp")
            out_folder = out_folder_obj.name

        """Step 1, preprocess compounds"""
        # refinedInputFile = "/data/local/Druglikness_prediction/external_test_set/approved_testset_final_withname.csv"

        # refinedInputFile = self.input_file
        # refinedInputFile =

        refinedOutputFolder = Path(out_folder)
        preprocess(
            moleculeCsv, refinedOutputFolder, smiles_col=smiles_col, id_col=id_col,
        )  # input: a dataframe with SMILES and ID, outpyut: a folder with the preprocessed files
        CheckOutputResult(
            refinedOutputFolder,
        )  # generate a csv file from log to check the output result

        """Step 2, separate multicomponents"""
        DB_processedDir = Path(out_folder) / "preprocessedSmiles"
        # the folder generated from step 1
        separate_multicomponents_test(
            DB_processedDir,
        )  # input: a folder with the preprocessed files, output: a folder with the separated multicomponents, for this step, I modified the script from anya

        """Step 3, remove duplicates"""
        refinedInputFolder = Path(out_folder) / "separate_multicom_GetMolFrags"
        # the folder generated from step 2
        refinedOutputFolder = refinedInputFolder.parent / "uniqueSmiles"
        if not refinedOutputFolder.exists():
            refinedOutputFolder.mkdir()
        remove_duplicate_splitted_files(
            refinedInputFolder, refinedOutputFolder, "approved_testset_final",
        )  # input: a folder with the separated multicomponents, output: a folder with the unique smiles files

        """Step 4, combine the unique smiles files, this step I haven't made it into a function"""
        unique_smiles_path = Path(out_folder) / "uniqueSmiles"

        df_list = []
        for filenames in os.listdir(unique_smiles_path):
            file = unique_smiles_path / filenames
            df = pd.read_csv(file, sep="\t", index_col=None, header=0)
            df_list.append(df)
        preprocessed_df = pd.concat(df_list, axis=0, ignore_index=True)

        """Step 5, remove stereochemical information from the preprocessed SMILES and filter the compounds to retain only small molecules"""
        preprocessed_df["Molecule"] = preprocessed_df["preprocessedSmiles"].apply(
            Chem.MolFromSmiles,
        )
        preprocessed_df["SmilesForDropDu"] = preprocessed_df["Molecule"].apply(
            Chem.MolToSmiles, isomericSmiles=False,
        )
        preprocessed_df["Molecule"] = preprocessed_df["SmilesForDropDu"].apply(
            Chem.MolFromSmiles,
        )
        preprocessed_df["mw"] = preprocessed_df.Molecule.apply(
            lambda mol: round(Descriptors.MolWt(mol), 3),
        )
        return preprocessed_df[preprocessed_df["mw"].between(150, 850)]

    def apply_tanimoto(self, preprocessed_df):
        """Step 7, calculate and process the Tanimoto similarity results, the query data is the preprocessed data from step 1-5, the output of this step is the target feature."""
        # load the preprocessed active and inactive targets from the ChEMBL database,
        # these targets are preprocessed through Step 1-5, but in the application it is not necessary to calculate it from the scratch

        temp_folder = tempfile.TemporaryDirectory()
        temp_folder2 = tempfile.TemporaryDirectory()
        output_path_temp_save_testdata_active = (
            temp_folder  # create a folder to save temporary files for the test data
        )
        output_path_temp_save_testdata_inactive = (
            temp_folder2  # create a folder to save temporary files for the test data
        )

        model_data = self.model_data
        (
            testset_active_binarize_list,
            testset_active_binarized_target_remain,
        ) = process_target_features(
            output_path_temp_save=output_path_temp_save_testdata_active,
            training_target_list=model_data[
                "training_target_list"
            ],  # only need to used the same targets as the training data
            fpe=model_data["active_fpe"],
            preprocessed_target_unique_smiles=model_data[
                "active_preprocessed_target_unique_smiles"
            ],
            query_data=preprocessed_df,
        )

        (
            testset_inactive_binarize_list,
            testset_inactive_binarized_target_remain,
        ) = process_target_features(
            output_path_temp_save=output_path_temp_save_testdata_inactive,
            training_target_list=model_data[
                "training_target_list"
            ],  # only need to used the same targets as the training data
            fpe=model_data["inactive_fpe"],
            preprocessed_target_unique_smiles=model_data[
                "inactive_preprocessed_target_unique_smiles"
            ],
            query_data=preprocessed_df,
        )
        testset_filtered_targets, testset_target_list = remove_tested_inactive_targets(
            testset_inactive_binarized_target_remain,
            testset_active_binarized_target_remain,
        )  # testset_filtered_targets is the target features I need for testset compounds

        testset_filtered_targets_id = testset_filtered_targets.merge(
            preprocessed_df[["SmilesForDropDu", "id"]],
            how="left",
            on="SmilesForDropDu",
        )

        """Step 8, calculate Morgan2 fingerprints for the training and test data"""
        morgan_cols = [f"morgan2_b{i}" for i in range(self.morgan_n_bits)]
        testset_filtered_targets_id = add_morgan_fingerprints(
            testset_filtered_targets_id, morgan_cols,
        )

        """Final step, employ the model"""
        # The input of Trialblazer is a dataframe of training featrues and the binary label of each compound, and the test set,
        # the output including a dataframe with the PrOCTOR socre and prediction results for each compound in test set, and the cloestest similairty between test compounds and training compounds
        return testset_filtered_targets_id

    def prepare_testset(self, out_folder=None, force=False) -> None:
        if not hasattr(self, "test_set") or force:
            preprocessed_df = self.preprocess(
                moleculeCsv=self.smiles, out_folder=out_folder,
            )
            test_set = self.apply_tanimoto(preprocessed_df=preprocessed_df)
            self.test_set = test_set
        else:
            pass

    def load_model(self, model_folder=None) -> None:
        """Load the model."""
        ##################### Model folder only
        """The following steps are used to calculate the descriptors for the compounds in dataset."""
        """Step 6, load the processed ChEMBL data and preprocessed training target features"""

        model_data = {}
        if model_folder is None:
            model_folder = self.model_folder
        inactive_preprocessed_target_unique_smiles = pd.read_csv(
            Path(model_folder)
            / "generated"
            / "target_preprocessed"
            / "inactive_target_preprocessed.csv",
            sep="|",
        )  # this preprocessed target smiles is precalculated by previous preprocessed steps base on chembl data, it doesn't need to be re-calculated.
        active_preprocessed_target_unique_smiles = pd.read_csv(
            Path(model_folder)
            / "generated"
            / "target_preprocessed"
            / "active_target_preprocessed.csv",
            sep="|",
        )  # this preprocessed target smiles is precalculated by previous preprocessed steps base on chembl data, it doesn't need to be re-calculated.
        active_preprocessed_target_unique_smiles[
            "target_id"
        ] = active_preprocessed_target_unique_smiles["target_id"].apply(
            ast.literal_eval,
        )  # this step is for converting the string type when I read the file from csv
        inactive_preprocessed_target_unique_smiles[
            "target_id"
        ] = inactive_preprocessed_target_unique_smiles["target_id"].apply(
            ast.literal_eval,
        )  # this step is for converting the string type when I read the file from csv

        # load the preprocessed training target features
        training_target_features = pd.read_csv(
            Path(model_folder) / "training_target_features.csv",
        )  # this training_target_features is calculated previously and don't need to be re-calculated

        # Load the generated active and inactive fpe (fingerprints engine)
        # These two fpe files are generated previously by using process_target_features function based on trainig data, and don't need to regenerated, I removed h5 files because if we have the fpe, then basically don't need the h5 file
        active_fpe = FPSim2Engine(
            Path(model_folder) / "generated" / "fingerprints" / "active_fpe.h5",
        )
        inactive_fpe = FPSim2Engine(
            Path(model_folder) / "generated" / "fingerprints" / "inactive_fpe.h5",
        )

        # load the preprocessed target features list, from the column names
        # with open(file_path, "r") as file:
        #     reader = csv.reader(file)
        #     training_target_list = [
        #         row[0] for row in reader
        #     ]  # get the list of traing_targets_list, number of targets: 777
        training_target_list = [
            c
            for c in training_target_features.columns
            if re.fullmatch(r"^CHEMBL[0-9]+$", c)
        ]
        morgan_cols = [f"morgan2_b{i}" for i in range(self.morgan_n_bits)]
        training_target_features = add_morgan_fingerprints(
            training_target_features, morgan_cols,
        )

        training_data_fpe = FPSim2Engine(
            Path(model_folder) / "generated" / "fingerprints" / "training_data_fpe.h5",
        )

        # M2FPs_PBFPs = morgan_cols + training_target_list
        model_data["y"] = training_target_features.Mark
        model_data["training_target_features"] = training_target_features
        model_data["active_fpe"] = active_fpe
        model_data["inactive_fpe"] = inactive_fpe
        model_data["training_target_list"] = training_target_list
        if self.M2FP_only:
            model_data["features"] = morgan_cols
        else:
            model_data["features"] = morgan_cols + training_target_list
        model_data["training_data_fpe"] = training_data_fpe
        model_data[
            "inactive_preprocessed_target_unique_smiles"
        ] = inactive_preprocessed_target_unique_smiles
        model_data[
            "active_preprocessed_target_unique_smiles"
        ] = active_preprocessed_target_unique_smiles
        self.model_data = model_data

        self.train_model()

    def save_classifier(self) -> None:
        classifier_path = Path(self.model_folder) / "generated" / "classifier.pkl"
        selector_path = Path(self.model_folder) / "generated" / "selector.pkl"
        if not os.path.exists(classifier_path):
            with open(classifier_path, "wb") as f:
                pickle.dump(self.classifier, file=f)
        if not os.path.exists(selector_path):
            with open(selector_path, "wb") as f:
                pickle.dump(self.selector, file=f)

    def load_classifier(self) -> None:
        classifier_path = Path(self.model_folder) / "generated" / "classifier.pkl"
        selector_path = Path(self.model_folder) / "generated" / "selector.pkl"
        with open(classifier_path, "rb") as f:
            self.classifier = pickle.load(f)
        with open(selector_path, "rb") as f:
            self.selector = pickle.load(f)

    def train_model(self, force=False, save=False, loadable=False) -> None:
        """Train the model if the file is not available. Save/Load is not advised: classifier depends on features chosen."""
        classifier_path = Path(self.model_folder) / "generated" / "classifier.pkl"
        if os.path.exists(classifier_path) and loadable:
            self.load_classifier()

        if not hasattr(self, "classifier") or force:
            self.classifier, self.selector = trialblazer_train(
                training_set=self.model_data["training_target_features"],
                y=self.model_data["y"],
                features=self.model_data["features"],
                k=self.k,
            )
            if save:
                self.save_classifier()
