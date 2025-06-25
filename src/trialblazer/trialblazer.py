import logging
import pandas as pd
import csv
import os

# Preprocess compounds
import os
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
import pandas as pd
import pickle
import ast
from . import models
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


logger = logging.getLogger(__name__).addHandler(logging.NullHandler())


class Trialblazer(object):
    """
    Wrapper to load the model, the input smiles, set the different parameters and the methods, and then store the results.
    """

    def __init__(
        self,
        input_file: None | str = None,
        model_folder: None | str = None,
        threshold: float = 0.06264154114736771,
        k: int = 900,
        unsure_if_toxic: bool = False,
        # features: None | list[str] = None,
        morgan_n_bits: int = 2048,
    ) -> None:
        """
        Create the triablazer object
        """
        self.input_file = input_file
        self.k = k
        self.threshold = threshold
        self.unsure_if_toxic = unsure_if_toxic
        # self.features = features if features is not None else []
        self.morgan_n_bits = morgan_n_bits
        if model_folder is None:
            self.model_folder = os.path.join(
                os.path.dirname(__file__), "data", "base_model"
            )
        else:
            self.model_folder = model_folder

    def import_smiles(self, smiles: list[str] = []) -> None:
        """
        Importing smiles either from the input file or from a list of smiles
        """
        if not hasattr(self, "smiles"):
            init_smiles = []
        else:
            init_smiles = self.smiles
        set_smiles = set(init_smiles)
        self.smiles = init_smiles + [s for s in smiles if s not in set_smiles]

    def import_smiles_file(
        self, smiles_file: str | None = None, force: bool = False
    ) -> None:
        """
        Importing smiles either from the input file or from a list of smiles
        """
        if not hasattr(self, "smiles") or force:
            if smiles_file is None:
                smiles_file = self.input_file
            if smiles_file is not None:
                with open(smiles_file, "r") as f:
                    read_smiles = f.read().split("\n")
                self.import_smiles(read_smiles)

    def run(self, force: bool = False) -> None:
        """
        Running model and storing results in self.result.
        If self.result already exists, recalculating only if "force" is specified.
        """
        if not hasattr(self, "result") or force:
            self.import_smiles_file(force=force)
            self.load_model()
            self.preprocess()
            result_with_score, closest_distance = self.run_model()
            self.result = dict(
                with_score=result_with_score,
                closest_distance=closest_distance,
            )

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns result as a dataframe
        """
        if not hasattr(self, "result"):
            raise IOError(
                "No result in Trialblazer object: Run the model first with the run() method"
            )
        else:
            df = self.result
            return df

    def write(self, output_file: str = "trialblazer_output.csv") -> None:
        """
        Write to file
        """
        if not hasattr(self, "result"):
            raise IOError(
                "No result in Trialblazer object: Run the model first with the run() method"
            )
        else:
            with open(output_file, "w") as f:
                f.write(str(self.result))

    def run_model(self):
        """
        Once the model is loaded and the input data preprocessed, run the prediction
        """
        if not hasattr(self, "model_data"):
            raise IOError(
                "No model data in Trialblazer object: Load the model first with the load_model() method"
            )
        return models.trialblazer_func(
            classifier=self.classifier,
            selector=self.selector,
            test_set=self.test_set,
            threshold=self.threshold,
            training_fpe=self.model_data["training_data_fpe"],
            unsure_if_toxic=self.unsure_if_toxic,
            features=self.model_data["features"],
            training_set=self.model_data["training_target_features"],
        )

    def preprocess(self):
        """
        Preprocess the input data
        """
        pass

    def load_model(self, model_folder=None):
        """
        Load the model
        """
        ##################### Model folder only
        """The following steps are used to calculate the descriptors for the compounds in dataset."""
        """Step 6, load the processed ChEMBL data and preprocessed training target features"""

        model_data = dict()
        if model_folder is None:
            model_folder = self.model_folder
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
        morgan_cols = [f"morgan2_b{i}" for i in range(self.morgan_n_bits)]
        training_target_features = add_morgan_fingerprints(
            training_target_features, morgan_cols
        )

        with open(
            Path(model_folder) / "training_data_fpe.pkl",
            "rb",
        ) as f:
            training_data_fpe = pickle.load(
                f
            )  # this fpe don't need to be generated again

        # M2FPs_PBFPs = morgan_cols + training_target_list
        model_data["y"] = training_target_features.Mark
        model_data["training_target_features"] = training_target_features
        model_data["active_fpe"] = active_fpe
        model_data["inactive_fpe"] = inactive_fpe
        model_data["training_target_list"] = training_target_list
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

    def save_classifier(self):
        pass

    def train_model(self, force=False, save=True):
        """
        Train the model if the file is not available
        """

        if not hasattr(self, "classifier") or force:
            self.classifier, self.selector = models.trialblazer_train(
                training_set=self.model_data["training_target_features"],
                y=self.model_data["y"],
                features=self.model_data["features"],
                k=self.k,
            )
            if save:
                self.save_classifier()
