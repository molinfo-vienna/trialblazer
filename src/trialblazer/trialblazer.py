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
from .models import Trialblazer as Triablazer_func
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
        self, input_file: None | str = None, model_folder: None | str = None
    ) -> None:
        """
        Create the triablazer object
        """
        self.input_file = input_file
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
            self.run_model()
        self.result = None

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

    def write(self, output_file: str = "output.csv") -> None:
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
        pass

    def preprocess(self):
        """
        Preprocess the input data
        """
        pass

    def load_model(self):
        """
        Load the model
        """
        pass

    def train_model(self):
        """
        Train the model if the file is not available
        """
        pass
