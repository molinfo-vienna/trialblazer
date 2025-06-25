# Preprocess compounds
import os
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
import pandas as pd
import pickle
import ast

# from .models import Trialblazer as Trialblazer_func
from .trialblazer import Trialblazer
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


def run(
    model_folder=base_model_folder,
    out_folder=None,
    data_folder=test_folder_data,
):
    tb = Trialblazer(input_file=os.path.join(data_folder, "test_input.csv"))

    # tb.test_set = test_set
    tb.run()
    # result_with_score, closest_distance = (
    #     tb.result["with_score"],
    #     tb.result["closest_distance"],
    # )


if __name__ == "__main__":
    run()
