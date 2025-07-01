## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/molinfo-vienna/trialblazer) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/molinfo-vienna/trialblazer)](https://github.com/molinfo-vienna/trialblazer) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-trialblazer-00a3e3.svg)](https://www.research-software.nl/software/trialblazer) [![workflow pypi badge](https://img.shields.io/pypi/v/trialblazer.svg?colorB=blue)](https://pypi.python.org/project/trialblazer/) |
| (4/5) citation                     | |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/trialblazer/badge/?version=latest)](https://trialblazer.readthedocs.io/en/latest/?badge=latest) || **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](https://github.com/molinfo-vienna/trialblazer/actions/workflows/build.yml/badge.svg)](https://github.com/molinfo-vienna/trialblazer/actions/workflows/build.yml) |

## Data

You can download the data, including training_and_test_data, precalculated_data_for_trialblazer_model and precomputed_data_for_reproduction_with_notebooks, from: https://doi.org/10.5281/zenodo.15783346

To download the data automatically, see below the description of the Command Line Interface.

## Reproduce experiments

To reproduce the experiments in the paper, you can check the notebooks here: 
https://github.com/molinfo-vienna/trialblazer_notebooks

## How to use Trialblazer

A Chemistry-Focused Predictor of Toxicity Risks in Late-Stage Drug Development

### Via Command Line

Several commands are made available:


#### Downloading the model
```
# Default model and default folder ($HOME/.trialblazer/models/base_model)
trialblazer-download

# Use other URL/folder
trialblazer-download --url=<MODEL-URL> --model-folder=<FOLDER>
```

#### Running the algorithm

The input data should be a CSV file with headers and a column named "SMILES". If present, the column "chembl_id" will also be used for the output.

The command `trialblazer --help` outputs:

```
Options:
  --input_file TEXT    Input File  [required]
  --output_file TEXT   Output File
  --model_folder TEXT  Model Folder
  --help               Show this message and exit.
```

The default output file is names `trialblazer.csv`.

### As a Python library

The library containers 2 main classes:

#### Trialblazer

This class loads and runs the model.

```
from trialblazer import Trialblazer

tb = Trialblazer(input_file=<INPUT_FILE>)
tb.run()  # Includes loading of the model, creation of the classifier, and running the algorithm

df = tb.get_dataframe() # This dataframe is augmented with RDKit Mol objects, and displaying it shows the visual representation of each molecule.

tb.write(output_file=<OUTPUT_FILE>)
```
#### Trialtrainer

This class is meant to preprocess training data to recreate a model from a single CSV file (`training_target_features.csv`). It downloads the Chembl database, extracts relevant info, preprocesses data for active and inactive targets, and creates fingerprints files for the 3 sets of molecules (training, active, inactive).

Simply put your `training_target_features.csv` in your `MODEL_FOLDER` and run:

```
from trialblazer import Trialtrainer

tt = Trialtrainer(model_folder=<MODEL_FOLDER>)
tt.build_model_data()

```

Then you can run the algorithm using:

```
from trialblazer import Trialblazer

tb = Trialblazer(input_file=<INPUT_FILE>, model_folder=<MODEL_FOLDER>)
tb.run()  # Includes loading of the model, creation of the classifier, and running the algorithm

```
## Installation

To install via PyPI, simply run:
```
pip install trialblazer
```

To install trialblazer from GitHub repository through SSH, do:
```console
git clone git@github.com:molinfo-vienna/trialblazer.git
cd trialblazer
python -m pip install .
```
or through HTTPS:
```console
git clone https://github.com/molinfo-vienna/trialblazer_notebooks.git
cd trialblazer
python -m pip install .
```


## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).


## Citation

```
Zhang, H., Welsch, M., Schueller, W., & Kirchmair, J. (2025). Trialblazer: A Chemistry-Focused Predictor of Toxicity Risks in Late-Stage Drug Development [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15783346
```
