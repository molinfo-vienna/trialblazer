## About trialblazer
Trialblazer is a machine learning classifier based on multilayer perceptrons (MLPs) for the prediction of compounds with a potentially increased risk of late-stage toxicity.

This repository contains data and code for trialblazer: 
Trialblazer: A chemistry-focused predictor of toxicity risks in late-stage drug development. (https://doi.org/10.1016/j.ejmech.2025.118306)

## Data

You can download the data, including training_and_test_data, precalculated_data_for_trialblazer_model and precomputed_data_for_reproduction_with_notebooks, from: https://doi.org/10.5281/zenodo.17311675

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

The input data should be a CSV file with headers and a column named "SMILES". If present, the column "your_id" will also be used for the output.

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

The library contains 2 main classes:

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
Zhang, H., Welsch, M., Schueller, W., & Kirchmair, J. (2025). Trialblazer: A Chemistry-Focused Predictor of Toxicity Risks in Late-Stage Drug Development [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17311675
```
