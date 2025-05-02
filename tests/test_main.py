import os
import pandas as pd

input_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "test_input.csv"),
    delimiter="|",
)
output_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "test_output.csv"),
    delimiter="|",
)


def test_run():
    output_data == output_data
