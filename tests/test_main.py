import os
import numpy as np
import pandas as pd
import trialblazer
from trialblazer import Trialblazer

base_model_folder = os.path.join(
    os.path.dirname(trialblazer.__file__),
    "data",
    "base_model",
)

input_file = os.path.join(os.path.dirname(__file__), "data", "test_input.csv")
input_data = pd.read_csv(
    input_file,
    delimiter="|",
)
output_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "test_output.csv"),
    delimiter="|",
)

output_data = output_data[["id", "pred_prob_positive", "PrOCTOR_score"]]
output_rm = output_data[~output_data["id"].str.contains(r"\d+x\d+")]

output_data = output_data.set_index("id")
output_data = output_data.sort_index()
output_rm = output_rm.set_index("id")
output_rm = output_rm.sort_index()


def test_run(tmpdir):
    tb = Trialblazer(input_file=input_file)
    tb.run()


def test_download():
    tb = Trialblazer()
    tb.download_model()


def test_run_no_remove(tmpdir):
    tb = Trialblazer(input_file=input_file, remove_MultiComponent_cpd=False)
    tb.run()
    tb.write(output_file=os.path.join(tmpdir, "trialblazer.csv"))
    df = tb.result.copy()
    df = df[["id", "pred_prob_positive", "PrOCTOR_score"]]
    df = df.set_index("id")
    df = df.sort_index()
    assert len(df.index) == len(output_data.index)
    assert np.isclose(df, output_data).all()
