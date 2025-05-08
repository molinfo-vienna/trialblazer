import trialblazer
import os

from trialblazer.original_code import trialblazer_draft

test_data_folder = os.path.join(os.path.dirname(__file__), "data")
base_model_folder = os.path.join(
    os.path.dirname(trialblazer.__file__), "data", "base_model"
)


def test_draft(tmpdir):
    trialblazer_draft.run(out_folder=tmpdir)
    trialblazer_draft.run(
        out_folder=tmpdir, data_folder=test_data_folder, model_folder=base_model_folder
    )
