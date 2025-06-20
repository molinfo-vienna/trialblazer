import trialblazer
import os

from trialblazer import trialblazer_draft, trialblazer_refactor

test_data_folder = os.path.join(os.path.dirname(__file__), "data")
base_model_folder = os.path.join(
    os.path.dirname(trialblazer.__file__),
    "data",
    "base_model",
)


# def test_draft(tmpdir):
#     trialblazer_draft.run(
#         out_folder=tmpdir, data_folder=test_data_folder, model_folder=base_model_folder
#     )


def test_refactor(tmpdir):
    trialblazer_refactor.run(
        out_folder=tmpdir, data_folder=test_data_folder, model_folder=base_model_folder
    )
