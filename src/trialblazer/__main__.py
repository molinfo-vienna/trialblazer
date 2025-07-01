import click
from trialblazer import Trialblazer


@click.command()
@click.option("--input_file", help="Input File", type=str, required=True)
@click.option("--output_file", help="Output File", default="output.csv", type=str)
@click.option("--model_folder", help="Model Folder", default=None, type=str)
def main(input_file, output_file, model_folder) -> None:
    tb = Trialblazer(input_file=input_file, model_folder=model_folder)
    tb.run()
    tb.write(output_file=output_file)


@click.command()
@click.option("--url", help="Archive Model URL", type=str, required=False,default="https://zenodo.org/records/15783346/files/precalculated_data_for_trialblazer_model.tar.gz")
@click.option("--archive_type", help="Archive Type", type=str, required=False, default="tar.gz")
@click.option("--top_folder", help="Whether the archive has a top folder", type=str, required=False, default=True)
@click.option("--model_folder", help="Model Folder", default=None, type=str)
def download(url, archive_type,top_folder, model_folder) -> None:
    tb = Trialblazer(model_url=url, model_folder=model_folder,archive_type=archive_type,top_folder=top_folder)
    tb.download_model()


if __name__ == "__main__":
    main()
