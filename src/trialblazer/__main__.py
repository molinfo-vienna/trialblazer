from trialblazer import Trialblazer

import click


@click.command()
@click.option("--input_file", help="Input File", type=str, required=True)
@click.option("--output_file", help="Output File", default="output.csv", type=str)
@click.option("--model_folder", help="Model Folder", default=None, type=str)
def main(input_file, output_file, model_folder):
    tb = Trialblazer(input_file=input_file, model_folder=model_folder)
    tb.run()
    tb.write(output_file=output_file)


@click.command()
@click.option("--url", help="ZIP Model URL", type=str, required=True)
@click.option("--model_folder", help="Model Folder", default=None, type=str)
def download(url, model_folder):
    tb = Trialblazer(model_url=url, model_folder=model_folder)
    tb.download_model()


if __name__ == "__main__":
    main()
