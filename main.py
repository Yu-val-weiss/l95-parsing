"""Defines CLI for evaluation and prediction."""

from __future__ import annotations

import click
from stanza.pipeline.core import DownloadMethod

from task.eval import eval_dep_rel
from task.predict import DataFrameFormat, DependencyParser


@click.group()
def cli() -> None:
    """Run CLI for evaluation and prediction."""


@cli.group()
def evaluate() -> None:
    """Evaluate."""


@evaluate.command(name="dep_rel")
@click.option(
    "--sentences-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the sentences file for evaluation.",
)
@click.option(
    "--gold-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the gold dependency relations file.",
)
@click.option(
    "--save-predictions",
    default=None,
    type=click.Path(),
    help="Path to save the prediction results.",
)
def _eval_dep_rel(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
) -> None:
    res = eval_dep_rel(sentences_file, gold_file, save_predictions)
    res.pretty_print()


@cli.group()
def predict() -> None:
    """Predict."""


@predict.command()
@click.option(
    "--df-format",
    type=click.Choice(
        [df_format.value for df_format in DataFrameFormat],
        case_sensitive=False,
    ),
    default=DataFrameFormat.DEPREL.value,
    help="Format of the output DataFrame as defined in `DataFrameFormat`.",
)
@click.option(
    "--device",
    type=str,
    default="mps",
    help="Device to run the pipeline on.",
)
@click.option(
    "--download-method",
    type=click.Choice([method.name for method in DownloadMethod], case_sensitive=False),
    default=DownloadMethod.REUSE_RESOURCES.name,
    help="Method to use for downloading Stanza models.",
)
@click.argument("text")
def dependency_parse(
    df_format: str,
    device: str,
    download_method: str,
    text: str,
) -> None:
    """Command-line interface for dependency parsing.

    Text is the input string to parse.
    """
    parser = DependencyParser(
        df_format=DataFrameFormat(df_format),
        device=device,
        download_method=DownloadMethod[download_method],
    )
    result = parser(text)
    click.echo(result)


if __name__ == "__main__":
    cli()
