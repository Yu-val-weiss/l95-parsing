"""Contains Python functions relating to prediction."""

import warnings
from enum import Enum
from typing import cast

import click
import pandas as pd
import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import DownloadMethod

from utils.stanza import doc_to_conllu_df, doc_to_deprel_df


class DataFrameFormat(Enum):
    """Enum for selecting a DataFrame format for dependency parsing."""

    CONLLU = "conllu"
    DEPREL = "deprel"


class DependencyParser:
    """Class wrapping Stanza functionality for dependency parsing."""

    @staticmethod
    def _get_dep_parse_pipeline(
        device: str,
        download_method: DownloadMethod,
    ) -> stanza.Pipeline:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return stanza.Pipeline(
                lang="en",
                processors="tokenize,mwt,pos,lemma,depparse",
                device=device,
                download_method=download_method,
            )

    def __init__(
        self,
        *,
        df_format: DataFrameFormat = DataFrameFormat.DEPREL,
        device: str = "mps",
        download_method: DownloadMethod = DownloadMethod.REUSE_RESOURCES,
    ) -> None:
        """Initalises the dependency parser.

        Args:
        df_format (DataFrameFormat, optional): Which pandas df format to use.
        Defaults to DEPREL.
        device (str, optional): Which device to use for the pipeline. Defaults to "mps".
        download_method (DownloadMethod, optional): Which download method to use.
        Defaults to DownloadMethod.REUSE_RESOURCES.

        """
        self._pipe = self._get_dep_parse_pipeline(device, download_method)
        self.df_format = df_format

    def __call__(self, string: str) -> pd.DataFrame:
        """Predict a dependency parse for the string.

        Args:
            pipeline (stanza.Pipeline): A stanza pipeline.
            string (str): String containing sentence/s to parse.

        Returns:
            pd.DataFrame: pandas DataFrame in the format according to `self.df_format`.

        """
        doc = self._pipe(string)
        doc = cast(Document, doc)  # cast to make type checking easier

        df_converters = {
            DataFrameFormat.CONLLU: doc_to_conllu_df,
            DataFrameFormat.DEPREL: doc_to_deprel_df,
        }

        return df_converters[self.df_format](doc)


@click.command()
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


@click.group()
def cli() -> None:
    """Run prediction CLI for dependency parsing and."""


cli.add_command(dependency_parse)

if __name__ == "__main__":
    cli()
