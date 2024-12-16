"""Contains Python functions relating to prediction."""

import warnings
from enum import Enum
from typing import cast

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
        df_format: DataFrameFormat = DataFrameFormat.CONLLU,
        device: str = "mps",
        download_method: DownloadMethod = DownloadMethod.REUSE_RESOURCES,
    ) -> None:
        """Initalises the dependency parser.

        Args:
        df_format (DataFrameFormat, optional): Which pandas df format to use.
        Defaults to CONLLU.
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


if __name__ == "__main__":
    parser = DependencyParser(df_format=DataFrameFormat.CONLLU)
    print(parser("As she walked past it, the driver's glass started to open."))
