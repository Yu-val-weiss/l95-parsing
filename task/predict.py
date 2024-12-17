"""Contains Python functions relating to prediction."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, cast

import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import DownloadMethod
from supar import Parser

from utils.constituency import clean_tree
from utils.stanza import doc_to_conllu_df, doc_to_deprel_df

if TYPE_CHECKING:
    import pandas as pd
    from nltk.tree import Tree

warnings.filterwarnings(action="ignore", category=FutureWarning)


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


class ConstituencyParser:
    """Class wrapping Supar functionality for constituence parsing."""

    def __init__(self, path: str = "con-crf-en", *, clean: bool = True) -> None:
        """Initialise the class.

        Args:
            path (str, optional): The path of the parser to load. Options are:
            "con-crf-en", "con-crf-zh", "con-crf-roberta-en", "con-crf-electra-zh",
            "con-crf-xlmr". Defaults to "con-crf-en".

            clean(bool, optional): Flag whether to clean the tree or not.

        """
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("ignore")  # Re-enable all warnings
            self._parser = Parser.load(path)
        # self._parser = Parser.load(path)
        self._clean = clean

    def __call__(self, string: str | list[str]) -> list[Tree]:
        """Predict a constituency parse for the string.

        Args:
            string (str | list): String containing sentence to parse.
            Or list containing one string per sentence.

        Returns:
            pd.DataFrame: pandas DataFrame in the format according to `self.df_format`.

        """
        res = self._parser.predict(string, lang="en", prob=False, verbose=True)
        return [
            clean_tree(sent.values[2]) if self._clean else sent.values[2]  # type: ignore
            for sent in res  # type: ignore
        ]


if __name__ == "__main__":
    parser = ConstituencyParser(path="con-crf-roberta-en", clean=False)
    sent = "All through August the rain hardly stopped."
    res = parser(sent)
    res[0].pretty_print()
    clean_tree(res[0]).pretty_print()
    # for tree in res:
    #     tree.pretty_print()
    #     simplify_tree(tree).pretty_print()
