"""Contains Python functions relating to prediction."""

import warnings
from enum import Enum
from typing import cast

import pandas as pd
import stanza
from nltk.tree import Tree
from stanza.models.common.doc import Document
from stanza.pipeline.core import DownloadMethod
from supar import Parser

from utils.stanza import doc_to_conllu_df, doc_to_deprel_df

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

    def __init__(self, path: str = "crf-con-en", clean: bool = True) -> None:
        """Initialise the class.

        Args:
            path (str, optional): The path of the parser to load. Options are:
            "crf-con-en", "crf-con-zh", "crf-con-roberta-en", "crf-con-electra-zh",
            "crf-con-xlmr". Defaults to "crf-con-en".

            clean(bool, optional): Flag whether to clean the tree or not.

        """
        self._parser = Parser.load(path)
        self._clean = clean

    def __call__(self, string: str) -> list[Tree]:
        """Predict a constituency parse for the string.

        Args:
            string (str): String containing sentence/s to parse.

        Returns:
            pd.DataFrame: pandas DataFrame in the format according to `self.df_format`.

        """
        res = self._parser.predict(string, lang="en", prob=False, verbose=True)
        return [
            clean_tree(sent.values[2]) if self._clean else sent.values[2]
            for sent in res
        ]


def wipe_empty_tags(tree: Tree) -> Tree:
    """Combine leaf nodes that had an empty POS with the one above.

    Args:
        tree (Tree): Tree to cleanup.

    Returns:
        Tree: Cleaned up tree.

    """
    new_tree = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            if subtree.label() == "_":
                new_tree.append(subtree.leaves()[0])
            else:
                new_tree.append(wipe_empty_tags(subtree))
    return Tree(tree.label(), new_tree)


def clean_tree(tree: Tree) -> Tree:
    """Clean tree for processing.

    Args:
        tree (Tree): Uncleaned up tree from parser.

    Returns:
        Tree: Clean tree.

    """
    if tree.label() == "TOP":
        return wipe_empty_tags(tree[0])  # type: ignore
    return wipe_empty_tags(tree)


if __name__ == "__main__":
    parser = ConstituencyParser()
    res = parser("As she walked past it, the driver's glass started to open.")
    res[0].pprint()
    # for tree in res:
    #     tree.pretty_print()
    #     simplify_tree(tree).pretty_print()
