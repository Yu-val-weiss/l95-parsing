"""Evaluation functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

from prettytable import PrettyTable

from src.task.predict import DataFrameFormat, DependencyParser
from src.utils.task_data import dump_dep_rel, load_dep_rel, load_sentences

from .score import Accuracy, EvalScore

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger("stanza")
logger.propagate = False


class DependencyRelationScore(NamedTuple):
    """NamedTuple for Dependency Relation scores.

    LAS (labelled attachment score)

    UAS (unlabelled attachment score)

    LS (label accuracy score)
    """

    LAS: Accuracy
    UAS: Accuracy
    LS: Accuracy

    def pretty_print(self) -> None:
        """Print table representation of DependencyRelationScore."""
        table = PrettyTable()
        table.title = "Dependency parse score"
        table.field_names = ["Metric", "Accuracy"]

        for metric_name, score in self._asdict().items():
            table.add_row(
                [
                    metric_name,
                    f"{score:.2f}",
                ],
            )

        print(table)


class SpecificDepRelScore(NamedTuple):
    """NamedTuple for Dependency Relation scores.

    label (the specific label being analysed)
    LAS (labelled attachment score)
    LS (label accuracy score)
    """

    label: str
    LAS: EvalScore
    LS: EvalScore

    def pretty_print(self) -> None:
        """Print table representation of SpecifcDepRel."""
        table = PrettyTable()
        table.title = f"{self.label} - specific evaluation"
        table.field_names = ["Metric", "Precision", "Recall", "F1"]

        for metric_name, score in self._asdict().items():
            if metric_name == "label":
                continue
            table.add_row(
                [
                    metric_name,
                    f"{score.precision:.2f}",
                    f"{score.recall:.2f}",
                    f"{score.f1:.2f}",
                ],
            )

        print(table)


def eval_dep_rel(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
    filter_label: None | str = None,
) -> DependencyRelationScore | SpecificDepRelScore:
    """Evaluate the Stanza's dependency relation parsing.

    Args:
        sentences_file (None | str, optional): The sentences file to predict on.
        Defaults to the one is task_files.
        gold_file (None | str, optional): The gold dependency file relation to use.
        Defaults to the one in task_files.
        save_predictions (None | str, optional): Where to save predictions.
        Will not save if set to None.
        filter_label (None | str, optional): Label to filter on.

    """
    sentences = (
        load_sentences() if sentences_file is None else load_sentences(sentences_file)
    )
    gold = load_dep_rel() if gold_file is None else load_dep_rel(gold_file)

    # need to convert sentences to a string for stanza
    sentences = " ".join(sentences)
    parser = DependencyParser(df_format=DataFrameFormat.DEPREL)

    logger.info("Parsing sentences...")
    result = parser(sentences)
    logger.info("...sentences parsed!")

    if save_predictions is not None:
        dump_dep_rel(result, save_predictions)

    logger.info("Evaluating...")

    pred_labelled_heads = df_to_heads(result, labelled=True)
    pred_unlabelled_heads = df_to_heads(result, labelled=False)
    pred_labels = df_to_labels(result)

    gold_labelled_heads = df_to_heads(gold, labelled=True)
    gold_unlabelled_heads = df_to_heads(gold, labelled=False)
    gold_labels = df_to_labels(gold)

    if not filter_label:
        res = DependencyRelationScore(
            Accuracy.from_sets(pred_labelled_heads, gold_labelled_heads),
            Accuracy.from_sets(pred_unlabelled_heads, gold_unlabelled_heads),
            Accuracy.from_sets(pred_labels, gold_labels),
        )

    else:
        filter_label = filter_label.lower()

        pred_labelled_heads = {
            x for x in pred_labelled_heads if x[2].lower() == filter_label
        }
        pred_labels = {x for x in pred_labels if x[2].lower() == filter_label}
        gold_labelled_heads = {
            x for x in gold_labelled_heads if x[2].lower() == filter_label
        }
        gold_labels = {x for x in gold_labels if x[2].lower() == filter_label}

        if len(pred_labelled_heads) == 0 and len(gold_labelled_heads) == 0:
            msg = "Label not found in either predicted or gold set."
            raise ValueError(msg)

        res = SpecificDepRelScore(
            filter_label,
            EvalScore.from_sets(pred_labelled_heads, gold_labelled_heads),
            EvalScore.from_sets(pred_labels, gold_labels),
        )

    logger.info("...evaluation complete!")

    return res


type LabelledDeps = tuple[int, int, str, int]
type UnlabelledDeps = tuple[int, int, int]


@overload
def df_to_heads(df: pd.DataFrame, *, labelled: Literal[True]) -> set[LabelledDeps]: ...


@overload
def df_to_heads(
    df: pd.DataFrame,
    *,
    labelled: Literal[False],
) -> set[UnlabelledDeps]: ...


def df_to_heads(
    df: pd.DataFrame,
    *,
    labelled: bool = True,
) -> set[LabelledDeps] | set[UnlabelledDeps]:
    """Generate a set containing each dependency relation.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame
        labelled (bool): Decides whether to return labelled or unlabelled tuples.

    Returns:
        set[tuple]: A set of (sent_id, word_id, deprel, head) tuples if labelled.
        And a set of (sent_id, word_id, head) if unlabelled.


    """
    return {
        (*row.Index, row.deprel.lower(), row.head)  # type: ignore
        if labelled
        else (*row.Index, row.head)  # type: ignore
        for row in df.itertuples()
    }


def df_to_labels(df: pd.DataFrame) -> set[tuple[int, int, str]]:
    """Generate a set containing each token with its label.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame

    Returns:
        set[tuple[int, int, str]]: A set of (sent_id, word_id, deprel) tuples

    """
    return {(*row.Index, row.deprel.lower()) for row in df.itertuples()}  # type: ignore


if __name__ == "__main__":
    r = eval_dep_rel()
    r.pretty_print()
