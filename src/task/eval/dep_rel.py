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
                    f"{score:.4f}",
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
        table.title = f"{self.label} - dependency evaluation"
        table.field_names = ["Metric", "Precision", "Recall", "F1"]

        for metric_name, score in self._asdict().items():
            if metric_name == "label":
                continue
            table.add_row(
                [
                    metric_name,
                    f"{score.precision:.4f}",
                    f"{score.recall:.4f}",
                    f"{score.f1:.4f}",
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

    pred_labelled_heads = df_to_heads(result, labelled=True, filter_label=filter_label)
    pred_unlabelled_heads = df_to_heads(result, labelled=False)

    gold_labelled_heads = df_to_heads(gold, labelled=True, filter_label=filter_label)
    gold_unlabelled_heads = df_to_heads(gold, labelled=False)

    if not filter_label:
        pred_labels = df_to_labels(result)
        gold_labels = df_to_labels(gold)

        res = DependencyRelationScore(
            Accuracy.from_sets(pred_labelled_heads, gold_labelled_heads),
            Accuracy.from_sets(pred_unlabelled_heads, gold_unlabelled_heads),
            Accuracy.from_sets(pred_labels, gold_labels),
        )

    else:
        pred_labels = df_get_label(result, filter_label)
        gold_labels = df_get_label(gold, filter_label)
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


def _deprel_label_filter_pred(filter_label: str, deprel: str) -> bool:
    """Predicate for filtering by deprel label.

    Args:
        filter_label (str): Label to filter by
        deprel (str): The deprel to check

    Returns:
        bool: whether filter matches or not

    """
    filter_label = filter_label.lower()
    deprel = deprel.lower()
    if ":" in filter_label:
        return filter_label == deprel
    main, *_ = deprel.split(":")
    return filter_label == main


type LabelledDeps = tuple[int, int, str, int]
type UnlabelledDeps = tuple[int, int, int]


@overload
def df_to_heads(
    df: pd.DataFrame,
    *,
    labelled: Literal[True],
    filter_label: str | None = None,
) -> set[LabelledDeps]: ...


@overload
def df_to_heads(
    df: pd.DataFrame,
    *,
    labelled: Literal[False],
    filter_label: str | None = None,
) -> set[UnlabelledDeps]: ...


def df_to_heads(
    df: pd.DataFrame,
    *,
    labelled: bool = True,
    filter_label: str | None = None,
) -> set[LabelledDeps] | set[UnlabelledDeps]:
    """Generate a set containing each dependency relation.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame
        labelled (bool): Decides whether to return labelled or unlabelled tuples.
        filter_label (None | str, optional): Label to filter on.

    Returns:
        set[tuple]: A set of (sent_id, word_id, deprel, head) tuples if labelled.
        And a set of (sent_id, word_id, head) if unlabelled.


    """
    if not labelled and filter_label:
        msg = "Cannot set labelled to false and filter by label."
        raise ValueError(msg)

    itertuples = df.itertuples()
    if filter_label:
        itertuples = (
            row
            for row in itertuples
            if _deprel_label_filter_pred(filter_label, row.deprel)  # type: ignore
        )

    return {
        (*row.Index, row.deprel.lower(), row.head)  # type: ignore
        if labelled
        else (*row.Index, row.head)  # type: ignore
        for row in itertuples
    }


def df_to_labels(df: pd.DataFrame) -> set[tuple[int, int, str]]:
    """Generate a set containing each token with its label.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame

    Returns:
        set[tuple[int, int, str]]: A set of (sent_id, word_id, deprel) tuples

    """
    return {(*row.Index, row.deprel.lower()) for row in df.itertuples()}  # type: ignore


def df_get_label(df: pd.DataFrame, label: str) -> set[tuple[int, int, str]]:
    """Generate a set containing each token of the given label.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame
        label (str): The label to get

    Returns:
        set[tuple[int, int, str]]: A set of (sent_id, word_id, deprel) tuples

    """
    label = label.lower()
    return {
        (*row.Index, row.deprel.lower())  # type: ignore
        for row in df.itertuples()
        if _deprel_label_filter_pred(label, row.deprel)  # type: ignore
    }


if __name__ == "__main__":
    r = eval_dep_rel()
    r.pretty_print()
