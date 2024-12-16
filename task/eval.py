"""Evaluation functions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from prettytable import PrettyTable

from task.predict import DataFrameFormat, DependencyParser
from utils.task_data import dump_dep_rel, load_dep_rel, load_sentences

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger("stanza")


class EvalScore(NamedTuple):
    """Tuple for evaluation scores."""

    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        """Create string representation.

        Returns:
            str: string depiction

        """
        return f"(P = {self.precision:.2f}, R = {self.recall:.2f}, F = {self.f1:.2f})"

    @classmethod
    def from_sets(cls, pred_set: set, gold_set: set) -> EvalScore:
        """Create an evaluation score from predicted and gold sets.

        Args:
            pred_set (set): Set containing predictions.
            gold_set (set): Set containing ground truth values.

        Returns:
            EvalScore: Returns corresponding EvalScore.

        """
        correct = len(pred_set & gold_set)
        pred = len(pred_set)
        gold = len(pred_set)

        p = correct / pred
        r = correct / gold
        f = (2 * p * r) / (p + r)

        return cls(p, r, f)


class DependencyRelationScore(NamedTuple):
    """Tuple for Dependency Relation scores.

    LAS (labelled attachment score)

    UAS (unlabelled attachment score)

    LS (label accuracy score)
    """

    LAS: EvalScore
    UAS: EvalScore
    LS: EvalScore

    def pretty_string(self) -> str:
        """Create table representation of DependencyRelationScore.

        Returns:
            str: table.

        """
        table = PrettyTable()
        table.field_names = ["Metric", "Precision", "Recall", "F1"]

        for metric_name, score in self._asdict().items():
            table.add_row(
                [
                    metric_name,
                    f"{score.precision:.2f}",
                    f"{score.recall:.2f}",
                    f"{score.f1:.2f}",
                ],
            )

        return table.get_string()


def eval_dep_rel(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
) -> DependencyRelationScore:
    """Evaluate the Stanza's dependency relation parsing.

    Args:
        sentences_file (None | str, optional): The sentences file to predict on.
        Defaults to the one is task_files.
        gold_file (None | str, optional): The gold dependency file relation to use.
        Defaults to the one in task_files.
        save_predictions (None | str, optional): Where to save predictions.
        Will not save if set None.

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

    pred_labelled_heads = df_to_labelled_heads(result)
    pred_unlabelled_heads = df_to_unlabelled_heads(result)
    pred_labels = df_to_labels(result)

    gold_labelled_heads = df_to_labelled_heads(gold)
    gold_unlabelled_heads = df_to_unlabelled_heads(gold)
    gold_labels = df_to_labels(gold)

    res = DependencyRelationScore(
        EvalScore.from_sets(pred_labelled_heads, gold_labelled_heads),
        EvalScore.from_sets(pred_unlabelled_heads, gold_unlabelled_heads),
        EvalScore.from_sets(pred_labels, gold_labels),
    )

    logger.info("...evaluation complete!")

    return res


def df_to_labelled_heads(df: pd.DataFrame) -> set[tuple[int, int, str, int]]:
    """Generate a set containing each dependency relation, labelled.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame

    Returns:
        set[tuple[int, int, str, int]]: A set of (sent_id, word_id, deprel, head) tuples

    """
    return {(*row.Index, row.deprel.lower(), row.head) for row in df.itertuples()}  # type: ignore


def df_to_unlabelled_heads(df: pd.DataFrame) -> set[tuple[int, int, int]]:
    """Generate a set containing each dependency relation, unlabelled.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame

    Returns:
        set[tuple[int, int, int]]: A set of (sent_id, word_id, head) tuples

    """
    return {(*row.Index, row.head) for row in df.itertuples()}  # type: ignore


def df_to_labels(df: pd.DataFrame) -> set[tuple[int, int, str]]:
    """Generate a set containing each token with its label.

    Args:
        df (pd.DataFrame): A `DEPREL_COLS` format pandas DataFrame

    Returns:
        set[tuple[int, int, str]]: A set of (sent_id, word_id, deprel) tuples

    """
    return {(*row.Index, row.deprel.lower()) for row in df.itertuples()}  # type: ignore


if __name__ == "__main__":
    eval_dep_rel()
