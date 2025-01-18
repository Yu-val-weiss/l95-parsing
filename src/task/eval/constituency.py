"""Evaluation functions and classes for constituency parsing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from prettytable import PrettyTable

from src.task.predict import ConstituencyParser
from src.utils.task_data import (
    dump_constituency_parses,
    load_constituency_parses,
    load_sentences,
)

from .score import EvalScore

if TYPE_CHECKING:
    from nltk.tree import Tree

logger = logging.getLogger()


class ConstituencyScore(NamedTuple):
    """NamedTuple for Dependency Relation scores.

    Use Parseval metrics.

    Labelled (Labelled Parseval score)
    Unlabelled (Unlabelled Parseval score)
    Cross-brackets
    """

    labelled: EvalScore
    unlabelled: EvalScore
    cross_brackets: int

    def pretty_print(self) -> None:
        """Print table representation of DependencyRelationScore."""
        table = PrettyTable()
        table.field_names = ["Metric", "Precision", "Recall", "F1"]

        cross_table = PrettyTable()
        cross_table.add_row([f"Cross brackets: {self.cross_brackets}"])
        cross_table.min_table_width = 42
        cross_table.header = False

        for metric_name, score in self._asdict().items():
            if metric_name == "cross_brackets":
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
        print(cross_table)


def eval_const(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
    parser_name: str = "con-crf-roberta-en",
) -> ConstituencyScore:
    """Evaluate the Stanza's constituency parsing.

    Args:
        sentences_file (None | str, optional): The sentences file to predict on.
        Defaults to the one is task_files.
        gold_file (None | str, optional): The gold constituency parse file to use.
        Defaults to the one in task_files.
        save_predictions (None | str, optional): Where to save predictions.
        Will not save if set to None.
        parser_name (str, optional): The parser to load. Options are: "con-crf-en",
        "con-crf-zh", "con-crf-roberta-en", "con-crf-electra-zh", "con-crf-xlmr".
        Defaults to "con-crf-en".

    """
    sentences = (
        load_sentences() if sentences_file is None else load_sentences(sentences_file)
    )
    gold = (
        load_constituency_parses()
        if gold_file is None
        else load_constituency_parses(gold_file)
    )

    # need to convert sentences to a string for stanza
    parser = ConstituencyParser(parser_name)

    logger.info("Parsing sentences...")
    result = parser(sentences)
    logger.info("...sentences parsed!")

    if save_predictions is not None:
        dump_constituency_parses(result, save_predictions)

    logger.info("Evaluating...")

    pred_labelled, pred_constituents = extract_all_constituents(result, labelled=True)
    pred_unlabelled, _ = extract_all_constituents(result, labelled=False)

    gold_labelled, gold_constituents = extract_all_constituents(gold, labelled=True)
    gold_unlabelled, _ = extract_all_constituents(gold, labelled=False)

    cc = sum(cross_count(g, p) for g, p in zip(gold_constituents, pred_constituents))

    res = ConstituencyScore(
        EvalScore.from_sets(pred_labelled, gold_labelled),
        EvalScore.from_sets(pred_unlabelled, gold_unlabelled),
        cc,
    )

    logger.info("...evaluation complete!")

    return res


def extract_constituents(
    tree: Tree,
    start_index: int = 0,
    *,
    labelled: bool = True,
) -> set[tuple]:
    """Recursively extract constituent spans from an NLTK Tree.

    Args:
        tree (Tree): NLTK Tree object
        start_index (int): Starting index for this subtree
        labelled (bool, optional): whether to return labelled or unlabelled constituents

    Returns:
        A set of tuples representing constituents as (start, end, label) if labelled
        and (start, end) otherwise.

    """
    constituents = set()

    if tree.height() > 1:
        end_index = start_index + len(tree.leaves())  # get span of current tree
        span = (
            (start_index, end_index, tree.label())
            if labelled
            else (start_index, end_index)
        )
        constituents.add(span)

        for subtree in tree:
            if isinstance(subtree, str):
                continue
            constituents.update(
                extract_constituents(subtree, start_index, labelled=labelled),
            )
            start_index += len(subtree.leaves())

    return constituents


def extract_all_constituents(
    trees: list[Tree],
    *,
    labelled: bool = True,
) -> tuple[set[tuple], list[set[tuple]]]:
    """Extract contituencies from each tree in trees.

    Args:
        trees (list[Tree]): Trees to extract from.
        labelled (bool, optional): Use labelled or unlabelled constituents.
        Defaults to True.

    Returns:
        tuple[set[tuple], list[set[tuple]]: First item of tuple is a list of either
        (sent_id, start_index, end_index, label)
        or (sent_id, start_index, end_index).
        Second item is a list of sets, each of which contains each tree's constituents.

    """
    big_set = set()
    all_consts = []

    for sent_id, tree in enumerate(trees, start=1):
        constituents = extract_constituents(tree, labelled=labelled)
        big_set.update({(sent_id, *tup) for tup in constituents})
        all_consts.append(constituents)

    return big_set, all_consts


def cross_count(gold_spans: set[tuple], pred_spans: set[tuple]) -> int:
    """Compute cross count. Can handle labelled or unlabelled spans.

    Args:
        gold_spans (set[tuple]): spans from gold tree.
        pred_spans (set[tuple]): spans from pred tree.

    Returns:
        int: cross count.

    """
    unmatched_spans = pred_spans - (gold_spans & pred_spans)

    crosses = 0
    for u_start, u_end, *_ in unmatched_spans:
        for g_start, g_end, *_ in gold_spans:  # noqa: PLW2901 Ruff
            if (u_start < g_start and u_end > g_start and u_end < g_end) or (
                u_start > g_start and u_start < g_end and u_end > g_end
            ):
                crosses += 1
                break
    return crosses


if __name__ == "__main__":
    res = eval_const(save_predictions="prediction_files/constituencies.txt")
    res.pretty_print()
