"""Data utils for each of the task files."""

import json
from collections.abc import Generator
from pathlib import Path

import pandas as pd

INDEX_COLS = ["sent_id", "word_id"]
DEP_REL_COLS = [*INDEX_COLS, "word", "deprel", "head"]
POS_TAG_COLS = [*INDEX_COLS, "word", "lemma", "ud_tag", "penn_tag"]


def _read_f(fp: str) -> str:
    with Path(fp).open() as f:
        return f.read().strip()


def load_task() -> dict[int, dict]:
    """Load task JSON.

    Returns:
        dict: Task loaded as dictionary

    """
    with Path("task/task.json").open() as f:
        j = json.load(f)
    return {int(k): v for k, v in j.items()}


def load_dep_rel() -> pd.DataFrame:
    """Load dependency relation tag file.

    Returns:
        pd.DataFrame: DataFrame containing dependency relations.
        Columns are defined in `DEP_REL_COLS` above.

    """
    data = _read_f("task/dep_rel.txt")

    def rows() -> Generator[dict]:
        for sent_id, sent in enumerate(data.split("\n\n"), start=1):
            for line in sent.split("\n"):
                word_id, *values = line.strip().split("\t")
                yield dict(
                    zip(
                        DEP_REL_COLS,
                        [sent_id, int(word_id), *values],
                    ),
                )

    return pd.DataFrame(rows()).set_index(INDEX_COLS)


def load_parses() -> list[str]:
    """Load parse tree file.

    Returns:
        list[str]: List of parse trees

    """
    return _read_f("task/parses.txt").split("\n\n")


def load_pos_tags() -> pd.DataFrame:
    """Load pos tag file.

    Returns:
        pd.DataFrame: DataFrame containing part-of-speech tags.
        Columns are defined in `POS_TAG_COLS` above.

    """
    data = _read_f("task/pos_tags.txt")

    def rows() -> Generator[dict]:
        for sent_id, sent in enumerate(data.split("\n\n"), start=1):
            for word_id, tagged_word in enumerate(sent.split("\t"), start=1):
                yield dict(
                    zip(
                        POS_TAG_COLS,
                        [sent_id, word_id, *tagged_word.strip().split("\\")],
                    ),
                )

    return pd.DataFrame(rows()).set_index(INDEX_COLS)


def load_sentences() -> list[str]:
    """Load sentences into a list.

    Returns:
        list[str]: list of sentences.

    """
    return _read_f("task/sentences.txt").split("\n\n")


if __name__ == "__main__":
    print(load_dep_rel())
