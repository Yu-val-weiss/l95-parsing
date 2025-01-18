"""Data utils for each of the task files."""

import json
from pathlib import Path

import pandas as pd
from nltk.tree import Tree

from src.utils import DEP_REL_COLS, INDEX_COLS, POS_TAG_COLS


def _read_f(fp: str) -> str:
    with Path(fp).open() as f:
        return f.read().strip()


def _dump(file: str, s: str):
    file_path = Path(file)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    with file_path.open("w") as f:
        f.write(s)


def load_task(file: str = "task_files/task.json") -> dict[int, dict]:
    """Load task JSON.

    Returns:
        dict: Task loaded as dictionary

    """
    with Path(file).open() as f:
        j = json.load(f)
    return {int(k): v for k, v in j.items()}


def load_constituency_parses(file: str = "task_files/constituencies.txt") -> list[Tree]:
    """Load parse tree file.

    Returns:
        list[str]: List of parse trees

    """
    return [Tree.fromstring(tree) for tree in _read_f(file).split("\n\n")]


def dump_constituency_parses(parses: list[Tree], file: str) -> None:
    """Dump constituency parses to file.

    Args:
        parses (list[Tree]): Constituency parses to dump.
        file (str): Where to dump.

    """
    s = "\n\n".join(x.pformat() for x in parses)
    _dump(file, s)


def load_dep_rel(file: str = "task_files/dep_rel_fixed.txt") -> pd.DataFrame:
    """Load dependency relation tag file.

    Returns:
        pd.DataFrame: DataFrame containing dependency relations.
        Columns are defined in `DEP_REL_COLS` above.
        Includes a `MultiIndex` made up of `INDEX_COLS`.

    """
    data = _read_f(file)

    def rows():
        for sent_id, sent in enumerate(data.split("\n\n"), start=1):
            for line in sent.split("\n"):
                word_id, word, deprel, head = line.strip().split("\t")
                yield [sent_id, int(word_id), word, deprel, int(head)]

    return pd.DataFrame(rows(), columns=INDEX_COLS + DEP_REL_COLS).set_index(INDEX_COLS)


def dump_dep_rel(dep_rel_df: pd.DataFrame, file: str) -> None:
    """Dump dependency relation DataFrame to file.

    Args:
        dep_rel_df (pd.DataFrame): DataFrame to dump.
        file (str): Where to dump.

    """
    x = [
        df.reset_index(level="sent_id", drop=True).to_csv(
            header=False,
            index=True,
            sep="\t",
            quotechar="`",
        )
        for _, df in dep_rel_df.groupby(level="sent_id")
    ]
    s = "\n".join(x).strip() + "\n\n"

    _dump(file, s)


def load_pos_tags(file: str = "task_files/pos_tags.txt") -> pd.DataFrame:
    """Load pos tag file.

    Returns:
        pd.DataFrame: DataFrame containing part-of-speech tags.
        Columns are defined in `POS_TAG_COLS` above.
        Includes a `MultiIndex` made up of `INDEX_COLS`.

    """
    data = _read_f(file)

    def rows():
        for sent_id, sent in enumerate(data.split("\n\n"), start=1):
            for word_id, tagged_word in enumerate(sent.split("\t"), start=1):
                yield [sent_id, word_id, *tagged_word.strip().split("\\")]

    return pd.DataFrame(rows(), columns=INDEX_COLS + POS_TAG_COLS).set_index(INDEX_COLS)


def dump_pos_tags(pos_tag_df: pd.DataFrame, file: str) -> None:
    """Dump dependency relation DataFrame to file.

    Args:
        pos_tag_df (pd.DataFrame): DataFrame to dump.
        file (str): Where to dump.

    """
    x = [
        "\t".join(["\\".join(t) for t in df.itertuples(index=False)])
        for _, df in pos_tag_df.groupby(level="sent_id")
    ]
    s = "\n\n".join(x).strip() + "\n\n"

    _dump(file, s)


def load_sentences(file: str = "task_files/sentences.txt") -> list[str]:
    """Load sentences into a list.

    Returns:
        list[str]: list of sentences.

    """
    return _read_f(file).split("\n\n")


if __name__ == "__main__":
    f = load_constituency_parses()
    dump_constituency_parses(f, "task_files/pretty_constituencies.txt")
