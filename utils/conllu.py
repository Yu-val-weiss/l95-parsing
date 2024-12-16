"""CONLL-U related utils."""

from pathlib import Path

import pandas as pd

from utils import CONLLU_COLS, DEP_REL_COLS, INDEX_COLS
from utils.task_data import _read_f, load_dep_rel, load_pos_tags


def generate_conll(
    file: str = "task_files/task.conllu",
    dep_rel_file: str = "task_files/dep_rel.txt",
    pos_tag_file: str = "task_files/pos_tags.txt",
) -> pd.DataFrame:
    """Dump dependence relation DataFrame to file in CONLL-U format.

    Args:
        file (str): Where to dump.
        dep_rel_file (str): Dependency relation file to use to create CONLL-U format.
        pos_tag_file (str): POS tag file to use to create CONLL-U format.

    Returns:
        pd.DataFrame: CONLL-U formatted DataFrame.

    """
    dep_rel_df = load_dep_rel(dep_rel_file)
    pos_tag_df = load_pos_tags(pos_tag_file)
    # map columns to conllu
    conllu_data = {
        "sent_id": dep_rel_df.index.get_level_values("sent_id"),
        "id": dep_rel_df.index.get_level_values("word_id"),
        "form": dep_rel_df["word"],
        "lemma": pos_tag_df["lemma"],
        "upos": pos_tag_df["ud_tag"],
        "xpos": "_",
        "feats": "_",
        "head": dep_rel_df["head"],  # Preserve "head"
        "deprel": dep_rel_df["deprel"],  # Preserve "deprel"
        "deps": "_",
        "misc": "_",
    }
    conllu_df = pd.DataFrame(conllu_data)
    x = [
        df.reset_index(drop=True)
        .drop("sent_id", axis="columns")
        .to_csv(
            header=False,
            index=False,
            sep="\t",
            quotechar="`",
        )
        for _, df in conllu_df.groupby(level="sent_id")
    ]
    s = "\n".join(x).strip() + "\n\n"

    file_path = Path(file)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    with file_path.open("w") as f:
        f.write(s)

    return conllu_df


def load_conll(file: str = "task_files/task.conllu") -> pd.DataFrame:
    """Load CONLL-U file.

    Returns:
        pd.DataFrame: DataFrame containing CONLL-U data.
        Columns are defined in `CONLLU_COLS`.

    """
    data = _read_f(file)

    def rows():
        for sent in data.split("\n\n"):
            for line in sent.split("\n"):
                word_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = (
                    line.strip().split("\t")
                )
                yield [
                    int(word_id),
                    form,
                    lemma,
                    upos,
                    xpos,
                    feats,
                    int(head),
                    deprel,
                    deps,
                    misc,
                ]

    return pd.DataFrame(rows(), columns=CONLLU_COLS)


def convert_to_dep_rel(conllu_df: pd.DataFrame) -> pd.DataFrame:
    """Convert CONLL-U format DataFrame to DEP_REL format.

    Args:
        conllu_df (pd.DataFrame): DataFrame to convert.

    Returns:
        pd.DataFrame: Converted DataFrame.

    """
    sentence_id = (conllu_df["id"] == 1).cumsum()
    dep_rel_data = {
        "sent_id": sentence_id,
        "word_id": conllu_df["id"],
        "word": conllu_df["form"],
        "deprel": conllu_df["deprel"],
        "head": conllu_df["head"],
    }
    return pd.DataFrame(dep_rel_data, columns=INDEX_COLS + DEP_REL_COLS).set_index(
        INDEX_COLS,
    )


if __name__ == "__main__":
    # generate_conll()
    conll_df = load_conll()
    print(convert_to_dep_rel(conll_df))
