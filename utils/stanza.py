"""File containing utils for interacting with stanza."""

from typing import cast

import pandas as pd
from stanza.models.common.doc import Document, Sentence, Word

from utils import CONLLU_COLS, DEP_REL_COLS, INDEX_COLS


def doc_to_conllu_df(doc: Document) -> pd.DataFrame:
    """Convert Stanza Document to pandas DataFrame.

    Args:
        doc (Document): Document to convert.

    Returns:
        pd.DataFrame: Convert to pandas DataFame with columns `CONLLU_COLS`.

    """

    def rows():
        for sent in doc.sentences:
            sent = cast(Sentence, sent)
            for word in sent.words:
                word = cast(Word, word)
                yield (
                    word.id,
                    word.text,
                    word.lemma,
                    word.upos,
                    word.xpos,
                    word.feats,
                    word.head,
                    word.deprel,
                    word.deps,
                    word.misc,
                )

    return pd.DataFrame(rows(), columns=CONLLU_COLS)


def doc_to_deprel_df(doc: Document) -> pd.DataFrame:
    """Convert Stanza Document to pandas DataFrame.

    Args:
        doc (Document): Document to convert.

    Returns:
        pd.DataFrame: Convert to pandas DataFame with columns `DEPREL_COLS`.
        Indices are `INDEX_COLS`.

    """

    def rows():
        for sent_id, sent in enumerate(doc.sentences, start=1):
            sent = cast(Sentence, sent)
            for word in sent.words:
                word = cast(Word, word)
                yield (
                    sent_id,
                    word.id,
                    word.text,
                    word.deprel,
                    word.head,
                )

    return pd.DataFrame(rows(), columns=INDEX_COLS + DEP_REL_COLS).set_index(INDEX_COLS)
