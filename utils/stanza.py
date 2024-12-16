"""File containing utils for interacting with stanza."""

from typing import cast

import pandas as pd
from stanza.models.common.doc import Document, Sentence, Word

from utils import CONLLU_COLS


def doc_to_conllu_df(doc: Document) -> pd.DataFrame:
    """Convert Stanza Document to pandas DataFrame.

    Args:
        doc (Document): Document to convert.

    Returns:
        pd.DataFrame: Convert to pandas DataFame with columns `CONLLU_COLS`.

    """

    def generate():
        for sent in doc.sentences:
            sent = cast(Sentence, sent)
            for word in sent.words:
                word = cast(Word, word)
                pass

    return pd.DataFrame(generate(), columns=CONLLU_COLS)
