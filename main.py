"""Main."""

import warnings

import stanza
from stanza.pipeline.core import DownloadMethod

# Function call with FutureWarnings filtered
with warnings.catch_warnings():
    warnings.simplefilter(action="ignore", category=FutureWarning)

    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        device="mps",
        download_method=DownloadMethod.REUSE_RESOURCES,
    )
    doc = nlp("Let me test this sentence!.")
    print(
        *[
            f"id: {word.id}\tword: {word.text}"
            f"\thead id: {word.head}"
            f"\thead: {sent.words[word.head-1].text if word.head > 0 else 'root'}"
            f"\tdeprel: {word.deprel}"
            for sent in doc.sentences  # type: ignore
            for word in sent.words
        ],
        sep="\n",
    )
