"""Contains Python functions relating to prediction."""

import warnings
from typing import cast

import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import DownloadMethod


def get_dep_parse_pipeline(
    *,
    device: str = "mps",
    download_method: DownloadMethod = DownloadMethod.REUSE_RESOURCES,
) -> stanza.Pipeline:
    """Get Stanza pipeline for dependency parse.

    Args:
        device (str, optional): Which device to use for the pipeline. Defaults to "mps".
        download_method (DownloadMethod, optional): Which download method to use.
        Defaults to DownloadMethod.REUSE_RESOURCES.

    Returns:
        stanza.Pipeline: The dependency parse pipeline.

    """
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        return stanza.Pipeline(
            lang="en",
            processors="tokenize,mwt,pos,lemma,depparse",
            device=device,
            download_method=download_method,
        )


def predict_dep_parse(pipeline: stanza.Pipeline, string: str) -> list[str]:
    """Predict a dependency parse for the string.

    Args:
        pipeline (stanza.Pipeline): A stanza pipeline.
        string (str): String containing sentence/s to parse.

    Returns:
        list[str]: _description_

    """
    doc = pipeline(string)
    doc = cast(Document, doc)  # cast to make type checking easier

    return [
        f"id: {word.id}\tword: {word.text}"
        f"\thead id: {word.head}"
        f"\thead: {sent.words[word.head-1].text if word.head > 0 else 'root'}"
        f"\tdeprel: {word.deprel}"
        for sent in doc.sentences
        for word in sent.words
    ]


if __name__ == "__main__":
    pipe = get_dep_parse_pipeline()
    print(
        predict_dep_parse(
            pipe,
            "As she walked past it, the driver's glass started to open.",
        ),
    )
