"""Defines CLI for evaluation and prediction."""

from __future__ import annotations

import click
import pyperclip
from stanza.pipeline.core import DownloadMethod

from src.task.eval import eval_const, eval_dep_rel
from src.task.predict import ConstituencyParser, DataFrameFormat, DependencyParser
from src.utils.constituency import remove_top, tree_to_latex, wipe_empty_tags
from src.utils.dep_rel import df_to_tikz_dependency
from src.utils.task_data import load_constituency_parses, load_dep_rel

BAD_FILTER_LABEL_MSG = (
    "--filter-label '{}' not found in either hypothesis or reference parse."
)
FILTER_LABEL_OPTION_NAME = "--filter-label"


class RangeType(click.ParamType):
    """Custom range type for visualisation CLI."""

    name = "index_range"

    def convert(
        self,
        value: str,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[int, ...]:
        """Convert RangeType to int tuple."""
        parts = value.replace(" ", "").split(",")
        rng = set()
        for p in parts:
            if p == "":
                continue
            if "-" in p:
                lo, hi = p.split("-")
                try:
                    lo = int(lo)
                    hi = int(hi)
                    rng.update(set(range(lo, hi + 1)))
                except ValueError:
                    self.fail(f"{lo!r} or {hi!r} is not a valid integer", param, ctx)
            else:
                try:
                    num = int(p)
                    rng.add(num)
                except ValueError:
                    self.fail(f"{p!r} is not a valid integer", param, ctx)

        return tuple(sorted(rng))


@click.group()
def cli() -> None:
    """CLI for L95 final task. Runs evaluation, prediction and visualisation."""


@cli.group(context_settings={"show_default": True})
def evaluate() -> None:
    """Run task file evaluation. Runs the relevant prediction and evaluation code."""


@evaluate.command(name="dep-rel")
@click.option(
    "--sentences-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the sentences file for evaluation. Leave unspecified for default.",
)
@click.option(
    "--gold-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the gold dependency relations file. Leave unspecified for default.",
)
@click.option(
    "--save-predictions",
    default=None,
    type=click.Path(),
    help="Path to save the prediction results. Leave unspecified to not save.",
)
@click.option(
    "--filter-label",
    default=None,
    type=str,
    help="Evaluate for a specific label",
)
@click.option(
    "--pretagged/--untagged",
    default=False,
    help="Flag if the input file is pretagged.",
)
def cli_eval_dep_rel(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
    filter_label: None | str = None,
    *,
    pretagged: bool,
) -> None:
    """Run dependency relation evaluation."""
    try:
        res = eval_dep_rel(
            sentences_file,
            gold_file,
            save_predictions,
            filter_label,
            pretagged=pretagged,
        )
    except ValueError as v:
        raise click.BadOptionUsage(
            FILTER_LABEL_OPTION_NAME,
            BAD_FILTER_LABEL_MSG.format(filter_label),
        ) from v

    res.pretty_print()


@evaluate.command(name="constituency")
@click.option(
    "--sentences-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the sentences file for evaluation. Leave unspecified for default.",
)
@click.option(
    "--gold-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the gold constituency relations file. Leave unspecified for default.",
)
@click.option(
    "--save-predictions",
    default=None,
    type=click.Path(),
    help="Path to save the prediction results. Leave unspecified to not save.",
)
@click.option(
    "--parser",
    "parser_name",
    default="con-crf-roberta-en",
    help="Which pre-trained parser to use",
    type=click.Choice(["con-crf-en", "con-crf-roberta-en"]),
)
@click.option(
    "--filter-label",
    default=None,
    type=str,
    help="Evaluate for a specific label",
)
def cli_eval_constituencies(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
    filter_label: None | str = None,
    *,
    parser_name: str,
) -> None:
    """Run constituency parsing evaluation."""
    try:
        res = eval_const(
            sentences_file,
            gold_file,
            save_predictions,
            filter_label,
            parser_name=parser_name,
        )
    except ValueError as v:
        raise click.BadOptionUsage(
            FILTER_LABEL_OPTION_NAME,
            BAD_FILTER_LABEL_MSG.format(filter_label),
        ) from v

    res.pretty_print()


@cli.group(context_settings={"show_default": True})
def predict() -> None:
    """Predict dependency relation or constituency parsing on input."""


@predict.command(name="dep-rel")
@click.option(
    "--df-format",
    type=click.Choice(
        [df_format.value for df_format in DataFrameFormat],
        case_sensitive=False,
    ),
    default=DataFrameFormat.DEPREL.value,
    help="Format of the output DataFrame as defined in 'DataFrameFormat'.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    show_default=False,
    help="Device to run the pipeline on. "
    "Defaults to 'auto' which automatically picks a device.",
)
@click.option(
    "--download-method",
    type=click.Choice([method.name for method in DownloadMethod], case_sensitive=False),
    default=DownloadMethod.REUSE_RESOURCES.name,
    help="Method to use for downloading Stanza models.",
)
@click.argument("text")
def pred_dep_rel(
    df_format: str,
    device: str,
    download_method: str,
    text: str,
) -> None:
    """Predict dependency relation parse on given text.

    Text is the input string to parse.
    """
    parser = DependencyParser(
        df_format=DataFrameFormat(df_format),
        device=device,
        download_method=DownloadMethod[download_method],
    )
    result = parser(text)
    click.echo(result)


@predict.command(name="constituency")
@click.option(
    "--pretty-print/--no-pretty-print",
    default=True,
    help="Pretty print the result.",
)
@click.option(
    "--parser",
    "parser_name",
    default="con-crf-roberta-en",
    help="Which pre-trained parser to use.",
    type=click.Choice(["con-crf-en", "con-crf-roberta-en"]),
)
@click.option(
    "--remove-top/--keep-top",
    "should_remove_top",
    default=True,
    help="Whether to remove the TOP node from the parse.",
)
@click.option(
    "--wipe-empty-tags/--keep-empty-tags",
    "should_wipe_empty",
    default=True,
    help="Whether to wipe empty tags from the parse.",
)
@click.option(
    "--latex",
    is_flag=True,
    help="Print forest LaTeX code to draw the parse. "
    "NOTE: needs to be enclosed in \\begin{forest} \\end{forest} tags.",
)
@click.argument("text")
def pred_const_parse(
    text: str,
    *,
    pretty_print: bool = True,
    parser_name: str,
    should_remove_top: bool,
    should_wipe_empty: bool,
    latex: bool,
) -> None:
    """Predict constituency parse on input text.

    Text is the input string to parse.
    """
    parser = ConstituencyParser(path=parser_name, clean=False)
    result = parser(text)

    for r in result:
        cleaned_r = remove_top(r) if should_remove_top else r
        cleaned_r = wipe_empty_tags(cleaned_r) if should_wipe_empty else cleaned_r
        click.echo(cleaned_r)
        if pretty_print:
            cleaned_r.pretty_print()
        if latex:
            click.echo(tree_to_latex(cleaned_r))


@cli.group(context_settings={"show_default": True})
def visualise() -> None:
    """Visualise constituency or dependency relation parses."""


@visualise.command()
@click.argument(
    "file",
    type=click.Path(exists=True),
)
@click.option(
    "--indices",
    default="1-10",
    type=RangeType(),
    help="Indices of sentences to visualise. Accepts ranges e.g. 1,3-5,6",
)
@click.option(
    "--copy/--no-copy",
    default=False,
    help="Copy (or don't) LaTeX to clipboard.",
)
def constituency(file: str, indices: tuple[int], *, copy: bool) -> None:
    """Visualise constituency parses from a file, and copy latex reprs to clipboard."""
    parses = load_constituency_parses(file)
    latexs = []
    for i in indices:
        click.echo(f"Sentence {i}")
        parses[i - 1].pretty_print()
        click.echo(ltx := tree_to_latex(parses[i]))
        latexs.append(f"% Sentence {i}\n{ltx}")
        click.echo("=" * 50)

    if copy:
        pyperclip.copy("\n\n".join(latexs))
        click.echo("\n\n📋 Successfully copied LaTeX trees to clipboard!")


@visualise.command("dep-rel")
@click.argument(
    "file",
    type=click.Path(exists=True),
)
@click.option(
    "--indices",
    default="1-10",
    type=RangeType(),
    help="Indices of sentences to visualise. Accepts ranges e.g. 1,3-5,6",
)
@click.option(
    "--copy/--no-copy",
    default=False,
    help="Copy (or don't) LaTeX to clipboard.",
)
def dep_rel(file: str, indices: tuple[int], *, copy: bool) -> None:
    """Visualise dependency relations from a file, and copy latex reprs to clipboard."""
    parses = load_dep_rel(file)
    latexs = []
    for i in indices:
        click.echo(f"Sentence {i}")
        click.echo(f"{parses.loc[i]}\n")
        click.echo(ltx := df_to_tikz_dependency(parses, i))
        latexs.append(f"% Sentence {i}\n{ltx}")
        click.echo("=" * 50)

    if copy:
        pyperclip.copy("\n\n".join(latexs))
        click.echo("\n\n📋 Successfully copied LaTeX trees to clipboard!")


if __name__ == "__main__":
    cli()
