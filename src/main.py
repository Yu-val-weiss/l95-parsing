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


class RangeType(click.ParamType):
    """Custom range type for visualisation CLI."""

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


@cli.group()
def evaluate() -> None:
    """Run task file evaluation. Runs the relevant prediction and evaluation code."""


@evaluate.command(name="dep_rel")
@click.option(
    "--sentences-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the sentences file for evaluation.",
)
@click.option(
    "--gold-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the gold dependency relations file.",
)
@click.option(
    "--save-predictions",
    default=None,
    type=click.Path(),
    help="Path to save the prediction results.",
)
def cli_eval_dep_rel(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
) -> None:
    """Run dependency relation evaluation."""
    res = eval_dep_rel(sentences_file, gold_file, save_predictions)
    res.pretty_print()


@evaluate.command(name="constituencies")
@click.option(
    "--sentences-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the sentences file for evaluation.",
)
@click.option(
    "--gold-file",
    default=None,
    type=click.Path(exists=True),
    help="Path to the gold constituency relations file.",
)
@click.option(
    "--save-predictions",
    default=None,
    type=click.Path(),
    help="Path to save the prediction results.",
)
def cli_eval_constituencies(
    sentences_file: None | str = None,
    gold_file: None | str = None,
    save_predictions: None | str = None,
) -> None:
    """Run constituency parsing evaluation."""
    res = eval_const(sentences_file, gold_file, save_predictions)
    res.pretty_print()


@cli.group()
def predict() -> None:
    """Predict dependency relation or constituency parsing on input."""


@predict.command(name="dep_rel")
@click.option(
    "--df-format",
    type=click.Choice(
        [df_format.value for df_format in DataFrameFormat],
        case_sensitive=False,
    ),
    default=DataFrameFormat.DEPREL.value,
    help="Format of the output DataFrame as defined in `DataFrameFormat`.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to run the pipeline on. Default is 'auto'.",
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
@click.option("--pretty-print/--no-pretty-print", default=True)
@click.argument("text")
def pred_const_parse(
    text: str,
    *,
    pretty_print: bool = True,
) -> None:
    """Predict constituency parse on input text.

    Text is the input string to parse.
    """
    parser = ConstituencyParser(clean=False)
    result = parser(text)

    for r in result:
        cleaned_r = remove_top(r)
        click.echo(wipe_empty_tags(cleaned_r))
        if pretty_print:
            cleaned_r.pretty_print()


@cli.group()
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
    help="Indices of sentences to visualise.",
)
def constituency(file: str, indices: tuple[int]) -> None:
    """Visualise constituency parses from a given file, and copy them to clipboard."""
    parses = load_constituency_parses(file)
    latexs = []
    for i in indices:
        click.echo(f"Sentence {i}")
        parses[i - 1].pretty_print()
        click.echo(ltx := tree_to_latex(parses[i]))
        latexs.append(f"% Sentence {i}\n{ltx}")
        click.echo("=" * 50)

    pyperclip.copy("\n\n".join(latexs))
    click.echo("\n\n📋 Successfully copied LaTeX trees to clipboard!")


@visualise.command("dep_rel")
@click.argument(
    "file",
    type=click.Path(exists=True),
)
@click.option(
    "--indices",
    default="1-10",
    type=RangeType(),
    help="Indices of sentences to visualise.",
)
def dep_rel(file: str, indices: tuple[int]) -> None:
    """Visualise dependency relations from a given file, and copy them to clipboard."""
    parses = load_dep_rel(file)
    latexs = []
    for i in indices:
        click.echo(f"Sentence {i}")
        click.echo(f"{parses.loc[i]}\n")
        click.echo(ltx := df_to_tikz_dependency(parses, i))
        latexs.append(f"% Sentence {i}\n{ltx}")
        click.echo("=" * 50)

    pyperclip.copy("\n\n".join(latexs))
    click.echo("\n\n📋 Successfully copied LaTeX trees to clipboard!")


if __name__ == "__main__":
    cli()
