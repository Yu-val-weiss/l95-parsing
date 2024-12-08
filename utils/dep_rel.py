"""Utils specifically for dependence relations."""

import networkx as nx
import pandas as pd
import pyperclip
from networkx.drawing.nx_latex import to_latex

from utils.task_data import load_dep_rel


def create_graph(df: pd.DataFrame, sent_id: int) -> nx.DiGraph:
    """Create a directed graph representing the dependency relations in a sentence.

    Args:
        df (pd.DataFrame): Dataframe containing dependency relations.
        sent_id (int): The sentence to make into a graph.

    Returns:
        nx.DiGraph: Graph representing the dependency relations.

    """
    sent = df.loc[sent_id]

    nodes = [(0, {"word": "ROOT"})]
    edges = []

    for t in sent.itertuples():  # note itertuples calls word_id by Index
        nodes.append((t.Index, {"word": t.word}))  # type: ignore
        edges.append((t.head, t.Index, {"deprel": t.deprel}))

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    return g


def graph_to_latex(g: nx.DiGraph) -> str:
    """Convert graph to LaTeX TikZ. Uses networkx's `to_latex` function.

    Args:
        g (nx.DiGraph): Graph representation a dependency relation.

    Returns:
        str: LaTeX string.

    """
    return to_latex(g, as_document=False)


def df_to_tikz_dependency(df: pd.DataFrame, sent_id: int) -> str:
    """Convert a sentence in a dataframe to a tikz-dependency graph.

    Args:
        df (pd.DataFrame): _description_
        sent_id (int): _description_

    Returns:
        str: the LaTeX string defining the dependence graph.
        The string is copied to the clipboard.

    """
    sent = df.loc[sent_id]

    latex_str = "\\begin{dependency} % see here for documentation\n"
    latex_str += "% https://gb.mirrors.cicku.me/ctan/graphics/pgf/contrib/tikz-dependency/tikz-dependency-doc.pdf\n"
    latex_str += "\t\\begin{deptext}[column sep=1em]\n"

    deptext = []
    depedges = []
    deproot = ""

    for t in sent.itertuples():  # note itertuples calls word_id by Index
        deptext.append(t.word)
        if t.head == 0:
            deproot = f"\t\\deproot{{{t.Index}}}{{root}}"
        else:
            depedges.append(f"\t\\depedge{{{t.head}}}{{{t.Index}}}{{{t.deprel}}}")

    latex_str += f"\t\t{' \\& '.join(deptext)} \\\\"
    latex_str += "\t\\end{deptext}\n"

    latex_str += "\n".join(depedges) + "\n"

    latex_str += deproot

    latex_str += "\n\\end{dependency}"

    pyperclip.copy(latex_str)
    print("Copied LaTeX string to clipboard!")

    return latex_str


if __name__ == "__main__":
    dep_rel_df = load_dep_rel()
    # print(dep_rel_df.loc[1])
    # g = create_graph(dep_rel_df, 1)
    print(df_to_tikz_dependency(dep_rel_df, 1))
