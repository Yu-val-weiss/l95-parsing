"""Utils specifically for dependence relations."""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
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


def graph_to_latex(G: nx.DiGraph) -> str:
    """Convert graph to LaTeX TikZ.

    Args:
        G (nx.DiGraph): Graph representation a dependency relation.

    Returns:
        str: LaTeX string.

    """
    return to_latex(G, as_document=False)


if __name__ == "__main__":
    dep_rel_df = load_dep_rel()
    g = create_graph(dep_rel_df, 1)
    print(graph_to_latex(g))
