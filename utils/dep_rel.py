"""Utils specifically for dependence relations."""

import networkx as nx
import pandas as pd

from utils.task_data import load_dep_rel


def create_graph(df: pd.DataFrame, sent_id: int):
    g = df.loc[1, 5]
    print(g)


if __name__ == "__main__":
    dep_rel_df = load_dep_rel()
    create_graph(dep_rel_df, 1)
