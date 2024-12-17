"""Utils specifically for dealing with constituency parses."""

import pyperclip
from nltk.tree import Tree


def get_latex(tree: Tree, *, copy_to_clipboard: bool = False) -> str:
    """Convert a tree to a `qtree` latex tree.

    Args:
        tree (Tree): the tree to convert.
        copy_to_clipboard (bool, optional): Set to true to copy

    Returns:
        str: the LaTeX string defining the dependency graph.
        The string is optionally copied to the clipboard.

    """
    s = tree.pformat_latex_qtree()
    if copy_to_clipboard:
        pyperclip.copy(s)
        print("Copied LaTeX string to clipboard!")
    return s
