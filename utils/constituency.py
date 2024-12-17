"""Utils specifically for dealing with constituency parses."""

import re

import pyperclip
from nltk.tree import Tree


def wipe_empty_tags(tree: Tree) -> Tree:
    """Combine leaf nodes that had an empty POS with the one above.

    Args:
        tree (Tree): Tree to cleanup.

    Returns:
        Tree: Cleaned up tree.

    """
    new_tree = []
    for subtree in tree:
        if isinstance(subtree, Tree):
            if subtree.label() == "_":
                new_tree.append(subtree.leaves()[0])
            else:
                new_tree.append(wipe_empty_tags(subtree))
        else:
            new_tree.append(subtree)
    return Tree(tree.label(), new_tree)


def remove_top(tree: Tree) -> Tree:
    """Remove "top" node in Tree.

    Args:
        tree (Tree): Tree to be cleaned.

    Returns:
        Tree: TOP-less tree.

    """
    if isinstance(tree.label(), str) and tree.label().lower() == "top":
        return tree[0]  # type: ignore
    return tree


def clean_tree(tree: Tree) -> Tree:
    """Clean tree for processing.

    Args:
        tree (Tree): Uncleaned up tree from parser.

    Returns:
        Tree: Clean tree.

    """
    return wipe_empty_tags(remove_top(tree))


def tree_to_latex(tree: Tree, *, copy_to_clipboard: bool = False) -> str:
    """Convert a tree to a `qtree` latex tree.

    Args:
        tree (Tree): the tree to convert.
        copy_to_clipboard (bool, optional): Set to true to copy

    Returns:
        str: the LaTeX string defining the dependency graph.
        The string is optionally copied to the clipboard.

    """
    s = tree.pformat_latex_qtree()
    s = re.sub(r"(\]\s+)\.(\s+\])", r"\g<1>{.}\g<2>", s)
    if copy_to_clipboard:
        pyperclip.copy(s)
        print("Copied LaTeX string to clipboard!")
    return s
