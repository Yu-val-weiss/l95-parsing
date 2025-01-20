"""Utils specifically for dealing with constituency parses."""

from typing import cast

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


def flatten_children(tree: Tree) -> Tree:
    """Flatten any children in the tree.

    Flattening children means patterns of (... (A x) y (A z) ...) -> (... (A x y z) ...)
    iff y is 'or' or 'and'.

    Args:
        tree (Tree): tree to flatten. The tree should have had all empty tags wiped.

    Returns:
        Tree: tree with flattened children.

    """
    # base case: if the input is not a Tree, return
    if not isinstance(tree, Tree):
        return tree

    # if the tree has fewer than 3 children, recursively flatten its children
    if len(tree) < 3:
        return Tree(tree.label(), [flatten_children(child) for child in tree])

    # check if all children are leaves (not Trees)
    if all(not isinstance(child, Tree) for child in tree):
        return tree

    new_tree = []
    i = 0

    while i < len(tree):
        current = tree[i]

        # if the current element is not a Tree, add it to new_tree and continue
        if not isinstance(current, Tree):
            new_tree.append(current)
            i += 1
            continue

        left = cast(Tree, current)
        if (
            i + 2 < len(tree)
            and isinstance(tree[i + 1], str)
            and isinstance(tree[i + 2], Tree)
        ):
            mid = cast(str, tree[i + 1])
            right = cast(Tree, tree[i + 2])

            # check if the middle token is "and" or "or"
            # and that the labels of left and right match
            if mid.lower() in {"and", "or"} and left.label() == right.label():
                new_tree.append(Tree(left.label(), [left[0], mid, right[0]]))
                i += 3
                continue

        # otherwise, flatten the current child and add it to the new tree
        new_tree.append(flatten_children(left))
        i += 1

    # if the new tree has a single child with the same label, return that child directly
    if (
        len(new_tree) == 1
        and isinstance(new_tree[0], Tree)
        and new_tree[0].label() == tree.label()
    ):
        return new_tree[0]

    return Tree(tree.label(), new_tree)


def tree_to_latex(tree: Tree, *, copy_to_clipboard: bool = False) -> str:
    """Convert a tree to a `forest` latex tree.

    Args:
        tree (Tree): the tree to convert.
        copy_to_clipboard (bool, optional): Set to true to copy

    Returns:
        str: the LaTeX string defining the constituency tree.
        The string is optionally copied to the clipboard.

    """
    s = _tree_to_latex(tree)
    if copy_to_clipboard:
        pyperclip.copy(s)
        print("Copied LaTeX string to clipboard!")
    return s


def _tree_to_latex(tree: Tree) -> str:
    str_parts = [f"[{_clean_tree_label(tree.label())}"]
    for subtree in tree:
        if isinstance(subtree, Tree):
            str_parts.append(_tree_to_latex(subtree))
        else:
            str_parts.append(f"[{_clean_tree_label(subtree)}]")
    if str_parts[-1][-1] == "]":
        str_parts[-1] += "]"
    else:
        str_parts.append("]")
    return " ".join(str_parts)


FOREST_SPECIAL_CHARS = [",", "=", "[", "]"]


def _clean_tree_label(label: str) -> str:
    if label == "_":
        return r"\_"
    if any(c in label for c in FOREST_SPECIAL_CHARS):
        return f"{{{label}}}"
    return label
