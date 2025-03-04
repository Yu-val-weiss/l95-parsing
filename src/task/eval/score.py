"""Contains EvalScore class definition."""

from __future__ import annotations

from typing import NamedTuple


class EvalScore(NamedTuple):
    """Tuple for evaluation scores."""

    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        """Create string representation.

        Returns:
            str: string depiction

        """
        return f"(P = {self.precision:.2f}, R = {self.recall:.2f}, F = {self.f1:.2f})"

    @classmethod
    def from_sets(cls, pred_set: set, gold_set: set) -> EvalScore:
        """Create an evaluation score from predicted and gold sets.

        Args:
            pred_set (set): Set containing predictions.
            gold_set (set): Set containing ground truth values.

        Returns:
            EvalScore: Returns corresponding EvalScore.

        """
        correct = len(pred_set & gold_set)
        pred = len(pred_set)
        gold = len(gold_set)

        p = correct / pred if pred != 0 else 0.0
        r = correct / gold if gold != 0 else 0.0
        f = (2 * p * r) / (p + r) if p + r > 0.0 else 0.0

        return cls(p, r, f)


class Accuracy(float):
    """Float wrapper class representing accuracy."""

    @classmethod
    def from_sets(cls, pred_set: set, gold_set: set) -> Accuracy:
        """Calculate accuracy from predicted and gold sets.

        Args:
            pred_set (set): Set containing predictions.
            gold_set (set): Set containing ground truth values.

        Returns:
            Accuracy: Returns corresponding accuracy.

        """
        correct = len(pred_set & gold_set)
        pred = len(pred_set)
        gold = len(gold_set)

        if pred != gold:
            msg = "Both sets must be the same length to compute accuracy correctly."
            raise ValueError(msg)

        p = correct / pred

        return cls(p)
