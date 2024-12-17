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
        gold = len(pred_set)

        p = correct / pred
        r = correct / gold
        f = (2 * p * r) / (p + r)

        return cls(p, r, f)
