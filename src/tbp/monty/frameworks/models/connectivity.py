# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from enum import Enum
from typing import Sequence

from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule

__all__ = ["Connection", "Connectivity"]


class Connection(Enum):
    """How one learning module's output reaches another learning module."""

    BOTTOM_UP = "bottom_up"
    """The receiver models the sender as an input channel, so the sender's output
    arrives as a percept to match against the receiver's models."""
    TOP_DOWN = "top_down"
    """The receiver does not model the sender, so the sender's output biases the
    receiver's hypotheses instead of being matched."""


class Connectivity:
    """The connectivity between learning modules.

    An `lm_to_lm_matrix` row says which learning modules a receiver gets messages from,
    but not the connection type, i.e., bottom-up or top-down. The connection type is
    decided by the receiver. If the message's sender is an input channel on the
    receiver's tolerance list, i.e., used for matching, the connection is bottom-up.
    Otherwise, the connection is classified as top-down. This is modeled based on how
    the hierarchical connections are defined in the cortex, i.e., by the asymmetry
    of L4 --> L3 vs. L6 --> L1 connections.

    Note that this class only models the `lm_to_lm_matrix` and does not include the
    voting connections, i.e., `lm_to_lm_vote_matrix`.
    """

    _senders: dict[Connection, list[list[int]]]
    """Stores the senders to receiving LMs by connection type"""

    def __init__(
        self,
        lm_to_lm_matrix: Sequence[Sequence[int]] | None,
        learning_modules: Sequence[LearningModule],
    ) -> None:
        """Classify every `lm_to_lm_matrix` edge by how it is delivered.

        Args:
            lm_to_lm_matrix: `lm_to_lm_matrix[i]` lists the learning modules whose
                output learning module i receives. `None` connects nothing.
            learning_modules: The learning modules the matrix indexes.

        Raises:
            ValueError: If the lengths of `learning_modules` and `lm_to_lm_matrix`
                do not match.
        """
        if lm_to_lm_matrix is None:
            lm_to_lm_matrix = [[] for _ in learning_modules]
        elif len(learning_modules) != len(lm_to_lm_matrix):
            raise ValueError(
                "The lengths of learning_modules and lm_to_lm_matrix must match"
            )

        self._senders = {
            connection: [[] for _ in learning_modules] for connection in Connection
        }
        for receiver, senders in enumerate(lm_to_lm_matrix):
            receiving_lm = learning_modules[receiver]
            for sender in senders:
                sender_id = learning_modules[sender].learning_module_id
                if receiving_lm.has_input_channel(sender_id):
                    self._senders[Connection.BOTTOM_UP][receiver].append(sender)
                else:
                    self._senders[Connection.TOP_DOWN][receiver].append(sender)

        # TODO: Reject bottom-up connections that form a cycle of any length, e.g.
        # LM0 -> LM1 -> LM2 -> LM0. Until then configs must define a valid
        # heterarchy connectivity

    def senders_to(self, receiver: int, connection: Connection) -> Sequence[int]:
        """The learning modules whose outputs reach a receiver over a connection.

        Args:
            receiver: Index of the receiving learning module.
            connection: How the senders' outputs reach the receiver.

        Returns:
            Indices of the sending learning modules.
        """
        return self._senders[connection][receiver]
