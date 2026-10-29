# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.connectivity import Connection, Connectivity


def build_learning_modules(
    input_channels_per_lm: list[list[str]],
) -> list[MagicMock]:
    """Build named learning module mocks with `has_input_channel`.

    Args:
        input_channels_per_lm: The input channels each learning module models.

    Returns:
        List of mocked learning modules.
    """
    learning_modules = []
    for i, input_channels in enumerate(input_channels_per_lm):
        learning_module = MagicMock(spec=LearningModule)
        learning_module.learning_module_id = f"learning_module_{i}"
        learning_module.has_input_channel.side_effect = input_channels.__contains__
        learning_modules.append(learning_module)
    return learning_modules


@st.composite
def unequal_lengths(draw: st.DrawFn) -> tuple[int, int]:
    """Draw a number of learning modules and a different number of matrix rows.

    Args:
        draw: Hypothesis' draw function.

    Returns:
        The number of learning modules and the number of rows, which differ.
    """
    num_lms = draw(st.integers(min_value=0, max_value=5))
    num_rows = draw(
        st.integers(min_value=0, max_value=5).filter(lambda n: n != num_lms)
    )
    return (num_lms, num_rows)


class ConnectivityTest(unittest.TestCase):
    @given(lengths=unequal_lengths())
    def test_init_raises_value_error_when_matrix_length_mismatches(
        self, lengths: tuple[int, int]
    ) -> None:
        num_lms, num_rows = lengths
        learning_modules = build_learning_modules([[] for _ in range(num_lms)])

        with self.assertRaises(ValueError):
            Connectivity([[] for _ in range(num_rows)], learning_modules)

    def test_no_matrix_connects_no_senders(self) -> None:
        learning_modules = build_learning_modules([[], [], ["learning_module_0"]])

        connectivity = Connectivity(None, learning_modules)

        for receiver in range(len(learning_modules)):
            for connection in Connection:
                self.assertEqual(connectivity.senders_to(receiver, connection), [])

    def test_sender_modeled_as_an_input_channel_connects_bottom_up(self) -> None:
        learning_modules = build_learning_modules([[], [], ["learning_module_0"]])

        connectivity = Connectivity([[], [], [0]], learning_modules)

        self.assertEqual(connectivity.senders_to(2, Connection.BOTTOM_UP), [0])
        self.assertEqual(connectivity.senders_to(2, Connection.TOP_DOWN), [])

    def test_sender_not_modeled_as_an_input_channel_connects_top_down(self) -> None:
        learning_modules = build_learning_modules([[], [], []])

        connectivity = Connectivity([[2], [], []], learning_modules)

        self.assertEqual(connectivity.senders_to(0, Connection.TOP_DOWN), [2])
        self.assertEqual(connectivity.senders_to(0, Connection.BOTTOM_UP), [])

    def test_reciprocal_edges_are_classified_from_each_receiver(self) -> None:
        learning_modules = build_learning_modules([[], [], ["learning_module_0"]])

        connectivity = Connectivity([[2], [], [0]], learning_modules)

        self.assertEqual(connectivity.senders_to(0, Connection.TOP_DOWN), [2])
        self.assertEqual(connectivity.senders_to(2, Connection.BOTTOM_UP), [0])

    def test_bottom_up_order_is_index_order_without_bottom_up_connections(
        self,
    ) -> None:
        learning_modules = build_learning_modules([[], [], []])

        top_down_only = Connectivity([[2], [2], []], learning_modules)
        unconnected = Connectivity(None, learning_modules)

        self.assertEqual(top_down_only.bottom_up_order, [0, 1, 2])
        self.assertEqual(unconnected.bottom_up_order, [0, 1, 2])

    def test_bottom_up_order_steps_every_sender_before_its_receiver(self) -> None:
        learning_modules = build_learning_modules(
            [["learning_module_1", "learning_module_2"], [], []]
        )

        connectivity = Connectivity([[1, 2], [], []], learning_modules)

        self.assertEqual(connectivity.bottom_up_order, [1, 2, 0])

    def test_bottom_up_order_follows_a_chain_of_senders(self) -> None:
        learning_modules = build_learning_modules(
            [["learning_module_1"], ["learning_module_2"], []]
        )

        connectivity = Connectivity([[1], [2], []], learning_modules)

        self.assertEqual(connectivity.bottom_up_order, [2, 1, 0])

    def test_init_raises_value_error_when_bottom_up_connections_form_a_cycle(
        self,
    ) -> None:
        learning_modules = build_learning_modules(
            [["learning_module_1"], ["learning_module_0"]]
        )

        with self.assertRaisesRegex(
            ValueError,
            r"Bottom-up connections between learning modules \[0, 1\] form a cycle",
        ):
            Connectivity([[1], [0]], learning_modules)
