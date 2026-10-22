# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Protocol, Sequence

from tbp.monty.cmp import AttentionRegion, Goal


class RegionProposer(Protocol):
    """Turns a learning module's goals for this step into an attention region."""

    def __call__(self, goals: Sequence[Goal]) -> AttentionRegion:
        """Propose a region from the learning module's goals.

        Args:
            goals: The goals the learning module proposed this step.

        Returns:
            The proposed region. Empty if there is nothing to propose.
        """
        ...
