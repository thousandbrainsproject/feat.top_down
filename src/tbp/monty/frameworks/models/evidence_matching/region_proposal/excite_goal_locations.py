# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.frameworks.models.evidence_matching.region_proposal.protocol import (
    RegionProposer,
)

DEFAULT_RADIUS = 0.01
"""Radius, in meters, of the excited ball around each goal location."""

DEFAULT_LATTICE_STEPS = 4
"""Lattice steps per radius. The point spacing (2.5 mm at the defaults) stays below
the attention voxel size so the ball voxelizes without gaps."""


def sample_ball(
    center: npt.ArrayLike, radius: float, lattice_steps: int
) -> npt.NDArray[np.float64]:
    """Sample a solid ball on a cubic lattice.

    Args:
        center: (3,) center of the ball.
        radius: Radius of the ball, in the units of `center`.
        lattice_steps: Lattice steps per radius. The lattice spans
            [-radius, radius] with 2 * lattice_steps + 1 points per axis.

    Returns:
        The (N, 3) lattice points within `radius` of `center`, the center among
        them.
    """
    axis = np.linspace(-radius, radius, 2 * lattice_steps + 1)
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    offsets = offsets.reshape(-1, 3)
    offsets = offsets[np.linalg.norm(offsets, axis=1) <= radius]
    return np.asarray(center, dtype=np.float64) + offsets


class ExciteGoalLocations(RegionProposer):
    """Excite a ball around each location the learning module's goals target.

    A goal is a place the learning module wants a sensor to go. Exciting the region
    around it keeps the attention system from filtering out that goal and goals near
    it. Goals without a location have nothing to excite and are skipped.
    """

    def __init__(
        self,
        radius: float = DEFAULT_RADIUS,
        lattice_steps: int = DEFAULT_LATTICE_STEPS,
        weight: float = AttentionRegion.MAX_WEIGHT,
    ) -> None:
        """Initialize the proposer.

        Args:
            radius: Radius of the excited ball around each goal location, in meters.
            lattice_steps: Lattice steps per radius used to fill the ball.
            weight: Attention weight given to every point of the ball.
        """
        self._radius = radius
        self._lattice_steps = lattice_steps
        self._weight = weight

    def __call__(self, goals: Sequence[Goal]) -> AttentionRegion:
        """Propose a ball around each goal location.

        Args:
            goals: The goals the learning module proposed this step.

        Returns:
            The balls at the configured weight. Empty if no goal has a location.
        """
        locations = [goal.location for goal in goals if goal.location is not None]
        if not locations:
            return AttentionRegion.empty()
        balls = np.vstack(
            [
                sample_ball(location, self._radius, self._lattice_steps)
                for location in locations
            ]
        )
        return AttentionRegion.uniform(balls, self._weight)
