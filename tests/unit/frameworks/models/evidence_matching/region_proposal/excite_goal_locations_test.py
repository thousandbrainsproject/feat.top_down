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

import numpy as np
import numpy.typing as npt

from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.frameworks.models.evidence_matching.region_proposal.excite_goal_locations import (  # noqa: E501
    ExciteGoalLocations,
    sample_ball,
)

RADIUS = 0.01
LATTICE_STEPS = 4


def goal_at(location: npt.ArrayLike | None) -> Goal:
    return Goal(
        location=None if location is None else np.asarray(location, dtype=float),
        morphological_features=None,
        non_morphological_features=None,
        confidence=1.0,
        pass_message=True,
        sender_id="learning_module_0",
        sender_type="GSG",
        process_features_in_lm=False,
        goal_tolerances=None,
    )


class SampleBallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.center = np.array([1.0, -2.0, 3.0])
        self.points = sample_ball(self.center, RADIUS, LATTICE_STEPS)

    def test_every_point_lies_within_the_radius(self) -> None:
        distances = np.linalg.norm(self.points - self.center, axis=1)
        self.assertLessEqual(distances.max(), RADIUS + 1e-12)

    def test_includes_the_center(self) -> None:
        self.assertTrue(np.isclose(self.points, self.center).all(axis=1).any())

    def test_points_are_spaced_by_radius_over_lattice_steps(self) -> None:
        np.testing.assert_allclose(
            np.unique(np.round(self.points[:, 0] - self.center[0], 12)),
            np.linspace(-RADIUS, RADIUS, 2 * LATTICE_STEPS + 1),
            atol=1e-12,
        )


class ExciteGoalLocationsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proposer = ExciteGoalLocations(radius=RADIUS, lattice_steps=LATTICE_STEPS)

    def test_no_goals_give_an_empty_region(self) -> None:
        self.assertEqual(len(self.proposer([])), 0)

    def test_goals_without_a_location_give_an_empty_region(self) -> None:
        self.assertEqual(len(self.proposer([goal_at(None)])), 0)

    def test_excites_a_ball_around_each_goal_location(self) -> None:
        near, far = np.zeros(3), np.array([1.0, 0.0, 0.0])

        region = self.proposer([goal_at(near), goal_at(None), goal_at(far)])

        ball = sample_ball(np.zeros(3), RADIUS, LATTICE_STEPS)
        np.testing.assert_allclose(
            region.locations, np.vstack([ball + near, ball + far])
        )
        np.testing.assert_array_equal(
            region.weights, np.full(len(region), AttentionRegion.MAX_WEIGHT)
        )

    def test_weight_is_configurable(self) -> None:
        region = ExciteGoalLocations(weight=0.5)([goal_at(np.zeros(3))])

        np.testing.assert_array_equal(region.weights, np.full(len(region), 0.5))
