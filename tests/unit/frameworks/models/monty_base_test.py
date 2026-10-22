# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch, sentinel

import numpy as np

from tbp.monty.attention.attention_system import DefaultAttentionSystem
from tbp.monty.cmp import AttentionRegion, Goal
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.evidence_matching.region_proposal.excite_goal_locations import (  # noqa: E501
    ExciteGoalLocations,
)
from tbp.monty.frameworks.models.monty_base import MontyBase


class MontyBasePrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sm1 = MagicMock()
        self.sm1.sensor_module_id = "sm1"
        self.sm2 = MagicMock()
        self.sm2.sensor_module_id = "sm2"
        self.lm1 = MagicMock()
        self.lm2 = MagicMock()
        self.lm3 = MagicMock()
        self.attention_system = MagicMock()
        self.monty_base = MontyBase(
            sensor_modules=[self.sm1, self.sm2],
            learning_modules=[self.lm1, self.lm2, self.lm3],
            motor_system=MagicMock(),
            sm_to_agent_dict={
                "sm1": AgentID("agent_id_0"),
                "sm2": AgentID("agent_id_0"),
            },
            sm_to_lm_matrix=[[], [], []],
            lm_to_lm_matrix=[[], [], []],
            lm_to_lm_vote_matrix=[[], [], []],
            min_eval_steps=10,
            min_train_steps=10,
            num_exploratory_steps=10,
            attention_system=self.attention_system,
        )

    def test_pass_goals_collects_all_goals_from_learning_and_sensor_modules_and_filters_them_through_the_attention_system(  # noqa: E501
        self,
    ) -> None:
        self.lm1.propose_goals.return_value = []
        self.lm2.propose_goals.return_value = [sentinel.lm2_goal]
        self.lm3.propose_goals.return_value = [
            sentinel.lm3_goal_1,
            sentinel.lm3_goal_2,
        ]
        self.sm1.propose_goals.return_value = []
        self.sm2.propose_goals.return_value = [
            sentinel.sm2_goal_1,
            sentinel.sm2_goal_2,
        ]
        self.lm1.propose_region.return_value = sentinel.lm1_region
        self.lm2.propose_region.return_value = sentinel.lm2_region
        self.lm3.propose_region.return_value = sentinel.lm3_region
        self.sm1.propose_region.return_value = sentinel.sm1_region
        self.sm2.propose_region.return_value = sentinel.sm2_region
        self.attention_system.step.return_value = sentinel.attention_system_goals

        self.monty_base._pass_goals()

        goals = [
            sentinel.lm2_goal,
            sentinel.lm3_goal_1,
            sentinel.lm3_goal_2,
            sentinel.sm2_goal_1,
            sentinel.sm2_goal_2,
        ]
        self.attention_system.step.assert_called_once_with(
            goals,
            [
                sentinel.lm1_region,
                sentinel.lm2_region,
                sentinel.lm3_region,
                sentinel.sm1_region,
                sentinel.sm2_region,
            ],
        )
        self.assertIs(self.monty_base._goals, sentinel.attention_system_goals)


class MontyBaseAttentionTest(unittest.TestCase):
    VOXEL_SIZE = 0.005

    def setUp(self) -> None:
        self.sm = MagicMock()
        self.sm.sensor_module_id = "sm"
        self.sm.propose_goals.return_value = []
        self.sm.propose_region.return_value = AttentionRegion.uniform(
            [[0.0, 0.0, 0.0], [self.VOXEL_SIZE, 0.0, 0.0]], AttentionRegion.MAX_WEIGHT
        )
        self.lm_goal = Goal(
            location=np.array([0.03, 0.0, 0.0]),
            morphological_features=None,
            non_morphological_features=None,
            confidence=1.0,
            pass_message=True,
            sender_id="learning_module_0",
            sender_type="GSG",
            process_features_in_lm=False,
            goal_tolerances=None,
        )

    def goals_after_attention(self, graph_lm: EvidenceGraphLM) -> list[Goal]:
        monty_base = MontyBase(
            sensor_modules=[self.sm],
            learning_modules=[graph_lm],
            motor_system=MagicMock(),
            sm_to_agent_dict={"sm": AgentID("agent_id_0")},
            sm_to_lm_matrix=[[]],
            lm_to_lm_matrix=[[]],
            lm_to_lm_vote_matrix=[[]],
            min_eval_steps=10,
            min_train_steps=10,
            num_exploratory_steps=10,
            attention_system=DefaultAttentionSystem(voxel_size=self.VOXEL_SIZE),
        )
        with patch.object(graph_lm, "propose_goals", return_value=[self.lm_goal]):
            monty_base._pass_goals()
        return monty_base._goals

    def elm(self, *proposers: ExciteGoalLocations) -> EvidenceGraphLM:
        return EvidenceGraphLM(
            max_match_distance=0.005,
            tolerances={},
            feature_weights={},
            region_proposers=proposers,
        )

    def test_lm_goal_off_the_sensor_region_is_filtered_out_without_proposers(
        self,
    ) -> None:
        self.assertEqual(self.goals_after_attention(self.elm()), [])

    def test_lm_goal_off_the_sensor_region_survives_with_excite_goal_locations(
        self,
    ) -> None:
        goals = self.goals_after_attention(self.elm(ExciteGoalLocations()))

        self.assertEqual(goals, [self.lm_goal])
