# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import logging
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.evidence_matching import graph_memory
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.object_model import GridTooSmallError
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    rotate_multiple_pose_vectors,
)
from tbp.monty.geometry import Rotation


class EvidenceGraphMemoryTest(TestCase):
    def setUp(self) -> None:
        self.memory = EvidenceGraphMemory(
            max_nodes_per_graph=10,
            max_graph_size=0.3,
            num_model_voxels_per_dim=50,
        )

    def test_build_graph_warns_when_grid_is_too_small(self) -> None:
        with patch.object(graph_memory, "GridObjectModel") as model_class:
            model_class.return_value.build_model.side_effect = GridTooSmallError
            with self.assertLogs(graph_memory.logger, level="WARNING") as captured:
                self.memory._build_graph(
                    locations=MagicMock(),
                    features=MagicMock(),
                    graph_id="test_object",
                    input_channel="patch",
                )

        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].levelno, logging.WARNING)
        self.assertEqual(
            captured.records[0].getMessage(),
            "Grid too small for given locations. Not building a model for test_object",
        )
        self.assertEqual(self.memory.models_in_memory, {})

    def test_build_graph_success_does_not_warn(self) -> None:
        with patch.object(graph_memory, "GridObjectModel") as model_class:
            with self.assertLogs(graph_memory.logger, level="INFO") as captured:
                self.memory._build_graph(
                    locations=MagicMock(),
                    features=MagicMock(),
                    graph_id="test_object",
                    input_channel="patch",
                )

        self.assertTrue(
            all(record.levelno < logging.WARNING for record in captured.records)
        )
        self.assertIs(
            self.memory.models_in_memory["test_object"]["patch"],
            model_class.return_value,
        )


class EvidenceGraphMemoryUpdateMemoryTest(TestCase):
    def setUp(self) -> None:
        self.memory = EvidenceGraphMemory(
            max_nodes_per_graph=10,
            max_graph_size=0.3,
            num_model_voxels_per_dim=50,
        )
        self.model_locations = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05],
            ]
        )
        self.model_pose_vectors = np.tile(
            np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (4, 1)
        )
        self.rotation = Rotation.from_euler("xyz", [30, 40, 50], degrees=True)
        self.location_rel_model = np.array([0.01, 0.0, 0.0])
        self.object_location_rel_body = np.array([0.2, 0.1, -0.3])
        self.body_locations = (
            self.rotation.apply(self.model_locations - self.location_rel_model)
            + self.object_location_rel_body
        )
        self.body_pose_vectors = rotate_multiple_pose_vectors(
            self.model_pose_vectors, self.rotation
        )

    def features(
        self, pose_vectors: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "pose_vectors": pose_vectors.copy(),
            "pose_fully_defined": np.ones((len(pose_vectors), 1)),
        }

    def test_new_input_channel_of_known_graph_is_stored_in_model_frame(
        self,
    ) -> None:
        self.memory.update_memory(
            locations=self.model_locations.copy(),
            features={"patch_a": self.features(self.model_pose_vectors)},
            graph_id="test_object",
            object_location_rel_body=np.zeros(3),
            location_rel_model=np.zeros(3),
            object_rotation=Rotation.identity(),
        )

        self.memory.update_memory(
            locations=self.body_locations,
            features={"patch_b": self.features(self.body_pose_vectors)},
            graph_id="test_object",
            object_location_rel_body=self.object_location_rel_body,
            location_rel_model=self.location_rel_model,
            object_rotation=self.rotation,
        )

        known_channel = self.memory.get_graph("test_object", "patch_a")
        new_channel = self.memory.get_graph("test_object", "patch_b")
        np.testing.assert_allclose(new_channel.pos, known_channel.pos, atol=1e-12)
        pose_vectors = new_channel.feature_mapping["pose_vectors"]
        np.testing.assert_allclose(
            new_channel.x[:, pose_vectors[0] : pose_vectors[1]],
            known_channel.x[:, pose_vectors[0] : pose_vectors[1]],
            atol=1e-12,
        )

    def test_new_graph_is_stored_in_body_frame(self) -> None:
        self.memory.update_memory(
            locations=self.body_locations.copy(),
            features={"patch_a": self.features(self.body_pose_vectors)},
            graph_id="test_object",
            object_location_rel_body=self.object_location_rel_body,
            location_rel_model=self.location_rel_model,
            object_rotation=self.rotation,
        )

        stored = self.memory.get_graph("test_object", "patch_a")
        np.testing.assert_allclose(
            np.sort(stored.pos, axis=0), np.sort(self.body_locations, axis=0)
        )
