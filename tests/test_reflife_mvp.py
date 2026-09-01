import unittest

import torch
import torch.nn.functional as F
from torch import nn

from models.reflife_mvp import (
    CPULedger,
    FEEDBACK_CONSISTENT,
    FEEDBACK_CONTRADICTORY,
    FEEDBACK_UNCERTAIN,
    FuturePredictor,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_DELETED,
    LIFECYCLE_DORMANT,
    REFERENCE_FROZEN,
    REFERENCE_OBSERVATION,
    REFERENCE_PREDICTION,
    LongIdentityMemory,
    PredictionFeedback,
    ReferenceController,
    ReliableGeometry,
    ReliableObservationGate,
)
from models.structures import Instances
from models.transrmot_pro import ClipMatcher, RuntimeTrackerBase, TransRMOT


def make_track_instances(
    scores,
    refer_probabilities,
    embeddings,
):
    """Build minimal track instances for RefLife unit tests."""
    scores = torch.tensor(
        scores,
        dtype=torch.float32,
    )

    refer_probabilities = torch.tensor(
        refer_probabilities,
        dtype=torch.float32,
    )

    embeddings = torch.tensor(
        embeddings,
        dtype=torch.float32,
    )

    num_tracks, feature_dim = embeddings.shape

    track_instances = Instances((1, 1))

    track_instances.scores = scores

    track_instances.pred_refers = torch.logit(
        refer_probabilities.clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )
    ).unsqueeze(-1)

    track_instances.pred_boxes = torch.zeros(
        (num_tracks, 4),
        dtype=torch.float32,
    )

    track_instances.output_embedding = embeddings

    track_instances.long_memory = torch.zeros(
        (num_tracks, feature_dim),
        dtype=torch.float32,
    )

    track_instances.long_memory_valid = torch.zeros(
        (num_tracks,),
        dtype=torch.bool,
    )

    track_instances.long_memory_age = torch.zeros(
        (num_tracks,),
        dtype=torch.long,
    )
    track_instances.next_box_prediction = torch.zeros(
        (num_tracks, 4), dtype=torch.float32)
    track_instances.prediction_valid = torch.zeros(
        (num_tracks,), dtype=torch.bool)
    track_instances.next_visibility_logit = torch.zeros(
        (num_tracks,), dtype=torch.float32)
    track_instances.prediction_uncertainty = torch.zeros(
        (num_tracks,), dtype=torch.float32)
    track_instances.box_residual = torch.zeros(
        (num_tracks,), dtype=torch.float32)
    track_instances.identity_similarity = torch.zeros(
        (num_tracks,), dtype=torch.float32)
    track_instances.feedback_valid = torch.zeros(
        (num_tracks,), dtype=torch.bool)
    track_instances.feedback_state = torch.zeros(
        (num_tracks,), dtype=torch.uint8)
    track_instances.last_reliable_box = torch.zeros(
        (num_tracks, 4), dtype=torch.float32)
    track_instances.velocity = torch.zeros(
        (num_tracks, 4), dtype=torch.float32)
    track_instances.reliable_state_valid = torch.zeros(
        (num_tracks,), dtype=torch.bool)
    track_instances.obj_idxes = torch.full(
        (num_tracks,), -1, dtype=torch.long)
    track_instances.disappear_time = torch.zeros(
        (num_tracks,), dtype=torch.long)
    track_instances.lifecycle_state = torch.zeros(
        (num_tracks,), dtype=torch.uint8)
    track_instances.dormant_age = torch.zeros(
        (num_tracks,), dtype=torch.long)
    track_instances.emit_mask = torch.zeros(
        (num_tracks,), dtype=torch.bool)
    track_instances.ref_pts = torch.zeros(
        (num_tracks, 2), dtype=torch.float32)
    track_instances.ref_source = torch.zeros(
        (num_tracks,), dtype=torch.uint8)

    return track_instances


def make_transrmot_shell(
    reflife_enabled,
    long_memory_enabled,
    prediction_feedback_enabled=False,
):
    """Build only the RefLife-related part of TransRMOT."""
    model = TransRMOT.__new__(TransRMOT)
    nn.Module.__init__(model)
    model.reflife_enabled = reflife_enabled
    model.long_memory_enabled = long_memory_enabled
    model.prediction_feedback_enabled = prediction_feedback_enabled
    model.reliable_observation_gate = ReliableObservationGate(
        object_threshold=0.6,
        refer_threshold=0.5,
    )
    model.long_identity_memory = LongIdentityMemory(momentum=0.95)
    model.prediction_feedback = PredictionFeedback(
        box_threshold=0.15,
        identity_threshold=0.5,
    )
    return model


class ReliableObservationGateTests(unittest.TestCase):
    def test_gate_checks_scores_refer_and_finite_values(self):
        track_instances = make_track_instances(
            scores=[0.90, 0.59, 0.90, 0.90],
            refer_probabilities=[0.80, 0.80, 0.49, 0.80],
            embeddings=[
                [3.0, 4.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
        )

        # The fourth query contains an invalid box value.
        track_instances.pred_boxes[3, 0] = float("nan")

        gate = ReliableObservationGate(
            object_threshold=0.6,
            refer_threshold=0.5,
        )

        write_mask = gate(track_instances)

        expected_mask = torch.tensor(
            [True, False, False, False],
            dtype=torch.bool,
        )

        self.assertTrue(
            torch.equal(write_mask, expected_mask)
        )


class LongIdentityMemoryTests(unittest.TestCase):
    def test_first_reliable_observation_initializes_memory(self):
        track_instances = make_track_instances(
            scores=[0.90],
            refer_probabilities=[0.80],
            embeddings=[[3.0, 4.0]],
        )

        gate = ReliableObservationGate(
            object_threshold=0.6,
            refer_threshold=0.5,
        )

        memory = LongIdentityMemory(momentum=0.95)

        write_mask = gate(track_instances)
        track_instances = memory(
            track_instances,
            write_mask,
        )

        expected_memory = torch.tensor(
            [[0.6, 0.8]],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            track_instances.long_memory,
            expected_memory,
        )

        self.assertTrue(
            track_instances.long_memory_valid[0].item()
        )

        self.assertEqual(
            track_instances.long_memory_age[0].item(),
            0,
        )

    def test_later_reliable_observation_uses_ema(self):
        track_instances = make_track_instances(
            scores=[0.90],
            refer_probabilities=[0.80],
            embeddings=[[0.0, 1.0]],
        )

        track_instances.long_memory[0] = torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        )

        track_instances.long_memory_valid[0] = True
        track_instances.long_memory_age[0] = 4

        memory = LongIdentityMemory(momentum=0.5)

        write_mask = torch.tensor(
            [True],
            dtype=torch.bool,
        )

        track_instances = memory(
            track_instances,
            write_mask,
        )

        expected_memory = F.normalize(
            torch.tensor(
                [[0.5, 0.5]],
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        )

        torch.testing.assert_close(
            track_instances.long_memory,
            expected_memory,
        )

        self.assertEqual(
            track_instances.long_memory_age[0].item(),
            0,
        )

    def test_rejected_observation_keeps_memory_unchanged(self):
        track_instances = make_track_instances(
            scores=[0.20],
            refer_probabilities=[0.80],
            embeddings=[[0.0, 1.0]],
        )

        track_instances.long_memory[0] = torch.tensor(
            [1.0, 0.0],
            dtype=torch.float32,
        )

        track_instances.long_memory_valid[0] = True
        track_instances.long_memory_age[0] = 2

        memory_before = (
            track_instances.long_memory.clone()
        )

        memory = LongIdentityMemory(momentum=0.5)

        write_mask = torch.tensor(
            [False],
            dtype=torch.bool,
        )

        track_instances = memory(
            track_instances,
            write_mask,
        )

        torch.testing.assert_close(
            track_instances.long_memory,
            memory_before,
        )

        self.assertEqual(
            track_instances.long_memory_age[0].item(),
            3,
        )


class PredictionFeedbackTests(unittest.TestCase):
    def test_feedback_separates_three_states(self):
        track_instances = make_track_instances(
            scores=[0.9, 0.9, 0.9],
            refer_probabilities=[0.8, 0.8, 0.8],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        )
        track_instances.pred_boxes = torch.tensor([
            [0.50, 0.50, 0.20, 0.20],
            [0.80, 0.80, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
        ])
        track_instances.next_box_prediction = torch.tensor([
            [0.51, 0.49, 0.20, 0.20],
            [0.10, 0.10, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
        ])
        track_instances.long_memory[:] = torch.tensor([1.0, 0.0])
        track_instances.long_memory_valid[:] = True
        track_instances.prediction_valid[:] = torch.tensor([True, True, False])

        feedback = PredictionFeedback(
            box_threshold=0.15,
            identity_threshold=0.5,
        )
        feedback(track_instances, torch.ones(3, dtype=torch.bool))

        self.assertEqual(track_instances.feedback_state.tolist(), [
            FEEDBACK_CONSISTENT,
            FEEDBACK_CONTRADICTORY,
            FEEDBACK_UNCERTAIN,
        ])
        self.assertEqual(track_instances.feedback_valid.tolist(), [True, True, False])


class ReliableGeometryTests(unittest.TestCase):
    def test_geometry_initializes_updates_and_freezes(self):
        track_instances = make_track_instances(
            scores=[0.9],
            refer_probabilities=[0.8],
            embeddings=[[1.0, 0.0]],
        )
        track_instances.last_reliable_box = torch.zeros((1, 4))
        track_instances.velocity = torch.zeros((1, 4))
        track_instances.reliable_state_valid = torch.zeros((1,), dtype=torch.bool)
        geometry = ReliableGeometry()

        track_instances.pred_boxes[0] = torch.tensor([0.4, 0.5, 0.2, 0.3])
        geometry(track_instances, torch.tensor([True]))
        torch.testing.assert_close(
            track_instances.last_reliable_box,
            torch.tensor([[0.4, 0.5, 0.2, 0.3]]),
        )
        torch.testing.assert_close(track_instances.velocity, torch.zeros((1, 4)))

        track_instances.pred_boxes[0] = torch.tensor([0.5, 0.45, 0.22, 0.3])
        geometry(track_instances, torch.tensor([True]))
        torch.testing.assert_close(
            track_instances.velocity,
            torch.tensor([[0.1, -0.05, 0.02, 0.0]]),
        )

        previous_box = track_instances.last_reliable_box.clone()
        previous_velocity = track_instances.velocity.clone()
        track_instances.pred_boxes[0] = 0.9
        geometry(track_instances, torch.tensor([False]))
        torch.testing.assert_close(track_instances.last_reliable_box, previous_box)
        torch.testing.assert_close(track_instances.velocity, previous_velocity)


class FuturePredictorTests(unittest.TestCase):
    def test_zero_initialized_head_uses_constant_velocity(self):
        track_instances = make_track_instances(
            scores=[0.9, 0.2],
            refer_probabilities=[0.8, 0.3],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        track_instances.pred_boxes = torch.tensor([
            [0.4, 0.5, 0.2, 0.3],
            [0.7, 0.4, 0.1, 0.2],
        ])
        track_instances.last_reliable_box[:] = track_instances.pred_boxes
        track_instances.velocity[0] = torch.tensor([0.1, -0.05, 0.0, 0.0])
        track_instances.reliable_state_valid[0] = True
        predictor = FuturePredictor(feature_dim=2, hidden_dim=8)

        predictor(track_instances, torch.tensor([True, False]))

        torch.testing.assert_close(
            track_instances.next_box_prediction[0],
            torch.tensor([0.5, 0.45, 0.2, 0.3]),
        )
        self.assertEqual(track_instances.prediction_valid.tolist(), [True, False])
        self.assertGreater(track_instances.prediction_uncertainty[0].item(), 0.0)

    def test_future_loss_uses_next_frame_of_same_identity(self):
        track_instances = make_track_instances(
            scores=[0.9, 0.9],
            refer_probabilities=[0.8, 0.8],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        track_instances.obj_idxes[:] = torch.tensor([10, 20])
        track_instances.prediction_valid[:] = True
        track_instances.next_visibility_logit = torch.zeros(2, requires_grad=True)
        track_instances.next_box_prediction = torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.2, 0.2, 0.1, 0.1],
        ], requires_grad=True)
        track_instances.prediction_uncertainty = torch.ones(2, requires_grad=True)

        current_gt = Instances((1, 1))
        current_gt.obj_ids = torch.tensor([10, 20])
        current_gt.boxes = torch.zeros((2, 4))
        next_gt = Instances((1, 1))
        next_gt.obj_ids = torch.tensor([10])
        next_gt.boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])

        criterion = ClipMatcher.__new__(ClipMatcher)
        nn.Module.__init__(criterion)
        criterion._current_frame_idx = 1
        criterion.gt_instances = [current_gt, next_gt]
        criterion.losses_dict = {}
        criterion.loss_future_prediction(track_instances)

        self.assertIn('frame_0_loss_future_presence', criterion.losses_dict)
        self.assertIn('frame_0_loss_future_box', criterion.losses_dict)
        self.assertIn('frame_0_loss_future_giou', criterion.losses_dict)
        total_loss = sum(criterion.losses_dict.values())
        total_loss.backward()
        self.assertIsNotNone(track_instances.next_box_prediction.grad)

    def test_future_loss_allows_clip_without_valid_match(self):
        track_instances = make_track_instances(
            scores=[0.1],
            refer_probabilities=[0.1],
            embeddings=[[1.0, 0.0]],
        )
        track_instances.next_box_prediction = torch.zeros(
            (1, 4), requires_grad=True
        )
        track_instances.next_visibility_logit = torch.zeros(
            (1,), requires_grad=True
        )
        current_gt = Instances((1, 1))
        current_gt.obj_ids = torch.empty((0,), dtype=torch.long)
        current_gt.boxes = torch.empty((0, 4))
        next_gt = Instances((1, 1))
        next_gt.obj_ids = torch.empty((0,), dtype=torch.long)
        next_gt.boxes = torch.empty((0, 4))
        criterion = ClipMatcher.__new__(ClipMatcher)
        nn.Module.__init__(criterion)
        criterion._current_frame_idx = 1
        criterion.gt_instances = [current_gt, next_gt]
        criterion.losses_dict = {}

        criterion.loss_future_prediction(track_instances)
        sum(criterion.losses_dict.values()).backward()

        self.assertIsNotNone(track_instances.next_box_prediction.grad)
        self.assertIsNotNone(track_instances.next_visibility_logit.grad)


class LifecycleTests(unittest.TestCase):
    def test_prediction_can_hold_track_beyond_native_delete_point(self):
        track_instances = make_track_instances(
            scores=[0.1, 0.1],
            refer_probabilities=[0.8, 0.8],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        track_instances.obj_idxes[:] = torch.tensor([4, 5])
        track_instances.disappear_time[:] = 9
        track_instances.prediction_valid[:] = True
        track_instances.next_visibility_logit[:] = torch.logit(
            torch.tensor([0.9, 0.1])
        )
        tracker = RuntimeTrackerBase(miss_tolerance=10)

        tracker.update(
            track_instances,
            reflife_enabled=True,
            presence_threshold=0.6,
            max_dormant_age=20,
        )

        self.assertEqual(track_instances.obj_idxes.tolist(), [4, -1])
        self.assertEqual(track_instances.lifecycle_state.tolist(), [
            LIFECYCLE_DORMANT,
            LIFECYCLE_DELETED,
        ])
        self.assertEqual(track_instances.emit_mask.tolist(), [False, False])

        track_instances.scores[0] = 0.9
        tracker.update(track_instances, reflife_enabled=True)
        self.assertEqual(track_instances.lifecycle_state[0].item(), LIFECYCLE_ACTIVE)
        self.assertEqual(track_instances.dormant_age[0].item(), 0)
        self.assertTrue(track_instances.emit_mask[0].item())


class ReferenceControllerTests(unittest.TestCase):
    def test_dormant_track_uses_prediction_or_freezes_reference(self):
        track_instances = make_track_instances(
            scores=[0.9, 0.1, 0.1],
            refer_probabilities=[0.8, 0.8, 0.8],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        )
        track_instances.obj_idxes[:] = torch.tensor([1, 2, 3])
        track_instances.lifecycle_state[:] = torch.tensor([
            LIFECYCLE_ACTIVE,
            LIFECYCLE_DORMANT,
            LIFECYCLE_DORMANT,
        ])
        track_instances.prediction_valid[:] = torch.tensor([True, True, False])
        track_instances.next_box_prediction[1] = torch.tensor([0.7, 0.4, 0.2, 0.2])
        track_instances.ref_pts[2] = torch.tensor([2.0, -1.0])
        controller = ReferenceController()

        controller(track_instances)

        self.assertEqual(track_instances.ref_source.tolist(), [
            REFERENCE_OBSERVATION,
            REFERENCE_PREDICTION,
            REFERENCE_FROZEN,
        ])
        torch.testing.assert_close(
            track_instances.ref_pts[1],
            torch.logit(torch.tensor([0.7, 0.4])),
        )
        torch.testing.assert_close(
            track_instances.ref_pts[2],
            torch.tensor([2.0, -1.0]),
        )


class CPULedgerTests(unittest.TestCase):
    def test_records_detached_track_state_and_clears(self):
        track_instances = make_track_instances(
            scores=[0.9],
            refer_probabilities=[0.8],
            embeddings=[[1.0, 0.0]],
        )
        track_instances.obj_idxes[0] = 7
        track_instances.emit_mask[0] = True
        ledger = CPULedger()

        ledger.record(track_instances)

        self.assertEqual(len(ledger.records), 1)
        self.assertEqual(ledger.records[0]['track_id'], 7)
        self.assertEqual(ledger.records[0]['frame'], 0)
        self.assertIsInstance(ledger.records[0]['box'], list)
        ledger.clear()
        self.assertEqual(ledger.records, [])
        self.assertEqual(ledger.frame_index, 0)


class TransRMOTLongMemoryIntegrationTests(unittest.TestCase):
    def test_master_switch_off_keeps_memory_unchanged(self):
        track_instances = make_track_instances(
            scores=[0.90],
            refer_probabilities=[0.80],
            embeddings=[[3.0, 4.0]],
        )
        model = make_transrmot_shell(
            reflife_enabled=False,
            long_memory_enabled=False,
        )

        model._update_reflife_long_memory(track_instances)

        torch.testing.assert_close(
            track_instances.long_memory,
            torch.zeros_like(track_instances.long_memory),
        )
        self.assertFalse(track_instances.long_memory_valid[0].item())

    def test_memory_switch_off_keeps_memory_unchanged(self):
        track_instances = make_track_instances(
            scores=[0.90],
            refer_probabilities=[0.80],
            embeddings=[[3.0, 4.0]],
        )
        model = make_transrmot_shell(
            reflife_enabled=True,
            long_memory_enabled=False,
        )

        model._update_reflife_long_memory(track_instances)

        torch.testing.assert_close(
            track_instances.long_memory,
            torch.zeros_like(track_instances.long_memory),
        )
        self.assertFalse(track_instances.long_memory_valid[0].item())

    def test_both_switches_on_update_memory(self):
        track_instances = make_track_instances(
            scores=[0.90],
            refer_probabilities=[0.80],
            embeddings=[[3.0, 4.0]],
        )
        model = make_transrmot_shell(
            reflife_enabled=True,
            long_memory_enabled=True,
        )

        model._update_reflife_long_memory(track_instances)

        torch.testing.assert_close(
            track_instances.long_memory,
            torch.tensor([[0.6, 0.8]], dtype=torch.float32),
        )
        self.assertTrue(track_instances.long_memory_valid[0].item())


if __name__ == "__main__":
    unittest.main()
