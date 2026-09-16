import unittest

import torch

from models.structures import Instances
from models.ttt_memory import TrackTTTMemory
from models.track_ttt_action import TrackActionHead


def tracks(ids=(1, 2), score=0.9, dim=8):
    t = Instances((1, 1))
    t.obj_idxes = torch.tensor(ids)
    t.output_embedding = torch.randn(len(ids), dim)
    t.cache_scores = torch.full((len(ids),), score)
    t.cache_pred_boxes = torch.rand(len(ids), 4)
    t.cache_pred_refers = torch.zeros(len(ids), 1)
    t.disappear_time = torch.zeros(len(ids), dtype=torch.long)
    t.matched_gt_idxes = torch.zeros(len(ids), dtype=torch.long)
    t.iou = torch.ones(len(ids))
    return t


class TTTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.memory = TrackTTTMemory(8, 16)

    def test_current_tokens_are_written_before_the_first_read(self):
        t = tracks()
        x = t.output_embedding.clone()
        self.memory(t)
        self.assertGreater(float((t.output_embedding - x).detach().abs().sum()), 0)
        self.assertEqual(set(self.memory.states), {1, 2})
        t = tracks()
        x = t.output_embedding.clone()
        self.memory(t)
        self.assertGreater(float((t.output_embedding - x).detach().abs().sum()), 0)

    def test_inner_and_outer_gradients(self):
        self.memory(tracks())
        t = self.memory(tracks())
        t.output_embedding.square().mean().backward()
        expected = ('tokens.identity_encoder.1.weight', 'to_qkv.0.weight',
                    'fast_model.0.weight', 'reader.in_proj_weight',
                    'output_projection.weight', 'alpha', 'log_lr')
        for name in expected:
            grad = dict(self.memory.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0, name)

    def test_reordering_does_not_mix_ids(self):
        self.memory(tracks())
        before = {i: tuple(w.clone() for w in s['weights'])
                  for i, s in self.memory.states.items()}
        self.memory(tracks(ids=(2, 1), score=0.1), gt_instances=None)
        for i in before:
            self.assertEqual(self.memory.states[i]['age'], 2)
            self.assertFalse(all(torch.equal(a, b) for a, b in zip(
                before[i], self.memory.states[i]['weights'])))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.memory(tracks(ids=(1, 1)))

    def test_id_isolation_pruning_and_reset(self):
        self.memory(tracks())
        old = tuple(x.clone() for x in self.memory.states[2]['weights'])
        self.memory(tracks(ids=(2,), score=0.1))
        self.assertNotIn(1, self.memory.states)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(
            old, self.memory.states[2]['weights'])))
        self.memory.reset()
        self.assertEqual(self.memory.states, {})

    def test_inference_no_grad_and_empty_queries(self):
        self.memory.eval()
        with torch.no_grad():
            self.memory(tracks(ids=(-1,)))
            self.assertEqual(self.memory.states, {})
            self.memory(tracks())
            t = self.memory(tracks())
        self.assertFalse(t.output_embedding.requires_grad)
        self.assertTrue(all(not w.requires_grad for s in self.memory.states.values()
                            for w in s['weights']))

    def test_zero_gate_is_exact_identity(self):
        self.memory.alpha.data.zero_()
        self.memory(tracks())
        t = tracks()
        x = t.output_embedding.clone()
        self.memory(t)
        torch.testing.assert_close(t.output_embedding, x, rtol=0, atol=0)

    def test_history_delta_only_detects_accumulated_fast_weights(self):
        observations = [tracks(), tracks()]
        normal = TrackTTTMemory(8, 16, mode='normal')
        reset = TrackTTTMemory(8, 16, mode='reset')
        no_update = TrackTTTMemory(8, 16, mode='no_update')

        for memory in (normal, reset, no_update):
            for observation in observations:
                copied = tracks(dim=8)
                copied.obj_idxes = observation.obj_idxes.clone()
                copied.output_embedding = observation.output_embedding.clone()
                copied.cache_scores = observation.cache_scores.clone()
                copied.cache_pred_boxes = observation.cache_pred_boxes.clone()
                copied.cache_pred_refers = observation.cache_pred_refers.clone()
                copied.disappear_time = observation.disappear_time.clone()
                copied.matched_gt_idxes = observation.matched_gt_idxes.clone()
                copied.iou = observation.iou.clone()
                memory(copied)

        self.assertEqual(normal.stats['history_delta_count'], 2)
        self.assertGreater(normal.stats['history_delta_sum'], 0)
        self.assertEqual(reset.stats['history_delta_sum'], 0)
        self.assertEqual(no_update.stats['history_delta_sum'], 0)

    def test_low_score_observations_still_write_evidence(self):
        low = tracks(score=0.01)
        self.memory(low)
        self.assertEqual(self.memory.stats['writes'], 6)
        self.assertEqual(set(self.memory.states), {1, 2})

    def test_untrained_action_defaults_to_trust_and_is_noop(self):
        self.memory.alpha.data.zero_()
        value = tracks(ids=(1, -1))
        before = value.output_embedding.clone()
        self.memory(value)
        self.assertEqual(int(value.ttt_action[0]), TrackActionHead.TRUST)
        probabilities = value.ttt_action_probabilities.detach()
        self.assertGreater(float(probabilities[0, 0]),
                           float(probabilities[0, 1:].max()))
        self.assertEqual(int(value.ttt_candidate_index[0]), 1)
        torch.testing.assert_close(value.output_embedding, before, rtol=0, atol=0)

    def test_disabling_feature_and_actions_is_exact_identity(self):
        memory = TrackTTTMemory(8, 16, apply_feature=False, apply_actions=False)
        value = tracks(ids=(1, -1))
        before_embedding = value.output_embedding.clone()
        before_boxes = value.cache_pred_boxes.clone()
        memory(value)
        torch.testing.assert_close(value.output_embedding, before_embedding, rtol=0, atol=0)
        torch.testing.assert_close(value.cache_pred_boxes, before_boxes, rtol=0, atol=0)
        self.assertEqual(int(value.ttt_action[0]), TrackActionHead.TRUST)

    def test_training_teacher_produces_finite_structured_losses(self):
        first = tracks(ids=(1, -1))
        first.matched_gt_idxes = torch.tensor([0, -1])
        self.memory(first)
        second = tracks(ids=(1, -1))
        second.matched_gt_idxes = torch.tensor([0, -1])
        second.iou = torch.tensor([0.1, 0.0])
        second.cache_pred_boxes[1] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        gt = Instances((1, 1))
        gt.boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        self.memory(second, gt_instances=gt)
        losses = self.memory.pop_frame_losses()
        self.assertEqual(set(losses), {
            'action', 'candidate', 'identity', 'feedback', 'safe'})
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))

    def test_action_loss_balances_present_teacher_classes(self):
        first = tracks(ids=(1, 2, 3, 4, -1))
        first.matched_gt_idxes = torch.tensor([0, 1, 2, 3, -1])
        self.memory(first)
        second = tracks(ids=(1, 2, 3, 4, -1))
        second.matched_gt_idxes = torch.tensor([0, 1, 2, -1, -1])
        second.iou = torch.tensor([0.9, 0.9, 0.1, 0.0, 0.0])
        second.cache_pred_boxes[4] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        gt = Instances((1, 1))
        gt.boxes = torch.tensor([
            [0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1],
            [0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]])
        self.memory(second, gt_instances=gt)
        loss = self.memory.pop_frame_losses()['action']
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.1)

    def test_runtime_reassociation_consumes_only_selected_candidate(self):
        value = tracks(ids=(4, -1, -1))
        value.ttt_action = torch.tensor([
            TrackActionHead.REASSOCIATE, TrackActionHead.TRUST,
            TrackActionHead.TRUST])
        value.ttt_candidate_index = torch.tensor([2, -1, -1])
        self.memory.apply_runtime_actions(value)
        torch.testing.assert_close(value.obj_idxes, torch.tensor([4, -1, -2]))

    def test_tbptt_detach_still_allows_the_next_inner_update(self):
        self.memory(tracks())
        self.memory.detach_states()
        value = self.memory(tracks())
        value.output_embedding.square().mean().backward()
        self.assertTrue(torch.isfinite(self.memory.log_lr.grad).all())

    def test_feedback_uses_the_previous_frame_action(self):
        first = tracks(ids=(1,))
        self.memory(first)
        self.memory.states[1]['action'] = TrackActionHead.HOLD
        second = tracks(ids=(1,))
        self.memory(second)
        feedback = second.ttt_feedback.detach()
        self.assertEqual(float(feedback[0, TrackActionHead.HOLD]), 1.0)
        self.assertEqual(float(feedback[0, 12]), 1.0)


if __name__ == '__main__':
    unittest.main()
