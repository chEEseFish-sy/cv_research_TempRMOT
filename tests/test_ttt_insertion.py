import unittest

import torch
from torch import nn

from models.spatial_temporal_reason import SpatialTemporalReasoner
from models.structures import Instances
from models.transrmot_pro import RuntimeTrackerBase


class OrderedReasoner(SpatialTemporalReasoner):
    def __init__(self):
        nn.Module.__init__(self)
        self.history_reasoning = True
        self.events = []

    def frame_shift(self, tracks):
        self.events.append('shift')
        return tracks

    def forward_history_reasoning(self, tracks):
        self.events.append('reason')
        tracks.output_embedding = tracks.output_embedding + 1
        return tracks

    def forward_history_refine(self, tracks, training):
        self.events.append('heads')
        tracks.cache_scores = tracks.output_embedding.sum(-1)
        return tracks


class TTTInsertionTests(unittest.TestCase):
    def test_adapter_runs_after_reasoning_before_heads(self):
        reasoner = OrderedReasoner()
        tracks = Instances((1, 1))
        tracks.output_embedding = torch.zeros(1, 2)

        def adapter(value):
            reasoner.events.append('ttt')
            value.output_embedding = value.output_embedding * 3
            return value

        result = reasoner(tracks, {}, feature_adapter=adapter)
        self.assertEqual(reasoner.events, ['shift', 'reason', 'ttt', 'heads'])
        torch.testing.assert_close(result.cache_scores, torch.tensor([6.0]))

    def test_hold_prevents_deletion_and_terminate_forces_it(self):
        tracker = RuntimeTrackerBase(miss_tolerance=10)
        tracks = Instances((1, 1))
        tracks.obj_idxes = torch.tensor([1, 2, 3])
        tracks.scores = torch.tensor([0.1, 0.1, 0.1])
        tracks.disappear_time = torch.tensor([9, 9, 0])
        tracks.ttt_action = torch.tensor([2, 0, 3])
        tracker.update(tracks)
        torch.testing.assert_close(tracks.obj_idxes, torch.tensor([1, -1, -1]))


if __name__ == '__main__':
    unittest.main()
