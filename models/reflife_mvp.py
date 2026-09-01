import torch
import torch.nn.functional as F
from torch import nn


FEEDBACK_UNCERTAIN = 0
FEEDBACK_CONSISTENT = 1
FEEDBACK_CONTRADICTORY = 2
LIFECYCLE_ACTIVE = 0
LIFECYCLE_DORMANT = 1
LIFECYCLE_DELETED = 2
REFERENCE_OBSERVATION = 0
REFERENCE_PREDICTION = 1
REFERENCE_FROZEN = 2


class LongIdentityMemory(nn.Module):
    """Maintain one slowly updated identity anchor for each track query."""

    def __init__(self, momentum=0.95):
        super().__init__()

        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"momentum must be in [0, 1), but got {momentum}"
            )

        self.momentum = momentum

    @torch.no_grad()
    def forward(self, track_instances, write_mask):
        """Update long-term identity memory on reliable observations only.

        Args:
            track_instances:
                TempRMOT Instances containing output_embedding and
                the RefLife persistent fields.

            write_mask:
                Boolean tensor of shape [N]. True means that the current
                observation is reliable enough to update identity memory.

        Returns:
            The same track_instances object with updated memory fields.
        """
        if len(track_instances) == 0:
            return track_instances

        if write_mask.ndim != 1:
            raise ValueError(
                "write_mask must be a one-dimensional tensor"
            )

        if len(write_mask) != len(track_instances):
            raise ValueError(
                "write_mask and track_instances must have the same length"
            )

        valid_before_update = (
            track_instances.long_memory_valid.clone()
        )

        write_mask = write_mask.to(
            device=track_instances.output_embedding.device,
            dtype=torch.bool,
        )

        current_embedding = F.normalize(
            track_instances.output_embedding.detach(),
            p=2,
            dim=-1,
        )

        # A valid memory becomes one frame older before this frame is written.
        track_instances.long_memory_age[
            valid_before_update
        ] += 1

        initialize_mask = write_mask & ~valid_before_update
        update_mask = write_mask & valid_before_update

        # The first reliable observation initializes the identity anchor.
        if initialize_mask.any():
            track_instances.long_memory[
                initialize_mask
            ] = current_embedding[initialize_mask]

        # Later reliable observations update it slowly using EMA.
        if update_mask.any():
            previous_memory = track_instances.long_memory[
                update_mask
            ]

            updated_memory = (
                self.momentum * previous_memory
                + (1.0 - self.momentum)
                * current_embedding[update_mask]
            )

            track_instances.long_memory[
                update_mask
            ] = F.normalize(
                updated_memory,
                p=2,
                dim=-1,
            )

        # Only successfully written memories become valid and age zero.
        track_instances.long_memory_valid[
            write_mask
        ] = True

        track_instances.long_memory_age[
            write_mask
        ] = 0

        return track_instances


class ReliableObservationGate(nn.Module):
    """Select observations that are safe to write into long memory."""

    def __init__(
        self,
        object_threshold=0.6,
        refer_threshold=0.5,
    ):
        super().__init__()

        if not 0.0 <= object_threshold <= 1.0:
            raise ValueError(
                "object_threshold must be in [0, 1], "
                f"but got {object_threshold}"
            )

        if not 0.0 <= refer_threshold <= 1.0:
            raise ValueError(
                "refer_threshold must be in [0, 1], "
                f"but got {refer_threshold}"
            )

        self.object_threshold = object_threshold
        self.refer_threshold = refer_threshold

    @torch.no_grad()
    def forward(self, track_instances):
        """Return a Boolean mask for reliable current observations."""
        if len(track_instances) == 0:
            return torch.zeros(
                (0,),
                dtype=torch.bool,
                device=track_instances.scores.device,
            )

        object_probability = (
            track_instances.scores.detach()
        )

        refer_probability = (
            track_instances.pred_refers
            .detach()
            .sigmoid()
            .squeeze(-1)
        )

        if object_probability.ndim != 1:
            raise ValueError(
                "track_instances.scores must have shape [N]"
            )

        if refer_probability.ndim != 1:
            raise ValueError(
                "track_instances.pred_refers must have shape [N, 1]"
            )

        finite_mask = (
            torch.isfinite(object_probability)
            & torch.isfinite(refer_probability)
            & torch.isfinite(
                track_instances.pred_boxes
            ).all(dim=-1)
            & torch.isfinite(
                track_instances.output_embedding
            ).all(dim=-1)
        )

        reliable_mask = (
            finite_mask
            & (
                object_probability
                >= self.object_threshold
            )
            & (
                refer_probability
                >= self.refer_threshold
            )
        )

        return reliable_mask


class PredictionFeedback(nn.Module):
    """Compare the previous one-step prediction with current evidence."""

    def __init__(
        self,
        box_threshold=0.15,
        identity_threshold=0.5,
    ):
        super().__init__()
        if box_threshold < 0.0:
            raise ValueError("box_threshold must be non-negative")
        if not -1.0 <= identity_threshold <= 1.0:
            raise ValueError("identity_threshold must be in [-1, 1]")
        self.box_threshold = box_threshold
        self.identity_threshold = identity_threshold

    @torch.no_grad()
    def forward(self, track_instances, observation_mask):
        if len(track_instances) == 0:
            return track_instances

        observation_mask = observation_mask.to(
            device=track_instances.pred_boxes.device,
            dtype=torch.bool,
        )
        if observation_mask.ndim != 1 or len(observation_mask) != len(track_instances):
            raise ValueError("observation_mask must have shape [N]")

        track_instances.box_residual.zero_()
        track_instances.identity_similarity.zero_()
        track_instances.feedback_valid.zero_()
        track_instances.feedback_state.fill_(FEEDBACK_UNCERTAIN)

        finite_mask = (
            torch.isfinite(track_instances.pred_boxes).all(dim=-1)
            & torch.isfinite(track_instances.next_box_prediction).all(dim=-1)
            & torch.isfinite(track_instances.output_embedding).all(dim=-1)
            & torch.isfinite(track_instances.long_memory).all(dim=-1)
        )
        valid_mask = (
            observation_mask
            & track_instances.prediction_valid
            & track_instances.long_memory_valid
            & finite_mask
        )
        if not valid_mask.any():
            return track_instances

        box_residual = torch.abs(
            track_instances.pred_boxes[valid_mask]
            - track_instances.next_box_prediction[valid_mask]
        ).mean(dim=-1)
        identity_similarity = F.cosine_similarity(
            track_instances.output_embedding[valid_mask],
            track_instances.long_memory[valid_mask],
            dim=-1,
        )

        track_instances.box_residual[valid_mask] = box_residual
        track_instances.identity_similarity[valid_mask] = identity_similarity
        track_instances.feedback_valid[valid_mask] = True

        consistent = (
            (box_residual <= self.box_threshold)
            & (identity_similarity >= self.identity_threshold)
        )
        contradictory = (
            (box_residual > self.box_threshold)
            & (identity_similarity < self.identity_threshold)
        )
        valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
        track_instances.feedback_state[valid_indices[consistent]] = FEEDBACK_CONSISTENT
        track_instances.feedback_state[valid_indices[contradictory]] = FEEDBACK_CONTRADICTORY
        return track_instances


class ReliableGeometry(nn.Module):
    """Keep the latest reliable box and its one-step velocity."""

    @torch.no_grad()
    def forward(self, track_instances, update_mask):
        if len(track_instances) == 0:
            return track_instances

        update_mask = update_mask.to(
            device=track_instances.pred_boxes.device,
            dtype=torch.bool,
        )
        if update_mask.ndim != 1 or len(update_mask) != len(track_instances):
            raise ValueError("update_mask must have shape [N]")
        update_mask &= torch.isfinite(track_instances.pred_boxes).all(dim=-1)

        initialized = update_mask & track_instances.reliable_state_valid
        first_update = update_mask & ~track_instances.reliable_state_valid
        track_instances.velocity[first_update] = 0.0
        track_instances.velocity[initialized] = (
            track_instances.pred_boxes[initialized]
            - track_instances.last_reliable_box[initialized]
        )
        track_instances.last_reliable_box[update_mask] = track_instances.pred_boxes[update_mask]
        track_instances.reliable_state_valid[update_mask] = True
        return track_instances


class FuturePredictor(nn.Module):
    """Predict the next box, presence, and uncertainty for each track."""

    def __init__(self, feature_dim=256, hidden_dim=256):
        super().__init__()
        input_dim = 2 * feature_dim + 12
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.box_head = nn.Linear(hidden_dim, 4)
        self.presence_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

        # A newly added head starts as constant-velocity prediction. This keeps
        # an old TempRMOT checkpoint numerically safe before RefLife training.
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.zeros_(self.presence_head.bias)
        nn.init.zeros_(self.uncertainty_head.weight)
        nn.init.zeros_(self.uncertainty_head.bias)

    @staticmethod
    def _decode_box(base_box, delta):
        width_height = base_box[:, 2:].clamp(min=1e-4)
        center = base_box[:, :2] + delta[:, :2] * width_height
        size = width_height * torch.exp(delta[:, 2:].clamp(min=-4.0, max=4.0))
        return torch.cat((center, size), dim=-1).clamp(min=0.0, max=1.0)

    def forward(self, track_instances, observation_mask):
        if len(track_instances) == 0:
            return track_instances

        observation_mask = observation_mask.to(
            device=track_instances.pred_boxes.device,
            dtype=torch.bool,
        )
        if observation_mask.ndim != 1 or len(observation_mask) != len(track_instances):
            raise ValueError("observation_mask must have shape [N]")

        reliable_box = torch.where(
            observation_mask[:, None],
            track_instances.pred_boxes,
            track_instances.last_reliable_box,
        )
        base_box = (reliable_box + track_instances.velocity).clamp(0.0, 1.0)
        memory = torch.where(
            track_instances.long_memory_valid[:, None],
            track_instances.long_memory,
            torch.zeros_like(track_instances.long_memory),
        )
        object_probability = track_instances.scores[:, None]
        refer_probability = track_instances.pred_refers.sigmoid()
        flags = torch.stack(
            (
                track_instances.long_memory_valid,
                track_instances.reliable_state_valid,
            ),
            dim=-1,
        ).to(dtype=base_box.dtype)
        predictor_input = torch.cat(
            (
                F.normalize(track_instances.output_embedding, p=2, dim=-1),
                memory,
                reliable_box,
                track_instances.velocity,
                object_probability,
                refer_probability,
                flags,
            ),
            dim=-1,
        )
        valid_mask = (
            (observation_mask | track_instances.reliable_state_valid)
            & torch.isfinite(predictor_input).all(dim=-1)
            & torch.isfinite(base_box).all(dim=-1)
        )

        hidden = self.encoder(predictor_input)
        next_box = self._decode_box(base_box, self.box_head(hidden))
        presence_logit = self.presence_head(hidden).squeeze(-1)
        uncertainty = F.softplus(
            self.uncertainty_head(hidden).squeeze(-1)
        ) + 1e-2

        track_instances.next_box_prediction = torch.where(
            valid_mask[:, None],
            next_box,
            torch.zeros_like(next_box),
        )
        track_instances.next_visibility_logit = torch.where(
            valid_mask,
            presence_logit,
            torch.zeros_like(presence_logit),
        )
        track_instances.prediction_uncertainty = torch.where(
            valid_mask,
            uncertainty,
            torch.zeros_like(uncertainty),
        )
        track_instances.prediction_valid = valid_mask
        return track_instances


class ReferenceController(nn.Module):
    """Choose observed, predicted, or frozen decoder reference points."""

    @torch.no_grad()
    def forward(self, track_instances):
        if len(track_instances) == 0:
            return track_instances

        active_mask = (
            (track_instances.obj_idxes >= 0)
            & (track_instances.lifecycle_state == LIFECYCLE_ACTIVE)
        )
        dormant_mask = (
            (track_instances.obj_idxes >= 0)
            & (track_instances.lifecycle_state == LIFECYCLE_DORMANT)
        )
        predicted_mask = dormant_mask & track_instances.prediction_valid
        frozen_mask = dormant_mask & ~track_instances.prediction_valid

        track_instances.ref_source[active_mask] = REFERENCE_OBSERVATION
        track_instances.ref_source[predicted_mask] = REFERENCE_PREDICTION
        track_instances.ref_source[frozen_mask] = REFERENCE_FROZEN
        if predicted_mask.any():
            predicted_center = track_instances.next_box_prediction[
                predicted_mask, :2
            ].clamp(min=1e-5, max=1.0 - 1e-5)
            track_instances.ref_pts[predicted_mask] = torch.logit(predicted_center)
        return track_instances


class CPULedger:
    """Keep detached per-track diagnostics outside the GPU graph."""

    def __init__(self):
        self.records = []
        self.frame_index = 0

    def clear(self):
        self.records.clear()
        self.frame_index = 0

    @torch.no_grad()
    def record(self, track_instances):
        selected = (
            (track_instances.obj_idxes >= 0)
            | (track_instances.lifecycle_state == LIFECYCLE_DELETED)
        ).nonzero(as_tuple=False).flatten()
        for index in selected.tolist():
            self.records.append({
                'frame': self.frame_index,
                'track_id': int(track_instances.obj_idxes[index].item()),
                'state': int(track_instances.lifecycle_state[index].item()),
                'emit': bool(track_instances.emit_mask[index].item()),
                'object_score': float(track_instances.scores[index].item()),
                'refer_score': float(
                    track_instances.pred_refers[index].sigmoid().item()
                ),
                'box': track_instances.pred_boxes[index].detach().cpu().tolist(),
                'next_box': (
                    track_instances.next_box_prediction[index]
                    .detach().cpu().tolist()
                ),
                'feedback': int(track_instances.feedback_state[index].item()),
                'reference_source': int(track_instances.ref_source[index].item()),
            })
        self.frame_index += 1
        return self.records
