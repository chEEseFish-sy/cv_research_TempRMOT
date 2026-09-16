"""Structured tokens and conservative actions for track-private TTT memory."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TrackTokenBuilder(nn.Module):
    """Build identity, evidence and delayed action-feedback tokens."""

    feedback_dim = 14

    def __init__(self, dim=256, identity_dim=128):
        super().__init__()
        self.identity_encoder = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, identity_dim))
        self.identity_token = nn.Linear(identity_dim + 2, dim)
        self.evidence_token = nn.Sequential(
            nn.Linear(dim + 12, dim), nn.GELU(), nn.Linear(dim, dim))
        self.feedback_token = nn.Sequential(
            nn.Linear(self.feedback_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.type_embedding = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.type_embedding, std=0.02)

    def encode_identity(self, embedding):
        return F.normalize(self.identity_encoder(embedding), dim=-1, eps=1e-6)

    def forward(self, embedding, object_score, refer_score, boxes,
                previous, has_history, disappear_time):
        identity = self.encode_identity(embedding)
        previous_identity = previous['identity']
        identity_cosine = (identity * previous_identity).sum(-1)
        identity_cosine = torch.where(has_history, identity_cosine,
                                      torch.zeros_like(identity_cosine))

        box_delta = boxes - previous['box']
        box_delta = torch.where(has_history[:, None], box_delta,
                                torch.zeros_like(box_delta))
        feedback = torch.cat([
            F.one_hot(previous['action'], 4).to(embedding.dtype),
            previous['action_confidence'][:, None],
            (object_score - previous['object_score'])[:, None],
            (refer_score - previous['refer_score'])[:, None],
            box_delta,
            identity_cosine[:, None],
            has_history[:, None].to(embedding.dtype),
            (previous['age'].to(embedding.dtype) / 32.0).clamp(max=1)[:, None],
        ], dim=-1)

        identity_meta = torch.stack(
            [has_history.to(embedding.dtype), identity_cosine], dim=-1)
        identity_token = self.identity_token(
            torch.cat([identity, identity_meta], dim=-1))
        evidence_scalars = torch.cat([
            object_score[:, None], refer_score[:, None], boxes, box_delta,
            has_history[:, None].to(embedding.dtype),
            (disappear_time.to(embedding.dtype) / 10.0).clamp(max=2)[:, None],
        ], dim=-1)
        evidence_token = self.evidence_token(
            torch.cat([embedding, evidence_scalars], dim=-1))
        feedback_token = self.feedback_token(feedback)
        tokens = torch.stack([identity_token, evidence_token, feedback_token], dim=1)
        tokens = tokens + self.type_embedding[None]
        return tokens, identity, feedback


class TrackActionHead(nn.Module):
    """Read detection candidates and select a conservative tracking action."""

    TRUST = 0
    REASSOCIATE = 1
    HOLD = 2
    TERMINATE = 3
    names = ('trust_track', 'reassociate_detect', 'hold_and_search', 'terminate')

    def __init__(self, dim=256):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(dim + 6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.memory_norm = nn.LayerNorm(dim)
        self.log_geometry_weight = nn.Parameter(torch.tensor(0.0))
        self.action = nn.Sequential(
            nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 4))
        self.feedback_success = nn.Sequential(
            nn.Linear(TrackTokenBuilder.feedback_dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1))
        self.candidate_alpha = nn.Parameter(torch.tensor(0.0))
        final = self.action[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        # Start as a safe no-op, but keep the four actions close enough for the
        # rare REASSOCIATE/HOLD teachers to move the classifier early.
        final.bias.data[self.TRUST] = 1.0

    def forward(self, enhanced, evidence, memory, track_boxes, all_embeddings,
                all_boxes, all_object_scores, all_refer_scores,
                candidate_mask, feedback):
        count = len(enhanced)
        candidate_index = torch.full(
            (count,), -1, dtype=torch.long, device=enhanced.device)
        candidate_context = torch.zeros_like(enhanced)
        candidate_box = torch.zeros_like(track_boxes)
        candidate_logits = enhanced.new_empty((count, 0))
        if candidate_mask.any():
            candidate_boxes = all_boxes[candidate_mask]
            candidates = self.candidate_encoder(torch.cat([
                all_embeddings[candidate_mask], candidate_boxes,
                all_object_scores[candidate_mask, None],
                all_refer_scores[candidate_mask, None]], dim=-1))
            candidate_rows = torch.where(candidate_mask)[0]
            # Candidate retrieval is the identity-memory task: ask the current
            # detect pool with the track-private fast-memory read, rather than
            # repeating the current-frame track embedding.
            candidate_logits = torch.matmul(
                self.query(self.memory_norm(memory)),
                self.key(candidates).transpose(0, 1))
            candidate_logits = candidate_logits / math.sqrt(enhanced.shape[-1])
            geometry_distance = (track_boxes[:, None] - candidate_boxes[None]).abs().mean(-1)
            candidate_logits = candidate_logits - F.softplus(
                self.log_geometry_weight) * geometry_distance
            candidate_weights = candidate_logits.softmax(dim=-1)
            candidate_context = torch.matmul(candidate_weights, self.value(candidates))
            candidate_box = torch.matmul(candidate_weights, candidate_boxes)
            candidate_index = candidate_rows[candidate_logits.argmax(dim=-1)]

        action_logits = self.action(
            torch.cat([enhanced, evidence, self.memory_norm(memory),
                       candidate_context], dim=-1))
        action_probabilities = action_logits.softmax(dim=-1)
        reassociate = action_probabilities[:, self.REASSOCIATE:self.REASSOCIATE + 1]
        candidate_delta = torch.tanh(self.candidate_alpha) * reassociate
        output = enhanced + candidate_delta * (candidate_context - enhanced)
        feedback_success = self.feedback_success(feedback).squeeze(-1)
        return dict(
            output=output,
            candidate_context=candidate_context,
            logits=action_logits,
            probabilities=action_probabilities,
            candidate_index=candidate_index,
            candidate_logits=candidate_logits,
            candidate_box=candidate_box,
            candidate_delta=candidate_delta.squeeze(-1),
            feedback_success=feedback_success,
        )
