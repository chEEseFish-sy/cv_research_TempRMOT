"""Structured, track-private test-time-training memory."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from util import box_ops
from .track_ttt_action import TrackActionHead, TrackTokenBuilder


class TrackTTTMemory(nn.Module):
    """Store long track context in private two-layer fast weights."""

    def __init__(self, dim=256, hidden_dim=512, identity_dim=128,
                 inner_lr=0.01, mode='normal', residual_limit=0.2,
                 apply_feature=True, apply_actions=True):
        super().__init__()
        if mode not in ('normal', 'reset', 'no_update'):
            raise ValueError('Unknown TTT mode: %s' % mode)
        self.dim = dim
        self.identity_dim = identity_dim
        self.mode = mode
        self.residual_limit = residual_limit
        self.apply_feature = apply_feature
        self.apply_actions = apply_actions
        self.tokens = TrackTokenBuilder(dim, identity_dim)
        self.to_qkv = nn.ModuleList([nn.Linear(dim, dim * 3) for _ in range(3)])
        self.fast_model = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
        self.reader = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.output_projection = nn.Linear(dim, dim)
        self.action_head = TrackActionHead(dim)
        self.alpha = nn.Parameter(torch.full((dim,), 0.001))
        initial_lr = math.log(math.expm1(inner_lr))
        self.log_lr = nn.Parameter(torch.full((3,), initial_lr))
        self.reset()

    def reset(self):
        self.states = {}
        self.frame_losses = {}
        self.stats = dict(
            writes=0, reads=0, frames=0, residual_sum=0.0,
            inner_loss_sum=0.0, max_tracks=0, action_counts=[0, 0, 0, 0],
            teacher_action_counts=[0, 0, 0, 0],
            reassociate_candidates=0, history_delta_sum=0.0,
            history_delta_count=0, history_delta_max=0.0,
            action_probability_sum=[0.0, 0.0, 0.0, 0.0],
            action_probability_count=0)

    def detach_states(self):
        for state in self.states.values():
            state['weights'] = tuple(value.detach() for value in state['weights'])
            for key in ('identity', 'box', 'object_score', 'refer_score',
                        'action_confidence'):
                state[key] = state[key].detach()

    def prune(self, ids):
        live = set(int(i) for i in ids.detach().cpu().tolist() if i >= 0)
        self.states = {i: state for i, state in self.states.items() if i in live}

    @staticmethod
    def apply_fast(weights, x):
        """Apply batched MLP weights to ``[tracks, tokens, dim]`` inputs."""
        w1, b1, w2, b2 = weights
        single_token = x.ndim == 2
        if single_token:
            x = x[:, None]
        hidden = torch.einsum('nhd,ntd->nth', w1, x) + b1[:, None]
        hidden = F.gelu(hidden)
        output = torch.einsum('ndh,nth->ntd', w2, hidden) + b2[:, None]
        return output[:, 0] if single_token else output

    def _initial_weights(self, count):
        return tuple(parameter.unsqueeze(0).expand(count, *parameter.shape)
                     for parameter in self.fast_model.parameters())

    def _previous(self, ids, embedding, boxes, object_score, refer_score):
        initial = tuple(self.fast_model.parameters())
        states = [self.states.get(track_id) for track_id in ids]
        history = torch.tensor([state is not None for state in states],
                               dtype=torch.bool, device=embedding.device)
        weights = [state['weights'] if state is not None else initial
                   for state in states]
        if self.mode == 'reset':
            weights = [initial for _ in states]
        stacked = tuple(torch.stack([state[j] for state in weights])
                        for j in range(4))
        # TBPTT detaches old graphs, but the next frame must still be able to
        # perform a fresh local fast-weight gradient step.
        stacked = tuple(value if value.requires_grad else value.requires_grad_()
                        for value in stacked)
        zeros_identity = embedding.new_zeros((len(ids), self.identity_dim))
        previous = dict(
            identity=torch.stack([
                state['identity'] if state is not None else zeros_identity[row]
                for row, state in enumerate(states)]),
            box=torch.stack([
                state['box'] if state is not None else boxes[row]
                for row, state in enumerate(states)]),
            object_score=torch.stack([
                state['object_score'] if state is not None else object_score[row]
                for row, state in enumerate(states)]),
            refer_score=torch.stack([
                state['refer_score'] if state is not None else refer_score[row]
                for row, state in enumerate(states)]),
            action=torch.tensor([
                state['action'] if state is not None else TrackActionHead.TRUST
                for state in states], dtype=torch.long, device=embedding.device),
            action_confidence=torch.stack([
                state['action_confidence'] if state is not None
                else object_score.new_tensor(1.0)
                for state in states]),
            age=torch.tensor([
                state['age'] if state is not None else 0 for state in states],
                dtype=torch.long, device=embedding.device),
        )
        return stacked, previous, history

    @staticmethod
    def _pairwise_iou(boxes1, boxes2):
        if not len(boxes1) or not len(boxes2):
            return boxes1.new_zeros((len(boxes1), len(boxes2)))
        boxes1 = box_ops.box_cxcywh_to_xyxy(boxes1)
        boxes2 = box_ops.box_cxcywh_to_xyxy(boxes2)
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
            boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
            boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        size = (right_bottom - left_top).clamp(min=0)
        intersection = size[..., 0] * size[..., 1]
        return intersection / (area1[:, None] + area2[None] - intersection + 1e-6)

    def _training_losses(self, tracks, rows, history, identity, previous,
                         result, gt_instances):
        if gt_instances is None or not history.any():
            self.frame_losses = {}
            return
        matched = tracks.matched_gt_idxes[rows]
        current_iou = tracks.iou[rows].clamp(0, 1)
        candidate_mask = tracks.obj_idxes < 0
        candidate_boxes = tracks.cache_pred_boxes[candidate_mask]
        target_action = torch.full_like(matched, TrackActionHead.TRUST)
        best_candidate = torch.zeros_like(current_iou)
        candidate_target = torch.full_like(matched, -1)
        visible = matched >= 0
        if candidate_mask.any() and visible.any():
            gt_boxes = gt_instances.boxes[matched[visible]]
            candidate_iou = self._pairwise_iou(gt_boxes, candidate_boxes)
            best_value, best_index = candidate_iou.max(dim=1)
            best_candidate[visible] = best_value
            candidate_target[visible] = best_index
            choose_candidate = visible & (best_candidate > current_iou)
            target_action[choose_candidate] = TrackActionHead.REASSOCIATE
        missing = ~visible
        target_action[missing] = TrackActionHead.HOLD
        target_action[missing & (tracks.disappear_time[rows] >= 9)] = (
            TrackActionHead.TERMINATE)
        supervise = history
        with torch.no_grad():
            counts = torch.bincount(target_action[supervise], minlength=4).cpu().tolist()
            self.stats['teacher_action_counts'] = [a + b for a, b in zip(
                self.stats['teacher_action_counts'], counts)]
        supervised_logits = result['logits'][supervise]
        supervised_target = target_action[supervise]
        action_ce = F.cross_entropy(
            supervised_logits, supervised_target, reduction='none')
        target_probability = supervised_logits.softmax(dim=-1).gather(
            1, supervised_target[:, None]).squeeze(1)
        focal_loss = (1 - target_probability).square() * action_ce
        # Average within each present teacher class before averaging classes.
        # This prevents easy TRUST frames from numerically drowning the rarer
        # REASSOCIATE/HOLD supervision, without hand-tuned class thresholds.
        present_classes = supervised_target.unique()
        action_loss = torch.stack([
            focal_loss[supervised_target == action].mean()
            for action in present_classes]).mean()

        candidate_loss = identity.sum() * 0
        candidate_supervise = supervise & visible & (best_candidate > 0)
        if candidate_supervise.any() and result['candidate_logits'].shape[1]:
            # Learn identity retrieval from every visible track, not only the
            # rare frames where a candidate already beats the track box. IoU is
            # a continuous confidence weight, so no hand-picked quality cutoff
            # is introduced.
            candidate_ce = F.cross_entropy(
                result['candidate_logits'][candidate_supervise],
                candidate_target[candidate_supervise], reduction='none')
            static_ce = F.cross_entropy(
                result['static_candidate_logits'][candidate_supervise],
                candidate_target[candidate_supervise], reduction='none')
            # Counterfactual adaptation objective: the post-write fast memory
            # should retrieve the correct identity better than the same module
            # with its initial (history-free) fast weights. Softplus supplies a
            # smooth gradient at equality and needs no manually chosen margin.
            adaptation = F.softplus(candidate_ce - static_ce.detach())
            candidate_loss = (
                (candidate_ce + adaptation)
                * best_candidate[candidate_supervise].detach()).sum()
            candidate_loss = candidate_loss / best_candidate[
                candidate_supervise].detach().sum().clamp_min(1e-6)

        history_rows = torch.where(history & visible)[0]
        identity_loss = identity.sum() * 0
        if history_rows.numel() == 1:
            row = history_rows[0]
            identity_loss = 1 - (identity[row] * previous['identity'][row]).sum()
        elif history_rows.numel() > 1:
            current = identity[history_rows]
            old = F.normalize(previous['identity'][history_rows], dim=-1, eps=1e-6)
            logits = torch.matmul(current, old.transpose(0, 1)) / 0.07
            labels = torch.arange(len(history_rows), device=current.device)
            identity_loss = F.cross_entropy(logits, labels)

        feedback_target = current_iou.detach()
        feedback_loss = F.binary_cross_entropy_with_logits(
            result['feedback_success'][supervise], feedback_target[supervise])
        trust = supervise & (target_action == TrackActionHead.TRUST)
        safe_loss = identity.sum() * 0
        if trust.any():
            safe_loss = result['residual_ratio'][trust].square().mean()
        self.frame_losses = dict(
            action=action_loss, candidate=candidate_loss,
            identity=identity_loss, feedback=feedback_loss, safe=safe_loss)

    def pop_frame_losses(self):
        losses, self.frame_losses = self.frame_losses, {}
        return losses

    def apply_runtime_actions(self, tracks):
        if not tracks.has('ttt_action'):
            return tracks
        reassociate = tracks.ttt_action == TrackActionHead.REASSOCIATE
        for row in torch.where(reassociate)[0].tolist():
            candidate = int(tracks.ttt_candidate_index[row])
            if candidate >= 0 and tracks.obj_idxes[candidate] < 0:
                tracks.obj_idxes[candidate] = -2
        return tracks

    def forward(self, tracks, gt_instances=None):
        self.prune(tracks.obj_idxes)
        rows = torch.where(tracks.obj_idxes >= 0)[0]
        self.stats['frames'] += 1
        self.frame_losses = {}
        if rows.numel() == 0:
            return tracks
        ids = tracks.obj_idxes[rows].detach().cpu().tolist()
        if len(set(ids)) != len(ids):
            raise ValueError('TTT requires unique live track IDs')
        embedding = tracks.output_embedding[rows]
        boxes = tracks.cache_pred_boxes[rows]
        object_score = tracks.cache_scores[rows]
        refer_score = tracks.cache_pred_refers[rows].sigmoid().flatten()
        values = (embedding, boxes, object_score, refer_score)
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError('Non-finite structured TTT input')

        weights, previous, history = self._previous(
            ids, embedding, boxes, object_score, refer_score)
        with torch.enable_grad():
            if not self.training:
                weights = tuple(value.detach().requires_grad_() for value in weights)
            tokens, identity, feedback = self.tokens(
                embedding, object_score, refer_score, boxes, previous, history,
                tracks.disappear_time[rows])
            qkv = [layer(tokens[:, index]).chunk(3, dim=-1)
                   for index, layer in enumerate(self.to_qkv)]
            query = torch.stack([value[0] for value in qkv], dim=1)
            key = torch.stack([value[1] for value in qkv], dim=1)
            target = torch.stack([value[2] for value in qkv], dim=1)

            with torch.no_grad():
                old_answer = self.apply_fast(
                    tuple(value.detach() for value in weights), query.detach())
                initial_answer = self.apply_fast(
                    tuple(value.detach() for value in self._initial_weights(len(ids))),
                    query.detach())
                history_delta = ((old_answer - initial_answer).flatten(1).norm(dim=-1)
                                 / initial_answer.flatten(1).norm(dim=-1).clamp_min(1e-8))

            prediction = self.apply_fast(weights, key)
            per_type_loss = (prediction - target).square().mean(dim=-1)
            per_track_loss = per_type_loss.mean(dim=-1)
            if self.mode == 'no_update':
                updated = weights
            else:
                learning_rate = F.softplus(self.log_lr)
                objective = (per_type_loss * learning_rate[None]).sum()
                gradients = torch.autograd.grad(
                    objective, weights, create_graph=self.training)
                updated = tuple(value - gradient
                                for value, gradient in zip(weights, gradients))

            memory_tokens = self.apply_fast(updated, query)
            memory, _ = self.reader(
                embedding[:, None], memory_tokens, memory_tokens)
            memory = self.output_projection(memory[:, 0])
            residual = torch.tanh(self.alpha) * memory
            residual_norm = residual.norm(dim=-1)
            limit = self.residual_limit * embedding.norm(dim=-1).clamp_min(1e-6)
            residual = residual * torch.minimum(
                torch.ones_like(limit), limit / residual_norm.clamp_min(1e-6))[:, None]
            memory_enhanced = embedding + residual if self.apply_feature else embedding

            candidate_mask = tracks.obj_idxes < 0
            action = self.action_head(
                memory_enhanced, tokens[:, 1], memory, boxes, tracks.output_embedding,
                tracks.cache_pred_boxes, tracks.cache_scores,
                tracks.cache_pred_refers.sigmoid().flatten(), candidate_mask,
                feedback)
            if self.training:
                static_tokens = self.apply_fast(
                    self._initial_weights(len(ids)), query)
                static_memory, _ = self.reader(
                    embedding[:, None], static_tokens, static_tokens)
                static_memory = self.output_projection(static_memory[:, 0])
                static_action = self.action_head(
                    memory_enhanced, tokens[:, 1], static_memory, boxes,
                    tracks.output_embedding, tracks.cache_pred_boxes,
                    tracks.cache_scores,
                    tracks.cache_pred_refers.sigmoid().flatten(),
                    candidate_mask, feedback)
                action['static_candidate_logits'] = static_action[
                    'candidate_logits']
            predicted_action = action['probabilities'].argmax(dim=-1)
            # Keep the candidate path differentiable while training. At inference,
            # however, a candidate may change the track only when REASSOCIATE is
            # the selected action; a small soft probability must not move every
            # otherwise trusted track on every frame.
            if self.training:
                candidate_strength = action['candidate_delta'].abs()
            else:
                candidate_strength = action['candidate_delta'].abs() * (
                    predicted_action == TrackActionHead.REASSOCIATE)
            candidate_residual = candidate_strength[:, None] * (
                action['candidate_context'] - memory_enhanced)
            enhanced = (memory_enhanced + candidate_residual
                        if self.apply_actions else memory_enhanced)
            blended_boxes = boxes
            if self.apply_actions:
                blended_boxes = boxes + candidate_strength[:, None] * (
                    action['candidate_box'] - boxes)
            all_boxes = tracks.cache_pred_boxes.clone()
            all_boxes[rows] = blended_boxes
            tracks.cache_pred_boxes = all_boxes
            residual_ratio = ((enhanced - embedding).norm(dim=-1)
                              / embedding.norm(dim=-1).clamp_min(1e-8))
            action['residual_ratio'] = residual_ratio
            self._training_losses(
                tracks, rows, history, identity, previous, action, gt_instances)

        if not torch.isfinite(enhanced).all():
            raise FloatingPointError('Non-finite structured TTT output')
        selected_action = predicted_action if self.apply_actions else torch.full_like(
            predicted_action, TrackActionHead.TRUST)
        selected_confidence = action['probabilities'].max(dim=-1).values
        if not self.training:
            updated = tuple(value.detach() for value in updated)
            enhanced = enhanced.detach()
            identity = identity.detach()
            selected_confidence = selected_confidence.detach()
        for row, track_id in enumerate(ids):
            self.states[track_id] = dict(
                weights=tuple(value[row] for value in updated),
                identity=identity[row], box=boxes[row],
                object_score=object_score[row], refer_score=refer_score[row],
                action=int(selected_action[row]),
                action_confidence=selected_confidence[row],
                age=int(previous['age'][row]) + 1)

        output = tracks.output_embedding.clone()
        output[rows] = enhanced
        tracks.output_embedding = output
        count = len(tracks)
        action_logits = output.new_zeros((count, 4))
        action_probabilities = output.new_zeros((count, 4))
        action_index = torch.full((count,), TrackActionHead.TRUST,
                                  dtype=torch.long, device=output.device)
        candidate_index = torch.full((count,), -1, dtype=torch.long,
                                     device=output.device)
        residual_field = output.new_zeros(count)
        feedback_field = output.new_zeros((count, TrackTokenBuilder.feedback_dim))
        identity_field = output.new_zeros((count, self.identity_dim))
        action_logits[rows] = action['logits']
        action_probabilities[rows] = action['probabilities']
        action_index[rows] = selected_action
        candidate_index[rows] = action['candidate_index']
        residual_field[rows] = residual_ratio
        feedback_field[rows] = feedback
        identity_field[rows] = identity
        tracks.ttt_action_logits = action_logits
        tracks.ttt_action_probabilities = action_probabilities
        tracks.ttt_action = action_index
        tracks.ttt_candidate_index = candidate_index
        tracks.ttt_residual_ratio = residual_field
        tracks.ttt_feedback = feedback_field
        tracks.ttt_identity = identity_field

        with torch.no_grad():
            self.stats['writes'] += len(ids) * 3 if self.mode != 'no_update' else 0
            self.stats['reads'] += len(ids) * 3
            self.stats['max_tracks'] = max(self.stats['max_tracks'], len(self.states))
            self.stats['inner_loss_sum'] += float(per_track_loss.sum())
            self.stats['residual_sum'] += float(residual_ratio.sum())
            self.stats['reassociate_candidates'] += int(
                (action['candidate_index'] >= 0).sum())
            counts = torch.bincount(selected_action, minlength=4).cpu().tolist()
            self.stats['action_counts'] = [a + b for a, b in zip(
                self.stats['action_counts'], counts)]
            probability_sum = action['probabilities'].sum(dim=0).cpu().tolist()
            self.stats['action_probability_sum'] = [a + b for a, b in zip(
                self.stats['action_probability_sum'], probability_sum)]
            self.stats['action_probability_count'] += len(ids)
            valid_delta = history_delta[history]
            self.stats['history_delta_sum'] += float(valid_delta.sum())
            self.stats['history_delta_count'] += int(valid_delta.numel())
            if valid_delta.numel():
                self.stats['history_delta_max'] = max(
                    self.stats['history_delta_max'], float(valid_delta.max()))
        return tracks
