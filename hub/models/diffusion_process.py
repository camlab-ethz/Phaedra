from __future__ import annotations

import math

import torch


class CosineMaskSchedule:
    def __init__(self, num_steps: int) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        self.num_steps = int(num_steps)

    def masked_fraction(self, timesteps: torch.Tensor) -> torch.Tensor:
        # t=0 -> 0 masked, t=T -> 1 masked
        t = timesteps.float().clamp(min=0.0, max=float(self.num_steps))
        phase = t / float(self.num_steps)
        return torch.sin(0.5 * math.pi * phase).clamp(min=0.0, max=1.0)

class RationalMaskSchedule:
    def __init__(self, num_steps: int) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        self.num_steps = int(num_steps)

    def masked_fraction(self, timesteps: torch.Tensor, ratio=3) -> torch.Tensor:
        # t=0 -> 0 masked, t=T -> 1 masked
        # Ratio of 1 is a linear schedule, higher ratio means less aggressive masking in early steps.
        t = timesteps.float().clamp(min=0.0, max=float(self.num_steps))
        phase = t / float(self.num_steps)
        return (phase / (phase + ratio * (1 - phase))).clamp(min=0.0, max=1.0)


class MaskedDiscreteDiffusion:
    def __init__(
        self,
        num_steps: int,
        morph_mask_id: int,
        amp_mask_id: int,
    ) -> None:
        self.num_steps = int(num_steps)
        self.morph_mask_id = int(morph_mask_id)
        self.amp_mask_id = int(amp_mask_id)
        self.schedule = CosineMaskSchedule(num_steps=num_steps)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,), device=device, dtype=torch.long)

    def forward_process(
        self,
        seq_morph: torch.Tensor,
        seq_amp: torch.Tensor,
        target_nonpad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, _ = seq_morph.shape
        device = seq_morph.device

        timesteps = self.sample_timesteps(batch_size=bsz, device=device)
        masked_frac = self.schedule.masked_fraction(timesteps).unsqueeze(1)
        
        random_u = torch.rand_like(seq_morph.float())
        supervised_mask = (random_u < masked_frac) & target_nonpad_mask

        # Ensure at least one supervised position when a sample has valid targets.
        target_counts = target_nonpad_mask.sum(dim=1)
        for i in range(bsz):
            if target_counts[i] > 0 and not supervised_mask[i].any():
                valid_idx = torch.nonzero(target_nonpad_mask[i], as_tuple=False).squeeze(1)
                choice = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=device)]
                supervised_mask[i, choice] = True

        corrupted_morph = seq_morph.clone()
        corrupted_amp = seq_amp.clone()
        corrupted_morph[supervised_mask] = self.morph_mask_id
        corrupted_amp[supervised_mask] = self.amp_mask_id

        return corrupted_morph, corrupted_amp, timesteps, supervised_mask

    @torch.no_grad()
    def generate(
        self,
        model,
        seq_morph: torch.Tensor,
        seq_amp: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        variable_ids: torch.Tensor,
        spatial_indices: torch.Tensor,
        lead_time: torch.Tensor,
        pde_type_id: torch.Tensor,
        target_nonpad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _ = seq_morph.shape
        device = seq_morph.device

        pred_morph = seq_morph.clone()
        pred_amp = seq_amp.clone()

        unknown_mask = target_nonpad_mask.clone()
        pred_morph[unknown_mask] = self.morph_mask_id
        pred_amp[unknown_mask] = self.amp_mask_id

        total_target = target_nonpad_mask.sum(dim=1)

        for tau in range(self.num_steps, 0, -1):
            timestep = torch.full((bsz,), tau, device=device, dtype=torch.long)
            morph_logits, amp_logits = model(
                morph_tokens=pred_morph,
                amp_tokens=pred_amp,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                variable_ids=variable_ids,
                spatial_indices=spatial_indices,
                lead_time=lead_time,
                pde_type_id=pde_type_id,
                diffusion_timestep=timestep,
            )

            morph_prob = torch.softmax(morph_logits, dim=-1)
            amp_prob = torch.softmax(amp_logits, dim=-1)
            morph_conf, morph_ids = morph_prob.max(dim=-1)
            amp_conf, amp_ids = amp_prob.max(dim=-1)

            confidence = 0.5 * (morph_conf + amp_conf)
            confidence = confidence.masked_fill(~unknown_mask, float("-inf"))

            next_step = torch.full((bsz,), max(0, tau - 1), device=device, dtype=torch.long)
            next_fraction = self.schedule.masked_fraction(next_step)
            desired_remaining = torch.floor(total_target.float() * next_fraction).long()
            current_remaining = unknown_mask.sum(dim=1)

            reveal_budget = (current_remaining - desired_remaining).clamp(min=0)
            if tau == 1:
                reveal_budget = current_remaining

            for i in range(bsz):
                budget = int(reveal_budget[i].item())
                if budget <= 0:
                    continue

                idx_unknown = torch.nonzero(unknown_mask[i], as_tuple=False).squeeze(1)
                if idx_unknown.numel() == 0:
                    continue
                budget = min(budget, int(idx_unknown.numel()))

                conf_vals = confidence[i, idx_unknown]
                topk_local = torch.topk(conf_vals, k=budget, dim=0).indices
                chosen_idx = idx_unknown[topk_local]

                pred_morph[i, chosen_idx] = morph_ids[i, chosen_idx]
                pred_amp[i, chosen_idx] = amp_ids[i, chosen_idx]
                unknown_mask[i, chosen_idx] = False

        return pred_morph, pred_amp
