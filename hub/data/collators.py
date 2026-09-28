from __future__ import annotations

from typing import Any, Callable

import torch


def _flatten_var_spatial(tokens: torch.Tensor) -> torch.Tensor:
    v, h, w = tokens.shape
    return tokens.reshape(v, h * w).reshape(-1)


class Seq2SeqCollator:
    def __init__(self, token_type: str = "phaedra") -> None:
        self.token_type = str(token_type).lower().strip()
        if self.token_type not in {"phaedra", "fsq"}:
            raise ValueError(f"Unsupported token_type for seq2seq collator: {self.token_type}")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.token_type == "phaedra":
            input_amp = torch.stack([x["input_amp"] for x in batch], dim=0)
            input_morph = torch.stack([x["input_morph"] for x in batch], dim=0)
            output_amp = torch.stack([x["output_amp"] for x in batch], dim=0)
            output_morph = torch.stack([x["output_morph"] for x in batch], dim=0)

            bsz, in_vars, _, _ = input_amp.shape
            _, out_vars, _, _ = output_amp.shape
        else:
            input_tokens = torch.stack([x["input_tokens"] for x in batch], dim=0)
            output_tokens = torch.stack([x["output_tokens"] for x in batch], dim=0)

            bsz, in_vars, _, _ = input_tokens.shape
            _, out_vars, _, _ = output_tokens.shape

        # Use repeat (not expand) so pinned-memory copies are valid for DataLoader workers.
        input_var_ids = torch.arange(in_vars, dtype=torch.long).unsqueeze(0).repeat(bsz, 1).contiguous()
        output_var_ids = torch.arange(out_vars, dtype=torch.long).unsqueeze(0).repeat(bsz, 1).contiguous()
        input_var_mask = torch.ones((bsz, in_vars), dtype=torch.bool)
        output_var_mask = torch.ones((bsz, out_vars), dtype=torch.bool)

        base = {
            "input_time_idx": torch.tensor([x["input_time_idx"] for x in batch], dtype=torch.long),
            "output_time_idx": torch.tensor([x["output_time_idx"] for x in batch], dtype=torch.long),
            "lead_time_idx": torch.tensor([x["lead_time_idx"] for x in batch], dtype=torch.long),
            "input_var_ids": input_var_ids,
            "output_var_ids": output_var_ids,
            "input_var_mask": input_var_mask,
            "output_var_mask": output_var_mask,
            "problem_type_id": torch.zeros((bsz,), dtype=torch.long),
            "member_idx": torch.tensor([x["member_idx"] for x in batch], dtype=torch.long),
            "var_names": [x["var_names"] for x in batch],
            "dataset_name": [x["dataset_name"] for x in batch],
        }

        if self.token_type == "phaedra":
            base.update(
                {
                    "input_amp": input_amp,
                    "input_morph": input_morph,
                    "output_amp": output_amp,
                    "output_morph": output_morph,
                }
            )
        else:
            base.update(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )

        return base


class DiffusionCollator:
    def __init__(self, pad_morph_id: int, pad_amp_id: int, variable_pad_id: int) -> None:
        self.pad_morph_id = int(pad_morph_id)
        self.pad_amp_id = int(pad_amp_id)
        self.variable_pad_id = int(variable_pad_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        bsz = len(batch)
        h = int(batch[0]["input_morph"].shape[-2])
        w = int(batch[0]["input_morph"].shape[-1])

        input_lengths = []
        target_lengths = []
        flat_in_morph = []
        flat_in_amp = []
        flat_out_morph = []
        flat_out_amp = []

        for sample in batch:
            in_morph_flat = _flatten_var_spatial(sample["input_morph"].long())
            in_amp_flat = _flatten_var_spatial(sample["input_amp"].long())
            out_morph_flat = _flatten_var_spatial(sample["output_morph"].long())
            out_amp_flat = _flatten_var_spatial(sample["output_amp"].long())

            flat_in_morph.append(in_morph_flat)
            flat_in_amp.append(in_amp_flat)
            flat_out_morph.append(out_morph_flat)
            flat_out_amp.append(out_amp_flat)

            input_lengths.append(int(in_morph_flat.numel()))
            target_lengths.append(int(out_morph_flat.numel()))

        max_input_len = max(input_lengths)
        max_target_len = max(target_lengths)
        total_len = max_input_len + max_target_len

        sequence_morph = torch.full((bsz, total_len), self.pad_morph_id, dtype=torch.long)
        sequence_amp = torch.full((bsz, total_len), self.pad_amp_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        target_nonpad_mask = torch.zeros((bsz, total_len), dtype=torch.bool)
        segment_ids = torch.zeros((bsz, total_len), dtype=torch.long)
        variable_ids = torch.full((bsz, total_len), self.variable_pad_id, dtype=torch.long)
        spatial_indices = torch.full((bsz, total_len), -1, dtype=torch.long)

        for i, sample in enumerate(batch):
            in_len = input_lengths[i]
            out_len = target_lengths[i]

            vars_per_sample = int(sample["input_morph"].shape[0])
            spatial = torch.arange(h * w, dtype=torch.long)
            var_ids_flat = torch.arange(vars_per_sample, dtype=torch.long).unsqueeze(1).expand(-1, h * w).reshape(-1)
            spatial_flat = spatial.repeat(vars_per_sample)

            sequence_morph[i, :in_len] = flat_in_morph[i]
            sequence_amp[i, :in_len] = flat_in_amp[i]
            variable_ids[i, :in_len] = var_ids_flat
            spatial_indices[i, :in_len] = spatial_flat
            attention_mask[i, :in_len] = True

            out_start = max_input_len
            out_end = max_input_len + out_len
            sequence_morph[i, out_start:out_end] = flat_out_morph[i]
            sequence_amp[i, out_start:out_end] = flat_out_amp[i]
            variable_ids[i, out_start:out_end] = var_ids_flat
            spatial_indices[i, out_start:out_end] = spatial_flat
            segment_ids[i, out_start: max_input_len + max_target_len] = 1
            attention_mask[i, out_start:out_end] = True
            target_nonpad_mask[i, out_start:out_end] = True

        return {
            "sequence_morph": sequence_morph,
            "sequence_amp": sequence_amp,
            "attention_mask": attention_mask,
            "target_nonpad_mask": target_nonpad_mask,
            "segment_ids": segment_ids,
            "variable_ids": variable_ids,
            "spatial_indices": spatial_indices,
            "lead_time": torch.tensor([x["lead_time_idx"] for x in batch], dtype=torch.long),
            "problem_type_id": torch.zeros((bsz,), dtype=torch.long),
            "input_time_idx": torch.tensor([x["input_time_idx"] for x in batch], dtype=torch.long),
            "output_time_idx": torch.tensor([x["output_time_idx"] for x in batch], dtype=torch.long),
            "member_idx": torch.tensor([x["member_idx"] for x in batch], dtype=torch.long),
            "target_amp_grid": torch.stack([x["output_amp"] for x in batch], dim=0),
            "target_morph_grid": torch.stack([x["output_morph"] for x in batch], dim=0),
            "target_start": torch.tensor(max_input_len, dtype=torch.long),
            "grid_height": torch.tensor(h, dtype=torch.long),
            "grid_width": torch.tensor(w, dtype=torch.long),
            "var_names": [x["var_names"] for x in batch],
            "dataset_name": [x["dataset_name"] for x in batch],
        }


class HybridCollator:
    def __init__(self, max_variables: int, variable_to_id: dict[str, int]) -> None:
        self.max_variables = int(max_variables)
        self.variable_to_id = variable_to_id

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        max_h = max(int(s["input_amp"].shape[1]) for s in samples)
        max_w = max(int(s["input_amp"].shape[2]) for s in samples)
        bsz = len(samples)

        source_amp = torch.zeros((bsz, self.max_variables, max_h, max_w), dtype=torch.long)
        source_morph = torch.zeros((bsz, self.max_variables, max_h, max_w), dtype=torch.long)
        target_amp = torch.zeros((bsz, self.max_variables, max_h, max_w), dtype=torch.long)
        target_morph = torch.zeros((bsz, self.max_variables, max_h, max_w), dtype=torch.long)
        active_var_mask = torch.zeros((bsz, self.max_variables), dtype=torch.bool)
        slot_var_ids = torch.zeros((bsz, self.max_variables), dtype=torch.long)
        spatial_mask = torch.zeros((bsz, max_h, max_w), dtype=torch.bool)

        for i, sample in enumerate(samples):
            vars_local, h, w = sample["input_amp"].shape
            names = list(sample["var_names"])
            for local_idx, name in enumerate(names):
                slot = int(self.variable_to_id[name])
                source_amp[i, slot, :h, :w] = sample["input_amp"][local_idx]
                source_morph[i, slot, :h, :w] = sample["input_morph"][local_idx]
                target_amp[i, slot, :h, :w] = sample["output_amp"][local_idx]
                target_morph[i, slot, :h, :w] = sample["output_morph"][local_idx]
                active_var_mask[i, slot] = True
                slot_var_ids[i, slot] = slot + 1
            spatial_mask[i, :h, :w] = True

        return {
            "source_amp": source_amp,
            "source_morph": source_morph,
            "target_amp": target_amp,
            "target_morph": target_morph,
            "active_var_mask": active_var_mask,
            "slot_var_ids": slot_var_ids,
            "spatial_mask": spatial_mask,
            "lead_time_idx": torch.tensor([s["lead_time_idx"] for s in samples], dtype=torch.long),
            "input_time_idx": torch.tensor([s["input_time_idx"] for s in samples], dtype=torch.long),
            "output_time_idx": torch.tensor([s["output_time_idx"] for s in samples], dtype=torch.long),
            "member_idx": torch.tensor([s["member_idx"] for s in samples], dtype=torch.long),
            "var_names": [s["var_names"] for s in samples],
            "dataset_name": [s["dataset_name"] for s in samples],
        }


def build_collator(model_type: str, dataset_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    kind = str(model_type)
    if kind == "seq2seq":
        return Seq2SeqCollator(token_type=str(dataset_cfg.get("token_type", "phaedra")))
    if kind == "diffusion":
        pad_morph_id = int(dataset_cfg["morph_vocab_size"]) + 1
        pad_amp_id = int(dataset_cfg["amp_vocab_size"]) + 1
        return DiffusionCollator(
            pad_morph_id=pad_morph_id,
            pad_amp_id=pad_amp_id,
            variable_pad_id=int(len(dataset_cfg["variable_order"])),
        )
    if kind == "hybrid":
        variable_to_id = {name: idx for idx, name in enumerate(dataset_cfg["variable_order"])}
        return HybridCollator(max_variables=int(model_cfg.get("max_variables", 4)), variable_to_id=variable_to_id)
    raise ValueError(f"Unsupported model_type for collator: {kind}")
