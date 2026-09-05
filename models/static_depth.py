"""Experimental fixed-batch depth cache; sampling and output heads stay eager."""

from __future__ import annotations

import torch
from transformers import StaticCache


class StaticDepthRunner:
    def __init__(self, model, *, compile_decode=True):
        if model.training:
            raise ValueError("Static depth requires an eval model")
        self.model = model
        config = model.config
        self.num_codebooks = int(config.num_codebooks)
        self.device, self.dtype = model.embed_tokens.weight.device, model.embed_tokens.weight.dtype
        self.backbone_hidden_size = int(config.backbone_hidden_size)
        self._next_step = None
        self.cache = StaticCache(config=config, max_cache_len=self.num_codebooks, batch_size=2)
        dummy = torch.zeros(
            (2, config.num_key_value_heads, 1, config.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        for layer in self.cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy)
        self.prefill_position = torch.arange(2, device=self.device)
        self.decode_positions = [
            torch.tensor([step + 1], device=self.device)
            for step in range(1, self.num_codebooks - 1)
        ]
        self.prefill_mask = self._mask(self.prefill_position)
        self.decode_masks = [self._mask(position) for position in self.decode_positions]
        self._decode_call = (
            torch.compile(
                self._decode, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
            )
            if compile_decode
            else self._decode
        )

    def _mask(self, position):
        allowed = torch.arange(self.num_codebooks, device=self.device)[None] <= position[:, None]
        mask = torch.zeros(allowed.shape, device=self.device, dtype=self.dtype).masked_fill(
            ~allowed, torch.finfo(self.dtype).min
        )
        return mask[None, None].expand(2, 1, -1, -1).contiguous()

    def _check(self, ids, length):
        if tuple(ids.shape) != (2, length):
            raise ValueError(f"Expected one CFG pair (2, {length})")
        if ids.dtype != torch.long or ids.device != self.device:
            raise ValueError("Static depth token dtype/device mismatch")

    def _decode(self, input_ids, cache_position, attention_mask):
        return self.model(
            input_ids=input_ids,
            backbone_last_hidden_state=None,
            past_key_values=self.cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=True,
        ).last_hidden_state

    @torch.inference_mode()
    def begin(self, input_ids, branch_hidden):
        self._check(input_ids, 2)
        if (
            tuple(branch_hidden.shape) != (2, self.backbone_hidden_size)
            or branch_hidden.device != self.device
            or branch_hidden.dtype != self.dtype
        ):
            raise ValueError("Static depth hidden-state shape/device/dtype mismatch")
        self._next_step = None
        self.cache.reset()
        hidden = self.model(
            input_ids=input_ids,
            backbone_last_hidden_state=branch_hidden,
            past_key_values=self.cache,
            cache_position=self.prefill_position,
            attention_mask=self.prefill_mask,
            use_cache=True,
        ).last_hidden_state
        self._next_step = 1
        return hidden[:, -1:, :]

    @torch.inference_mode()
    def step(self, input_ids, step):
        self._check(input_ids, 1)
        if (
            type(step) is not int
            or step != self._next_step
            or not 1 <= step < self.num_codebooks - 1
        ):
            raise ValueError("Static depth steps must follow codebook order")
        hidden = self._decode_call(
            input_ids, self.decode_positions[step - 1], self.decode_masks[step - 1]
        )
        self._next_step = step + 1
        return hidden

    @torch.inference_mode()
    def warmup(self):
        hidden = torch.zeros((2, self.backbone_hidden_size), device=self.device, dtype=self.dtype)
        self.begin(torch.zeros((2, 2), device=self.device, dtype=torch.long), hidden)
        for step in range(1, self.num_codebooks - 1):
            self.step(torch.zeros((2, 1), device=self.device, dtype=torch.long), step)
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.cache.reset()
        self._next_step = None
