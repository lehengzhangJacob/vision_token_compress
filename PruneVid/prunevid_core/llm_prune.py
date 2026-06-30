from __future__ import annotations

import numpy as np
import torch


class VTPWindowCache:
    """Prune visual KV-cache entries using text-to-image attention at a selected layer."""

    def __init__(
        self,
        alpha: float = 0.2,
        total_num_layers: int = 32,
        selected_layer: int = 9,
        pooling_shape=(16, 12, 12),
        num_frames: int = 16,
        pad_token_id=None,
        head: int = 0,
        softmax: float = 1.0,
    ) -> None:
        self.alpha = alpha
        self.total_num_layers = total_num_layers
        self.selected_layer = selected_layer
        self.pooling_shape = pooling_shape
        self.num_frames = num_frames
        self.pad_token_id = pad_token_id
        self.head = head
        self.softmax = softmax
        self.img_start = None
        self.img_end = None

    def process_attention(self, text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes):
        batch_size, _, _, _ = text_to_image_attentions.shape
        if batch_size != 1:
            raise ValueError("PruneVid VTP currently expects batch size 1 during generation")
        if not (len(static_sizes) == len(dynamic_sizes) == len(window_sizes)):
            raise ValueError("static_sizes, dynamic_sizes, and window_sizes must have the same length")

        text_to_image_attentions = text_to_image_attentions[0].max(dim=0)[0].max(dim=0)[0]

        start_idx = 0
        topk_indices_list = []
        for static_size, dynamic_size, window_size in zip(static_sizes, dynamic_sizes, window_sizes):
            end_idx = start_idx + static_size + dynamic_size
            window_attentions = text_to_image_attentions[start_idx:end_idx]

            static_attentions = window_attentions[:static_size]
            num_retain_static_tokens = int(static_size * self.alpha)
            if num_retain_static_tokens > 0:
                _, static_topk_indices = torch.topk(static_attentions, k=num_retain_static_tokens, dim=-1)
                topk_indices_list.append(static_topk_indices + start_idx)

            dynamic_attentions = window_attentions[static_size:].view(window_size, -1)
            num_retain_dynamic_tokens = int(dynamic_attentions.shape[-1] * self.alpha)
            if num_retain_dynamic_tokens > 0:
                _, dynamic_topk_indices = torch.topk(dynamic_attentions, k=num_retain_dynamic_tokens, dim=-1)
                dynamic_topk_indices = dynamic_topk_indices + start_idx + static_size
                dynamic_topk_indices_list = []
                for idx in range(window_size):
                    dynamic_topk_indices_list.append(dynamic_topk_indices[idx] + dynamic_attentions.shape[-1] * idx)
                topk_indices_list.append(torch.cat(dynamic_topk_indices_list, dim=0))

            start_idx = end_idx

        if not topk_indices_list:
            return torch.empty(0, dtype=torch.long, device=text_to_image_attentions.device)
        return torch.cat(topk_indices_list, dim=0)

    def obtain_language_attention(self, input_ids, attentions, pad_token, text_indices=None):
        pad_token = self.pad_token_id
        _, seq_img_indices = torch.where(input_ids == pad_token)
        img_start, img_end = seq_img_indices[0].item(), seq_img_indices[-1].item()

        text_to_image_attentions = attentions[:, :, img_end + 1 :, img_start : img_end + 1]
        image_to_image_attentions = attentions[:, :, img_start : img_end + 1, img_start : img_end + 1]
        text_to_text_attentions = attentions[:, :, img_end + 1 :, img_end + 1 :]

        return text_to_image_attentions, text_to_text_attentions, image_to_image_attentions, img_start, img_end, attentions.shape[-1]

    def prompt_prefill(
        self,
        past_key_values=None,
        input_ids=None,
        attentions=None,
        hidden_states=None,
        past_hidden_states=None,
        causal_mask=None,
        attention_mask=None,
        pad_token_id=None,
        position_ids=None,
        text_indices=None,
        attn_shallower=None,
        static_sizes=None,
        dynamic_sizes=None,
        window_sizes=None,
        decoding_flag=False,
    ):
        static_sizes = static_sizes or []
        dynamic_sizes = dynamic_sizes or []
        window_sizes = window_sizes or []
        text_to_image_attentions, _, _, img_start, img_end, seq_len = self.obtain_language_attention(
            input_ids, attentions, pad_token_id
        )

        topk_indices = self.process_attention(text_to_image_attentions, static_sizes, dynamic_sizes, window_sizes)
        index_list = (topk_indices + img_start).sort(descending=False)[0]
        self.img_start = img_start
        self.img_end = img_start + len(index_list) - 1

        index_list_pre_image = torch.arange(0, img_start, device=input_ids.device)
        index_list_post_image = torch.arange(img_end + 1, seq_len, device=input_ids.device)
        index_list = torch.cat([index_list_pre_image, index_list, index_list_post_image], dim=0)

        for layer_idx in range(self.selected_layer + 1):
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, index_list, :].contiguous()
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, index_list, :].contiguous()

        hidden_states = hidden_states[:, index_list, :]
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, index_list, :][:, :, :, index_list]
        if attention_mask is not None:
            attention_mask = attention_mask[:, index_list]

        updated_hidden_states = () if past_hidden_states is not None else None
        if updated_hidden_states is not None:
            for old_hidden_state in past_hidden_states:
                updated_hidden_states += (old_hidden_state[:, index_list, :],)

        position_ids = position_ids[:, index_list]
        cache_position = torch.arange(index_list.shape[0], device=input_ids.device)

        return past_key_values, hidden_states, updated_hidden_states, causal_mask, attention_mask, position_ids, cache_position

    def __call__(self, *args, **kwargs):
        return self.prompt_prefill(*args, **kwargs)
