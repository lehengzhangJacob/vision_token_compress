from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import einsum

from .clustering import cluster_dpc_knn, refine_clusters, segment_lengths


class PruneVidVisionMerger:
    """Backbone-agnostic implementation of PruneVid's visual-token merge stage."""

    def __init__(
        self,
        num_frames: int,
        tokens_per_frame: int,
        tau: float = 0.8,
        temporal_segment_ratio: float = 0.25,
        cluster_ratio: float = 0.5,
        knn_k: int = 7,
    ) -> None:
        self.num_frames = int(num_frames)
        self.tokens_per_frame = int(tokens_per_frame)
        self.tau = float(tau)
        self.temporal_segment_ratio = float(temporal_segment_ratio)
        self.cluster_ratio = float(cluster_ratio)
        self.knn_k = int(knn_k)

    @staticmethod
    def compute_cluster_vectors(
        image_key_vectors: torch.Tensor,
        cluster_key_idx: torch.Tensor,
        num_cluster: int,
    ) -> torch.Tensor:
        batch_size, _, _ = image_key_vectors.shape
        cluster_key_idx_onehot = F.one_hot(cluster_key_idx, num_classes=num_cluster).to(dtype=image_key_vectors.dtype)
        cluster_sums = torch.bmm(cluster_key_idx_onehot.permute(0, 2, 1), image_key_vectors)
        cluster_counts = cluster_key_idx_onehot.sum(dim=1)
        cluster_counts_nonzero = cluster_counts.clone()
        cluster_counts_nonzero[cluster_counts_nonzero == 0] = 1
        cluster_features = cluster_sums / cluster_counts_nonzero.unsqueeze(-1)
        zero_mask = (cluster_counts == 0).unsqueeze(-1)
        return cluster_features.masked_fill(zero_mask, 0).view(batch_size, num_cluster, -1)

    def spatial_merge_tokens(self, feature: torch.Tensor, num_cluster: int, k: int | None = None) -> torch.Tensor:
        cluster_idx, num_cluster = cluster_dpc_knn(feature, cluster_num=num_cluster, k=k or self.knn_k)
        return self.compute_cluster_vectors(feature, cluster_idx, num_cluster=num_cluster)

    def merge_frames_dynamic(self, frames: torch.Tensor):
        """Merge visual features and return metadata used by the LLM attention-prune stage.

        Args:
            frames: Tensor shaped ``[B, T * tokens_per_frame, C]``.
        """

        batch_size, token_count, channel_count = frames.shape
        expected_tokens = self.num_frames * self.tokens_per_frame
        if token_count != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} visual tokens, got {token_count}")

        frames = frames.view(batch_size, self.num_frames, self.tokens_per_frame, channel_count)
        temporal_clusters = max(1, int(self.num_frames * self.temporal_segment_ratio))
        idx_clusters, _ = cluster_dpc_knn(frames.mean(dim=2), cluster_num=temporal_clusters, k=self.knn_k)
        idx_clusters = refine_clusters(idx_clusters)
        window_list = segment_lengths(idx_clusters)

        static_features = []
        dynamic_features = []
        static_sizes = []
        dynamic_sizes = []

        start_idx = 0
        for window_size in window_list[0]:
            window_size = int(window_size.item())
            if window_size <= 0:
                continue

            current_frames = frames[:, start_idx : start_idx + window_size, :, :]
            frames_normed = F.normalize(current_frames, p=2, dim=-1)
            frames_sim = einsum("b w l c, b t l c -> b w t l", frames_normed, frames_normed)
            denom = max(1, window_size * (window_size - 1))
            frames_sim = (frames_sim.sum(dim=-2) - 1).sum(dim=-2) / denom

            mask = frames_sim > self.tau
            mask_expand = mask.view(batch_size, 1, self.tokens_per_frame, 1).expand(
                -1, window_size, -1, channel_count
            )

            static_feat = torch.masked_select(current_frames, mask_expand).view(batch_size, window_size, -1, channel_count)
            static_feat = static_feat.mean(dim=1)
            if static_feat.shape[1] > 14:
                static_feat = self.spatial_merge_tokens(
                    static_feat,
                    num_cluster=max(1, int(static_feat.shape[1] * self.cluster_ratio)),
                    k=self.knn_k,
                )
            static_features.append(static_feat)
            static_sizes.append(static_feat.shape[1])

            dynamic_feat = torch.masked_select(current_frames, ~mask_expand).view(batch_size, window_size, -1, channel_count)
            dynamic_window_list = []
            for frame_idx in range(window_size):
                dynamic_feat_window = dynamic_feat[:, frame_idx, :, :]
                if dynamic_feat_window.shape[1] > 14:
                    dynamic_feat_window = self.spatial_merge_tokens(
                        dynamic_feat_window,
                        num_cluster=max(1, int(dynamic_feat_window.shape[1] * self.cluster_ratio)),
                        k=self.knn_k,
                    )
                dynamic_window_list.append(dynamic_feat_window)

            dynamic_feat = torch.cat(dynamic_window_list, dim=1)
            dynamic_features.append(dynamic_feat)
            dynamic_sizes.append(dynamic_feat.shape[1])

            start_idx += window_size

        final_features = []
        for static_feature, dynamic_feature in zip(static_features, dynamic_features):
            final_features.append(static_feature)
            final_features.append(dynamic_feature)

        return torch.cat(final_features, dim=1), static_sizes, dynamic_sizes, window_list[0].tolist()
