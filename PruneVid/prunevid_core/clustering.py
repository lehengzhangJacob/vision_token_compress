import torch


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather point features by batched indices."""

    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def cluster_dpc_knn(x: torch.Tensor, cluster_num: int, k: int = 5, token_mask: torch.Tensor | None = None):
    """Density-peaks clustering with k-nearest-neighbor density."""

    with torch.no_grad():
        batch_size, token_count, channel_count = x.shape
        cluster_num = max(1, min(int(cluster_num), token_count))
        k = max(1, min(int(k), token_count))

        dist_matrix = torch.cdist(x.float(), x.float()) / (channel_count**0.5)
        if token_mask is not None:
            token_mask = token_mask > 0
            dist_matrix = dist_matrix * token_mask[:, None, :] + (dist_matrix.max() + 1) * (~token_mask[:, None, :])

        dist_nearest, _ = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
        density = (-(dist_nearest**2).mean(dim=-1)).exp()
        density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6
        if token_mask is not None:
            density = density * token_mask

        mask = (density[:, None, :] > density[:, :, None]).type(x.dtype)
        dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
        dist, _ = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

        score = dist * density
        _, index_down = torch.topk(score, k=cluster_num, dim=-1)

        dist_matrix = index_points(dist_matrix, index_down)
        idx_cluster = dist_matrix.argmin(dim=1)

        idx_batch = torch.arange(batch_size, device=x.device)[:, None].expand(batch_size, cluster_num)
        idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(batch_size, cluster_num)
        idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

    return idx_cluster, cluster_num


def segment_lengths(tensor: torch.Tensor) -> torch.Tensor:
    """Return lengths of contiguous label segments for each batch element."""

    device = tensor.device
    batch_size, token_count = tensor.shape
    segment_lengths_list = []
    max_segments = 0

    for batch_idx in range(batch_size):
        seq = tensor[batch_idx]
        change_points = torch.where(seq[1:] != seq[:-1])[0] + 1
        boundaries = torch.cat(
            [
                torch.tensor([0], device=device),
                change_points,
                torch.tensor([token_count], device=device),
            ]
        )
        lengths = boundaries[1:] - boundaries[:-1]
        segment_lengths_list.append(lengths)
        max_segments = max(max_segments, lengths.numel())

    result = torch.zeros((batch_size, max_segments), dtype=torch.long, device=device)
    for batch_idx, lengths in enumerate(segment_lengths_list):
        result[batch_idx, : lengths.numel()] = lengths
    return result


def refine_clusters(cluster_idx: torch.Tensor) -> torch.Tensor:
    """Make temporal cluster ids contiguous by reassigning short disjoint fragments."""

    batch_size, _ = cluster_idx.shape
    refined_cluster_idx = cluster_idx.clone()
    for batch_idx in range(batch_size):
        clusters = torch.unique(cluster_idx[batch_idx])
        segment_info = {}
        for cluster_label in clusters:
            indices = (cluster_idx[batch_idx] == cluster_label).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue

            segments = []
            start = indices[0].item()
            prev = indices[0].item()
            for idx in indices[1:]:
                idx = idx.item()
                if idx == prev + 1:
                    prev = idx
                else:
                    segments.append((start, prev))
                    start = idx
                    prev = idx
            segments.append((start, prev))
            segment_info[cluster_label.item()] = segments

        for cluster_label, segments in segment_info.items():
            longest = max(segments, key=lambda seg: seg[1] - seg[0] + 1)
            for start, end in segments:
                if (start, end) == longest:
                    continue
                for pos in range(start, end + 1):
                    left = refined_cluster_idx[batch_idx, pos - 1] if pos > 0 else None
                    right = refined_cluster_idx[batch_idx, pos + 1] if pos + 1 < refined_cluster_idx.shape[1] else None
                    if left is not None and left != cluster_label:
                        refined_cluster_idx[batch_idx, pos] = left
                    elif right is not None and right != cluster_label:
                        refined_cluster_idx[batch_idx, pos] = right

    return refined_cluster_idx
