from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PruneVidConfig:
    """Configuration shared by PruneVid backbone adapters."""

    tau: float = 0.8
    temporal_segment_ratio: float = 0.25
    cluster_ratio: float = 0.5
    alpha: float = 0.4
    selected_layer: int = 10
    num_frames: int = 16
    pooling_shape: Tuple[int, int, int] = (16, 12, 12)
    pad_token_id: int | None = None
    head: int = 0
    softmax: float = 1.0
