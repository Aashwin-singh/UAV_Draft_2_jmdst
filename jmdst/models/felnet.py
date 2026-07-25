"""FELNet: Feature Encoding and Location Network (paper Sec. 2.2, Table 1).

A channel- and depth-reduced Darknet-53 derivative that consumes a 64x64
small-sized image (SSI) and predicts, for each of 4 anchor boxes arranged on
a 2x2 grid (each anchor covering a 32x32 region of the SSI):

    o  -- overlap vector (4 values, paper Eq. 1) locating the target
    E  -- embedding feature (16-D by default) for appearance matching
    C  -- confidence that this anchor is responsible for a target

Backbone, exactly as tabulated in PROJECT_CONTEXT.md A.3 / paper Table 1:

    Stem         Conv 16 3x3/1, Conv 32 3x3/2            64x64 -> 32x32
    1x block     Conv 16 3x3/1, Conv 32 3x3/1, residual  32x32
    Downsample   Conv 64 3x3/2                           16x16
    2x block     Conv 32 3x3/1, Conv 64 3x3/1, residual  16x16
    Downsample   Conv 128 3x3/2                          8x8
    6x block     Conv 64 3x3/1, Conv 128 3x3/1, residual 8x8
    Downsample   Conv 128 3x3/2                          4x4
    6x block     Conv 128 3x3/1, Conv 128 3x3/1, resid.  4x4
    Downsample   Conv 128 3x3/2                          2x2
    4x block     Conv 128 3x3/1, Conv 128 3x3/1, resid.  2x2

Three fully-convolutional heads (replacing Darknet's FC layer), each a stack
of 1x1 convolutions 128 -> 64 -> 32 -> 16 -> out_channels.

Note on faithfulness: the paper's Table 1 lists 3x3 kernels for *both* convs
inside each residual block (standard Darknet-53 uses 1x1 then 3x3). We follow
the paper's table. The paper does not specify the normalization, activation,
or output activations; we use Darknet-53's conventional BatchNorm +
LeakyReLU(0.1), a sigmoid on confidence (labels are 0/1 and inference
thresholds at >0.9), and optional L2-normalized embeddings so that the
paper's cross-correlation similarity is bounded in [-1, 1] to match its
T(i, j) = +/-1 targets (paper Eq. 5/6).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class FELNetConfig:
    """Configuration for FELNet.

    Defaults reproduce the paper. ``embedding_dim`` is the paper's 16-D
    feature length (Table 7); ``grid_size`` 2 yields the paper's four
    32x32 anchors over a 64x64 SSI.
    """

    input_size: int = 64
    grid_size: int = 2
    embedding_dim: int = 16
    in_channels: int = 3
    negative_slope: float = 0.1
    normalize_embedding: bool = True

    @property
    def num_anchors(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def anchor_size(self) -> float:
        return self.input_size / self.grid_size


class ConvUnit(nn.Sequential):
    """Darknet convolution unit: Conv -> BatchNorm -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope, inplace=True),
        )


class ResidualBlock(nn.Module):
    """Darknet-style residual block with the paper's two 3x3 convolutions.

    ``channels`` in and out, squeezed to ``hidden_channels`` in between, with
    an identity shortcut around both convolutions.
    """

    def __init__(self, channels: int, hidden_channels: int, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.conv1 = ConvUnit(channels, hidden_channels, 3, 1, negative_slope)
        self.conv2 = ConvUnit(hidden_channels, channels, 3, 1, negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))


class PredictionHead(nn.Sequential):
    """Fully-convolutional 1x1 head: 128 -> 64 -> 32 -> 16 -> out_channels."""

    def __init__(self, in_channels: int, out_channels: int, negative_slope: float = 0.1) -> None:
        super().__init__(
            ConvUnit(in_channels, 64, 1, 1, negative_slope),
            ConvUnit(64, 32, 1, 1, negative_slope),
            ConvUnit(32, 16, 1, 1, negative_slope),
            nn.Conv2d(16, out_channels, kernel_size=1, stride=1),
        )


class FELNet(nn.Module):
    """Feature Encoding and Location Network (paper Table 1).

    Forward returns a dict of anchor-flattened predictions:

        ``overlap``    (B, num_anchors, 4)  -- raw overlap vectors, SSI pixels
        ``embedding``  (B, num_anchors, embedding_dim)
        ``confidence`` (B, num_anchors)     -- sigmoid-activated, in [0, 1]

    Anchors are flattened row-major (index = row * grid_size + col), matching
    the ordering produced by ``jmdst.data.crops.anchor_boxes`` so predictions
    line up with the targets from ``make_felnet_targets`` without reordering.
    """

    def __init__(self, config: FELNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or FELNetConfig()
        slope = self.config.negative_slope

        self.backbone = nn.Sequential(
            # Stem: 64x64 -> 32x32
            ConvUnit(self.config.in_channels, 16, 3, 1, slope),
            ConvUnit(16, 32, 3, 2, slope),
            # 1x residual block @ 32x32
            ResidualBlock(32, 16, slope),
            # -> 16x16
            ConvUnit(32, 64, 3, 2, slope),
            *[ResidualBlock(64, 32, slope) for _ in range(2)],
            # -> 8x8
            ConvUnit(64, 128, 3, 2, slope),
            *[ResidualBlock(128, 64, slope) for _ in range(6)],
            # -> 4x4
            ConvUnit(128, 128, 3, 2, slope),
            *[ResidualBlock(128, 128, slope) for _ in range(6)],
            # -> 2x2
            ConvUnit(128, 128, 3, 2, slope),
            *[ResidualBlock(128, 128, slope) for _ in range(4)],
        )

        self.overlap_head = PredictionHead(128, 4, slope)
        self.embedding_head = PredictionHead(128, self.config.embedding_dim, slope)
        self.confidence_head = PredictionHead(128, 1, slope)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.backbone(x)

        overlap = self.overlap_head(features)
        embedding = self.embedding_head(features)
        confidence = torch.sigmoid(self.confidence_head(features))

        batch = x.shape[0]
        # (B, C, H, W) -> (B, H*W, C); H*W flattens row-major to match
        # jmdst.data.crops.anchor_boxes ordering.
        overlap = overlap.flatten(2).transpose(1, 2)
        embedding = embedding.flatten(2).transpose(1, 2)
        confidence = confidence.flatten(2).transpose(1, 2).reshape(batch, -1)

        if self.config.normalize_embedding:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return {"overlap": overlap, "embedding": embedding, "confidence": confidence}


def decode_boxes(overlap: Tensor, anchors_xywh: Tensor) -> Tensor:
    """Convert predicted overlap vectors to boxes via the paper's Eq. 2.

    Args:
        overlap: (..., 4) overlap vectors [o1, o2, o3, o4] in SSI pixels.
        anchors_xywh: (..., 4) anchor boxes [l1, u1, w1, h1] in SSI pixels,
            broadcastable against ``overlap``.

    Returns:
        (..., 4) boxes as [left, top, width, height] in SSI pixel coordinates.
    """

    l1, u1, w1, h1 = anchors_xywh.unbind(-1)
    o1, o2, o3, o4 = overlap.unbind(-1)
    return torch.stack(
        (
            l1 + w1 - o1,
            u1 + h1 - o3,
            o1 + o2 - w1,
            o3 + o4 - h1,
        ),
        dim=-1,
    )


def select_anchor_output(
    overlap: Tensor,
    confidence: Tensor,
    reference_overlap: Tensor,
    confidence_threshold: float = 0.9,
) -> Tensor:
    """Pick the best anchor per sample, following the paper's Sec. 2.2 rule.

    "the predicted bounding box is converted into an overlap vector relative
    to the cropped image. Then, the Euclidean distance is computed between
    this overlap vector and each of the four output overlap vectors predicted
    by FELNet. Among the output sets with confidence greater than 0.9, the one
    with the smallest Euclidean distance is selected."

    Args:
        overlap: (B, A, 4) predicted overlap vectors.
        confidence: (B, A) predicted confidences in [0, 1].
        reference_overlap: (B, 4) overlap vector of the Kalman-predicted box
            relative to the SSI crop.
        confidence_threshold: paper's 0.9 floor.

    Returns:
        (B,) long tensor of selected anchor indices. If no anchor for a sample
        clears the threshold, that sample falls back to its highest-confidence
        anchor, so a selection is always defined.
    """

    distances = torch.linalg.vector_norm(overlap - reference_overlap.unsqueeze(1), dim=-1)

    eligible = confidence > confidence_threshold
    masked = distances.masked_fill(~eligible, float("inf"))
    selected = masked.argmin(dim=1)

    # Samples with no eligible anchor fall back to highest confidence.
    none_eligible = ~eligible.any(dim=1)
    if bool(none_eligible.any()):
        selected = torch.where(none_eligible, confidence.argmax(dim=1), selected)
    return selected
