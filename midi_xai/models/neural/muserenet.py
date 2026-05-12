from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        current_channels = in_channels
        padding = kernel_size // 2

        for _ in range(depth):
            layers.extend(
                [
                    nn.Conv1d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_channels = out_channels

        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResolutionBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int,
        block_depth: int,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        current_channels = in_channels

        for out_channels in channels:
            blocks.append(
                ConvBlock1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    depth=block_depth,
                    dropout=dropout,
                )
            )
            current_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.output_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return F.adaptive_avg_pool1d(x, output_size=1).squeeze(-1)


class MuSeReNetClassifier(nn.Module):
    """Multiple Sequence Resolution CNN"""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int] = (32, 64, 128),
        resolutions: Sequence[int] = (1, 2, 4),
        kernel_size: int = 5,
        block_depth: int = 1,
        dropout: float = 0.2,
        classifier_hidden_dim: int = 128,
    ):
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.branches = nn.ModuleList(
            [
                ResolutionBranch(
                    in_channels=input_channels,
                    channels=channels,
                    kernel_size=kernel_size,
                    block_depth=block_depth,
                    dropout=dropout,
                )
                for _ in self.resolutions
            ]
        )

        total_channels = sum(branch.output_channels for branch in self.branches)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_channels),
            nn.Linear(total_channels, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = []
        for resolution, branch in zip(self.resolutions, self.branches):
            branch_input = x
            if resolution > 1:
                branch_input = F.avg_pool1d(
                    x,
                    kernel_size=resolution,
                    stride=resolution,
                    ceil_mode=True,
                )
            branch_outputs.append(branch(branch_input))

        return self.classifier(torch.cat(branch_outputs, dim=1))
