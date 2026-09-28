"""Gated visibility-grid residual network of the deployed correction.

One definition for training (pipeline/fit_correction.py) and runtime
(reliability.commissioned_visibility), so the deployed model is exactly the trained one.
"""
from __future__ import annotations

import torch
from torch import nn

GRID_SIZE = 16


class VisibilityPatchResidualNet(nn.Module):
    """Add a gated visibility-grid residual to a frozen tabular correction.

    The last residual layer is initialized to zero.  Before training, and after
    loading an explicitly zero residual, the output is exactly the supplied box
    correction.  This makes degradation an observed training outcome rather than
    an architectural consequence of replacing the baseline.
    """

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature count must be positive")
        self.feature_count = int(feature_count)
        self.visibility = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.features = nn.Sequential(
            nn.Linear(self.feature_count, 32), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(24 * 4 * 4 + 32, 96), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(96, 48), nn.ReLU(),
        )
        self.residual = nn.Linear(48, 2)
        self.gate = nn.Linear(48, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        visibility_grid: torch.Tensor,
        standardized_features: torch.Tensor,
        base_correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visibility_grid.ndim != 4 or visibility_grid.shape[1:] != (1, GRID_SIZE, GRID_SIZE):
            raise ValueError("visibility input must have shape [batch,1,16,16]")
        if standardized_features.ndim != 2 or standardized_features.shape[1] != self.feature_count:
            raise ValueError("feature input has the wrong shape")
        if base_correction.ndim != 2 or base_correction.shape[1] != 2:
            raise ValueError("base correction must have shape [batch,2]")
        encoded = torch.cat(
            (self.visibility(visibility_grid), self.features(standardized_features)), dim=1
        )
        hidden = self.trunk(encoded)
        residual = self.residual(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return base_correction + gate * residual, residual, gate
