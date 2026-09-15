from __future__ import annotations

import torch
from torch import nn

from stage1_relation_layers import RelationConv1D


class _UNetBranch1D(nn.Module):
    """Two-level 1D U-Net branch with length-safe skip connections."""

    def __init__(self, in_channels: int, base_width: int, out_channels: int) -> None:
        super().__init__()
        w = int(base_width)
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, w, 5, padding=2), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv1d(w, 2 * w, 5, padding=2), nn.GELU())
        self.bottleneck = nn.Sequential(nn.Conv1d(2 * w, 4 * w, 5, padding=2), nn.GELU())
        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec2 = nn.Sequential(nn.Conv1d(6 * w, 2 * w, 5, padding=2), nn.GELU())
        self.dec1 = nn.Sequential(nn.Conv1d(3 * w, w, 5, padding=2), nn.GELU())
        self.out = nn.Conv1d(w, out_channels, 1)

    @staticmethod
    def _match_length(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target.shape[-1] - x.shape[-1]
        if diff > 0:
            return nn.functional.pad(x, (0, diff))
        if diff < 0:
            return x[..., : target.shape[-1]]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        xb = self.bottleneck(self.pool(x2))
        y = self._match_length(self.up(xb), x2)
        y = self.dec2(torch.cat([y, x2], dim=1))
        y = self._match_length(self.up(y), x1)
        y = self.dec1(torch.cat([y, x1], dim=1))
        return self.out(y)


class SharedUNet1D(nn.Module):
    """One shared U-Net that jointly predicts all complex-field channels."""

    def __init__(self, in_channels: int, base_width: int = 64, out_channels: int = 2) -> None:
        super().__init__()
        self.net = _UNetBranch1D(in_channels, base_width, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualUNet1D(nn.Module):
    """MAMBA-style independent U-Nets for real and imaginary field outputs."""

    def __init__(self, in_channels: int, base_width: int = 45, out_channels: int = 2) -> None:
        super().__init__()
        if out_channels != 2:
            raise ValueError("DualUNet1D is defined for real_imag output with exactly 2 channels.")
        self.real_net = _UNetBranch1D(in_channels, base_width, 1)
        self.imag_net = _UNetBranch1D(in_channels, base_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.real_net(x), self.imag_net(x)], dim=1)


class _PaperUNetBranch1D(nn.Module):
    """Reduced 1D reproduction of the five-convolution MAMBA U-Net branch."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_width: int = 16,
    ) -> None:
        super().__init__()
        w = int(base_width)
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, w, 5, padding=2), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(w, 2 * w, 5, padding=2), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv1d(2 * w, 4 * w, 5, padding=2), nn.ReLU())
        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(nn.Conv1d(6 * w, 2 * w, 5, padding=2), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv1d(3 * w, w, 5, padding=2), nn.ReLU())
        self.out = nn.Conv1d(w, out_channels, 1)

    @staticmethod
    def _match_length(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = target.shape[-1] - x.shape[-1]
        if difference > 0:
            return nn.functional.pad(x, (0, difference))
        if difference < 0:
            return x[..., : target.shape[-1]]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        bottleneck = self.bottleneck(self.pool(x2))
        y = self._match_length(self.up(bottleneck), x2)
        y = self.dec2(torch.cat([y, x2], dim=1))
        y = self._match_length(self.up(y), x1)
        y = self.dec1(torch.cat([y, x1], dim=1))
        return self.out(y)


class PaperDualUNet1D(nn.Module):
    """Two independent paper-style U-Nets for real and imaginary field parts.

    This reproduces the paper's central architecture at reduced one-rod scale.
    It does not reproduce the paper's 16-rod LAO/STO dataset.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.real_net = _PaperUNetBranch1D(in_channels, 1)
        self.imag_net = _PaperUNetBranch1D(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.real_net(x), self.imag_net(x)], dim=1)


class PaperWidthDualUNet1D(nn.Module):
    """Paper-style dual U-Net with configurable width for capacity-matched controls."""

    def __init__(self, in_channels: int = 1, base_width: int = 20) -> None:
        super().__init__()
        self.base_width = int(base_width)
        self.real_net = _PaperUNetBranch1D(in_channels, 1, self.base_width)
        self.imag_net = _PaperUNetBranch1D(in_channels, 1, self.base_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.real_net(x), self.imag_net(x)], dim=1)


class _PaperRelationUNetBranch1D(nn.Module):
    """Paper-style branch with a gated nonlocal polynomial relation after each feature convolution."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        relation_dim: int = 16,
        num_heads: int = 2,
        degree: int = 3,
        relation_mode: str = "joint",
        nonlocal_radius: int = 2,
        relation_layers: tuple[str, ...] = ("enc1", "enc2", "bottleneck", "dec2", "dec1"),
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        valid_layers = {"enc1", "enc2", "bottleneck", "dec2", "dec1"}
        unknown_layers = set(relation_layers) - valid_layers
        if unknown_layers:
            raise ValueError(f"Unknown relation layers: {sorted(unknown_layers)}")
        self.relation_layers = tuple(relation_layers)

        def feature_layer(name: str, input_channels: int, output_channels: int) -> nn.Module:
            if name not in self.relation_layers:
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, 5, padding=2),
                    nn.ReLU(),
                )
            layer_relation_dim = min(int(relation_dim), output_channels)
            if layer_relation_dim % num_heads != 0:
                layer_relation_dim = max(
                    num_heads,
                    (layer_relation_dim // num_heads) * num_heads,
                )
            return RelationConv1D(
                input_channels,
                output_channels,
                kernel_size=5,
                padding=2,
                activation="relu",
                relation_dim=layer_relation_dim,
                num_heads=num_heads,
                degree=degree,
                relation_mode=relation_mode,
                nonlocal_radius=nonlocal_radius,
                gate_init=gate_init,
            )

        self.enc1 = feature_layer("enc1", in_channels, 16)
        self.enc2 = feature_layer("enc2", 16, 32)
        self.bottleneck = feature_layer("bottleneck", 32, 64)
        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = feature_layer("dec2", 64 + 32, 32)
        self.dec1 = feature_layer("dec1", 32 + 16, 16)
        # Keep the final continuous-regression projection linear.
        self.out = nn.Conv1d(16, out_channels, 1)

    @staticmethod
    def _match_length(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = target.shape[-1] - x.shape[-1]
        if difference > 0:
            return nn.functional.pad(x, (0, difference))
        if difference < 0:
            return x[..., : target.shape[-1]]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        bottleneck = self.bottleneck(self.pool(x2))
        y = self._match_length(self.up(bottleneck), x2)
        y = self.dec2(torch.cat([y, x2], dim=1))
        y = self._match_length(self.up(y), x1)
        y = self.dec1(torch.cat([y, x1], dim=1))
        return self.out(y)


class PaperRelationDualUNet1D(nn.Module):
    """Two independent paper-style branches using RelationConv1D feature layers."""

    def __init__(
        self,
        in_channels: int = 1,
        relation_dim: int = 16,
        num_heads: int = 2,
        degree: int = 3,
        relation_mode: str = "joint",
        nonlocal_radius: int = 2,
        relation_layers: tuple[str, ...] = ("enc1", "enc2", "bottleneck", "dec2", "dec1"),
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        branch_kwargs = {
            "in_channels": in_channels,
            "out_channels": 1,
            "relation_dim": relation_dim,
            "num_heads": num_heads,
            "degree": degree,
            "relation_mode": relation_mode,
            "nonlocal_radius": nonlocal_radius,
            "relation_layers": relation_layers,
            "gate_init": gate_init,
        }
        self.real_net = _PaperRelationUNetBranch1D(**branch_kwargs)
        self.imag_net = _PaperRelationUNetBranch1D(**branch_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.real_net(x), self.imag_net(x)], dim=1)


def initialize_relation_backbone_from_paper(
    relation_model: PaperRelationDualUNet1D,
    paper_model: PaperDualUNet1D,
) -> None:
    """Copy the ordinary U-Net path so a zero-gated relation model has matched initialization."""

    for branch_name in ("real_net", "imag_net"):
        source_branch = getattr(paper_model, branch_name)
        target_branch = getattr(relation_model, branch_name)
        for layer_name in ("enc1", "enc2", "bottleneck", "dec2", "dec1"):
            source_layer = getattr(source_branch, layer_name)[0]
            target_layer = getattr(target_branch, layer_name)
            if isinstance(target_layer, RelationConv1D):
                target_layer.local_conv.load_state_dict(source_layer.state_dict())
            else:
                target_layer[0].load_state_dict(source_layer.state_dict())
        target_branch.out.load_state_dict(source_branch.out.state_dict())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
