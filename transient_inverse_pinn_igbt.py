from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "T-acm-new2.csv"
RESULTS_DIR = ROOT / "transient_inverse_pinn_results"
NOTEBOOK_PATH = ROOT / "transient_inverse_pinn_igbt_report.ipynb"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def simple_kmeans(points: np.ndarray, n_clusters: int, seed: int = 2026, n_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(points) < n_clusters:
        centers = points.copy()
        labels = np.arange(len(points))
        return centers, labels
    chosen = rng.choice(len(points), size=n_clusters, replace=False)
    centers = points[chosen].copy()
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(n_iter):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cid in range(n_clusters):
            cluster = points[labels == cid]
            if len(cluster) == 0:
                centers[cid] = points[rng.integers(0, len(points))]
            else:
                centers[cid] = cluster.mean(axis=0)
    return centers.astype(np.float32), labels


@dataclass
class ThermalDataset:
    coords: np.ndarray
    temps: np.ndarray
    temps_norm: np.ndarray
    ambient_c: float
    temp_scale: float
    time_values: np.ndarray
    power_values: np.ndarray
    time_features: np.ndarray
    surface_idx: np.ndarray
    interior_idx: np.ndarray
    boundary_idx: np.ndarray
    prominent_z: np.ndarray
    solder_bottom: float
    solder_top: float
    chip_bottom: float
    surface_z: float
    hot_centers_xy: np.ndarray
    hot_sigmas_xy: np.ndarray
    train_surface_idx: np.ndarray
    val_surface_idx: np.ndarray
    test_surface_idx: np.ndarray
    nominal_solder_k_wmk: float


@dataclass
class TrainConfig:
    name: str
    seed: int = 2026
    epochs: int = 260
    lr: float = 2e-3
    collocation_batch: int = 96
    boundary_batch: int = 96
    use_physics: bool = True
    use_transient_term: bool = True
    train_inverse_k: bool = True
    use_rba: bool = False
    warmup_epochs: int = 40
    weight_data: float = 8.0
    weight_ic: float = 4.0
    weight_bc: float = 0.25
    weight_pde: float = 1.0
    weight_param: float = 0.02
    weight_rollout: float = 0.05
    lbfgs_steps: int = 0
    hidden_dim: int = 96
    nheads: int = 4
    nlayers: int = 3
    n_space_freq: int = 18
    n_time_freq: int = 10
    early_stopping_patience: int = 35
    early_stopping_min_epochs: int = 60
    early_stopping_min_delta: float = 1e-4
    warm_start_from: str | None = None


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_freq: int, scale: float = 3.0, learnable: bool = False):
        super().__init__()
        basis = torch.randn(in_dim, n_freq) * scale
        if learnable:
            self.basis = nn.Parameter(basis)
        else:
            self.register_buffer("basis", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * x @ self.basis
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, dims: Iterable[int]):
        super().__init__()
        dims = list(dims)
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalResidualBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=0)
        self.norm = nn.LayerNorm(d_model)

    def _causal(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        return conv(F.pad(x, (2, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = F.silu(self._causal(y, self.conv1))
        y = self._causal(y, self.conv2).transpose(1, 2)
        return self.norm(residual + y)


class TransientInversePINN(nn.Module):
    def __init__(self, config: TrainConfig, dataset: ThermalDataset):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.n_time = len(dataset.time_values)

        self.space_ff = FourierFeatures(3, config.n_space_freq, scale=3.0, learnable=True)
        self.time_ff = FourierFeatures(dataset.time_features.shape[1], config.n_time_freq, scale=2.0, learnable=False)

        self.space_net = MLP(
            [
                3 + 2 * config.n_space_freq,
                config.hidden_dim,
                config.hidden_dim,
                config.hidden_dim,
            ]
        )
        self.time_proj = nn.Linear(dataset.time_features.shape[1] + 2 * config.n_time_freq, config.hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_time, config.hidden_dim))
        self.readout = MLP([config.hidden_dim, config.hidden_dim, config.hidden_dim // 2, 1])
        self.temporal_blocks = nn.ModuleList([CausalResidualBlock(config.hidden_dim) for _ in range(config.nlayers)])
        self.temporal_norm = nn.LayerNorm(config.hidden_dim)

        self.log_alpha_base = nn.Parameter(torch.tensor(-1.2))
        self.log_source_gain = nn.Parameter(torch.tensor(-0.5))
        self.log_k_ratio = nn.Parameter(torch.tensor(0.0))
        self.nominal_k = float(dataset.nominal_solder_k_wmk)

        if not config.train_inverse_k:
            self.log_k_ratio.requires_grad_(False)

    @property
    def alpha_base(self) -> torch.Tensor:
        return torch.exp(self.log_alpha_base)

    @property
    def source_gain(self) -> torch.Tensor:
        return torch.exp(self.log_source_gain)

    @property
    def k_ratio(self) -> torch.Tensor:
        return torch.exp(self.log_k_ratio)

    @property
    def k_solder_wmk(self) -> torch.Tensor:
        return self.nominal_k * self.k_ratio

    def solder_mask(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords[:, 2:3]
        width = max(1e-6, 0.12 * (self.dataset.surface_z - self.dataset.solder_bottom))
        low = torch.tensor(self.dataset.solder_bottom, device=coords.device, dtype=coords.dtype)
        high = torch.tensor(self.dataset.solder_top, device=coords.device, dtype=coords.dtype)
        return torch.sigmoid((z - low) / width) - torch.sigmoid((z - high) / width)

    def chip_mask(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords[:, 2:3]
        width = max(1e-6, 0.12 * (self.dataset.surface_z - self.dataset.chip_bottom))
        low = torch.tensor(self.dataset.chip_bottom, device=coords.device, dtype=coords.dtype)
        return torch.sigmoid((z - low) / width)

    def source_mask(self, coords: torch.Tensor) -> torch.Tensor:
        xy = coords[:, :2]
        centers = torch.tensor(self.dataset.hot_centers_xy, device=coords.device, dtype=coords.dtype)
        sigmas = torch.tensor(self.dataset.hot_sigmas_xy, device=coords.device, dtype=coords.dtype)
        diffs = xy[:, None, :] - centers[None, :, :]
        scaled = (diffs**2).sum(dim=-1) / (2.0 * sigmas[None, :] ** 2 + 1e-8)
        hot = torch.exp(-scaled).sum(dim=1, keepdim=True)
        hot = hot / hot.amax(dim=0, keepdim=True).clamp_min(1e-6)
        return hot * self.chip_mask(coords)

    def effective_alpha(self, coords: torch.Tensor) -> torch.Tensor:
        return self.alpha_base * (1.0 + (self.k_ratio - 1.0) * self.solder_mask(coords))

    def forward(self, coords: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        space_feat = self.space_net(self.space_ff(coords))
        time_feat = self.time_proj(self.time_ff(time_features.reshape(-1, time_features.shape[-1])))
        time_feat = time_feat.reshape(coords.shape[0], self.n_time, self.config.hidden_dim)
        tokens = time_feat + space_feat[:, None, :] + self.pos_embedding
        encoded = tokens
        for block in self.temporal_blocks:
            encoded = block(encoded)
        encoded = self.temporal_norm(encoded)
        return self.readout(encoded).squeeze(-1)


def load_dataset(nominal_solder_k_wmk: float = 50.0) -> ThermalDataset:
    raw_meta = pd.read_csv(CSV_PATH, header=None, nrows=9, low_memory=False)
    time_tokens = [float(str(v).split("=")[-1]) for v in raw_meta.iloc[7, 3:].tolist()]
    power_tokens = []
    for item in raw_meta.iloc[8, 3:].tolist():
        text = str(item)
        power_tokens.append(float(text.split("=")[-1]))

    cols = ["x", "y", "z"] + [f"T_{int(p) if float(p).is_integer() else p}" for p in power_tokens]
    df = pd.read_csv(CSV_PATH, skiprows=9, names=cols)

    coords = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    temps = df[cols[3:]].to_numpy(dtype=np.float32)

    ambient_c = float(np.mean(temps[:, 0]))
    temp_scale = float(max(temps.max() - ambient_c, 1.0))
    temps_norm = (temps - ambient_c) / temp_scale

    z_rounded = np.round(coords[:, 2], 6)
    unique_z, counts_z = np.unique(z_rounded, return_counts=True)
    prominent_mask = counts_z > 500
    prominent_z = np.sort(unique_z[prominent_mask])[::-1]
    positive_z = prominent_z[prominent_z > 0]

    surface_z = float(positive_z[0])
    chip_bottom = float(positive_z[min(4, len(positive_z) - 2)])
    solder_top = chip_bottom
    solder_bottom = float(positive_z[min(5, len(positive_z) - 1)])
    if solder_bottom > solder_top:
        solder_bottom, solder_top = solder_top - 3e-4, solder_top

    surface_idx = np.flatnonzero(np.isclose(coords[:, 2], coords[:, 2].max()))
    interior_idx = np.setdiff1d(np.arange(coords.shape[0]), surface_idx, assume_unique=False)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    tol = np.maximum((maxs - mins) * 0.01, 5e-4)
    boundary_mask = (
        (np.abs(coords[:, 0] - mins[0]) < tol[0])
        | (np.abs(coords[:, 0] - maxs[0]) < tol[0])
        | (np.abs(coords[:, 1] - mins[1]) < tol[1])
        | (np.abs(coords[:, 1] - maxs[1]) < tol[1])
        | (np.abs(coords[:, 2] - mins[2]) < tol[2])
    )
    boundary_idx = np.flatnonzero(boundary_mask)

    surface_xy = coords[surface_idx, :2]
    surface_final = temps[surface_idx, -1]
    hot_threshold = np.quantile(surface_final, 0.90)
    hot_xy = surface_xy[surface_final >= hot_threshold]
    n_clusters = int(np.clip(len(hot_xy) // 350, 2, 4))
    hot_centers_xy, labels = simple_kmeans(hot_xy, n_clusters=n_clusters, seed=2026)
    hot_sigmas_xy = []
    for cid in range(n_clusters):
        cluster_points = hot_xy[labels == cid]
        sigma = np.std(cluster_points - hot_centers_xy[cid], axis=0).mean()
        hot_sigmas_xy.append(max(float(sigma), 0.004))
    hot_sigmas_xy = np.asarray(hot_sigmas_xy, dtype=np.float32)

    surface_bins = pd.qcut(surface_final, q=min(8, len(surface_final) // 300), labels=False, duplicates="drop")
    train_surface_idx, temp_surface_idx = train_test_split(
        surface_idx,
        test_size=0.30,
        random_state=2026,
        stratify=surface_bins,
    )
    temp_bins = pd.qcut(temps[temp_surface_idx, -1], q=min(6, len(temp_surface_idx) // 200), labels=False, duplicates="drop")
    val_surface_idx, test_surface_idx = train_test_split(
        temp_surface_idx,
        test_size=0.50,
        random_state=2027,
        stratify=temp_bins,
    )

    time_values = np.asarray(time_tokens, dtype=np.float32)
    power_values = np.asarray(power_tokens, dtype=np.float32)
    time_norm = (time_values - time_values.min()) / max(time_values.max() - time_values.min(), 1.0)
    power_norm = (power_values - power_values.min()) / max(power_values.max() - power_values.min(), 1.0)
    dp = np.gradient(power_norm, time_norm + 1e-6)
    energy = np.concatenate([[0.0], np.cumsum(np.diff(time_norm) * (power_norm[1:] + power_norm[:-1]) / 2.0)])
    energy = energy / max(float(energy.max()), 1.0)
    time_features = np.stack([time_norm, power_norm, dp, energy], axis=1).astype(np.float32)

    return ThermalDataset(
        coords=coords,
        temps=temps,
        temps_norm=temps_norm,
        ambient_c=ambient_c,
        temp_scale=temp_scale,
        time_values=time_values,
        power_values=power_values,
        time_features=time_features,
        surface_idx=surface_idx.astype(np.int64),
        interior_idx=interior_idx.astype(np.int64),
        boundary_idx=boundary_idx.astype(np.int64),
        prominent_z=prominent_z.astype(np.float32),
        solder_bottom=solder_bottom,
        solder_top=solder_top,
        chip_bottom=chip_bottom,
        surface_z=surface_z,
        hot_centers_xy=hot_centers_xy,
        hot_sigmas_xy=hot_sigmas_xy,
        train_surface_idx=train_surface_idx.astype(np.int64),
        val_surface_idx=val_surface_idx.astype(np.int64),
        test_surface_idx=test_surface_idx.astype(np.int64),
        nominal_solder_k_wmk=nominal_solder_k_wmk,
    )


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def model_inputs_for_points(dataset: ThermalDataset, indices: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = to_tensor(dataset.coords[indices], device)
    targets = to_tensor(dataset.temps_norm[indices], device)
    time_features = to_tensor(dataset.time_features, device).unsqueeze(0).repeat(coords.shape[0], 1, 1)
    return coords, targets, time_features


def spatial_derivatives(
    model: TransientInversePINN,
    coords: torch.Tensor,
    time_features: torch.Tensor,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    coords_req = coords.detach().clone().requires_grad_(True)
    pred = model(coords_req, time_features)
    laplacians: List[torch.Tensor] = []
    gradz: List[torch.Tensor] = []
    for step in range(n_steps):
        field = pred[:, step : step + 1]
        grad = torch.autograd.grad(field.sum(), coords_req, create_graph=True, retain_graph=True)[0]
        lap = 0.0
        for axis in range(3):
            second = torch.autograd.grad(
                grad[:, axis : axis + 1].sum(),
                coords_req,
                create_graph=True,
                retain_graph=True,
            )[0][:, axis : axis + 1]
            lap = lap + second
        laplacians.append(lap)
        gradz.append(grad[:, 2:3])
    lap_tensor = torch.stack(laplacians, dim=1)
    gradz_tensor = torch.stack(gradz, dim=1)
    alpha = model.effective_alpha(coords_req)
    alpha_grad = torch.autograd.grad(alpha.sum(), coords_req, create_graph=True, retain_graph=True)[0]
    return pred, lap_tensor, gradz_tensor, alpha, alpha_grad[:, 2:3]


def compute_losses(
    model: TransientInversePINN,
    dataset: ThermalDataset,
    config: TrainConfig,
    train_coords: torch.Tensor,
    train_targets: torch.Tensor,
    train_time_features: torch.Tensor,
    device: torch.device,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_train = model(train_coords, train_time_features)
    loss_data = F.mse_loss(pred_train, train_targets)
    loss_rollout = F.mse_loss(pred_train[:, 1:] - pred_train[:, :-1], train_targets[:, 1:] - train_targets[:, :-1])

    stats = {
        "loss_data": float(loss_data.detach().cpu()),
        "loss_rollout": float(loss_rollout.detach().cpu()),
        "loss_ic": 0.0,
        "loss_bc": 0.0,
        "loss_pde": 0.0,
        "loss_param": 0.0,
    }

    total_loss = config.weight_data * loss_data + config.weight_rollout * loss_rollout

    if not config.use_physics or epoch < config.warmup_epochs:
        return total_loss, stats

    rng = np.random.default_rng(config.seed + epoch)
    collocation_idx = rng.choice(dataset.interior_idx, size=min(config.collocation_batch, len(dataset.interior_idx)), replace=False)
    boundary_idx = rng.choice(dataset.boundary_idx, size=min(config.boundary_batch, len(dataset.boundary_idx)), replace=False)

    col_coords, col_targets, col_time_features = model_inputs_for_points(dataset, collocation_idx, device)
    pred_col, lap_col, gradz_col, alpha_col, alpha_grad_z = spatial_derivatives(
        model,
        col_coords,
        col_time_features,
        n_steps=len(dataset.time_values) - 1,
    )

    dt = torch.tensor(np.diff(dataset.time_values), dtype=torch.float32, device=device).view(1, -1)
    power = torch.tensor(dataset.time_features[:-1, 1], dtype=torch.float32, device=device).view(1, -1, 1)

    dtdt = ((pred_col[:, 1:] - pred_col[:, :-1]) / dt).unsqueeze(-1)
    div_term = alpha_col[:, None, :] * lap_col + alpha_grad_z[:, None, :] * gradz_col
    source = model.source_gain * model.source_mask(col_coords)[:, None, :] * power
    if config.use_transient_term:
        residual = dtdt - div_term - source
    else:
        residual = -div_term - source

    if config.use_rba:
        weights = residual.detach().abs()
        weights = 1.0 + 0.25 * weights / weights.amax().clamp_min(1e-6)
        loss_pde = ((weights * residual) ** 2).mean()
    else:
        loss_pde = (residual**2).mean()

    loss_ic = F.mse_loss(pred_col[:, 0], col_targets[:, 0])

    bc_coords, _, bc_time_features = model_inputs_for_points(dataset, boundary_idx, device)
    bc_coords_req = bc_coords.detach().clone().requires_grad_(True)
    pred_bc = model(bc_coords_req, bc_time_features)
    grad_penalty = 0.0
    for step in range(len(dataset.time_values)):
        grad_bc = torch.autograd.grad(
            pred_bc[:, step : step + 1].sum(),
            bc_coords_req,
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_penalty = grad_penalty + grad_bc.pow(2).mean()
    loss_bc = grad_penalty / len(dataset.time_values)

    loss_param = (model.k_ratio - torch.tensor(1.0, device=device)).pow(2)

    total_loss = (
        total_loss
        + config.weight_ic * loss_ic
        + config.weight_bc * loss_bc
        + config.weight_pde * loss_pde
        + config.weight_param * loss_param
    )
    stats["loss_ic"] = float(loss_ic.detach().cpu())
    stats["loss_bc"] = float(loss_bc.detach().cpu())
    stats["loss_pde"] = float(loss_pde.detach().cpu())
    stats["loss_param"] = float(loss_param.detach().cpu())
    return total_loss, stats


def evaluate_model(
    model: TransientInversePINN,
    dataset: ThermalDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> Dict[str, float]:
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            coords, targets, time_features = model_inputs_for_points(dataset, chunk, device)
            pred = model(coords, time_features)
            preds.append(pred.cpu().numpy())
            truths.append(targets.cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0)
    true_norm = np.concatenate(truths, axis=0)
    pred_c = pred_norm * dataset.temp_scale + dataset.ambient_c
    true_c = true_norm * dataset.temp_scale + dataset.ambient_c
    err = pred_c - true_c
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(true_c), 1e-6)) * 100.0)
    r2 = float(1.0 - np.sum(err**2) / np.maximum(np.sum((true_c - true_c.mean()) ** 2), 1e-12))
    peak_error = float(abs(pred_c[:, -1].max() - true_c[:, -1].max()))
    return {
        "rmse_c": rmse,
        "mae_c": mae,
        "mape_pct": mape,
        "r2": r2,
        "peak_error_c": peak_error,
    }


def evaluate_full_field(
    model: TransientInversePINN,
    dataset: ThermalDataset,
    device: torch.device,
    sample_size: int = 12000,
) -> Dict[str, float]:
    if sample_size >= len(dataset.coords):
        full_idx = np.arange(len(dataset.coords))
    else:
        rng = np.random.default_rng(77)
        full_idx = rng.choice(np.arange(len(dataset.coords)), size=sample_size, replace=False)
    return evaluate_model(model, dataset, full_idx, device=device)


def plot_geometry(dataset: ThermalDataset, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(dataset.coords[:, 2], bins=120, color="#1f77b4", alpha=0.85)
    axes[0].axvspan(dataset.solder_bottom, dataset.solder_top, color="#ff7f0e", alpha=0.25, label="auto solder band")
    axes[0].axvline(dataset.surface_z, color="#2ca02c", linestyle="--", label="surface")
    axes[0].set_title("Z-level distribution")
    axes[0].set_xlabel("z [m]")
    axes[0].set_ylabel("count")
    axes[0].legend()

    surf_xy = dataset.coords[dataset.surface_idx, :2]
    surf_t = dataset.temps[dataset.surface_idx, -1]
    scatter = axes[1].scatter(surf_xy[:, 0], surf_xy[:, 1], c=surf_t, s=7, cmap="inferno")
    axes[1].scatter(dataset.hot_centers_xy[:, 0], dataset.hot_centers_xy[:, 1], c="cyan", s=90, marker="x")
    axes[1].set_title("Surface hot spots at final step")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    fig.colorbar(scatter, ax=axes[1], label="T [degC]")
    fig.tight_layout()
    fig.savefig(out_dir / "geometry_inference.png", dpi=200)
    plt.close(fig)


def plot_training_curves(history: List[Dict[str, float]], out_path: Path) -> None:
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val_rmse_c"], label="val rmse")
    axes[0].set_title("Training history")
    axes[0].set_xlabel("epoch")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(hist_df["epoch"], hist_df["k_ratio"], label="k_ratio")
    axes[1].plot(hist_df["epoch"], hist_df["alpha_base"], label="alpha_base")
    axes[1].plot(hist_df["epoch"], hist_df["source_gain"], label="source_gain")
    axes[1].set_title("Inverse parameters")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_rollout(
    dataset: ThermalDataset,
    pred_surface_c: np.ndarray,
    true_surface_c: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(dataset.time_values, true_surface_c.mean(axis=0), marker="o", label="true")
    axes[0].plot(dataset.time_values, pred_surface_c.mean(axis=0), marker="s", label="pred")
    axes[0].set_title(f"{title}: mean surface temperature")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("T [degC]")
    axes[0].legend()

    axes[1].plot(dataset.time_values, true_surface_c.max(axis=0), marker="o", label="true")
    axes[1].plot(dataset.time_values, pred_surface_c.max(axis=0), marker="s", label="pred")
    axes[1].set_title(f"{title}: peak surface temperature")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("T [degC]")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def infer_surface_predictions(
    model: TransientInversePINN,
    dataset: ThermalDataset,
    indices: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    coords, targets, time_features = model_inputs_for_points(dataset, indices, device)
    with torch.no_grad():
        pred = model(coords, time_features)
    pred_c = pred.cpu().numpy() * dataset.temp_scale + dataset.ambient_c
    true_c = targets.cpu().numpy() * dataset.temp_scale + dataset.ambient_c
    return pred_c, true_c


def train_single_model(
    dataset: ThermalDataset,
    config: TrainConfig,
    device: torch.device,
    out_dir: Path,
    emit_plots: bool = False,
) -> Dict[str, object]:
    set_seed(config.seed)
    model = TransientInversePINN(config, dataset).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=1e-5)
    if config.warm_start_from:
        warm_path = out_dir / config.warm_start_from
        if warm_path.exists():
            warm_payload = torch.load(warm_path, map_location=device)
            state_dict = warm_payload.get("model_state_dict", warm_payload)
            model.load_state_dict(state_dict, strict=False)
            print(f"[warm-start] {config.name} <- {warm_path.name}", flush=True)
    print(f"[train] {config.name} | epochs={config.epochs} | physics={config.use_physics} | transient={config.use_transient_term} | inverse_k={config.train_inverse_k} | rba={config.use_rba}", flush=True)

    train_coords, train_targets, train_time_features = model_inputs_for_points(dataset, dataset.train_surface_idx, device)
    val_metrics_best = {"rmse_c": float("inf")}
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, components = compute_losses(
            model=model,
            dataset=dataset,
            config=config,
            train_coords=train_coords,
            train_targets=train_targets,
            train_time_features=train_time_features,
            device=device,
            epoch=epoch,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        val_metrics = evaluate_model(model, dataset, dataset.val_surface_idx, device=device)
        improved = val_metrics["rmse_c"] < (val_metrics_best["rmse_c"] - config.early_stopping_min_delta)
        if improved:
            val_metrics_best = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_metrics": val_metrics_best,
                    "config": vars(config),
                },
                out_dir / f"{config.name}_best.pt",
            )
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "val_rmse_c": val_metrics["rmse_c"],
                "k_ratio": float(model.k_ratio.detach().cpu()),
                "alpha_base": float(model.alpha_base.detach().cpu()),
                "source_gain": float(model.source_gain.detach().cpu()),
                **components,
            }
        )
        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(
                f"[epoch {epoch:03d}/{config.epochs}] {config.name} "
                f"loss={history[-1]['train_loss']:.4e} "
                f"val_rmse={history[-1]['val_rmse_c']:.4f}C "
                f"k_ratio={history[-1]['k_ratio']:.4f}",
                flush=True,
            )
        if epoch >= config.early_stopping_min_epochs and bad_epochs >= config.early_stopping_patience:
            print(
                f"[early-stop] {config.name} at epoch {epoch} | best_epoch={best_epoch} | best_val_rmse={val_metrics_best['rmse_c']:.4f}C",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if config.lbfgs_steps > 0:
        train_coords, train_targets, train_time_features = model_inputs_for_points(dataset, dataset.train_surface_idx, device)
        lbfgs = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad], lr=0.35, max_iter=config.lbfgs_steps)

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss, _ = compute_losses(
                model=model,
                dataset=dataset,
                config=config,
                train_coords=train_coords,
                train_targets=train_targets,
                train_time_features=train_time_features,
                device=device,
                epoch=config.epochs + 1,
            )
            loss.backward()
            return loss

        lbfgs.step(closure)

    runtime_sec = time.time() - start_time
    print(f"[done] {config.name} | runtime={runtime_sec:.1f}s", flush=True)
    surface_metrics = evaluate_model(model, dataset, dataset.test_surface_idx, device=device)
    full_field_metrics = evaluate_full_field(model, dataset, device=device)
    pred_surface_c, true_surface_c = infer_surface_predictions(model, dataset, dataset.test_surface_idx, device=device)

    aging_ratio = max(0.0, 1.0 - float(model.k_ratio.detach().cpu()))
    summary = {
        "model": config.name,
        "runtime_sec": runtime_sec,
        "best_epoch": best_epoch,
        "surface_test_rmse_c": surface_metrics["rmse_c"],
        "surface_test_mae_c": surface_metrics["mae_c"],
        "surface_test_mape_pct": surface_metrics["mape_pct"],
        "surface_test_r2": surface_metrics["r2"],
        "surface_test_peak_error_c": surface_metrics["peak_error_c"],
        "full_field_rmse_c": full_field_metrics["rmse_c"],
        "full_field_mae_c": full_field_metrics["mae_c"],
        "full_field_r2": full_field_metrics["r2"],
        "k_ratio": float(model.k_ratio.detach().cpu()),
        "k_solder_wmk": float(model.k_solder_wmk.detach().cpu()),
        "aging_ratio": aging_ratio,
        "alpha_base": float(model.alpha_base.detach().cpu()),
        "source_gain": float(model.source_gain.detach().cpu()),
    }

    torch.save(model.state_dict(), out_dir / f"{config.name}_state.pt")
    torch.save(
        {
            "epoch": history[-1]["epoch"] if history else 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
        },
        out_dir / f"{config.name}_last.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / f"{config.name}_history.csv", index=False)
    if emit_plots:
        plot_training_curves(history, out_dir / f"{config.name}_history.png")
        plot_rollout(
            dataset=dataset,
            pred_surface_c=pred_surface_c,
            true_surface_c=true_surface_c,
            title=config.name,
            out_path=out_dir / f"{config.name}_rollout.png",
        )
    np.save(out_dir / f"{config.name}_surface_pred.npy", pred_surface_c)
    np.save(out_dir / f"{config.name}_surface_true.npy", true_surface_c)
    with open(out_dir / f"{config.name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def plot_ablation_table(results_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].bar(results_df["model"], results_df["surface_test_rmse_c"], color="#1f77b4")
    axes[0].set_title("Surface test RMSE")
    axes[0].set_ylabel("degC")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(results_df["model"], results_df["full_field_r2"], color="#ff7f0e")
    axes[1].set_title("Full-field R2")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_notebook(dataset: ThermalDataset, results_df: pd.DataFrame) -> None:
    nb = nbf.v4.new_notebook()
    rel_results = RESULTS_DIR.name
    prominent_display = ", ".join(f"{z:.6f}" for z in dataset.prominent_z[:10])

    md_intro = f"""
# IGBT transient inverse PINN report

This notebook was generated automatically from `transient_inverse_pinn_igbt.py`.

## What is inside

- Surface-temperature-constrained transient PINN training
- Automatic solder-band inference from the COMSOL point cloud
- Inverse estimation of solder thermal conductivity ratio
- Ablations:
  - `data_only_seq`
  - `steady_like_no_time`
  - `transient_fixed_k`
  - `transient_inverse_no_rba`
  - `transient_inverse_full`

## Core assumptions

- The CSV is interpreted as a transient 3D thermal field with 13 time steps.
- The top surface (`z = {dataset.surface_z:.6f} m`) acts as the infrared-observed surface.
- The solder band is auto-detected in `z in [{dataset.solder_bottom:.6f}, {dataset.solder_top:.6f}] m`.
- Conductivity is reported both as a relative ratio and via a configurable nominal value (`{dataset.nominal_solder_k_wmk:.1f} W/mK`).

## Prominent z planes

{prominent_display}
"""

    code_setup = f"""
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

root = Path.cwd()
results_dir = root / "{rel_results}"
metrics = pd.read_csv(results_dir / "ablation_metrics.csv")
metrics
"""

    code_figs = f"""
display(Image(filename=str(results_dir / "geometry_inference.png")))
display(Image(filename=str(results_dir / "ablation_summary.png")))
"""

    code_best = """
best = metrics.sort_values("surface_test_rmse_c").iloc[0]
best
"""

    code_rollout = """
best_name = best["model"]
display(Image(filename=str(results_dir / f"{best_name}_history.png")))
display(Image(filename=str(results_dir / f"{best_name}_rollout.png")))
"""

    code_summaries = """
summaries = {}
for path in sorted(results_dir.glob("*_summary.json")):
    with open(path, "r", encoding="utf-8") as f:
        summaries[path.stem.replace("_summary", "")] = json.load(f)
summaries
"""

    nb.cells = [
        nbf.v4.new_markdown_cell(md_intro),
        nbf.v4.new_code_cell(code_setup),
        nbf.v4.new_code_cell(code_figs),
        nbf.v4.new_code_cell(code_best),
        nbf.v4.new_code_cell(code_rollout),
        nbf.v4.new_code_cell(code_summaries),
    ]
    nbf.write(nb, NOTEBOOK_PATH)


def write_dataset_summary(dataset: ThermalDataset) -> None:
    dataset_summary = {
        "n_points": int(dataset.coords.shape[0]),
        "n_surface_points": int(len(dataset.surface_idx)),
        "n_train_surface": int(len(dataset.train_surface_idx)),
        "n_val_surface": int(len(dataset.val_surface_idx)),
        "n_test_surface": int(len(dataset.test_surface_idx)),
        "ambient_c": float(dataset.ambient_c),
        "temp_scale": float(dataset.temp_scale),
        "time_values": [float(x) for x in dataset.time_values.tolist()],
        "power_values": [float(x) for x in dataset.power_values.tolist()],
        "surface_z": float(dataset.surface_z),
        "chip_bottom": float(dataset.chip_bottom),
        "solder_bottom": float(dataset.solder_bottom),
        "solder_top": float(dataset.solder_top),
        "hot_centers_xy": dataset.hot_centers_xy.tolist(),
        "hot_sigmas_xy": dataset.hot_sigmas_xy.tolist(),
        "nominal_solder_k_wmk": float(dataset.nominal_solder_k_wmk),
    }
    with open(RESULTS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)


def generate_report(dataset: ThermalDataset) -> None:
    if not (RESULTS_DIR / "ablation_metrics.csv").exists():
        raise FileNotFoundError("ablation_metrics.csv not found. Run training first.")
    results_df = pd.read_csv(RESULTS_DIR / "ablation_metrics.csv")
    plot_geometry(dataset, RESULTS_DIR)
    for model_name in results_df["model"].tolist():
        hist_path = RESULTS_DIR / f"{model_name}_history.csv"
        pred_path = RESULTS_DIR / f"{model_name}_surface_pred.npy"
        true_path = RESULTS_DIR / f"{model_name}_surface_true.npy"
        if hist_path.exists():
            hist = pd.read_csv(hist_path).to_dict(orient="records")
            plot_training_curves(hist, RESULTS_DIR / f"{model_name}_history.png")
        if pred_path.exists() and true_path.exists():
            plot_rollout(
                dataset=dataset,
                pred_surface_c=np.load(pred_path),
                true_surface_c=np.load(true_path),
                title=model_name,
                out_path=RESULTS_DIR / f"{model_name}_rollout.png",
            )
    plot_ablation_table(results_df, RESULTS_DIR / "ablation_summary.png")
    build_notebook(dataset, results_df)
    execute_notebook()


def execute_notebook() -> None:
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(NOTEBOOK_PATH),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def run_pipeline(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    dataset = load_dataset(nominal_solder_k_wmk=args.nominal_solder_k)
    write_dataset_summary(dataset)
    print(
        f"[dataset] n_points={dataset.coords.shape[0]} surface={len(dataset.surface_idx)} "
        f"train/val/test={len(dataset.train_surface_idx)}/{len(dataset.val_surface_idx)}/{len(dataset.test_surface_idx)} "
        f"solder_z=[{dataset.solder_bottom:.6f}, {dataset.solder_top:.6f}]",
        flush=True,
    )

    if args.report_only:
        generate_report(dataset)
        print(f"[report-only] Report regenerated at {RESULTS_DIR}", flush=True)
        return

    device = select_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    else:
        torch.backends.cudnn.benchmark = True
    print(f"[device] using {device}", flush=True)

    configs = [
        TrainConfig(
            name="data_only_seq",
            use_physics=False,
            use_transient_term=False,
            train_inverse_k=False,
            epochs=120,
            warmup_epochs=9999,
            weight_data=10.0,
            weight_rollout=0.08,
        ),
        TrainConfig(
            name="steady_like_no_time",
            use_physics=True,
            use_transient_term=False,
            train_inverse_k=False,
            use_rba=False,
            epochs=140,
            warmup_epochs=55,
            weight_data=10.0,
            weight_ic=2.0,
            weight_bc=0.05,
            weight_pde=0.05,
            weight_rollout=0.06,
            warm_start_from="data_only_seq_best.pt",
        ),
        TrainConfig(
            name="transient_fixed_k",
            use_physics=True,
            use_transient_term=True,
            train_inverse_k=False,
            use_rba=False,
            epochs=160,
            warmup_epochs=55,
            weight_data=10.0,
            weight_ic=2.0,
            weight_bc=0.05,
            weight_pde=0.05,
            weight_rollout=0.06,
            warm_start_from="data_only_seq_best.pt",
        ),
        TrainConfig(
            name="transient_inverse_no_rba",
            use_physics=True,
            use_transient_term=True,
            train_inverse_k=True,
            use_rba=False,
            epochs=170,
            warmup_epochs=60,
            weight_data=12.0,
            weight_ic=2.0,
            weight_bc=0.03,
            weight_pde=0.03,
            weight_param=0.005,
            weight_rollout=0.06,
            lbfgs_steps=0,
            warm_start_from="data_only_seq_best.pt",
        ),
        TrainConfig(
            name="transient_inverse_full",
            use_physics=True,
            use_transient_term=True,
            train_inverse_k=True,
            use_rba=True,
            epochs=180,
            warmup_epochs=60,
            weight_data=12.0,
            weight_ic=2.0,
            weight_bc=0.03,
            weight_pde=0.03,
            weight_param=0.005,
            weight_rollout=0.06,
            lbfgs_steps=0,
            warm_start_from="data_only_seq_best.pt",
        ),
    ]

    all_results = []
    for cfg in configs:
        result = train_single_model(
            dataset=dataset,
            config=cfg,
            device=device,
            out_dir=RESULTS_DIR,
            emit_plots=args.emit_plots,
        )
        all_results.append(result)
        print(f"[summary] {cfg.name}: surface_rmse={result['surface_test_rmse_c']:.4f}C full_r2={result['full_field_r2']:.4f}", flush=True)

    results_df = pd.DataFrame(all_results).sort_values("surface_test_rmse_c").reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "ablation_metrics.csv", index=False)
    plot_ablation_table(results_df, RESULTS_DIR / "ablation_summary.png")

    if args.emit_plots:
        generate_report(dataset)
    print(f"[complete] Results saved to {RESULTS_DIR}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transient inverse PINN for the IGBT thermal field dataset.")
    parser.add_argument("--nominal-solder-k", type=float, default=50.0, help="Reference solder conductivity in W/mK.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda:0, cuda:1, ...")
    parser.add_argument("--emit-plots", action="store_true", help="Generate plots and notebook in the current environment.")
    parser.add_argument("--report-only", action="store_true", help="Skip training and only regenerate plots/notebook from saved artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
