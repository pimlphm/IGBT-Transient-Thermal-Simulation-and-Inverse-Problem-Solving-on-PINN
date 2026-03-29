# PIM — Physics-Informed Model for IGBT Thermal Simulation

> **Transient Inverse PINN for 3-D Temperature-Field Reconstruction and Solder Degradation Identification in IGBT Power Modules**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## Overview

This project implements a **Transient Inverse Physics-Informed Neural Network (PINN)** that reconstructs the full 3-D spatio-temporal temperature field of an IGBT power module from surface measurements alone, while **simultaneously identifying the solder layer thermal conductivity** — enabling non-destructive solder aging detection without disassembly or internal sensors.

The method combines a causal temporal neural architecture with the transient heat conduction PDE as a hard physics constraint. By solving the forward and inverse problems jointly, the network infers latent material parameters (thermal diffusivity, solder conductivity ratio, heat-source gain) alongside the complete temperature distribution at every spatial node and time step.

---

## Key Features

| Feature | Description |
|---|---|
| **Transient PDE Loss** | Enforces ∂T/∂t = ∇·(α∇T) + Q at interior collocation points |
| **Inverse Parameter Estimation** | Jointly learns solder thermal conductivity ratio `k_ratio`, thermal diffusivity `α_base`, and heat-source gain |
| **Causal Temporal Architecture** | Causal 1-D residual convolution blocks prevent information leakage from future time steps |
| **Fourier Feature Encoding** | Learnable spatial Fourier features + fixed temporal Fourier features to capture multi-scale variations |
| **Warm-Start Curriculum** | Sequential training pipeline: data-only pretraining → steady physics → transient physics → full inverse |
| **Residual-Based Attention (RBA)** | Optional adaptive PDE residual weighting to focus training on high-error collocation regions |
| **Automatic Solder-Band Detection** | Infers solder layer z-bounds from the point-cloud z-level histogram without manual annotation |
| **Ablation Study** | Five model variants with systematic comparison of architectural and physics choices |

---

## Method

### Architecture: `TransientInversePINN`

```
Input: (x, y, z) coords  +  time-feature vector [t_norm, P_norm, dP/dt, energy]
         │                           │
   Learnable Fourier           Fixed Fourier
   Features (3D space)         Features (time)
         │                           │
    Space MLP                   Time Projection
    (3 hidden layers)           (Linear)
         │                           │
         └──────── Fusion ───────────┘
                       │
              Positional Embedding
                       │
          ┌────────────────────────┐
          │ Causal Residual Blocks │  × 3
          │ (Conv1d, stride-free,  │
          │  causal padding)       │
          └────────────────────────┘
                       │
              Layer Norm + MLP Readout
                       │
            Output: T(x,y,z,t)  [n_points × n_timesteps]

Inverse parameters (learnable scalars):
  log_k_ratio   →  k_solder = k_nominal × exp(log_k_ratio)
  log_alpha_base →  α_base  = exp(log_alpha_base)
  log_source_gain → source_gain = exp(log_source_gain)
```

### Loss Function

The total loss combines five terms (with configurable weights):

```
L = w_data  × L_data          # MSE on observed surface temperatures
  + w_rollout × L_rollout      # Temporal consistency (Δt residuals)
  + w_ic  × L_ic               # Initial condition at t=0
  + w_bc  × L_bc               # Neumann BC (zero-flux on side walls)
  + w_pde × L_pde              # Transient heat equation residual
  + w_param × L_param          # Parameter regularization (k_ratio ≈ 1)
```

For the full model (`transient_inverse_full`), **Residual-Based Attention (RBA)** re-weights the PDE loss by the local residual magnitude:

```
L_pde = mean( (1 + 0.25 × |r| / max|r|)² × r² )
```

### Training Curriculum (Warm-Start)

Models are trained sequentially, with each stage warm-starting from the previous best checkpoint:

```
Stage 1: data_only_seq          (physics OFF, k fixed)
    ↓  warm-start
Stage 2: steady_like_no_time    (steady-state PDE, k fixed)
    ↓  warm-start
Stage 3: transient_fixed_k      (transient PDE, k fixed)
    ↓  warm-start
Stage 4: transient_inverse_no_rba  (transient PDE, k learned, RBA OFF)
    ↓  warm-start
Stage 5: transient_inverse_full    (transient PDE, k learned, RBA ON)
```

---

## Dataset

The input data is a CSV file exported from a COMSOL Multiphysics transient thermal simulation of an IGBT module.

| Property | Value |
|---|---|
| Total spatial nodes | 131,507 |
| Surface measurement points | 4,398 |
| Train / Val / Test split | 3,078 / 660 / 660 |
| Time steps | 13 (0 – 10 s) |
| Power ramp | 0 → 114 W |
| Ambient temperature | ≈ 65 °C |
| Nominal solder conductivity | 50 W/mK |
| Chip hot-spot centers | 2 (auto-detected by k-means on top-10% surface temps) |

**Geometry (auto-detected from z-level histogram):**

| Layer | z range |
|---|---|
| Surface (IR observation plane) | z = 5.827 × 10⁻³ m |
| Chip layer bottom | z = 5.327 × 10⁻³ m |
| Solder layer | 5.007 × 10⁻³ m ≤ z ≤ 5.327 × 10⁻³ m |

**Time features** fed to the model per step:
- Normalized time `t_norm`
- Normalized power `P_norm`
- Power rate of change `dP/dt`
- Cumulative energy `∫P dt`

---

## Results

### Ablation Study

| Model | Surface RMSE (°C) | Surface MAE (°C) | Surface MAPE (%) | Surface R² | Peak Error (°C) | Runtime (s) |
|---|---|---|---|---|---|---|
| **data_only_seq** | 3.553 | 2.644 | 2.443 | 0.9775 | 8.35 | 4.2 |
| **steady_like_no_time** | 3.763 | 2.704 | 2.433 | 0.9748 | 11.17 | 57.6 |
| **transient_fixed_k** | 3.763 | 2.704 | 2.433 | 0.9748 | 11.17 | 57.1 |
| **transient_inverse_no_rba** | 3.625 | 2.535 | 2.312 | 0.9766 | 7.23 | 57.9 |
| **transient_inverse_full** | **3.625** | **2.535** | **2.312** | **0.9766** | **7.23** | 57.9 |

> Best model: `transient_inverse_full` — Surface R² = **0.977**, RMSE ≈ **3.6 °C**, MAPE ≈ **2.3 %**, Peak error ≈ **7.2 °C**

### Recovered Inverse Parameters (full model)

| Parameter | Value | Physical Meaning |
|---|---|---|
| `k_ratio` | 1.000 | Solder conductivity ratio (1.0 = no aging) |
| `k_solder` | 50.0 W/mK | Identified solder thermal conductivity |
| `aging_ratio` | 0.0 | Estimated solder degradation (0 = pristine) |
| `alpha_base` | 0.301 | Normalized thermal diffusivity |
| `source_gain` | 0.607 | Heat-source amplitude gain |

---

## Repository Structure

```
IGBT仿真/
├── transient_inverse_pinn_igbt.py          # Main training & inference script
├── transient_inverse_pinn_igbt_report.ipynb # Auto-generated analysis notebook
├── T-acm-new2.csv                           # Input dataset (COMSOL point cloud)
├── README.md                                # This file
└── transient_inverse_pinn_results/          # Training outputs
    ├── dataset_summary.json                 # Dataset statistics
    ├── ablation_metrics.csv                 # Summary metrics for all variants
    ├── ablation_summary.png                 # Bar-chart comparison
    ├── geometry_inference.png               # Geometry + hot-spot visualization
    ├── {model}_best.pt                      # Best checkpoint (by val RMSE)
    ├── {model}_last.pt                      # Last checkpoint
    ├── {model}_state.pt                     # Model state dict
    ├── {model}_history.csv                  # Per-epoch training log
    ├── {model}_history.png                  # Training curves
    ├── {model}_rollout.png                  # Mean/peak surface T over time
    ├── {model}_surface_pred.npy             # Predicted test surface temps
    └── {model}_surface_true.npy             # Ground-truth test surface temps
```

---

## Installation

**Prerequisites:** Python ≥ 3.9, CUDA (optional but recommended)

```bash
# Clone the repository
git clone https://github.com/<your-username>/PIM.git
cd PIM

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib nbformat jupyter
```

---

## Usage

### 1. Run the Full Training Pipeline

```bash
python transient_inverse_pinn_igbt.py
```

This sequentially trains all 5 model variants and saves results to `transient_inverse_pinn_results/`.

### 2. Generate Plots and Notebook (requires saved checkpoints)

```bash
python transient_inverse_pinn_igbt.py --report-only --emit-plots
```

### 3. Training Options

```
usage: transient_inverse_pinn_igbt.py [-h]
                                       [--nominal-solder-k NOMINAL_SOLDER_K]
                                       [--device DEVICE]
                                       [--emit-plots]
                                       [--report-only]

Arguments:
  --nominal-solder-k FLOAT   Reference solder conductivity in W/mK
                             (default: 50.0)
  --device STR               Compute device: auto | cpu | cuda:0 | cuda:1
                             (default: auto)
  --emit-plots               Save PNG plots and execute the Jupyter notebook
  --report-only              Skip training; regenerate plots from saved artifacts
```

### 4. Inspect Results in Jupyter

```bash
jupyter notebook transient_inverse_pinn_igbt_report.ipynb
```

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Hidden dim | 96 | Feature dimension of all MLP and temporal blocks |
| Temporal blocks | 3 | Number of causal residual conv layers |
| Space Fourier freqs | 18 | Learnable spatial frequency components |
| Time Fourier freqs | 10 | Fixed temporal frequency components |
| Learning rate | 2 × 10⁻³ | AdamW initial LR |
| Weight decay | 1 × 10⁻⁵ | AdamW regularization |
| Early stopping patience | 35 epochs | (after min 60 epochs) |
| Gradient clipping | 1.0 (max norm) | Prevents gradient explosion |
| Warmup epochs | 40–60 | PDE loss disabled during warmup |

---

## Reference

> Yang et al., 2025 — *A Parameterized Thermal Simulation Method Based on Physics-Informed Neural Networks*  
> (PDF included in this repository: `Yang et al_2025_A Parameterized Thermal Simulation Method Based on Physics-Informed Neural.pdf`)

---

## License

This project is released under the [MIT License](LICENSE).

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{PIM_IGBT_PINN_2026,
  title  = {PIM: Physics-Informed Model for IGBT Transient Thermal Simulation},
  year   = {2026},
  url    = {https://github.com/<your-username>/PIM}
}
```
