# SURGIN: SURrogate-guided Generative INversion

[![arXiv](https://img.shields.io/badge/arXiv-2509.13189-b31b1b.svg)](https://arxiv.org/abs/2509.13189)

SURGIN integrates a **U-Net enhanced Fourier Neural Operator** (UFNO) surrogate with a **score-based generative model** (SGM), framing the conditional generation as a surrogate prediction-guidance process in a Bayesian perspective. Instead of directly learning the conditional generation of geological parameters, an unconditional SGM is first pretrained in a self-supervised manner to capture the geological prior, after which posterior sampling is performed by leveraging a differentiable U-FNO surrogate to enable efficient forward evaluations conditioned on unseen observations.

**Key capabilities:**
- **Unconditional generation** of realistic permeability fields via diffusion prior
- **Surrogate approximation** from permeability fields to spatiotemporal pressure and saturation fields
- **Conditional generation** guided by a pre-trained UFNO surrogate, solving inverse problems from pressure/saturation/permeability measurements

---

## 🚀 Quick Start: Using Pre-trained Models for Conditional Inference

This section describes how to use the pre-trained diffusion prior and UFNO surrogate to solve inverse problems from observations without retraining.

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n surgin python=3.10
conda activate surgin

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Checkpoints & Datasets

Download the compressed archive and extract it in the repository root:

```bash
# After downloading surgin_pretrained_data.tar.gz to the repo root
tar -xzvf surgin_pretrained_data.tar.gz
```

This extracts the following files into their expected directories:

| File | Location | Description |
|------|----------|-------------|
| `ufno_pre.pth` | `checkpoint/` | UFNO pressure surrogate (horizontal) |
| `ufno_sat.pth` | `checkpoint/` | UFNO saturation surrogate (horizontal) |
| `ufno_pre_vertical.pth` | `checkpoint/` | UFNO pressure surrogate (vertical) |
| `ufno_sat_vertical.pth` | `checkpoint/` | UFNO saturation surrogate (vertical) |
| `ema_0.9999_160000.pt` | `UnconditionalDiffusionTraining_and_Generation/output/logs_gaussian_dit/` | DiT diffusion prior (horizontal) |
| `ema_0.9999_350000.pt` | `UnconditionalDiffusionTraining_and_Generation/output/logs_gaussian_vertical_dit/` | DiT diffusion prior (vertical) |
| `K_gstools_1w.npy` | `dataset/` | Horizontal permeability training data |
| `K_vertical.npy` | `dataset/` | Vertical permeability training data |
| `Multi_Cartesian_Gaussian.hdf5` | `dataset/` | Horizontal flow simulation dataset |
| `Multi_Cartesian_Gaussian_vertical.hdf5` | `dataset/` | Vertical flow simulation dataset |

**Download links:** [surgin_pretrained_data.tar.gz on Hugging Face](https://huggingface.co/datasets/zhaoffeng/SURGIN/resolve/main/surgin_pretrained_data.tar.gz)

### 3. Run Unconditional Generation

Generate permeability field samples from the diffusion prior:

```bash
cd UnconditionalDiffusionTraining_and_Generation
python scripts/inference.py training_recipes/Multi_Cartesian.yml
```

For vertical cross-sections:

```bash
python scripts/inference.py training_recipes/Multi_Cartesian_vertical.yml
```

Generated samples are saved to the path specified by `save_gen_path` in the YAML config.

### 4. Run Conditional Inverse Modeling

Conditional generation (surrogate-guided diffusion) is performed via Jupyter notebooks in `ConditionalDiffusionGeneration/inference_scripts/Case/Generation/`:

| Notebook | Description |
|----------|-------------|
| `Cartesian_inverse_sparse.ipynb` | Horizontal inverse with sparse well data |
| `Vertical_inverse_3wells_injection.ipynb` | Vertical case with 3-well injection observations |

- TODO: please define your forward function $\mathcal{M}$ in `ConditionalDiffusionGeneration/src/guided_diffusion/measurements.py` to create arbitrary conditioning.

---

## 🔥 Training from Scratch

### Step 1: Configure Weights & Biases (W&B)

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking. Set the following environment variables before training:

```bash
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_PROJECT="YOUR_WANDB_PROJECT_NAME"
export WANDB_ENTITY="YOUR_WANDB_ENTITY"
```

> **Tip:** You can get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize). Create a free account and a new project at [https://wandb.ai](https://wandb.ai). If the `WANDB_API_KEY` environment variable is not set, training will proceed without W&B logging.

### Step 2: Train the UFNO Surrogate Model

The UFNO learns the forward mapping from permeability fields to pressure/saturation solutions.

```bash
python train_ufno.py
```

Training configuration is defined directly in the `Config` class within `train_ufno.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_path` | `dataset/Multi_Cartesian_Gaussian_vertical.hdf5` | Training HDF5 dataset |
| `batch_size` | 50 | Batch size |
| `num_epochs` | 150 | Number of training epochs |
| `learning_rate` | 0.001 | Initial learning rate |
| `mode1, mode2, mode3` | 10, 10, 8 | Fourier modes per dimension |
| `width` | 36 | Hidden channel width |

Checkpoints are saved to `checkpoint/`.

### Step 3: Train the Unconditional Diffusion Prior (DiT)

```bash
cd UnconditionalDiffusionTraining_and_Generation
python scripts/train.py training_recipes/Multi_Cartesian.yml
```

For the vertical case:

```bash
python scripts/train.py training_recipes/Multi_Cartesian_vertical.yml
```

Training recipe configuration (YAML):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 64 | Spatial resolution |
| `patch_size` | 2 | DiT patch size |
| `hidden_size` | 256 | Transformer hidden dim |
| `depth` | 12 | Number of transformer blocks |
| `num_heads` | 8 | Attention heads |
| `steps` | 1000 | Diffusion timesteps |
| `noise_schedule` | cosine | Noise schedule type |
| `lr` | 5e-5 | Learning rate |
| `ema_rate` | 0.9999 | EMA decay rate |
| `save_interval` | 10000 | Checkpoint save frequency |

Model checkpoints (including EMA weights) are saved to the `log_path` directory specified in the YAML config.

### Step 4: Run Conditional Generation

Once both the UFNO surrogate and diffusion prior are trained, use the notebooks in `ConditionalDiffusionGeneration/inference_scripts/Case/Generation/` to run surrogate-guided inverse modeling.

---

## 📁 Repository Structure

```
SURGIN/
├── README.md
├── train_ufno.py                          # Train UFNO surrogate model
├── basicutility/
│   ├── ReadInput.py                       # YAML config loader
├── Surrogate/
│   ├── ufno.py                            # U-shaped Fourier Neural Operator (UFNO)
│   ├── lploss.py                          # Relative Lp loss function
│   └── utility.py                         # Dataset loaders & HDF5 utilities
├── UnconditionalDiffusionTraining_and_Generation/
│   ├── scripts/
│   │   ├── train.py                       # Train unconditional DiT diffusion model
│   │   └── inference.py                   # Generate unconditional samples
│   ├── training_recipes/
│   │   ├── Multi_Cartesian.yml            # Config for horizontal case
│   │   └── Multi_Cartesian_vertical.yml   # Config for vertical case
│   ├── src/                               # Diffusion model source code
│   │   ├── dit.py                         # Diffusion Transformer (DiT) architecture
│   │   ├── gaussian_diffusion.py          # DDPM forward/reverse diffusion
│   │   ├── train_util.py                  # Training loop with EMA & DDP
│   │   ├── script_util.py                 # Model/diffusion factory functions
│   │   ├── logger.py                      # Logging backends (stdout, W&B, etc.)
│   │   └── ...                            # fp16, losses, resampling, etc.
│   └── latents/
│       └── create_dataset.py              # Dataset class for diffusion training
└── ConditionalDiffusionGeneration/
    ├── inference_scripts/
    │   └── Case/Generation/               # Jupyter notebooks for conditional inference
    │       ├── Cartesian_inverse_sparse.ipynb
    │       └── Vertical_inverse_3wells_injection.ipynb
    └── src/guided_diffusion/
        ├── gaussian_diffusion.py          # Guided diffusion with conditioning hooks
        ├── condition_methods.py           # Conditioning strategies
        ├── measurements.py                # Measurement operators & UFNO integration
        ├── dit.py                         # DiT for conditional generation
        ├── posterior_mean_variance.py     # Posterior computation utilities
        └── ...
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{feng2025surgin,
  title={SURGIN: SURrogate-guided Generative INversion for subsurface multiphase flow with quantified uncertainty},
  author={Feng, Zhao and Yan, Bicheng and Zhao, Luanxiao and Shen, Xianda and Zhao, Renyu and Wang, Wenhao and Zhang, Fengshou},
  journal={arXiv preprint arXiv:2509.13189},
  year={2025}
}
```
