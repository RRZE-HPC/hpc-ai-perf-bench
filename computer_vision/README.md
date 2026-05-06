# Benchmarking Image Classification (ViT) and Image Generation (Stable Diffusion)

## Overview

This benchmark suite evaluates the performance of common deep learning models, specifically Vision Transformers (ViT) for image classification and Stable Diffusion for image generation, primarily focusing on single-GPU throughput. The code is designed to be runnable on both NVIDIA and AMD GPUs, utilizing PyTorch Lightning.

While adaptable for multi-GPU and multi-node configurations, optimization and testing have primarily focused on single-device scenarios.

## Prerequisites

Before you begin, ensure you have the following installed:

*   Python (3.8+ recommended)
*   `pip` (Python package installer)
*   **For AMD GPUs:** Apptainer (formerly Singularity) for building and running the ROCm container.
*   **For SLURM:** Access to a SLURM cluster and the SLURM client tools installed locally.
*   Git (for cloning if necessary)

## Installation

### NVIDIA GPUs

1.  Ensure you have the appropriate NVIDIA drivers installed on your system.
2.  The recommended method is to use the provided Apptainer container definition, which bundles CUDA libraries and Python dependencies.
     ```bash
     # Ensure you are in the 'computer_vision' directory
     # For standard NVIDIA setup:
     apptainer build nvidia_cuda12_9.sif scripts/environment/nvidia_cuda12_9.def
     
     # For NVIDIA setup with xformers optimization:
     apptainer build nvidia_cuda12_9_xformers.sif scripts/environment/nvidia_cuda12_9_xformers.def
     ```
     Using the container ensures compatibility and isolates dependencies.

### AMD GPUs

1.  Ensure you have the necessary ROCm drivers installed on your system.
2.  The recommended method is to use one of the provided Apptainer container definitions, which bundle ROCm libraries and Python dependencies.
     ```bash
     # Ensure you are in the 'computer_vision' directory
     apptainer build amd_rocm6_3_xformers.sif scripts/environment/amd_rocm6_3_xformers.def

     # Alternative ROCm 7.1-based image:
     apptainer build amd_rocm7_1_xformers.sif scripts/environment/amd_rocm7_1_xformers.def
     ```
     Using the container ensures compatibility and isolates dependencies.

### Optional: Using xformers (NVIDIA Only)

[xformers](https://github.com/facebookresearch/xformers) provides optimized attention mechanisms that can accelerate Stable Diffusion on NVIDIA GPUs.

*   To enable it, uncomment the `xformers` line in `requirements.txt` *before* building the container.

## Running the Benchmarks

Benchmarks are executed using the `main_vit.py` and `main_ldm.py` scripts.

The default Stable Diffusion configuration loaded by `main_ldm.py` is `configs/train_SDv2.yaml`.

### Direct Execution (Primarily for single-GPU testing outside SLURM)

You can run the scripts directly using `python` after building the container.

**ViT Example (Single GPU):**
```bash
# Example using ViT-Large
apptainer exec --nv nvidia_cuda12_9.sif python main_vit.py --batch_size 256 --model vit_l_16
```

**Stable Diffusion Example (Single GPU):**
```bash
apptainer exec --nv nvidia_cuda12_9_xformers.sif python main_ldm.py --batch_size 192
```
*Note: Adjust `--batch_size` based on available GPU memory and desired configuration.* 

### SLURM Execution (Recommended for Cluster Environments)

Example SLURM submission scripts are provided in the `scripts/slurm` directory. These scripts handle environment setup (like loading modules or running inside containers) and resource allocation.

**Important:**
*   These scripts are intended as starting points and may require adaptation for different cluster environments (paths, partition names, account details, module loads, and container paths).
*   Resource requests (`--gres=gpu:<devices>`, `--nodes=<nodes>`) are often defined *within* the script using `#SBATCH` directives. The command-line arguments passed to the script might supplement or override these, depending on the script's logic.

Recommended entry points:

- `scripts/slurm/slurm_helma.sh` for NVIDIA-based cluster runs
- `scripts/slurm/slurm_mi300x.sh` for AMD MI300X runs
- `scripts/slurm/run_helma_apptainer.sh` as the NVIDIA runner used by `slurm_helma.sh`
- `scripts/slurm/run_mi300x.sh` as the AMD runner currently used by `slurm_mi300x.sh`
- `scripts/slurm/run_mi300x_apptainer.sh` as an alternative AMD runner variant

**Command Structure (Current Wrappers):**
```bash
sh scripts/slurm/slurm_helma.sh <python_script> <batch_size> [<partition>] [<nodes>] [<devices>] [<tasks_per_node>]
sh scripts/slurm/slurm_mi300x.sh <python_script> <batch_size> [<devices>]
```
*   `slurm_helma.sh`: wraps NVIDIA jobs and constructs `--batch_size <batch_size>` internally.
*   `slurm_mi300x.sh`: wraps AMD jobs and constructs `--batch_size <batch_size>` internally.
*   `<python_script>`: e.g., `main_vit.py` or `main_ldm.py`.
*   `<batch_size>`: Batch size value passed into the wrapper.
*   `[<partition>]`, `[<nodes>]`, `[<devices>]`, `[<tasks_per_node>]`: Optional NVIDIA wrapper arguments used to control resource requests.

**Example (NVIDIA H100 using `scripts/slurm/slurm_helma.sh`):**

*Uses `scripts/slurm/slurm_helma.sh` as an example NVIDIA submission wrapper.*

ViT (Example uses ViT-Large):
```bash
sh scripts/slurm/slurm_helma.sh main_vit.py 256 h100 1 1
```
Stable Diffusion:
```bash
sh scripts/slurm/slurm_helma.sh main_ldm.py 192 h100 1 1
```

**Example (AMD MI300X using `scripts/slurm/slurm_mi300x.sh`):**

*Uses `scripts/slurm/slurm_mi300x.sh` as an example AMD submission wrapper.*

ViT (Example uses ViT-Large):
```bash
sh scripts/slurm/slurm_mi300x.sh main_vit.py 576 1
```
Stable Diffusion:
```bash
sh scripts/slurm/slurm_mi300x.sh main_ldm.py 224 1
```