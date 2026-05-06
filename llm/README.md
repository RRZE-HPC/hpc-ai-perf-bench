# LLM Benchmark Suite — Power-Capped GPU Benchmarks

## Overview

Benchmark suite for measuring LLM performance and energy efficiency on HPC GPU clusters. Includes tooling for **training** and **inference** workloads with GPU power-capping analysis.

Both benchmarks use Meta LLaMA 3 8B, sweep GPU power limits (default: 200–700W), and measure **tokens/sec** and **tokens/joule** at each level. Containerized with Apptainer for reproducible execution. Training tokens and inference prompts are generated on the fly, so no benchmark dataset needs to be downloaded or unpacked.

| Benchmark | Framework | What it measures |
|-----------|-----------|-----------------|
| [Training](training/README.md) | [LitGPT](https://github.com/Lightning-AI/litgpt) | Pre-training throughput (tokens/sec, tokens/joule) |
| [Inference](inference/README.md) | [SGLang](https://github.com/sgl-project/sglang) | Serving throughput (output tokens/sec, tokens/joule) |

## Repository Structure

```
repo/
├── README.md                       # This file — shared setup
├── training/
│   ├── litgpt.def                  # Apptainer container definition
│   ├── litgpt/                     # LitGPT source (included in repo)
│   ├── configs/
│   │   └── llama3_8b_4gpu.yaml     # Training config (4 GPUs, LLaMA 3 8B)
│   ├── slurm_scripts/
│   │   └── powercap_training_4x.sh # SLURM job script for 4x GPUs
│   ├── analysis/
│   │   └── analyze_powercap_results.py
│   ├── training_logs/              # Benchmark output logs (GPU metrics, power, utilization)
│   └── results/                    # Analysis outputs and visualizations (CSVs, charts)
├── inference/
│   ├── slurm_scripts/
│   │   └── powercap_inference.sh   # SLURM job script (all GPU types)
│   ├── analysis/
│   │   └── analyze_inference_results.py
│   ├── inference_logs/             # Benchmark output logs (GPU metrics, power, utilization)
│   └── results/                    # Analysis outputs and visualizations (CSVs, charts)
└── models/                         # Model weights (not in repo)
    └── Meta-Llama-3-8B/
```

## Prerequisites

- SLURM-managed HPC cluster with NVIDIA H100 or H200 GPUs
- [Apptainer](https://apptainer.org/) installed on the cluster
- `sudo` access on compute nodes for `nvidia-smi --power-limit` (or a SLURM reservation with power-capping privileges)
- Access to [Meta LLaMA 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) model weights (requires HuggingFace account and access approval)

## Setup

### 1. Obtain the Repository

```bash
# Clone the repository and enter the llm/ directory
git clone <repository-url>
cd <repository>/llm
```

### 2. Download the Model

Both benchmarks share the same model weights. Download once and both will use them.

```bash
export HF_TOKEN="your_huggingface_token"
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/Meta-Llama-3-8B/meta-llama/Meta-Llama-3-8B
```

> **Note:** The training benchmark can also download via `litgpt download meta-llama/Meta-Llama-3-8B` inside the container.

#### Expected `models/` directory layout

```
models/
└── Meta-Llama-3-8B/
    └── meta-llama/
        └── Meta-Llama-3-8B/
            ├── .cache/
            ├── config.json
            ├── generation_config.json
            ├── lit_model.pth
            ├── model-00001-of-00004.safetensors
            ├── model-00002-of-00004.safetensors
            ├── model-00003-of-00004.safetensors
            ├── model-00004-of-00004.safetensors
            ├── model.safetensors.index.json
            ├── model_config.yaml
            ├── tokenizer.json
            └── tokenizer_config.json
```

### 3. Set `APPTAINER_CACHEDIR`

Before building or pulling any container, set `APPTAINER_CACHEDIR` to avoid filling your home directory with build cache. Point it to a location with sufficient storage on your system.

```bash
export APPTAINER_CACHEDIR=/path/to/apptainer/cache
```

### 4. Set Up Python Environment for Analysis Scripts

Both benchmarks include Python analysis scripts that require `pandas` and `matplotlib`. Set up a Python virtual environment to run the analysis:

```bash
# Load a suitable Python environment
module load python/3.12-conda

# Create virtual environment
python3 -m venv env

# Activate environment
source env/bin/activate

# Install required packages
pip install pandas matplotlib
```

> **Note:** Keep this environment activated when running the analysis scripts in `training/analysis/` and `inference/analysis/` directories.

### 5. Continue with Benchmark-Specific Setup

Each benchmark has its own container and SLURM configuration:

- **Training** — see [training/README.md](training/README.md) for container build, synthetic data configuration, and SLURM configuration
- **Inference** — see [inference/README.md](inference/README.md) for container pull, synthetic prompt generation, and SLURM configuration
