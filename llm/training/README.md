# LLM Training Benchmark — Power-Capped Pre-Training

Benchmark suite for measuring LLM training throughput and energy efficiency under GPU power capping. Runs Meta LLaMA 3 8B continued pre-training at multiple power limits (default: 200–700W) and measures **tokens/sec** and **tokens/joule** at each level.

Built on [LitGPT](https://github.com/Lightning-AI/litgpt), containerized with Apptainer for reproducible execution on HPC clusters. Training samples are generated synthetically on the fly, so no text corpus archive is required.

> **Shared setup** (prerequisites, model download, directory layout) is in the [main README](../README.md). Complete those steps first.

## Repository Structure

```
training/
├── litgpt.def                              # Apptainer container definition
├── litgpt/                                 # LitGPT source (included in repo)
├── configs/
│   └── llama3_8b_4gpu.yaml                 # Training config (4 GPUs, LLaMA 3 8B)
├── slurm_scripts/
│   └── powercap_training_4x.sh             # SLURM job script for 4x GPUs (edit partition)
├── analysis/
│   └── analyze_powercap_results.py         # Post-hoc analysis of benchmark results
├── training_logs/                          # Benchmark output logs
│   └── llama_benchmark_<JOBID>_4x<GPU>_<TIMESTAMP>_logs/
│       ├── <POWERWatts>/                   # Per power-limit subdirectories
│       │   ├── nvidia_smi_gpu_usage.csv    # GPU metrics (power, clocks, utilization, temperature)
│       │   ├── metrics.csv                 # Training metrics (step, loss, tokens/sec, time)
│       │   └── gpu_allocation.log          # GPU allocation info
│       └── analysis_results.csv            # Summary analysis results
└── results/                                # Analysis outputs and visualizations
    ├── training_<GPU>.csv                  # Per-GPU type summary CSVs
    └── training_zplot_*.png                # Energy efficiency vs throughput charts
```

## Setup

### 1. Build the Apptainer Container

```bash
apptainer build litgpt.sif litgpt.def
```

This creates `litgpt.sif` — a container with CUDA 12.1, PyTorch, InfiniBand support, and LitGPT pre-installed.

### 2. Configure the SLURM Script

Edit `slurm_scripts/powercap_training_4x.sh`:

1. **Adjust `#SBATCH` directives** as needed for your cluster:
   - `--partition` — your GPU partition name (e.g. `h100` or `h200`)
   - `--reservation` — uncomment and set if you have a reservation
   - `--time` — wall time (default: 12 hours, covers 6 power caps with 10-min pauses)
2. **Optionally adjust `POWER_LIMITS`** array (default: `200 300 400 500 600 700` watts)
3. **Optionally adjust `PAUSE_SECONDS`** — stabilization time between power cap changes (default: 600s = 10 min)
4. **Optionally adjust synthetic data size** in `configs/llama3_8b_4gpu.yaml` via `data.init_args.train_samples` and `data.init_args.val_samples`

> **No manual path configuration needed.** The script uses `$SLURM_SUBMIT_DIR` to locate all paths automatically — as long as you submit from inside the `llm/` directory.

## Running

Always submit from inside the `llm/` directory.

### Smoke test (no power capping)

Use this first to verify the container, model weights, and config are working:

```bash
cd /path/to/hpc-ai-perf-bench/llm
sbatch training/slurm_scripts/training_4x.sh
```

### Full power-cap sweep

Once the smoke test passes, run the full benchmark:

```bash
cd /path/to/hpc-ai-perf-bench/llm
sbatch training/slurm_scripts/powercap_training_4x.sh

# Optional: control SLURM stdout/stderr location
# sbatch -o /some/path/training_%x_%j.out training/slurm_scripts/powercap_training_4x.sh
```

### What Happens

For each power limit in the `POWER_LIMITS` array, the script:

1. Sets the GPU power cap via `sudo nvidia-smi --power-limit=<watts>`
2. Pauses for `PAUSE_SECONDS` to allow thermal stabilization
3. Starts GPU metrics logging at 100ms resolution (`nvidia-smi` → CSV)
4. Runs LLaMA 3 8B pre-training inside the Apptainer container using synthetically generated token sequences
5. Copies `metrics.csv` (training throughput) to the log directory
6. Moves on to the next power limit

This script trains on local scratch and `rsync`s logs to the shared filesystem after each run (better for slow shared storage).

### Output Structure

Logs are written to `<WORKSPACE>/training_logs/`:

Each job writes job metadata, per-power-limit metrics, and GPU monitoring logs into a timestamped directory under `training_logs/`.


### Options

```
--trim N          Number of training iterations to exclude from the end (default: 3)
--results-dir DIR Directory to save charts and CSVs (default: ./results)
```

## Customization

### Training Config

Edit `configs/llama3_8b_4gpu.yaml` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `global_batch_size` | 64 | Total batch size across all GPUs |
| `micro_batch_size` | 4 | Per-GPU batch size (gradient accumulation = global/micro/devices) |
| `max_seq_length` | 512 | Sequence length |
| `max_tokens` | 30,000,000 | Total tokens to process before stopping |
| `devices` | 4 | Number of GPUs |
| `train_samples` | 10000 | Number of synthetic training samples generated per run |
| `val_samples` | 1000 | Number of synthetic validation samples generated per run |

### Power Limits

Edit the `POWER_LIMITS` array in the SLURM script:

```bash
POWER_LIMITS=(200 300 400 500 600 700)
```

### Stabilization Pause

Override the pause between power cap changes (default 10 minutes):

```bash
PAUSE_SECONDS=300 sbatch slurm_scripts/powercap_training_4x.sh
```

## License

See the repository [LICENSE](../../LICENSE).
