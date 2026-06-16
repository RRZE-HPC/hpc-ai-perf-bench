# LLM Inference Benchmark — Power-Capped Serving

Benchmark suite for measuring LLM inference throughput and energy efficiency under GPU power capping. Serves Meta LLaMA 3 8B with [SGLang](https://github.com/sgl-project/sglang) and runs `bench_serving` at multiple power limits (default: 200–700W), measuring **tokens/sec** and **tokens/joule** at each level.

Containerized with Apptainer for reproducible execution on HPC clusters. Benchmark prompts are generated on the fly with SGLang's random dataset mode, so no prompt dataset file is required.

> **Shared setup** (prerequisites, model download, directory layout) is in the [main README](../README.md). Complete those steps first.

## Repository Structure

```
inference/
├── slurm_scripts/
│   ├── inference_4x.sh                              # SLURM smoke test (no power capping)
│   ├── powercap_inference.sh                        # SLURM job script (NVIDIA GPUs)
│   └── powercap_inference_mi300x_4x.sh              # SLURM job script (AMD MI300X GPUs)
├── analysis/
│   └── analyze_inference_results.py                 # Post-hoc analysis of benchmark results
├── inference_logs/                                  # Benchmark output logs (GPU metrics, power, utilization, temperature)
│   └── inference_<JOBID>_<N>x<GPU>_<TIMESTAMP>/    # Per-job log directory
└── results/                                         # Analysis outputs and visualizations
    ├── inference_zplot_*.png                        # Z-plot charts (energy efficiency vs throughput)
    ├── inference_zplot_*.svg                        # Z-plot charts (vector format)
```

## Setup

### 1. Pull the SGLang Container

**For NVIDIA GPUs:**

```bash
cd /path/to/hpc-ai-perf-bench/llm/inference
apptainer pull docker://lmsysorg/sglang:v0.4.4.post1-cu124
```

This creates `sglang_v0.4.4.post1-cu124.sif` in the `inference/` directory — a container with SGLang, CUDA 12.4, and all serving dependencies.

**For AMD MI300X GPUs:**

```bash
cd /path/to/hpc-ai-perf-bench/llm/inference
apptainer pull docker://lmsysorg/sglang:v0.4.4.post1-rocm630
```

This creates `sglang_v0.4.4.post1-rocm630.sif` in the `inference/` directory — a container with SGLang, ROCm 6.3.0, and all serving dependencies.

### 2. Configure the SLURM Script

**For NVIDIA GPUs,** edit `slurm_scripts/powercap_inference.sh`:

1. **Adjust `#SBATCH` directives** as needed for your cluster:
   - `--partition` — your GPU partition name (e.g. `h200`, `h100`)
   - `--reservation` — uncomment and set if you have a reservation
   - `--gres` and `--ntasks` — must match `NUM_GPUS`
   - `--time` — wall time (default: 4 hours, covers 6 power caps with 10-min pauses)
2. **Optionally adjust other variables:**
   - `NUM_GPUS` — number of GPUs (must match `--gres` and `--ntasks`; also used as tensor-parallel degree)
   - `NUM_PROMPTS` — number of prompts per benchmark run (default: 2048)
   - `POWER_LIMITS` — array of power caps in watts (default: `200 300 400 500 600 700`)
   - `PAUSE_SECONDS` — stabilization time between power cap changes (default: 600s = 10 min)

**For AMD MI300X GPUs,** edit `slurm_scripts/powercap_inference_mi300x_4x.sh`:

1. **Adjust `#SBATCH` directives** as needed:
   - `--nodelist` — your MI300X node name
   - `--constraint` — GPU constraint (default: `mi300x`)
   - Remove or comment out `--partition` if not using partitions
2. **Set `SIF_PATH`** to point to the ROCm container: `sglang_v0.4.4.post1-rocm630.sif`
3. **Verify `HIP_VISIBLE_DEVICES`** matches your GPU configuration (default: `0,1,2,3`)
4. **Optionally adjust** the same variables as NVIDIA script above

> **No manual path configuration needed** for NVIDIA scripts. Both use `$SLURM_SUBMIT_DIR` to locate paths automatically — as long as you submit from inside the `llm/` directory.

## Running

Always submit from inside the `llm/` directory.

### Smoke test (no power capping)

Use this first to verify the container, model weights, and server start correctly:

```bash
cd /path/to/hpc-ai-perf-bench/llm
sbatch inference/slurm_scripts/inference_4x.sh
```

### Full power-cap sweep

Once the smoke test passes, run the full benchmark:

```bash
cd /path/to/hpc-ai-perf-bench/llm
sbatch inference/slurm_scripts/powercap_inference.sh
```

You can also override the partition at submit time:

```bash
sbatch --partition=h100 inference/slurm_scripts/powercap_inference.sh
```

Or override the pause duration via environment variable:

```bash
PAUSE_SECONDS=300 sbatch inference/slurm_scripts/powercap_inference.sh
```

### AMD MI300X GPUs

Submit the MI300X job:

```bash
sbatch slurm_scripts/powercap_inference_mi300x_4x.sh
```

Override pause duration:

```bash
PAUSE_SECONDS=300 sbatch slurm_scripts/powercap_inference_mi300x_4x.sh
```

Skip power capping (for testing):

```bash
SKIP_POWERCAP=1 sbatch slurm_scripts/powercap_inference_mi300x_4x.sh
```

### What Happens

The SLURM job runs end-to-end without manual intervention:

1. **Starts the SGLang server** — launches `sglang.launch_server` in tensor-parallel mode across all GPUs
2. **Waits for readiness** — polls the server health endpoint until it responds (up to 10 minutes)
3. **Runs a warmup pass** — 100 prompts at 200W to warm caches and JIT (results discarded)
4. **Sweeps power limits** — for each cap in `POWER_LIMITS`:
   - Sets the GPU power cap:
     - **NVIDIA:** `sudo nvidia-smi --power-limit=<watts>`
     - **AMD:** `sudo rocm-smi --setpoweroverdrive <watts>`
   - Pauses for `PAUSE_SECONDS` to allow thermal stabilization
   - Starts GPU metrics logging at 100ms resolution:
     - **NVIDIA:** `nvidia-smi` → CSV
     - **AMD:** `rocm_smi.py` (sysfs-based monitoring) → CSV
   - Runs `sglang.bench_serving` with randomly generated synthetic prompts
   - Stops the GPU logger
5. **Cleans up** — kills the SGLang server and resets GPU power settings

### Output Structure

Logs are written to `<WORKSPACE>/inference_logs/`:

Each job writes benchmark outputs, GPU metrics, and job metadata into a timestamped directory under `inference_logs/`.

## Analysis

After the job completes, run the analysis script on the job logs directory. The script analyzes benchmark results, prints summaries, and automatically generates a **Z-plot chart** (energy efficiency vs throughput with clock speeds on a secondary axis). Charts and CSVs are saved to the results directory.

> **Note:** The analysis script automatically detects and supports both NVIDIA (`nvidia_smi_*.CSV`) and AMD (`rocm_smi_*.CSV`) log formats.

### Single GPU type

```bash
# NVIDIA GPU
python analysis/analyze_inference_results.py inference_logs/inference_<JOBID>_4xh200_<TIMESTAMP>/ --gpu-type h200

# AMD MI300X GPU
python analysis/analyze_inference_results.py inference_logs/inference_<JOBID>_4xmi300x_<TIMESTAMP>/ --gpu-type mi300x
```

### Multiple GPU types

To compare GPU types on a single chart, pass multiple `--gpu-type` labels (the log directory must contain results for all specified types):

```bash
# Compare NVIDIA GPUs
python analysis/analyze_inference_results.py inference_logs/ --gpu-type h100 h200

# Compare NVIDIA and AMD
python analysis/analyze_inference_results.py inference_logs/ --gpu-type h200 mi300x
```

### Custom results directory

By default, charts and CSV copies are saved to `./results/`. Override with `--results-dir`:

```bash
python analysis/analyze_inference_results.py inference_logs/ --gpu-type h200 --results-dir ./my_results
```

### Output

The script produces:

1. **Per-power-limit analysis** — tokens/sec, average power draw, SM/memory clock speeds, tokens/joule
2. **Summary table** — all power limits side by side
3. **Optimal configuration** — the power limit with the best tokens/joule ratio
4. **Copy-pasteable numpy arrays** — `power_limits`, `output_tokens_per_sec`, `tokens_per_joule`, `total_avg_power`, `total_avg_gfx_clock`, `total_avg_mem_clock`
5. **`analysis_results_4x<gpu_type>.csv`** — saved to both the log directory and the results directory
6. **Z-plot chart** (PNG + SVG) — energy efficiency vs throughput with power-cap annotations and clock speeds on a secondary axis. When multiple GPU types are analyzed, all series appear on a single chart.

## Customization

### Power Limits

Edit the `POWER_LIMITS` array in the SLURM script:

```bash
POWER_LIMITS=(200 300 400 500 600 700)
```

### Stabilization Pause

Override the pause between power cap changes (default 10 minutes):

```bash
PAUSE_SECONDS=300 sbatch slurm_scripts/powercap_inference.sh
```

### Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_GPUS` | 4 | Number of GPUs (= tensor-parallel degree) |
| `NUM_PROMPTS` | 2048 | Prompts per benchmark run |
| `SERVER_PORT` | 6000 | SGLang server port |
| `--random-input-len` | 1024 | Input token length per prompt (must fit within context window with output) |
| `--random-output-len` | 256 | Output token length per prompt |

## License

See the repository [LICENSE](../../LICENSE).
