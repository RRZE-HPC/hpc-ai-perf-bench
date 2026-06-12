# HPC AI Performance Benchmark

**Throughput benchmarks for vision and language model workloads on HPC GPUs.**

This repository provides a benchmarking framework built around popular deep learning
applications from **computer vision** (image classification and generation) and
**large language models** (continued pre-training and inference). It focuses on
**throughput** rather than time-to-completion and is designed to run on both NVIDIA
and AMD GPUs.

## Benchmarks

| Suite | Workloads | Models | Frameworks |
|-------|-----------|--------|------------|
| [Computer Vision](computer_vision/README.md) | Image classification, image generation | ViT, ResNet, Stable Diffusion | PyTorch Lightning |
| [LLM](llm/README.md) | Continued pre-training, inference | LLaMA 3 8B | LitGPT, SGLang |

See each suite's documentation for usage and setup details.

## Citation

If you use this benchmark in your research, please cite our paper:

> Martin Mayr, Sebastian Wind, Lukas Schröder, Georg Hager, Harald Köstler, Gerhard Wellein.
> *AI Application Benchmarking: Power-Aware Performance Analysis for Vision and Language Models.*
> arXiv:2603.16164, 2026. <https://arxiv.org/abs/2603.16164>

```bibtex
@article{mayr2026aibenchmarking,
  title         = {AI Application Benchmarking: Power-Aware Performance Analysis for Vision and Language Models},
  author        = {Mayr, Martin and Wind, Sebastian and Schr{\"o}der, Lukas and Hager, Georg and K{\"o}stler, Harald and Wellein, Gerhard},
  year          = {2026},
  eprint        = {2603.16164},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PF},
  url           = {https://arxiv.org/abs/2603.16164}
}
```
