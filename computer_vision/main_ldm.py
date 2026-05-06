import os
# Enable NCCL debug information to verify InfiniBand usage
# os.environ["NCCL_DEBUG"] = "INFO"
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.callbacks.performance_tracking2_rocm import PerformanceTrackingCallbackAsync
from ldm.callbacks.communication_profiler import CommunicationProfilerCallback
import time
import argparse
try:
    import pynvml
except:
    pynvml = None


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, image_size=512, use_fp16=True, device="cpu"):
        self.num_samples = num_samples
        self.image_size = image_size
        self.use_fp16 = use_fp16
        self.device = device

        # Set precision based on config
        dtype = torch.float16 if self.use_fp16 else torch.float32

        self.images = torch.randn(
            10, 3, self.image_size, self.image_size, 
            dtype=dtype,
            device=self.device
        )

        self.captions = ["dummy caption"] * 10

    def __getitem__(self, index):
        sample_id = index % 10
        return self.images[sample_id], self.captions[sample_id]

    def __len__(self):
        return self.num_samples

class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, image_size=64, num_samples=(100000, 10000, 10000), use_fp16=True, num_workers=8, device="cuda"):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_samples, self.val_samples, self.test_samples = num_samples
        self.device = device
        self.use_fp16 = use_fp16
        self.num_workers = num_workers if device=="cpu" else 0

        # Create separate datasets for train, validation and test
        self.train_data = DummyDataset(num_samples = self.train_samples, image_size=self.image_size, use_fp16=self.use_fp16, device=self.device)
        self.val_data = DummyDataset(num_samples = self.val_samples, image_size=self.image_size, use_fp16=self.use_fp16, device=self.device)
        self.test_data = DummyDataset(num_samples = self.test_samples, image_size=self.image_size, use_fp16=self.use_fp16, device=self.device)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        # Return both the original keys and the keys needed for validation
        return {
            "jpg": images,
            "txt": captions,
            "caption": captions,  # Add this key for validation
            "image_id": [f"dummy_{i}.jpg" for i in range(len(batch))]  # Use image_id as specified in config
        }

def main():
    print("starting main...")

    # check if gpu is available, if not stop the script
    if not torch.cuda.is_available():
        print("No GPU available, exiting...")
        exit()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Stable Diffusion model')
    parser.add_argument('--batch_size', type=str, default="2,4,8", help='Comma-separated batch sizes per GPU')
    args = parser.parse_args()

    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    ntasks_per_node = max(ntasks_per_node, torch.cuda.device_count())

    # Parse batch sizes
    args.batch_size = [int(size) for size in args.batch_size.split(",")]

    # Set random seed for reproducibility
    seed_everything(42)

    # only sleep if job was on slurm hold
    if os.environ.get("SLURM_JOB_HOLD", ""):
        # sleep for 2 minutes
        print("sleeping for 2 minutes... Start: ", time.strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(120)
        print("sleeping for 2 minutes... End: ", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Load config
    print("loading config...")
    config_path = os.path.join("configs/train_SDv2.yaml")
    config = OmegaConf.load(config_path)

    img_size = 512

    # Calculate number of workers per GPU (not total)
    # This prevents creating too many worker processes when using multiple GPUs
    total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    num_total_gpus = ntasks_per_node * int(os.environ.get('SLURM_JOB_NUM_NODES', 1))

    # Divide CPUs by number of GPUs per node to get CPUs available per GPU
    cpus_per_gpu = total_cpus // ntasks_per_node if ntasks_per_node > 0 else total_cpus

    # Set workers per GPU: leave 2 CPUs for main process, minimum 4 workers
    num_workers = max(4, cpus_per_gpu - 2)

    print(f"Using {num_workers} workers per GPU ({total_cpus} total CPUs, {ntasks_per_node} GPUs per node)")
    print(f"Using {os.environ.get('SLURM_JOB_NUM_NODES', 1)} nodes")

    # Set precision based on config
    precision = "16-mixed" if config.model.params.unet_config.params.use_fp16 else "32"

    # Loop over batch sizes
    for bs in args.batch_size:
        print(f"\n{'='*80}")
        print(f"Training with batch size: {bs}")
        print(f"{'='*80}\n")

        # Create model
        print("creating model...")
        model = instantiate_from_config(config.model)
        model.learning_rate = 1e-4
        model.set_logging_params(config=os.path.split(config_path)[1], image_size=img_size, batch_size=bs)

        # Calculate number of samples
        sample_estimate = 50*bs*num_total_gpus
        num_samples = (sample_estimate, sample_estimate, sample_estimate)

        # Create dummy data module
        # Note: When device='cuda', num_workers must be 0 (CUDA tensors can't be shared across processes)
        data_module = DummyDataModule(
            batch_size=bs,
            image_size=img_size,
            num_samples=num_samples,
            use_fp16=config.model.params.unet_config.params.use_fp16,
            num_workers=num_workers,  
            device="cuda"
        )

        # Setup callbacks
        callbacks = [
            PerformanceTrackingCallbackAsync(
                metrics_file="ldm_benchmark_async.csv",
                sampling_interval=0.1
            )
        ]

        if ntasks_per_node > 1 and int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) > 1:
            strategy = DDPStrategy(find_unused_parameters=False, process_group_backend='nccl')
        else:
            strategy = "auto"

        devices = [i for i in range(ntasks_per_node)]
        print(f"Using devices: {devices}")

        # Create trainer
        print("creating trainer with {} devices and {} nodes".format(devices, int(os.environ.get("SLURM_JOB_NUM_NODES", 1))))
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=-1,
            num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
            strategy=strategy,
            precision=precision,
            enable_checkpointing=False,
            callbacks=callbacks,
            max_epochs=11,
            limit_val_batches=0,
            enable_progress_bar=False,
        )

        # Train model
        print("training model...")
        trainer.fit(model, data_module)

if __name__ == "__main__":
    main()

