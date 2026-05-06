import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torchvision import models
from omegaconf import OmegaConf
import time
import argparse
try:
    import pynvml
except:
    pynvml = None

from ldm.callbacks.performance_tracking2_rocm import PerformanceTrackingCallbackAsync
from vit.model.lit_model import LitModel


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, image_size=512):
        self.num_samples = num_samples
        self.image_size = image_size
        self.device = 'cuda'

        self.images = torch.randn(
            10, 3, self.image_size, self.image_size, 
            device=self.device
        )

        self.labels = torch.randint(0, 1000, (10,), dtype=torch.long)

    def __getitem__(self, index):
        sample_id = index % 10
        return self.images[sample_id], self.labels[sample_id]

    def __len__(self):
        return self.num_samples

class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, image_size=224, num_samples=(100000, 10000, 10000), num_workers=8, device="cpu"):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_samples, self.val_samples, self.test_samples = num_samples
        self.device = device
        self.num_workers = num_workers if device=="cpu" else 0

        # Create separate datasets for train, validation and test
        self.train_data = DummyDataset(num_samples = self.train_samples, image_size=self.image_size)
        self.val_data = DummyDataset(num_samples = self.val_samples, image_size=self.image_size)
        self.test_data = DummyDataset(num_samples = self.test_samples, image_size=self.image_size)

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
        labels = torch.stack([item[1] for item in batch])
        # Return both the original keys and the keys needed for validation
        return {
            "img": images,
            "label": labels,  # Add this key for validation
        }

def main():
    print("starting main...")

    # check if gpu is available, if not stop the script
    if not torch.cuda.is_available():
        print("No GPU available, exiting...")
        exit()

    #TESTING!!!!!!!!!!!!!!!!
    #torch.set_float32_matmul_precision('highest')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Stable Diffusion model')
    parser.add_argument('--batch_size', type=str, default="16,32,64", help='Batch size per GPU')
    parser.add_argument('--model', type=str, default="vit_l_16") # Available models: vit_b_16, vit_l_16, vit_b_32, vit_l_32
    args = parser.parse_args()

    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    ntasks_per_node = max(ntasks_per_node, torch.cuda.device_count())

    # Set random seed for reproducibility
    seed_everything(42)

    args.batch_size = [int(size) for size in args.batch_size.split(",")]

    img_size = 224
    max_steps = 250

    # Calculate number of workers per GPU (not total)
    # This prevents creating too many worker processes when using multiple GPUs
    total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    num_total_gpus = ntasks_per_node * int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    
    # Divide CPUs by number of GPUs per node to get CPUs available per GPU
    cpus_per_gpu = total_cpus // ntasks_per_node if ntasks_per_node > 0 else total_cpus
    
    # Set workers per GPU: leave 2 CPUs for main process, minimum 4 workers
    num_workers = max(4, cpus_per_gpu - 2)
    
    run_local = False
    if run_local:
        max_steps = 100
        img_size = 224
        num_samples = (500,50,50)
        args.model = "vit_b_16"


    for bs in args.batch_size:

        # Create model
        print("creating model...", "Batch size: {}".format(bs))
        model = LitModel(args.model, img_size=img_size, batch_size=bs)

        num_samples=(50*bs*num_total_gpus, 1000, 1000)

        # Create dummy data module
        # Note: When device='cuda', num_workers must be 0 (CUDA tensors can't be shared across processes)
        data_module = DummyDataModule(
            batch_size=bs,
            image_size=img_size,
            num_samples=num_samples,
            num_workers=num_workers,
            device="cuda"
        )

        print("Using {} CPU cores as number of workers".format(num_workers))
        
        # Setup callbacks
        callbacks = [
            PerformanceTrackingCallbackAsync(
                metrics_file="vit_benchmark_async.csv",
                sampling_interval=0.1
            )
        ]

        # Get timestamp and SLURM job ID for unique identification
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        
        amd_backup_gpus = num_total_gpus
        devices = int(os.environ.get("SLURM_GPUS_ON_NODE", amd_backup_gpus))
        devices = int(os.environ.get("LIMIT_GPUS", devices))

        if (
            os.environ.get("LIMIT_GPUS") is not None
            and os.environ.get("CUDA_VISIBLE_DEVICES") is None
            and os.environ.get("HIP_VISIBLE_DEVICES") is None
            and os.environ.get("ROCR_VISIBLE_DEVICES") is None
        ):
            visible = ",".join(str(i) for i in range(max(0, devices)))
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            os.environ["HIP_VISIBLE_DEVICES"] = visible
            os.environ["ROCR_VISIBLE_DEVICES"] = visible

        gpu_name = torch.cuda.get_device_name(0).upper()

        print("Using {} {} GPUs".format(devices, gpu_name))
        
        if int(devices) > 1 and int(os.environ.get("SLURM_JOB_NUM_NODES", 1)) > 1:
            # Check if AMD GPU is present by looking for 'AMD' in the GPU name
            backend = 'nccl' if 'AMD' in gpu_name else 'nccl' # Use 'nccl' for both AMD (ROCm) and NVIDIA
            strategy = DDPStrategy(find_unused_parameters=False, process_group_backend=backend)
            print("Using DDP strategy with backend: {}".format(backend))
        else:
            strategy = "auto"

        # Create trainer
        print("creating trainer...")
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=devices,  # Use all available devices
            num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),  # Get number of nodes from SLURM
            strategy=strategy,
            enable_checkpointing=False,
            callbacks=callbacks,
            max_epochs=11,
            limit_val_batches=0,  # This disables validation
            enable_progress_bar=False,
        )

        # Train model
        print("training model...")
        trainer.fit(model, data_module)

if __name__ == "__main__":
    main()