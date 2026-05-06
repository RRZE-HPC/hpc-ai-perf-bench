import time
import logging
import csv
import os
from datetime import datetime
from collections import deque
from typing import Any, Dict, Optional
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
try:
    import pynvml  # Add this import for NVIDIA GPU monitoring
    NVMLError = pynvml.NVMLError
except:
    pynvml = None
    class NVMLError(Exception): pass
try:
    import amdsmi
    from amdsmi import AmdSmiLibraryException, AMDSMI_MEM_TYPE_VRAM # Import necessary enum
    amdsmi_imported = True
except ImportError:
    amdsmi = None
    class AmdSmiLibraryException(Exception): pass # Define dummy for except block if import fails
    AMDSMI_MEM_TYPE_VRAM = None # Define dummy
    amdsmi_imported = False
    # logging.warning("AMDSMI library not found. AMD GPU monitoring disabled.")

class PerformanceTrackingCallback(pl.Callback):
    """Callback to track samples per second during training and validation."""
    
    def __init__(self, metrics_file: str = "performance_metrics.csv"):
        """Initialize the callback.
        
        Args:
            metrics_file: Path to CSV file where metrics will be saved. Defaults to 'performance_metrics.csv'.
        """
        global pynvml # Declare intent to modify global pynvml at the start
        global amdsmi # Declare intent to modify global amdsmi at the start
        super().__init__()
        
        # Store metrics file path
        self.metrics_file = metrics_file
        
        # Initialize tracking variables
        self.train_start_time = None
        self.val_start_time = None
        self.train_samples_in_epoch = 0
        self.val_samples_in_epoch = 0
        
        # Store metrics for final summary
        self.train_epoch_sps = []
        self.val_epoch_sps = []
        self.gpu_metrics_history = [] 
        
        if pynvml is not None:
            # Initialize NVIDIA Management Library
            try: # Add try-except for robustness
                pynvml.nvmlInit()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())]
            except pynvml.NVMLError as error:
                print(f"Failed to initialize NVML: {error}. GPU monitoring disabled.")
                pynvml = None # Ensure pynvml checks fail later
                self.gpu_handles = []
        elif amdsmi is not None:
            try:
                amdsmi.amdsmi_init()
                # Get handles for all GPUs. This includes integrated GPUs if present.
                # We might need more logic to filter for discrete GPUs if necessary.
                all_handles = amdsmi.amdsmi_get_processor_handles()
                # Simple check: Assume handles with memory > 1GB are discrete GPUs
                self.gpu_handles = []
                for handle in all_handles:
                    try:
                         # Check memory size to potentially filter out iGPUs
                         mem_info = amdsmi.amdsmi_get_gpu_memory_usage(handle, mem_type=AMDSMI_MEM_TYPE_VRAM) # Pass mem_type
                         mem_total = amdsmi.amdsmi_get_gpu_memory_total(handle, mem_type=AMDSMI_MEM_TYPE_VRAM) / (1024 * 1024) # VRAM Total
                         if mem_total > 1 * (1024**3): # More than 1GB VRAM
                            self.gpu_handles.append(handle)
                    except amdsmi.AmdSmiLibraryException:
                        # Handle cases where memory info might not be available for a device
                        pass # Or log a warning
                if len(self.gpu_handles) > 0:
                    print(f"Initialized AMDSMI for {len(self.gpu_handles)} AMD GPUs.")
                else:
                    print("AMDSMI initialized, but no suitable AMD GPUs found/filtered.")
                    amdsmi.amdsmi_shut_down() # Shutdown if no usable GPUs
                    amdsmi = None
            except amdsmi.AmdSmiLibraryException as e:
                print(f"Failed to initialize AMDSMI: {e}. AMDSMI monitoring disabled.")
                amdsmi = None # Ensure amdsmi is None if init failed
        else:
            self.gpu_handles = []
        
        # Get GPU name for logging
        self.gpu_name = self._get_gpu_name()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize power tracking
        self.power_readings = deque(maxlen=500)  # Store last 900 readings (~15 minutes)
        self.power_readings_time = deque(maxlen=500)
    
    def __del__(self):
        if pynvml is not None: # Check if nvml was initialized
            try:
                pynvml.nvmlShutdown()
            except:
                pass # Ignore shutdown errors
        elif amdsmi is not None: # Check if amdsmi was initialized
            try:
                amdsmi.amdsmi_shut_down()
            except:
                pass # Ignore shutdown errors

    def _get_gpu_name(self):
        """Get the name of the GPU being used."""
        try:
            if pynvml is not None and self.gpu_handles:
                # Get name from first GPU (assuming homogeneous setup)
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handles[0])
                # Decode if bytes (Python 3)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
                return gpu_name
            elif amdsmi is not None and self.gpu_handles:
                # Get AMD GPU name
                gpu_info = amdsmi.amdsmi_get_gpu_asic_info(self.gpu_handles[0])
                return gpu_info.get('market_name', 'AMD GPU')
            elif torch.cuda.is_available():
                # Fallback to PyTorch
                return torch.cuda.get_device_name(0)
            else:
                return 'N/A'
        except Exception as e:
            return 'Unknown GPU'
    
    def _setup_logging(self):
        """Setup logging configuration. Only runs on rank 0."""
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False # Prevent messages from propagating to the root logger
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    

    def _get_gpu_info(self, trainer, log_interval=10):
        """Retrieve GPU usage statistics, logging only every `log_interval` steps."""
        global pynvml 
        global amdsmi
        
        if trainer.global_step % log_interval != 0:
            return None

        try:
            total_power = 0
            num_gpus = len(self.gpu_handles)
            gpu_info = {
                "step": trainer.global_step,
                "gpu/memory_used_mb": 0,
                "gpu/memory_total_mb": 0,
                "gpu/utilization_pct": 0,
                "gpu/power_usage_watts": 0,
                "gpu/vram_used_mb": 0,
                "gpu/vram_total_mb": 0,
                "gpu/graphics_clock_mhz": 0,
                "gpu/memory_clock_mhz": 0,
                # "gpu/avg_power_watts_15min": 0,
            }

            if pynvml is not None:
                for i, handle in enumerate(self.gpu_handles):
                    # Memory
                    try:
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used = memory_info.used / 1024**2
                        mem_total = memory_info.total / 1024**2
                    except NVMLError:
                        # Fallback to torch if NVML memory info is not supported
                        try:
                            mem_used = torch.cuda.memory_reserved(i) / 1024**2
                            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**2
                        except:
                            mem_used = 0
                            mem_total = 0

                    # Utilization
                    try:
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = utilization.gpu
                    except NVMLError:
                        gpu_util = 0

                    # Power
                    try:
                        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    except NVMLError:
                        power_usage = 0
                    
                    # Clock speeds
                    try:
                        graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)  # MHz
                        memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)  # MHz
                    except NVMLError:
                        graphics_clock = 0
                        memory_clock = 0

                    gpu_info["gpu/memory_used_mb"] += mem_used
                    gpu_info["gpu/memory_total_mb"] += mem_total
                    # For NVIDIA, VRAM is the same as memory_info
                    gpu_info["gpu/vram_used_mb"] += mem_used
                    gpu_info["gpu/vram_total_mb"] += mem_total
                    gpu_info["gpu/utilization_pct"] += gpu_util
                    gpu_info["gpu/graphics_clock_mhz"] += graphics_clock
                    gpu_info["gpu/memory_clock_mhz"] += memory_clock
                    total_power += power_usage
            elif amdsmi is not None:
                for handle in self.gpu_handles:
                    mem_info = amdsmi.amdsmi_get_gpu_memory_usage(handle, mem_type=AMDSMI_MEM_TYPE_VRAM) # Pass mem_type
                    total_mem = amdsmi.amdsmi_get_gpu_memory_total(handle, mem_type=AMDSMI_MEM_TYPE_VRAM) / (1024 * 1024) # VRAM Total
                    activity = amdsmi.amdsmi_get_gpu_activity(handle) # dict with gfx_activity etc.
                    
                    # Clock speeds for AMD
                    try:
                        clocks = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
                        graphics_clock = clocks['cur_clk']  # Current graphics clock in MHz
                        mem_clocks = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
                        memory_clock = mem_clocks['cur_clk']  # Current memory clock in MHz
                    except:
                        graphics_clock = 0
                        memory_clock = 0

                    gpu_info["gpu/memory_used_mb"] += mem_info['vram_used'] / (1024**2) # Bytes
                    gpu_info["gpu/memory_total_mb"] += total_mem # VRAM Total
                    # Explicitly track VRAM for AMD GPUs
                    gpu_info["gpu/vram_used_mb"] += mem_info['vram_used'] / (1024**2) # Bytes
                    gpu_info["gpu/vram_total_mb"] += total_mem # VRAM Total
                    # Use gfx_activity as utilization metric. Might need adjustment based on desired metric.
                    gpu_info["gpu/utilization_pct"] += activity['gfx_activity'] # Percentage
                    gpu_info["gpu/graphics_clock_mhz"] += graphics_clock
                    gpu_info["gpu/memory_clock_mhz"] += memory_clock
                    power_info = amdsmi.amdsmi_get_power_ave(handle) / 1000.0 # Convert mW to W
                    total_power += power_info['average_socket_power'] / 1_000_000.0 # Convert total uW to W

            # Store power readings
            self.power_readings.append(total_power)
            self.power_readings_time.append(time.time())

            # Compute moving average over the last 15 minutes
            # if self.power_readings:
            #     gpu_info["gpu/avg_power_watts_500steps"] = sum(self.power_readings) / len(self.power_readings)

            # Normalize values per GPU
            if num_gpus > 0:
                gpu_info["gpu/memory_used_mb"] /= num_gpus
                gpu_info["gpu/memory_total_mb"] /= num_gpus
                gpu_info["gpu/utilization_pct"] /= num_gpus
                gpu_info["gpu/vram_used_mb"] /= num_gpus
                gpu_info["gpu/vram_total_mb"] /= num_gpus
                gpu_info["gpu/graphics_clock_mhz"] /= num_gpus
                gpu_info["gpu/memory_clock_mhz"] /= num_gpus
                # Normalize power to per-GPU average
                gpu_info["gpu/power_usage_watts"] = total_power / num_gpus
            else:
                gpu_info["gpu/power_usage_watts"] = total_power

            return gpu_info

        except (NVMLError, AmdSmiLibraryException) as e: # Catch specific NVML errors
            # Log error only once to avoid flooding
            if not getattr(self, '_nvml_error_logged', False):
                self.logger.error(f"NVML/AMDSMI error during GPU monitoring: {e}. Disabling further NVML/AMDSMI calls.")
                self._nvml_error_logged = True
                # Optionally disable pynvml/amdsmi for future calls
                pynvml = None 
                amdsmi = None 
            return None # Return None if fetching failed
        except Exception as e:
            # Log other errors only once
            if not getattr(self, '_other_gpu_error_logged', False):
                self.logger.error(f"Unexpected error during GPU monitoring: {e}")
                self._other_gpu_error_logged = True
            return None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset counters at the start of each training epoch."""
        if trainer.is_global_zero:
            self.train_start_time = time.time()
            self.train_samples_in_epoch = 0
    
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset counters at the start of each validation epoch."""
        if trainer.is_global_zero:
            self.val_start_time = time.time()
            self.val_samples_in_epoch = 0
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Dict[str, Any]],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Count samples processed in this batch."""
        try:
            # Try to get actual batch size if it's a tensor or dict/list containing tensors
            if isinstance(batch, torch.Tensor):
                current_batch_size = batch.size(0)
            elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], torch.Tensor):
                 current_batch_size = batch[0].size(0)
            elif isinstance(batch, dict) and batch:
                 # Find the first tensor value to determine batch size
                 first_tensor_key = next((k for k, v in batch.items() if isinstance(v, torch.Tensor)), None)
                 if first_tensor_key:
                     current_batch_size = batch[first_tensor_key].size(0)
                 else: # Fallback if no tensor found
                     current_batch_size = getattr(trainer.datamodule, 'batch_size', 1)
            else: # Fallback for unknown batch types
                current_batch_size = getattr(trainer.datamodule, 'batch_size', 1)
        except Exception: # Broad except just in case datamodule or batch structure is unexpected
             current_batch_size = getattr(trainer.datamodule, 'batch_size', 1)

        # Use gather_all_tensors for distributed summation
        total_batch_size_tensor = torch.tensor(current_batch_size, device=pl_module.device)
        gathered_sizes = trainer.strategy.all_gather(total_batch_size_tensor)
        total_batch_size = gathered_sizes.sum().item()

        # Only rank 0 needs to track total samples for epoch timing
        if trainer.is_global_zero:
            self.train_samples_in_epoch += total_batch_size
        
        # Gather GPU info only on rank 0
        if trainer.is_global_zero and (pynvml is not None or amdsmi is not None):
            gpu_info = self._get_gpu_info(trainer)
            if gpu_info is not None:
                 # Store GPU info instead of logging immediately
                 self.gpu_metrics_history.append(gpu_info)
            # REMOVED: trainer.logger.log_metrics(gpu_info, step=trainer.global_step)
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Log performance metrics at the end of each validation batch."""
         # Accumulate samples processed across all ranks correctly
        try:
            if isinstance(batch, torch.Tensor):
                current_batch_size = batch.size(0)
            elif isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], torch.Tensor):
                 current_batch_size = batch[0].size(0)
            elif isinstance(batch, dict) and batch:
                 first_tensor_key = next((k for k, v in batch.items() if isinstance(v, torch.Tensor)), None)
                 if first_tensor_key:
                     current_batch_size = batch[first_tensor_key].size(0)
                 else:
                     current_batch_size = getattr(trainer.datamodule, 'val_batch_size', getattr(trainer.datamodule, 'batch_size', 1)) # Check val_batch_size first
            else:
                 current_batch_size = getattr(trainer.datamodule, 'val_batch_size', getattr(trainer.datamodule, 'batch_size', 1))

        except Exception:
             current_batch_size = getattr(trainer.datamodule, 'val_batch_size', getattr(trainer.datamodule, 'batch_size', 1))

        total_batch_size_tensor = torch.tensor(current_batch_size, device=pl_module.device)
        # Handle cases where validation might not be distributed correctly depending on PL version/setup
        if trainer.world_size > 1:
             try:
                 gathered_sizes = trainer.strategy.all_gather(total_batch_size_tensor)
                 total_batch_size = gathered_sizes.sum().item()
             except Exception as e: # Fallback if all_gather fails in validation
                 # self.logger.warning(f"Could not gather batch sizes in validation: {e}. Using local batch size * world_size.")
                 total_batch_size = current_batch_size * trainer.world_size
        else:
             total_batch_size = current_batch_size

        # Only rank 0 needs to track total samples
        if trainer.is_global_zero:
             self.val_samples_in_epoch += total_batch_size
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Calculate and log training performance at the end of the epoch."""
        if not trainer.is_global_zero:
            return
            
        epoch_time = time.time() - self.train_start_time
        if epoch_time > 1e-6 and self.train_samples_in_epoch > 0:
             samples_per_second = self.train_samples_in_epoch / epoch_time
             # Store SPS for final summary
             self.train_epoch_sps.append(samples_per_second)
        else:
             samples_per_second = 0.0
             # Optionally log a warning or skip appending
             self.logger.warning(f"Train Epoch {trainer.current_epoch}: Invalid time ({epoch_time:.4f}s) or samples ({self.train_samples_in_epoch}) for SPS calculation.")

        
        self.logger.info(
            f"Train Epoch {trainer.current_epoch}: "
            f"Processed {self.train_samples_in_epoch} samples in {epoch_time:.2f}s "
            f"({samples_per_second:.2f} samples/sec)"
        )
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Calculate and log validation performance at the end of the epoch."""
        if not trainer.is_global_zero:
            return
            
        # Ensure val_start_time was set
        if self.val_start_time is None:
             self.logger.warning(f"Val Epoch {trainer.current_epoch}: Validation start time not recorded. Skipping SPS calculation.")
             return

        epoch_time = time.time() - self.val_start_time
        if epoch_time > 1e-6 and self.val_samples_in_epoch > 0:
            samples_per_second = self.val_samples_in_epoch / epoch_time
            # Store SPS for final summary
            self.val_epoch_sps.append(samples_per_second)
        else:
            samples_per_second = 0.0
            self.logger.warning(f"Val Epoch {trainer.current_epoch}: Invalid time ({epoch_time:.4f}s) or samples ({self.val_samples_in_epoch}) for SPS calculation.")

        # Keep console log for epoch summary
        self.logger.info(
            f"Val Epoch {trainer.current_epoch}: "
            f"Processed {self.val_samples_in_epoch} samples in {epoch_time:.2f}s "
            f"({samples_per_second:.2f} samples/sec)"
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the final performance summary after training finishes."""
        if not trainer.is_global_zero:
            return

        self.logger.info("="*30 + " Run Summary " + "="*30)

        # Log configuration from the Lightning Module
        model_name = getattr(pl_module, 'model_name', 'N/A')
        img_size = getattr(pl_module, 'img_size', 'N/A')
        batch_size = getattr(pl_module, 'batch_size', 'N/A') # Assuming you added batch_size to LitModel
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Model Name: {model_name}")
        self.logger.info(f"  - Image Size: {img_size}")
        self.logger.info(f"  - Batch Size (per device): {batch_size}")
        self.logger.info("-"*73) # Separator

        # Training Summary
        if self.train_epoch_sps:
            # remove first epoch
            self.train_epoch_sps = self.train_epoch_sps[1:]
            avg_train_sps = np.mean(self.train_epoch_sps)
            std_train_sps = np.std(self.train_epoch_sps)
            max_train_sps = np.max(self.train_epoch_sps)
            min_train_sps = np.min(self.train_epoch_sps)
            self.logger.info(f"Average Training Performance: {avg_train_sps:.2f} +/- {std_train_sps:.2f} samples/sec")
            self.logger.info(f"Maximum Training Performance: {max_train_sps:.2f} samples/sec")
            self.logger.info(f"Minimum Training Performance: {min_train_sps:.2f} samples/sec")
        else:
            self.logger.info("No training epochs were completed or recorded.")

        # Validation Summary
        if self.val_epoch_sps:
            # remove first epoch
            self.val_epoch_sps = self.val_epoch_sps[1:]
            avg_val_sps = np.mean(self.val_epoch_sps)
            std_val_sps = np.std(self.val_epoch_sps)
            max_val_sps = np.max(self.val_epoch_sps)
            min_val_sps = np.min(self.val_epoch_sps)
            self.logger.info(f"Average Validation Performance: {avg_val_sps:.2f} +/- {std_val_sps:.2f} samples/sec")
            self.logger.info(f"Maximum Validation Performance: {max_val_sps:.2f} samples/sec")
            self.logger.info(f"Minimum Validation Performance: {min_val_sps:.2f} samples/sec")
        else:
            self.logger.info("No validation epochs were completed or recorded.")
            
        # GPU Summary
        # Initialize variables that will be used in CSV writing
        num_gpus = len(self.gpu_handles) if self.gpu_handles else 0
        avg_gpu_mem_used = None
        avg_gpu_util = None
        avg_gpu_power = None
        avg_graphics_clock = None
        avg_memory_clock = None
        
        if self.gpu_metrics_history:
            avg_gpu_mem_used = np.mean([m["gpu/memory_used_mb"] for m in self.gpu_metrics_history])
            avg_gpu_util = np.mean([m["gpu/utilization_pct"] for m in self.gpu_metrics_history])
            avg_gpu_power = np.mean([m["gpu/power_usage_watts"] for m in self.gpu_metrics_history])
            avg_graphics_clock = np.mean([m.get("gpu/graphics_clock_mhz", 0) for m in self.gpu_metrics_history])
            avg_memory_clock = np.mean([m.get("gpu/memory_clock_mhz", 0) for m in self.gpu_metrics_history])
            # Get total memory from the first record (assuming it's constant)
            total_gpu_mem = self.gpu_metrics_history[0].get("gpu/memory_total_mb", "N/A")
            
            # Add VRAM tracking metrics
            avg_gpu_vram_used = np.mean([m.get("gpu/vram_used_mb", 0) for m in self.gpu_metrics_history])
            total_gpu_vram = self.gpu_metrics_history[0].get("gpu/vram_total_mb", "N/A")
            vram_utilization_pct = (avg_gpu_vram_used / total_gpu_vram * 100) if isinstance(total_gpu_vram, (int, float)) and total_gpu_vram > 0 else "N/A"

            self.logger.info(f"Average GPU Usage:")
            self.logger.info(f"  - Memory Used: {avg_gpu_mem_used:.2f} MB (of {total_gpu_mem:.2f} MB per GPU)")
            self.logger.info(f"  - VRAM Used: {avg_gpu_vram_used:.2f} MB (of {total_gpu_vram:.2f} MB per GPU, {vram_utilization_pct if isinstance(vram_utilization_pct, str) else f'{vram_utilization_pct:.2f}%'})")
            self.logger.info(f"  - Utilization: {avg_gpu_util:.2f}%")
            self.logger.info(f"  - Power Draw: {avg_gpu_power:.2f} Watts per GPU (avg across {num_gpus} GPUs)")
            self.logger.info(f"  - Graphics Clock: {avg_graphics_clock:.0f} MHz")
            self.logger.info(f"  - Memory Clock: {avg_memory_clock:.0f} MHz")

        elif (pynvml is not None and self.gpu_handles) or (amdsmi is not None and self.gpu_handles):
             self.logger.info("GPU monitoring was active, but no metrics were recorded (check intervals and NVML/AMDSMI errors).")
        else:
             self.logger.info("GPU monitoring was not active (pynvml/amdsmi not found or failed to initialize).")
             
        self.logger.info("="*73) # Match the length of the header
        
        # Write metrics to CSV file
        if self.metrics_file:
            self._write_metrics_to_file(
                model_name=model_name,
                img_size=img_size,
                batch_size=batch_size,
                gpu_name=self.gpu_name,
                avg_train_sps=avg_train_sps if self.train_epoch_sps else None,
                max_train_sps=max_train_sps if self.train_epoch_sps else None,
                min_train_sps=min_train_sps if self.train_epoch_sps else None,
                avg_val_sps=avg_val_sps if self.val_epoch_sps else None,
                max_val_sps=max_val_sps if self.val_epoch_sps else None,
                min_val_sps=min_val_sps if self.val_epoch_sps else None,
                avg_gpu_mem_used=avg_gpu_mem_used if self.gpu_metrics_history else None,
                avg_gpu_util=avg_gpu_util if self.gpu_metrics_history else None,
                avg_gpu_power=avg_gpu_power if self.gpu_metrics_history else None,
                avg_graphics_clock=avg_graphics_clock if self.gpu_metrics_history else None,
                avg_memory_clock=avg_memory_clock if self.gpu_metrics_history else None,
                num_gpus=num_gpus
            )

    def _write_metrics_to_file(self, **metrics):
        """Write performance metrics to a CSV file.
        
        Args:
            **metrics: Dictionary of metric names and values to write
        """
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(self.metrics_file)
        
        # Prepare the row data
        row_data = {
            'timestamp': datetime.now().isoformat(),
            'gpu_name': metrics.get('gpu_name', 'N/A'),
            'num_gpus': metrics.get('num_gpus', 0),
            'batch_size': metrics.get('batch_size', 'N/A'),
            'img_size': metrics.get('img_size', 'N/A'),
            'avg_train_sps': f"{metrics['avg_train_sps']:.2f}" if metrics.get('avg_train_sps') is not None else 'N/A',
            'max_train_sps': f"{metrics['max_train_sps']:.2f}" if metrics.get('max_train_sps') is not None else 'N/A',
            'min_train_sps': f"{metrics['min_train_sps']:.2f}" if metrics.get('min_train_sps') is not None else 'N/A',
            'avg_val_sps': f"{metrics['avg_val_sps']:.2f}" if metrics.get('avg_val_sps') is not None else 'N/A',
            'max_val_sps': f"{metrics['max_val_sps']:.2f}" if metrics.get('max_val_sps') is not None else 'N/A',
            'min_val_sps': f"{metrics['min_val_sps']:.2f}" if metrics.get('min_val_sps') is not None else 'N/A',
            'avg_gpu_mem_mb': f"{metrics['avg_gpu_mem_used']:.2f}" if metrics.get('avg_gpu_mem_used') is not None else 'N/A',
            'avg_gpu_util_pct': f"{metrics['avg_gpu_util']:.2f}" if metrics.get('avg_gpu_util') is not None else 'N/A',
            'avg_gpu_power_watts': f"{metrics['avg_gpu_power']:.2f}" if metrics.get('avg_gpu_power') is not None else 'N/A',
            'avg_graphics_clock_mhz': f"{metrics['avg_graphics_clock']:.0f}" if metrics.get('avg_graphics_clock') is not None else 'N/A',
            'avg_memory_clock_mhz': f"{metrics['avg_memory_clock']:.0f}" if metrics.get('avg_memory_clock') is not None else 'N/A',
        }
        
        try:
            with open(self.metrics_file, 'a', newline='') as csvfile:
                fieldnames = list(row_data.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                    self.logger.info(f"Created new metrics file: {self.metrics_file}")
                
                # Write the metrics row
                writer.writerow(row_data)
                self.logger.info(f"Metrics written to: {self.metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to write metrics to file: {e}")
