import time
import logging
import csv
import os
import threading
import time
import queue
from datetime import datetime
from collections import deque
from typing import Any, Dict, Optional
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

try:
    import pynvml  # Add this import for NVIDIA GPU monitoring
    NVMLError = pynvml.NVMLError
except ImportError:
    pynvml = None
    pynvml_import_error = True
    class NVMLError(Exception): pass
else:
    pynvml_import_error = False


try:
    #from pyrsmi import rocml
    import ldm.callbacks.rocml as rocml
    rocml_imported  = True
except ImportError:
    rocml = None
    rocml_imported = False
    rocml_import_error = True
    # logging.warning("rocml library not found. AMD GPU monitoring disabled.")
else:
    rocml_import_error = False

class PerformanceTrackingCallbackAsync(pl.Callback):
    """Callback to track samples per second during training and validation with async GPU monitoring."""

    @rank_zero_only
    def __init__(self, metrics_file: str = "performance_metrics.csv",
                 sampling_interval: float = 0.1):
        """Initialize the callback.

        Args:
            metrics_file: Path to CSV file where metrics will be saved. Defaults to 'performance_metrics.csv'.
            sampling_interval: How often to sample GPU metrics in seconds (default: 0.1 = 100ms)
        """

        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return
        except RuntimeError:
            return

        global pynvml # Declare intent to modify global pynvml at the start
        global rocml # Declare intent to modify global rocml at the start
        super().__init__()

        # Store metrics file path
        self.metrics_file = metrics_file
        self.sampling_interval = sampling_interval

        # Initialize tracking variables
        self.train_start_time = None
        self.val_start_time = None
        self.train_samples_in_epoch = 0
        self.val_samples_in_epoch = 0

        # Store metrics for final summary
        self.train_epoch_sps = []
        self.val_epoch_sps = []
        self.gpu_metrics_history = []

        # Async monitoring setup
        self.metrics_queue = queue.Queue()
        self.monitoring_active = threading.Event()  # Controls when to monitor
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()  # Signal to stop thread completely
        self.current_step = 0  # Track current global step
        self.step_lock = threading.Lock()  # Thread-safe step updates

        self._gpu_monitor_backend = "none"
        self._gpu_monitor_init_error = None
        self._rocml_raw_logged = False

        self.num_gpus_measured = 0

        if pynvml is not None:
            # Initialize NVIDIA Management Library
            try: # Add try-except for robustness
                pynvml.nvmlInit()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())]
                self._gpu_monitor_backend = "nvml"
                rocml = None # Ensure rocml is not used if NVML is available
                rocml_imported = False
            except pynvml.NVMLError as error:
                print(f"Failed to initialize NVML: {error}. GPU monitoring disabled.")
                pynvml = None # Ensure pynvml checks fail later
                self.gpu_handles = []
    
        elif rocml is not None:
            try:
                rocml.smi_initialize()

                # Store device IDs as "handles" (rocml uses integer device IDs)
                self.gpu_handles = []
                try:
                    device_count = rocml.smi_get_device_count()
                except Exception as e:
                    self.logger.error(f"Failed to get AMD device count: {e}")
                    device_count = 0

                for device_id in range(device_count):
                    try:
                        # Verify device has memory (to filter out non-GPU devices)
                        mem_total = rocml.smi_get_device_memory_total(device_id)
                        if mem_total and mem_total > 1 * (1024**3):  # More than 1GB
                            self.gpu_handles.append(device_id)
                    except Exception:
                        pass
                    
                if len(self.gpu_handles) > 0:
                    print(f"Initialized rocml for {len(self.gpu_handles)} AMD GPUs.")
                    self._gpu_monitor_backend = "rocml"
                else:
                    print("rocml initialized, but no AMD GPUs were found.")
                    rocml.smi_shutdown()
                    rocml = None


            except Exception as e:
                self._gpu_monitor_init_error = f"rocml init failed: {e}"
                print(f"Failed to initialize rocml: {e}. rocml monitoring disabled.")
                rocml = None

        else:
            self.gpu_handles = []

        # Get GPU name for logging
        self.gpu_name = self._get_gpu_name()

        # Get GPU hardware limits (power cap, max clocks)
        self.gpu_limits = self._get_gpu_limits()

        # Setup logging
        self._setup_logging()

        if getattr(self, 'logger', None) is not None:
            if pynvml is not None and self.gpu_handles:
                self.logger.info(f"GPU monitoring backend: NVML (handles={len(self.gpu_handles)})")
            elif rocml is not None and self.gpu_handles:
                self.logger.info(f"GPU monitoring backend: rocml (handles={len(self.gpu_handles)})")
            else:
                self.logger.info(f"GPU monitoring backend: none (pynvml={'yes' if pynvml is not None else 'no'}, rocml={'yes' if rocml is not None else 'no'}, handles={len(self.gpu_handles)})")
                self.logger.info(f"GPU monitoring debug (interpreter): sys.executable={sys.executable}")
                self.logger.info(f"GPU monitoring debug (imports): pynvml_import_error={'yes' if pynvml_import_error else 'no'}, rocml_import_error={'yes' if rocml_import_error else 'no'}")
            if self._gpu_monitor_init_error:
                self.logger.error(self._gpu_monitor_init_error)

        # Initialize power tracking
        self.power_readings = deque(maxlen=500)  # Store last 500 readings
        self.power_readings_time = deque(maxlen=500)

        # Start monitoring thread if GPUs are available
        if (pynvml is not None or rocml is not None) and self.gpu_handles:
            self.monitor_thread = threading.Thread(target=self._continuous_monitor, daemon=True)
            self.monitor_thread.start()

    @rank_zero_only
    def __del__(self):
        # Stop monitoring thread
        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                return
        except RuntimeError:
            return

        if self.monitor_thread is not None:
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=2.0)

        if pynvml is not None: # Check if nvml was initialized
            try:
                pynvml.nvmlShutdown()
            except:
                pass # Ignore shutdown errors
        elif rocml is not None: # Check if rocml was initialized
            try:
                # Check if we actually initialized it
                if hasattr(self, 'gpu_handles') and self.gpu_handles:
                    rocml.smi_shutdown()
            except Exception as e:
                # Silent failure is okay in __del__, but log if possible
                pass

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
            elif rocml is not None and self.gpu_handles:
                # Get AMD GPU name
                return rocml.smi_get_device_name(self.gpu_handles[0])
            elif torch.cuda.is_available():
                # Fallback to PyTorch
                return torch.cuda.get_device_name(0)
            else:
                return 'N/A'
        except Exception as e:
            return 'Unknown GPU'

    def _get_gpu_limits(self):
        """Get GPU hardware limits (power cap, max clocks) - both configured and absolute hardware max."""
        limits = {
            # Current configured limits
            'power_limit_watts': None,
            'max_graphics_clock_mhz': None,
            'max_memory_clock_mhz': None,
            # Absolute hardware maximums
            'hw_max_power_watts': None,
            'hw_max_graphics_clock_mhz': None,
            'hw_max_memory_clock_mhz': None,
        }

        try:
            if pynvml is not None and self.gpu_handles:
                handle = self.gpu_handles[0]  # Assuming homogeneous setup

                # Get current power limit (configured cap)
                try:
                    power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    limits['power_limit_watts'] = power_limit_mw / 1000.0  # Convert mW to W
                except NVMLError:
                    limits['power_limit_watts'] = None

                # Get hardware maximum power limit (absolute max the hardware can do)
                try:
                    constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                    # constraints is a tuple: (min_limit, max_limit) in milliwatts
                    limits['hw_max_power_watts'] = constraints[1] / 1000.0  # Convert mW to W
                except NVMLError:
                    limits['hw_max_power_watts'] = None

                # Get max boost clock (this is typically the maximum the GPU will boost to)
                try:
                    limits['max_graphics_clock_mhz'] = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                except NVMLError:
                    limits['max_graphics_clock_mhz'] = None

                # Get max memory clock
                try:
                    limits['max_memory_clock_mhz'] = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except NVMLError:
                    limits['max_memory_clock_mhz'] = None

                # For NVIDIA, the max boost clock IS the hardware max, so set both to same value
                limits['hw_max_graphics_clock_mhz'] = limits['max_graphics_clock_mhz']
                limits['hw_max_memory_clock_mhz'] = limits['max_memory_clock_mhz']

            elif rocml is not None and self.gpu_handles:
                handle = self.gpu_handles[0]

                power_cap_info = rocml.smi_get_power_cap_info(handle)
                limits['power_limit_watts'] = power_cap_info[0] / 1000000.0 if power_cap_info[0] != -1 else None
                limits['hw_max_power_watts'] = power_cap_info[1] / 1000000.0 if power_cap_info[1] != -1 else None

                clk_info_gfx = rocml.smi_get_clk_info(handle, 0)
                limits['max_graphics_clock_mhz'] = clk_info_gfx[2] if clk_info_gfx[2] != -1 else None

                clk_info_mem = rocml.smi_get_clk_info(handle, 4)
                limits['max_memory_clock_mhz'] = clk_info_mem[2] if clk_info_mem[2] != -1 else None

                # cannot differentiate as well
                limits['hw_max_graphics_clock_mhz'] = limits['max_graphics_clock_mhz']
                limits['hw_max_memory_clock_mhz'] = limits['max_memory_clock_mhz']

        except Exception as e:
            # If any error occurs, just return None values
            pass

        return limits

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

    def _continuous_monitor(self):
        """Background thread that continuously monitors GPU metrics."""
        global pynvml
        global rocml

        while not self.stop_monitoring.is_set():
            # Only collect metrics when monitoring is active (during epochs)
            if self.monitoring_active.is_set():
                try:
                    gpu_info = self._get_gpu_metrics()
                    if gpu_info is not None:
                        # Get current step in thread-safe manner
                        with self.step_lock:
                            gpu_info["step"] = self.current_step
                        # Store in queue
                        self.metrics_queue.put(gpu_info)
                except Exception as e:
                    if not getattr(self, '_monitor_error_logged', False):
                        self.logger.error(f"Error in monitoring thread: {e}")
                        self._monitor_error_logged = True

            # Sleep for the sampling interval
            time.sleep(self.sampling_interval)

    def _get_gpu_metrics(self):
        """Retrieve GPU usage statistics (called from background thread)."""
        global pynvml
        global rocml

        def _read_sysfs_int(path: str) -> Optional[int]:
            try:
                with open(path, 'r') as f:
                    return int(f.read().strip())
            except Exception:
                return None

        try:
            total_power = 0

            num_gpus = len(self.gpu_handles)
            handles = self.gpu_handles
            self.num_gpus_measured = 0

            max_mem_util = 0
            gpu_info = {
                "gpu/memory_used_mb": 0,
                "gpu/memory_total_mb": 0,
                "gpu/utilization_pct": 0,
                "gpu/memory_utilization_pct": 0,
                "gpu/memory_utilization_max_pct": 0,
                "gpu/power_usage_watts": 0,
                "gpu/vram_used_mb": 0,
                "gpu/vram_total_mb": 0,
                "gpu/graphics_clock_mhz": 0,
                "gpu/memory_clock_mhz": 0,
            }

            if pynvml is not None:
                for i, handle in enumerate(handles):
                    self.num_gpus_measured = len(handles)

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
                        mem_util = utilization.memory
                    except NVMLError:
                        gpu_util = 0
                        mem_util = 0

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
                    gpu_info["gpu/memory_utilization_pct"] += mem_util
                    max_mem_util = max(max_mem_util, mem_util)
                    gpu_info["gpu/graphics_clock_mhz"] += graphics_clock
                    gpu_info["gpu/memory_clock_mhz"] += memory_clock
                    total_power += power_usage

            elif rocml is not None:
                for handle in self.gpu_handles:

                    try:
                        used_bytes = rocml.smi_get_device_memory_used(handle, 'VRAM')
                        total_bytes = rocml.smi_get_device_memory_total(handle, 'VRAM') #Seems to be broken in rocml
                    except:
                        used_bytes = 0
                        total_bytes = 0
                    used_mb = used_bytes / (1024**2)
                    total_mb = total_bytes / (1024**2)

                    # Do not account GPUs with a memory usage below 1%
                    if used_mb <= 5000:
                        continue
                    else:
                        self.num_gpus_measured += 1

                    try:
                        mem_util = rocml.smi_get_device_memory_busy(handle)
                    except:
                        mem_util = 0


                    try:
                        gpu_util = rocml.smi_get_device_utilization(handle)
                    except:
                        gpu_util = 0

                    try:
                        power_watts = rocml.smi_get_device_average_power(handle)
                    except:
                        power_watts = 0


                    try:
                        gpu_clocks = rocml.smi_get_clk_info(handle, 0)
                        graphics_clock = gpu_clocks[0]
                    except:
                        graphics_clock = 0

                    try:
                        mem_clocks = rocml.smi_get_clk_info(handle, 4)
                        memory_clock = mem_clocks[0]
                    except:
                        memory_clock = 0

                    gpu_info["gpu/memory_used_mb"] += used_mb
                    gpu_info["gpu/vram_used_mb"] += used_mb
                    gpu_info["gpu/memory_total_mb"] += total_mb
                    gpu_info["gpu/vram_total_mb"] += total_mb
                    gpu_info["gpu/utilization_pct"] += gpu_util
                    gpu_info["gpu/memory_utilization_pct"] += mem_util
                    max_mem_util = max(max_mem_util, mem_util)
                    total_power += power_watts
                    gpu_info["gpu/graphics_clock_mhz"] += graphics_clock
                    gpu_info["gpu/memory_clock_mhz"] += memory_clock




            # Store power readings
            self.power_readings.append(total_power)
            self.power_readings_time.append(time.time())

            # Normalize values per GPU
            if self.num_gpus_measured > 0:
                gpu_info["gpu/memory_used_mb"] /= self.num_gpus_measured
                gpu_info["gpu/memory_total_mb"] /= self.num_gpus_measured
                gpu_info["gpu/utilization_pct"] /= self.num_gpus_measured
                gpu_info["gpu/memory_utilization_pct"] /= self.num_gpus_measured
                gpu_info["gpu/memory_utilization_max_pct"] = max_mem_util
                gpu_info["gpu/vram_used_mb"] /= self.num_gpus_measured
                gpu_info["gpu/vram_total_mb"] /= self.num_gpus_measured
                gpu_info["gpu/graphics_clock_mhz"] /= self.num_gpus_measured
                gpu_info["gpu/memory_clock_mhz"] /= self.num_gpus_measured
                # Normalize power to per-GPU average
                gpu_info["gpu/power_usage_watts"] = total_power / self.num_gpus_measured
            else:
                gpu_info["gpu/power_usage_watts"] = total_power
                gpu_info["gpu/memory_utilization_max_pct"] = 0

            return gpu_info

        except NVMLError as e: # Catch specific NVML errors
            # Log error only once to avoid flooding
            if not getattr(self, '_nvml_error_logged', False):
                self.logger.error(f"NVML error during GPU monitoring: {e}.")
                self._nvml_error_logged = True
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
            # Start async monitoring
            self.monitoring_active.set()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset counters at the start of each validation epoch."""
        if trainer.is_global_zero:
            self.val_start_time = time.time()
            self.val_samples_in_epoch = 0
            # Start async monitoring
            self.monitoring_active.set()

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
            # Enhanced batch size detection with better error handling
            current_batch_size = None

            # Method 1: Check if batch is a tensor
            if isinstance(batch, torch.Tensor):
                current_batch_size = batch.size(0)
            # Method 2: Check if batch is a list/tuple containing tensors
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                if isinstance(batch[0], torch.Tensor):
                    current_batch_size = batch[0].size(0)
            # Method 3: Check if batch is a dictionary containing tensors
            elif isinstance(batch, dict) and len(batch) > 0:
                first_tensor_key = next((k for k, v in batch.items() if isinstance(v, torch.Tensor)), None)
                if first_tensor_key is not None:
                    current_batch_size = batch[first_tensor_key].size(0)

            # Fallback hierarchy if tensor inspection failed
            if current_batch_size is None:
                # Try pl_module first (most accurate after tuning)
                current_batch_size = getattr(pl_module, 'batch_size', None)

                # Then try datamodule
                if current_batch_size is None:
                    current_batch_size = getattr(trainer.datamodule, 'batch_size', None)

                # Ultimate fallback
                if current_batch_size is None:
                    current_batch_size = 1
                    if not getattr(self, '_batch_size_fallback_warned', False):
                        self.logger.warning("Could not determine batch size, using fallback value of 1")
                        self._batch_size_fallback_warned = True

        except Exception as e:
            # Catch-all for any unexpected errors
            if not getattr(self, '_batch_size_error_logged', False):
                self.logger.error(f"Error detecting batch size: {e}. Using fallback of 1.")
                self._batch_size_error_logged = True
            current_batch_size = 1

        # Use gather_all_tensors for distributed summation
        total_batch_size_tensor = torch.tensor(current_batch_size, device=pl_module.device)
        gathered_sizes = trainer.strategy.all_gather(total_batch_size_tensor)
        total_batch_size = gathered_sizes.sum().item()

        # Only rank 0 needs to track total samples for epoch timing
        if trainer.is_global_zero:
            self.train_samples_in_epoch += total_batch_size

            # Update current step for monitoring thread
            with self.step_lock:
                self.current_step = trainer.global_step

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

             # Update current step for monitoring thread
             with self.step_lock:
                 self.current_step = trainer.global_step

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Calculate and log training performance at the end of the epoch."""
        if not trainer.is_global_zero:
            return

        # Stop async monitoring
        self.monitoring_active.clear()

        # Collect all metrics from queue
        self._collect_metrics_from_queue()

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

        # Stop async monitoring
        self.monitoring_active.clear()

        # Collect all metrics from queue
        self._collect_metrics_from_queue()

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

    def _collect_metrics_from_queue(self):
        """Collect all pending metrics from the queue and add to history."""
        while not self.metrics_queue.empty():
            try:
                gpu_info = self.metrics_queue.get_nowait()
                self.gpu_metrics_history.append(gpu_info)
            except queue.Empty:
                break

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the final performance summary after training finishes."""
        if not trainer.is_global_zero:
            return

        # Ensure all remaining metrics are collected
        self._collect_metrics_from_queue()

        self.logger.info("="*30 + " Run Summary " + "="*30)

        # Log configuration from the Lightning Module
        model_name = getattr(pl_module, 'model_name', 'N/A')
        img_size = getattr(pl_module, 'img_size', 'N/A')
        batch_size = getattr(pl_module, 'batch_size', getattr(trainer.datamodule, 'batch_size', 'N/A'))

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
        num_gpus = self.num_gpus_measured
        avg_gpu_mem_used = None
        avg_gpu_util = None
        avg_gpu_mem_util = None
        max_gpu_mem_util = None
        avg_gpu_power = None
        avg_graphics_clock = None
        avg_memory_clock = None
        vram_utilization_pct = None

        if self.gpu_metrics_history:
            avg_gpu_mem_used = np.mean([m["gpu/memory_used_mb"] for m in self.gpu_metrics_history])
            avg_gpu_util = np.mean([m["gpu/utilization_pct"] for m in self.gpu_metrics_history])
            avg_gpu_mem_util = np.mean([m.get("gpu/memory_utilization_pct", 0) for m in self.gpu_metrics_history])
            max_gpu_mem_util = np.mean([m.get("gpu/memory_utilization_max_pct", 0) for m in self.gpu_metrics_history])
            avg_gpu_power = np.mean([m["gpu/power_usage_watts"] for m in self.gpu_metrics_history])
            avg_graphics_clock = np.mean([m.get("gpu/graphics_clock_mhz", 0) for m in self.gpu_metrics_history])
            avg_memory_clock = np.mean([m.get("gpu/memory_clock_mhz", 0) for m in self.gpu_metrics_history])
            # Get total memory from the first record (assuming it's constant)
            total_gpu_mem = self.gpu_metrics_history[0].get("gpu/memory_total_mb", "N/A")

            # Add VRAM tracking metrics
            avg_gpu_vram_used = np.mean([m.get("gpu/vram_used_mb", 0) for m in self.gpu_metrics_history])
            total_gpu_vram = self.gpu_metrics_history[0].get("gpu/vram_total_mb", "N/A")
            vram_utilization_pct = (avg_gpu_vram_used / total_gpu_vram * 100) if isinstance(total_gpu_vram, (int, float)) and total_gpu_vram > 0 else None

            self.logger.info(f"Average GPU Usage:")
            self.logger.info(f"  - Memory Used: {avg_gpu_mem_used:.2f} MB (of {total_gpu_mem:.2f} MB per GPU)")
            self.logger.info(f"  - VRAM Used: {avg_gpu_vram_used:.2f} MB (of {total_gpu_vram:.2f} MB per GPU, {f'{vram_utilization_pct:.2f}%' if vram_utilization_pct is not None else 'N/A'})")
            self.logger.info(f"  - Utilization: {avg_gpu_util:.2f}%")
            self.logger.info(f"  - Memory Utilization (avg): {avg_gpu_mem_util:.2f}%")
            self.logger.info(f"  - Memory Utilization (max): {max_gpu_mem_util:.2f}%")
            self.logger.info(f"  - Power Draw: {avg_gpu_power:.2f} Watts per GPU (avg across {num_gpus} GPUs)")
            self.logger.info(f"  - Graphics Clock: {avg_graphics_clock:.0f} MHz")
            self.logger.info(f"  - Memory Clock: {avg_memory_clock:.0f} MHz")
            self.logger.info(f"  - Total Samples Collected: {len(self.gpu_metrics_history)} (async @ {int(1/self.sampling_interval)}Hz)")

        # Log GPU hardware limits
        self.logger.info(
            f"GPU monitoring debug: pynvml={'yes' if pynvml is not None else 'no'}, "
            f"rocml={'yes' if rocml is not None else 'no'}, "
            f"gpu_handles={len(self.gpu_handles) if self.gpu_handles else 0}, "
            f"gpu_metrics_samples={len(self.gpu_metrics_history)}"
        )

        if self.gpu_limits and any(v is not None for v in self.gpu_limits.values()):
            self.logger.info(f"GPU Configuration:")
            if self.gpu_limits.get('power_limit_watts') is not None:
                self.logger.info(f"  - Power Limit (Configured): {self.gpu_limits['power_limit_watts']:.2f} Watts")
            if self.gpu_limits.get('hw_max_power_watts') is not None:
                self.logger.info(f"  - Power Limit (HW Max): {self.gpu_limits['hw_max_power_watts']:.2f} Watts")
            if self.gpu_limits.get('max_graphics_clock_mhz') is not None:
                self.logger.info(f"  - Max Graphics Clock: {self.gpu_limits['max_graphics_clock_mhz']:.0f} MHz")
            if self.gpu_limits.get('hw_max_graphics_clock_mhz') is not None and \
               self.gpu_limits.get('hw_max_graphics_clock_mhz') != self.gpu_limits.get('max_graphics_clock_mhz'):
                self.logger.info(f"  - Max Graphics Clock (HW Max): {self.gpu_limits['hw_max_graphics_clock_mhz']:.0f} MHz")
            if self.gpu_limits.get('max_memory_clock_mhz') is not None:
                self.logger.info(f"  - Max Memory Clock: {self.gpu_limits['max_memory_clock_mhz']:.0f} MHz")
            if self.gpu_limits.get('hw_max_memory_clock_mhz') is not None and \
               self.gpu_limits.get('hw_max_memory_clock_mhz') != self.gpu_limits.get('max_memory_clock_mhz'):
                self.logger.info(f"  - Max Memory Clock (HW Max): {self.gpu_limits['hw_max_memory_clock_mhz']:.0f} MHz")

        elif self._gpu_monitor_backend != "none" and self.gpu_handles and self.gpu_metrics_history:
             self.logger.info("GPU monitoring was active and metrics were recorded.")
        elif self._gpu_monitor_backend != "none" and self.gpu_handles and not self.gpu_metrics_history:
             self.logger.info("GPU monitoring was active, but no metrics were recorded (check intervals and NVML/rocml errors).")
        else:
             self.logger.info("GPU monitoring was not active (pynvml/rocml not found or failed to initialize).")

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
                vram_utilization_pct=vram_utilization_pct if self.gpu_metrics_history else None,
                avg_gpu_util=avg_gpu_util if self.gpu_metrics_history else None,
                avg_gpu_mem_util=avg_gpu_mem_util if self.gpu_metrics_history else None,
                max_gpu_mem_util=max_gpu_mem_util if self.gpu_metrics_history else None,
                avg_gpu_power=avg_gpu_power if self.gpu_metrics_history else None,
                avg_graphics_clock=avg_graphics_clock if self.gpu_metrics_history else None,
                avg_memory_clock=avg_memory_clock if self.gpu_metrics_history else None,
                num_gpus=self.num_gpus_measured,
                power_limit_watts=self.gpu_limits.get('power_limit_watts'),
                hw_max_power_watts=self.gpu_limits.get('hw_max_power_watts'),
                max_graphics_clock_mhz=self.gpu_limits.get('max_graphics_clock_mhz'),
                hw_max_graphics_clock_mhz=self.gpu_limits.get('hw_max_graphics_clock_mhz'),
                max_memory_clock_mhz=self.gpu_limits.get('max_memory_clock_mhz'),
                hw_max_memory_clock_mhz=self.gpu_limits.get('hw_max_memory_clock_mhz')
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
            'avg_vram_util_pct': f"{metrics['vram_utilization_pct']:.2f}" if metrics.get('vram_utilization_pct') is not None else 'N/A',
            'avg_gpu_util_pct': f"{metrics['avg_gpu_util']:.2f}" if metrics.get('avg_gpu_util') is not None else 'N/A',
            'avg_gpu_mem_util_pct': f"{metrics['avg_gpu_mem_util']:.2f}" if metrics.get('avg_gpu_mem_util') is not None else 'N/A',
            'avg_gpu_mem_util_max_pct': f"{metrics['max_gpu_mem_util']:.2f}" if metrics.get('max_gpu_mem_util') is not None else 'N/A',
            'avg_gpu_power_watts': f"{metrics['avg_gpu_power']:.2f}" if metrics.get('avg_gpu_power') is not None else 'N/A',
            'avg_graphics_clock_mhz': f"{metrics['avg_graphics_clock']:.0f}" if metrics.get('avg_graphics_clock') is not None else 'N/A',
            'avg_memory_clock_mhz': f"{metrics['avg_memory_clock']:.0f}" if metrics.get('avg_memory_clock') is not None else 'N/A',
            'power_limit_watts': f"{metrics['power_limit_watts']:.2f}" if metrics.get('power_limit_watts') is not None else 'N/A',
            'hw_max_power_watts': f"{metrics['hw_max_power_watts']:.2f}" if metrics.get('hw_max_power_watts') is not None else 'N/A',
            'max_graphics_clock_mhz': f"{metrics['max_graphics_clock_mhz']:.0f}" if metrics.get('max_graphics_clock_mhz') is not None else 'N/A',
            'hw_max_graphics_clock_mhz': f"{metrics['hw_max_graphics_clock_mhz']:.0f}" if metrics.get('hw_max_graphics_clock_mhz') is not None else 'N/A',
            'max_memory_clock_mhz': f"{metrics['max_memory_clock_mhz']:.0f}" if metrics.get('max_memory_clock_mhz') is not None else 'N/A',
            'hw_max_memory_clock_mhz': f"{metrics['hw_max_memory_clock_mhz']:.0f}" if metrics.get('hw_max_memory_clock_mhz') is not None else 'N/A',
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
