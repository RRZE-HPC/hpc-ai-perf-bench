import time
import logging
import csv
import os
import threading
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
    import amdsmi
    amdsmi_imported = True

    AmdSmiLibraryException = getattr(amdsmi, 'AmdSmiLibraryException', None)
    if AmdSmiLibraryException is None:
        AmdSmiLibraryException = getattr(amdsmi, 'AmdSmiException', Exception)

    AMDSMI_MEM_TYPE_VRAM = getattr(amdsmi, 'AMDSMI_MEM_TYPE_VRAM', None)
    if AMDSMI_MEM_TYPE_VRAM is None:
        mem_enum = getattr(amdsmi, 'AmdSmiMemoryType', None)
        if mem_enum is not None:
            AMDSMI_MEM_TYPE_VRAM = getattr(mem_enum, 'VRAM', None)
        else:
            AMDSMI_MEM_TYPE_VRAM = None

except ImportError:
    amdsmi = None
    class AmdSmiLibraryException(Exception):
        pass
    AMDSMI_MEM_TYPE_VRAM = None
    amdsmi_imported = False
    amdsmi_import_error = True
    # logging.warning("AMDSMI library not found. AMD GPU monitoring disabled.")
else:
    amdsmi_import_error = False

class PerformanceTrackingCallbackAsync(pl.Callback):
    """Callback to track samples per second during training and validation with async GPU monitoring."""

    def __init__(self, metrics_file: str = "performance_metrics.csv",
                 sampling_interval: float = 0.1):
        """Initialize the callback.

        Args:
            metrics_file: Path to CSV file where metrics will be saved. Defaults to 'performance_metrics.csv'.
            sampling_interval: How often to sample GPU metrics in seconds (default: 0.1 = 100ms)
        """
        global pynvml # Declare intent to modify global pynvml at the start
        global amdsmi # Declare intent to modify global amdsmi at the start
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
        self._amdsmi_raw_logged = False

        if pynvml is not None:
            # Initialize NVIDIA Management Library
            try: # Add try-except for robustness
                pynvml.nvmlInit()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())]
                self._gpu_monitor_backend = "nvml"
            except pynvml.NVMLError as error:
                print(f"Failed to initialize NVML: {error}. GPU monitoring disabled.")
                pynvml = None # Ensure pynvml checks fail later
                self.gpu_handles = []
        if pynvml is None and amdsmi is not None:
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
                         if AMDSMI_MEM_TYPE_VRAM is not None:
                             amdsmi.amdsmi_get_gpu_memory_usage(handle, mem_type=AMDSMI_MEM_TYPE_VRAM)
                             mem_total_bytes = amdsmi.amdsmi_get_gpu_memory_total(handle, mem_type=AMDSMI_MEM_TYPE_VRAM)
                         else:
                             amdsmi.amdsmi_get_gpu_memory_usage(handle)
                             mem_total_bytes = amdsmi.amdsmi_get_gpu_memory_total(handle)
                         if isinstance(mem_total_bytes, dict):
                             mem_total_bytes = mem_total_bytes.get('vram_total', 0)
                         if mem_total_bytes and mem_total_bytes > 1 * (1024**3): # More than 1GB VRAM
                            self.gpu_handles.append(handle)
                    except (AmdSmiLibraryException, TypeError):
                        # Handle cases where memory info might not be available for a device
                        pass # Or log a warning
                if len(self.gpu_handles) > 0:
                    print(f"Initialized AMDSMI for {len(self.gpu_handles)} AMD GPUs.")
                    self._gpu_monitor_backend = "amdsmi"
                else:
                    self.gpu_handles = list(all_handles)
                    if len(self.gpu_handles) > 0:
                        print(f"Initialized AMDSMI for {len(self.gpu_handles)} AMD GPUs (filter fallback).")
                        self._gpu_monitor_backend = "amdsmi"
                    else:
                        print("AMDSMI initialized, but no AMD GPUs were returned by amdsmi_get_processor_handles().")
                        amdsmi.amdsmi_shut_down() # Shutdown if no usable GPUs
                        amdsmi = None
            except Exception as e:
                self._gpu_monitor_init_error = f"amdsmi init failed: {e}"
                print(f"Failed to initialize AMDSMI: {e}. AMDSMI monitoring disabled.")
                amdsmi = None # Ensure amdsmi is None if init failed
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
            elif amdsmi is not None and self.gpu_handles:
                self.logger.info(f"GPU monitoring backend: AMDSMI (handles={len(self.gpu_handles)})")
            else:
                self.logger.info(f"GPU monitoring backend: none (pynvml={'yes' if pynvml is not None else 'no'}, amdsmi={'yes' if amdsmi is not None else 'no'}, handles={len(self.gpu_handles)})")
                self.logger.info(f"GPU monitoring debug (interpreter): sys.executable={sys.executable}")
                self.logger.info(f"GPU monitoring debug (imports): pynvml_import_error={'yes' if pynvml_import_error else 'no'}, amdsmi_import_error={'yes' if amdsmi_import_error else 'no'}")
            if self._gpu_monitor_init_error:
                self.logger.error(self._gpu_monitor_init_error)

        # Initialize power tracking
        self.power_readings = deque(maxlen=500)  # Store last 500 readings
        self.power_readings_time = deque(maxlen=500)

        # Start monitoring thread if GPUs are available
        if (pynvml is not None or amdsmi is not None) and self.gpu_handles:
            self.monitor_thread = threading.Thread(target=self._continuous_monitor, daemon=True)
            self.monitor_thread.start()

    def __del__(self):
        # Stop monitoring thread
        if self.monitor_thread is not None:
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=2.0)

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

            elif amdsmi is not None and self.gpu_handles:
                handle = self.gpu_handles[0]  # Assuming homogeneous setup

                # Get current power cap (configured)
                try:
                    power_cap_info = amdsmi.amdsmi_get_power_cap(handle)
                    # power_cap_info is a dict with 'power_cap' in microwatts
                    limits['power_limit_watts'] = power_cap_info.get('power_cap', 0) / 1_000_000.0  # Convert uW to W
                except (AmdSmiLibraryException, KeyError):
                    limits['power_limit_watts'] = None

                # Try to get hardware max power (TDP or max power cap)
                try:
                    # Get power info which may contain max power
                    power_info = amdsmi.amdsmi_get_power_info(handle)
                    # power_info might have 'power_limit' or 'max_power' in microwatts
                    hw_max = power_info.get('power_limit', power_info.get('max_power', None))
                    if hw_max:
                        limits['hw_max_power_watts'] = hw_max / 1_000_000.0  # Convert uW to W
                except (AmdSmiLibraryException, KeyError, AttributeError):
                    limits['hw_max_power_watts'] = None

                # Get max clocks (current configured max)
                try:
                    gfx_clocks = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX)
                    limits['max_graphics_clock_mhz'] = gfx_clocks.get('max_clk', None)
                    # For AMD, max_clk should be the hardware max
                    limits['hw_max_graphics_clock_mhz'] = gfx_clocks.get('max_clk', None)
                except (AmdSmiLibraryException, KeyError):
                    limits['max_graphics_clock_mhz'] = None
                    limits['hw_max_graphics_clock_mhz'] = None

                try:
                    mem_clocks = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM)
                    limits['max_memory_clock_mhz'] = mem_clocks.get('max_clk', None)
                    # For AMD, max_clk should be the hardware max
                    limits['hw_max_memory_clock_mhz'] = mem_clocks.get('max_clk', None)
                except (AmdSmiLibraryException, KeyError):
                    limits['max_memory_clock_mhz'] = None
                    limits['hw_max_memory_clock_mhz'] = None

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
        global amdsmi

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
        global amdsmi

        def _read_sysfs_int(path: str) -> Optional[int]:
            try:
                with open(path, 'r') as f:
                    return int(f.read().strip())
            except Exception:
                return None

        try:
            total_power = 0
            # num_gpus = len(self.gpu_handles)
            num_gpus = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)) * int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
            visible_gpus = len(self.gpu_handles)
            if visible_gpus > 0:
                num_gpus = max(1, min(num_gpus, visible_gpus))
            handles = self.gpu_handles[:num_gpus] if visible_gpus > 0 else []
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
            elif amdsmi is not None:
                for i, handle in enumerate(handles):
                    try:
                        if AMDSMI_MEM_TYPE_VRAM is not None:
                            used = amdsmi.amdsmi_get_gpu_memory_usage(handle, mem_type=AMDSMI_MEM_TYPE_VRAM)
                            total = amdsmi.amdsmi_get_gpu_memory_total(handle, mem_type=AMDSMI_MEM_TYPE_VRAM)
                        else:
                            used = amdsmi.amdsmi_get_gpu_memory_usage(handle)
                            total = amdsmi.amdsmi_get_gpu_memory_total(handle)
                    except (AmdSmiLibraryException, TypeError):
                        used = 0
                        total = 0

                    if isinstance(used, dict):
                        used_bytes = used.get('vram_used', 0)
                    else:
                        used_bytes = used or 0

                    if isinstance(total, dict):
                        total_bytes = total.get('vram_total', 0)
                    else:
                        total_bytes = total or 0

                    used_mb = used_bytes / (1024**2)
                    total_mb = total_bytes / (1024**2)

                    activity = {}
                    try:
                        activity = amdsmi.amdsmi_get_gpu_activity(handle) or {}
                    except AmdSmiLibraryException:
                        activity = {}
                    if not self._amdsmi_raw_logged and getattr(self, 'logger', None) is not None:
                        self.logger.info(f"AMDSMI raw activity: {activity}")

                    metrics = {}
                    if not activity:
                        get_metrics = getattr(amdsmi, 'amdsmi_get_gpu_metrics_info', None)
                        if callable(get_metrics):
                            try:
                                metrics = get_metrics(handle) or {}
                            except Exception:
                                metrics = {}
                            if not self._amdsmi_raw_logged and getattr(self, 'logger', None) is not None:
                                self.logger.info(f"AMDSMI raw metrics: {metrics}")

                    # Ensure we only emit raw AMDSMI dict logging once per process
                    if not self._amdsmi_raw_logged and getattr(self, 'logger', None) is not None:
                        if activity == {} and metrics == {}:
                            self._amdsmi_raw_logged = True

                    gpu_util = 0
                    for k in ('gfx_activity', 'GFX_ACTIVITY', 'gpu_activity', 'GPU_ACTIVITY', 'gfx', 'GFX'):
                        v = activity.get(k)
                        if isinstance(v, (int, float)):
                            gpu_util = v
                            break
                    if gpu_util == 0 and metrics:
                        for k in (
                            'gfx_activity', 'GFX_ACTIVITY', 'gpu_activity', 'GPU_ACTIVITY',
                            'gfx_busy', 'GFX_BUSY', 'gfx_util', 'GFX_UTIL',
                            'average_gfx_activity', 'AVERAGE_GFX_ACTIVITY'
                        ):
                            v = metrics.get(k)
                            if isinstance(v, (int, float)):
                                gpu_util = v
                                break

                    # Sysfs fallback (ROCm): /sys/class/drm/cardX/device/gpu_busy_percent
                    if gpu_util == 0:
                        busy = _read_sysfs_int(f"/sys/class/drm/card{i}/device/gpu_busy_percent")
                        if busy is not None:
                            gpu_util = busy

                    mem_util = 0
                    for k in ('mem_activity', 'MEM_ACTIVITY', 'memory_activity', 'MEMORY_ACTIVITY', 'mem', 'MEM'):
                        v = activity.get(k)
                        if isinstance(v, (int, float)):
                            mem_util = v
                            break
                    if mem_util == 0 and metrics:
                        for k in (
                            'mem_activity', 'MEM_ACTIVITY', 'memory_activity', 'MEMORY_ACTIVITY',
                            'mem_busy', 'MEM_BUSY', 'mem_util', 'MEM_UTIL',
                            'average_mem_activity', 'AVERAGE_MEM_ACTIVITY'
                        ):
                            v = metrics.get(k)
                            if isinstance(v, (int, float)):
                                mem_util = v
                                break

                    graphics_clock = 0
                    memory_clock = 0
                    try:
                        clocks = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.GFX) or {}
                        graphics_clock = clocks.get('cur_clk', 0) or 0
                    except Exception:
                        pass
                    try:
                        mem_clocks = amdsmi.amdsmi_get_clock_info(handle, amdsmi.AmdSmiClkType.MEM) or {}
                        memory_clock = mem_clocks.get('cur_clk', 0) or 0
                    except Exception:
                        pass

                    power_watts = 0.0
                    try:
                        power_info = amdsmi.amdsmi_get_power_ave(handle) or {}
                        if not self._amdsmi_raw_logged and getattr(self, 'logger', None) is not None:
                            self.logger.info(f"AMDSMI raw power: {power_info}")
                            self._amdsmi_raw_logged = True

                        avg_uw = 0
                        for k in ('average_socket_power', 'average_power', 'power', 'avg_power', 'socket_power'):
                            v = power_info.get(k)
                            if isinstance(v, (int, float)):
                                avg_uw = v
                                break

                        if avg_uw > 10_000:
                            power_watts = avg_uw / 1_000_000.0
                        else:
                            power_watts = float(avg_uw)
                    except Exception:
                        power_watts = 0.0

                    gpu_info["gpu/memory_used_mb"] += used_mb
                    gpu_info["gpu/memory_total_mb"] += total_mb
                    gpu_info["gpu/vram_used_mb"] += used_mb
                    gpu_info["gpu/vram_total_mb"] += total_mb
                    gpu_info["gpu/utilization_pct"] += gpu_util
                    gpu_info["gpu/memory_utilization_pct"] += mem_util
                    max_mem_util = max(max_mem_util, mem_util)
                    gpu_info["gpu/graphics_clock_mhz"] += graphics_clock
                    gpu_info["gpu/memory_clock_mhz"] += memory_clock
                    total_power += power_watts

            # Store power readings
            self.power_readings.append(total_power)
            self.power_readings_time.append(time.time())

            # Normalize values per GPU
            if num_gpus > 0:
                gpu_info["gpu/memory_used_mb"] /= num_gpus
                gpu_info["gpu/memory_total_mb"] /= num_gpus
                gpu_info["gpu/utilization_pct"] /= num_gpus
                gpu_info["gpu/memory_utilization_pct"] /= num_gpus
                gpu_info["gpu/memory_utilization_max_pct"] = max_mem_util
                gpu_info["gpu/vram_used_mb"] /= num_gpus
                gpu_info["gpu/vram_total_mb"] /= num_gpus
                gpu_info["gpu/graphics_clock_mhz"] /= num_gpus
                gpu_info["gpu/memory_clock_mhz"] /= num_gpus
                # Normalize power to per-GPU average
                gpu_info["gpu/power_usage_watts"] = total_power / num_gpus
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
        num_gpus = len(self.gpu_handles) if self.gpu_handles else 0
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
            f"amdsmi={'yes' if amdsmi is not None else 'no'}, "
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
                vram_utilization_pct=vram_utilization_pct if self.gpu_metrics_history else None,
                avg_gpu_util=avg_gpu_util if self.gpu_metrics_history else None,
                avg_gpu_mem_util=avg_gpu_mem_util if self.gpu_metrics_history else None,
                max_gpu_mem_util=max_gpu_mem_util if self.gpu_metrics_history else None,
                avg_gpu_power=avg_gpu_power if self.gpu_metrics_history else None,
                avg_graphics_clock=avg_graphics_clock if self.gpu_metrics_history else None,
                avg_memory_clock=avg_memory_clock if self.gpu_metrics_history else None,
                num_gpus=num_gpus,
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
