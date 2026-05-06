import time
import logging
import os
import subprocess
import re
from typing import Any, Dict, Optional
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
from collections import defaultdict
from functools import wraps


class CommunicationProfilerCallback(pl.Callback):
    """Callback to profile and track NCCL/MPI communication during distributed training.
    
    This callback uses PyTorch's built-in profiler to track communication operations
    (all-reduce, broadcast, etc.) and reports statistics about data transfer volumes
    and communication overhead.
    """
    
    def __init__(
        self,
        profile_steps: int = 10,
        wait_steps: int = 5,
        warmup_steps: int = 2,
        active_steps: int = 3,
        repeat: int = 1,
        log_interval: int = 50,
        output_dir: str = "./profiler_logs"
    ):
        """Initialize the communication profiler callback.
        
        Args:
            profile_steps: Total number of steps to profile per cycle
            wait_steps: Number of steps to wait before starting profiling
            warmup_steps: Number of warmup steps before active profiling
            active_steps: Number of steps to actively profile
            repeat: Number of times to repeat the profiling cycle
            log_interval: How often to log communication statistics (in steps)
            output_dir: Directory to save profiler traces
        """
        super().__init__()
        
        self.profile_steps = profile_steps
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat = repeat
        self.log_interval = log_interval
        self.output_dir = output_dir
        
        # Statistics tracking
        self.comm_stats = defaultdict(lambda: {
            'count': 0,
            'total_time_us': 0,
            'total_bytes': 0,
            'cuda_time_us': 0
        })
        
        # Track all operations for percentage breakdown
        self.all_ops_stats = defaultdict(lambda: {
            'count': 0,
            'total_time_us': 0,
            'cuda_time_us': 0
        })
        
        self.profiler = None
        self.profiling_enabled = False
        self.total_profiled_time_us = 0
        
        # Network interface detection
        self.network_interface = None
        
        # NCCL bandwidth tracking
        self.nccl_bytes_transferred = defaultdict(int)
        self.nccl_operation_count = defaultdict(int)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize profiler when training starts."""
        # Only enable profiling if using distributed training
        if trainer.world_size > 1:
            self.profiling_enabled = True
            self.logger.info(
                f"Communication profiling enabled for {trainer.world_size} processes"
            )
        else:
            self.logger.info(
                "Communication profiling disabled (single GPU training)"
            )
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Profiler output directory: {self.output_dir}")
        
        # Detect network interface
        self._detect_network_interface()
        
        # Create profiler with schedule
        schedule = torch.profiler.schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat
        )
        
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=self._on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Set to True for detailed stack traces (higher overhead)
        )
        
        
        self.profiler.__enter__()
        self.logger.info("PyTorch Profiler initialized for communication tracking")
    
    def _detect_network_interface(self):
        """Detect if InfiniBand or Ethernet is being used."""
        try:
            # Check for InfiniBand devices
            result = subprocess.run(['ibstat'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and 'State: Active' in result.stdout:
                self.network_interface = 'InfiniBand'
                self.logger.info("✓ Detected InfiniBand network interface")
                
                # Try to get IB link speed
                if 'Rate:' in result.stdout:
                    for line in result.stdout.split('\n'):
                        if 'Rate:' in line:
                            self.logger.info(f"  {line.strip()}")
                            break
            else:
                self.network_interface = 'Ethernet'
                self.logger.info("✓ Detected Ethernet network interface")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # ibstat not available, check for ib interfaces
            try:
                result = subprocess.run(['ip', 'link'], capture_output=True, text=True, timeout=2)
                if 'ib' in result.stdout.lower():
                    self.network_interface = 'InfiniBand'
                    self.logger.info("✓ Detected InfiniBand network interface (via ip link)")
                else:
                    self.network_interface = 'Ethernet'
                    self.logger.info("✓ Detected Ethernet network interface")
            except:
                self.network_interface = 'Unknown'
                self.logger.warning("Could not detect network interface type")
    

    
    def _on_trace_ready(self, prof):
        """Callback when profiler trace is ready."""
        # Export trace for visualization
        trace_path = f"{self.output_dir}/trace_{int(time.time())}.json"
        try:
            prof.export_chrome_trace(trace_path)
            self.logger.info(f"Profiler trace saved to {trace_path}")
        except Exception as e:
            self.logger.warning(f"Failed to export trace: {e}")
        
        # Analyze communication events
        self._analyze_comm_events(prof)
    
    def _analyze_comm_events(self, prof):
        """Analyze communication events from profiler."""
        events = prof.key_averages()
        
        # Track total time across all operations
        total_time_this_batch = 0
        
        # Keywords that indicate communication operations
        comm_keywords = [
            'nccl', 'allreduce', 'allgather', 'broadcast', 
            'reduce_scatter', 'all_to_all', 'barrier',
            'ncclKernel', 'ncclAllReduce', 'ncclBroadcast'
        ]
        
        comm_events = []
        
        # First pass: categorize all events and track total time
        for event in events:
            # Track all operations for percentage breakdown
            cuda_time = getattr(event, 'cuda_time_total', getattr(event, 'device_time_total', 0))
            self.all_ops_stats[event.key]['count'] += event.count
            self.all_ops_stats[event.key]['total_time_us'] += event.cpu_time_total
            self.all_ops_stats[event.key]['cuda_time_us'] += cuda_time
            total_time_this_batch += cuda_time if cuda_time > 0 else event.cpu_time_total
            
            # Check if this is a communication event
            event_key_lower = event.key.lower()
            if any(keyword in event_key_lower for keyword in comm_keywords):
                comm_events.append(event)
                
                # Update communication statistics
                self.comm_stats[event.key]['count'] += event.count
                self.comm_stats[event.key]['total_time_us'] += event.cpu_time_total
                self.comm_stats[event.key]['cuda_time_us'] += cuda_time
                
        
        self.total_profiled_time_us += total_time_this_batch
        
        if comm_events:
            self.logger.info("=" * 80)
            self.logger.info("Communication Events Summary:")
            self.logger.info("-" * 80)
            
            for event in comm_events:
                cpu_time_ms = event.cpu_time_total / 1000.0
                # Use cuda_time_total if available, otherwise try device_time_total, or default to 0
                cuda_time = getattr(event, 'cuda_time_total', getattr(event, 'device_time_total', 0))
                cuda_time_ms = cuda_time / 1000.0
                
                # Calculate percentage of total time
                time_percentage = (cuda_time / total_time_this_batch * 100) if total_time_this_batch > 0 else 0
                
                info_msg = (
                    f"  {event.key}:\n"
                    f"    Count: {event.count}\n"
                    f"    CPU Time: {cpu_time_ms:.2f} ms\n"
                    f"    CUDA Time: {cuda_time_ms:.2f} ms ({time_percentage:.1f}% of batch)\n"
                    f"    Avg CPU Time: {cpu_time_ms/max(event.count, 1):.2f} ms"
                )
                
                # Calculate data transfer rate if we have memory info
                if hasattr(event, 'cpu_memory_usage') and event.cpu_memory_usage != 0:
                    mem_bytes = abs(event.cpu_memory_usage)
                    mem_mb = mem_bytes / (1024 * 1024)
                    info_msg += f"\n    Memory: {mem_mb:.2f} MB"
                    
                    # Calculate transfer rate (GB/s)
                    if cuda_time_ms > 0:
                        transfer_rate_gbps = (mem_bytes / (cuda_time_ms / 1000.0)) / (1024**3)
                        info_msg += f"\n    Transfer Rate: {transfer_rate_gbps:.2f} GB/s"
                
                self.logger.info(info_msg)
            
            self.logger.info("=" * 80)
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Dict[str, Any]],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Step the profiler after each batch."""
        if self.profiler is not None and self.profiling_enabled:
            self.profiler.step()
            
            # Log periodic statistics
            if trainer.is_global_zero and trainer.global_step % self.log_interval == 0:
                self._log_cumulative_stats(trainer.global_step)
    
    def _log_cumulative_stats(self, step: int):
        """Log cumulative communication statistics with time breakdown."""
        if not self.comm_stats:
            return
        
        self.logger.info("=" * 80)
        self.logger.info(f"Cumulative Statistics (Step {step}):")
        if self.network_interface:
            self.logger.info(f"Network Interface: {self.network_interface}")
        self.logger.info("-" * 80)
        
        # Calculate total time across all operations
        total_all_ops_time_us = sum(stats['cuda_time_us'] if stats['cuda_time_us'] > 0 else stats['total_time_us'] 
                                     for stats in self.all_ops_stats.values())
        
        # Calculate communication time
        total_comm_time_us = 0
        
        for op_name, stats in sorted(self.comm_stats.items()):
            cuda_time_us = stats['cuda_time_us'] if stats['cuda_time_us'] > 0 else stats['total_time_us']
            total_comm_time_us += cuda_time_us
        
        # Calculate computation time (everything that's not communication)
        total_compute_time_us = total_all_ops_time_us - total_comm_time_us
        
        # Log high-level breakdown
        self.logger.info("TIME BREAKDOWN:")
        if total_all_ops_time_us > 0:
            comm_percentage = (total_comm_time_us / total_all_ops_time_us) * 100
            compute_percentage = (total_compute_time_us / total_all_ops_time_us) * 100
            
            self.logger.info(f"  Communication: {total_comm_time_us/1000:.2f} ms ({comm_percentage:.1f}%)")
            self.logger.info(f"  Computation:   {total_compute_time_us/1000:.2f} ms ({compute_percentage:.1f}%)")
            self.logger.info(f"  Total:         {total_all_ops_time_us/1000:.2f} ms")
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("COMMUNICATION OPERATIONS DETAIL:")
        
        for op_name, stats in sorted(self.comm_stats.items(), key=lambda x: x[1]['cuda_time_us'], reverse=True):
            cpu_time_ms = stats['total_time_us'] / 1000.0
            cuda_time_ms = stats['cuda_time_us'] / 1000.0
            time_used = cuda_time_ms if cuda_time_ms > 0 else cpu_time_ms
            
            # Percentage of total time
            percentage = (stats['cuda_time_us'] / total_all_ops_time_us * 100) if total_all_ops_time_us > 0 else 0
            
            # Percentage of communication time
            comm_percentage = (stats['cuda_time_us'] / total_comm_time_us * 100) if total_comm_time_us > 0 else 0
            
            avg_time_ms = time_used / max(stats['count'], 1)
            
            info_msg = (
                f"  {op_name}:\n"
                f"    Calls: {stats['count']}\n"
                f"    Time: {time_used:.2f} ms ({percentage:.1f}% of total, {comm_percentage:.1f}% of comm)\n"
                f"    Avg/Call: {avg_time_ms:.2f} ms"
            )
            
            self.logger.info(info_msg)
        
        self.logger.info("=" * 80)
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Cleanup profiler when training ends."""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.logger.info("Communication profiler stopped")
            
            # Log final statistics
            if trainer.is_global_zero:
                self._log_final_summary()
    
    def _log_final_summary(self):
        """Log final summary of communication statistics with comprehensive breakdown."""
        if not self.comm_stats:
            self.logger.info("No communication events were captured during profiling")
            return
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINAL PROFILING SUMMARY")
        if self.network_interface:
            self.logger.info(f"Network Interface: {self.network_interface}")
        self.logger.info("=" * 80)
        
        # Calculate total time across all operations
        total_all_ops_time_us = sum(stats['cuda_time_us'] if stats['cuda_time_us'] > 0 else stats['total_time_us'] 
                                     for stats in self.all_ops_stats.values())
        
        # Calculate communication totals
        total_calls = sum(stats['count'] for stats in self.comm_stats.values())
        total_comm_time_us = sum(stats['cuda_time_us'] if stats['cuda_time_us'] > 0 else stats['total_time_us'] 
                                  for stats in self.comm_stats.values())
        
        # Calculate computation time
        total_compute_time_us = total_all_ops_time_us - total_comm_time_us
        
        # High-level summary
        self.logger.info("OVERALL TIME BREAKDOWN:")
        if total_all_ops_time_us > 0:
            comm_percentage = (total_comm_time_us / total_all_ops_time_us) * 100
            compute_percentage = (total_compute_time_us / total_all_ops_time_us) * 100
            
            self.logger.info(f"  Communication: {total_comm_time_us/1000:.2f} ms ({comm_percentage:.1f}%)")
            self.logger.info(f"  Computation:   {total_compute_time_us/1000:.2f} ms ({compute_percentage:.1f}%)")
            self.logger.info(f"  Total Time:    {total_all_ops_time_us/1000:.2f} ms ({total_all_ops_time_us/1000000:.2f} s)")
        
        self.logger.info(f"\nTotal Communication Operations: {total_calls}")
        
        self.logger.info("\n" + "-" * 80)
        self.logger.info("BREAKDOWN BY OPERATION TYPE:")
        self.logger.info("-" * 80)
        
        # Sort by total time (descending)
        sorted_ops = sorted(
            self.comm_stats.items(),
            key=lambda x: x[1]['cuda_time_us'] if x[1]['cuda_time_us'] > 0 else x[1]['total_time_us'],
            reverse=True
        )
        
        for op_name, stats in sorted_ops:
            time_us = stats['cuda_time_us'] if stats['cuda_time_us'] > 0 else stats['total_time_us']
            time_ms = time_us / 1000.0
            
            # Percentage of total time
            percentage = (time_us / total_all_ops_time_us * 100) if total_all_ops_time_us > 0 else 0
            
            # Percentage of communication time
            comm_percentage = (time_us / total_comm_time_us * 100) if total_comm_time_us > 0 else 0
            
            info_msg = (
                f"  {op_name}:\n"
                f"    Calls: {stats['count']}\n"
                f"    Time: {time_ms:.2f} ms ({percentage:.1f}% of total, {comm_percentage:.1f}% of comm)\n"
                f"    Avg: {time_ms/max(stats['count'], 1):.2f} ms/call"
            )
            
            self.logger.info(info_msg)
        
        self.logger.info("=" * 80 + "\n")
