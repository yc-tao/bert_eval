#!/usr/bin/env python3
"""
Device management utilities for CUDA configuration and multi-GPU support
"""

import os
import logging
import subprocess
from typing import List, Optional, Tuple, Dict, Any
import torch
import psutil

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages CUDA device configuration and multi-GPU setup"""

    def __init__(self):
        self.available_devices = self._get_available_devices()
        self.device_info = self._get_device_info()

    def set_cuda_devices(self, cuda_visible_devices: str) -> bool:
        """
        Set CUDA_VISIBLE_DEVICES environment variable

        Args:
            cuda_visible_devices: Comma-separated string of device IDs (e.g., "0,1,4,7")

        Returns:
            bool: True if devices were set successfully
        """
        if not cuda_visible_devices:
            logger.warning("No CUDA devices specified")
            return False

        # Parse device IDs
        try:
            device_ids = [int(d.strip()) for d in cuda_visible_devices.split(',')]
        except ValueError:
            logger.error(f"Invalid CUDA device specification: {cuda_visible_devices}")
            return False

        # Validate device availability
        if not self._validate_devices(device_ids):
            return False

        # Set environment variable
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

        # Update available devices after setting environment variable
        self.available_devices = self._get_available_devices()

        return True

    def _validate_devices(self, device_ids: List[int]) -> bool:
        """
        Validate that specified devices are available and not heavily used

        Args:
            device_ids: List of device IDs to validate

        Returns:
            bool: True if all devices are available
        """
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return False

        total_devices = torch.cuda.device_count()

        for device_id in device_ids:
            if device_id >= total_devices:
                logger.error(f"Device {device_id} not available. Only {total_devices} devices found.")
                return False

        # Check GPU utilization
        busy_devices = self._get_busy_devices()
        busy_requested = [d for d in device_ids if d in busy_devices]

        if busy_requested:
            logger.warning(f"Devices {busy_requested} appear to be in use. Proceeding anyway...")
            logger.warning("Consider using different devices or stopping other processes")

        return True

    def _get_available_devices(self) -> List[int]:
        """Get list of available CUDA devices"""
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    def _get_device_info(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed information about each device"""
        device_info = {}

        if not torch.cuda.is_available():
            return device_info

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info[i] = {
                'name': props.name,
                'total_memory': props.total_memory,
                'memory_gb': props.total_memory / (1024**3),
                'multi_processor_count': props.multi_processor_count,
                'major': props.major,
                'minor': props.minor
            }

        return device_info

    def _get_busy_devices(self, memory_threshold: float = 0.1) -> List[int]:
        """
        Get list of devices that are currently busy (using significant memory)

        Args:
            memory_threshold: Memory usage threshold (fraction) to consider device busy

        Returns:
            List of device IDs that are busy
        """
        busy_devices = []

        try:
            # Try using nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()

            for i in range(torch.cuda.device_count()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                memory_used_fraction = memory_info.used / memory_info.total
                if memory_used_fraction > memory_threshold:
                    busy_devices.append(i)

        except ImportError:
            # Fallback to nvidia-smi parsing
            busy_devices = self._get_busy_devices_nvidia_smi(memory_threshold)

        except Exception as e:
            logger.warning(f"Could not check device usage: {e}")

        return busy_devices

    def _get_busy_devices_nvidia_smi(self, memory_threshold: float = 0.1) -> List[int]:
        """Fallback method using nvidia-smi to check device usage"""
        busy_devices = []

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        device_id = int(parts[0])
                        memory_used = float(parts[1])
                        memory_total = float(parts[2])

                        if memory_used / memory_total > memory_threshold:
                            busy_devices.append(device_id)

        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not run nvidia-smi: {e}")

        return busy_devices

    def setup_distributed_training(self, local_rank: Optional[int] = None) -> Dict[str, Any]:
        """
        Setup configuration for distributed training

        Args:
            local_rank: Local rank for distributed training

        Returns:
            Dictionary with distributed training configuration
        """
        config = {
            'use_distributed': False,
            'world_size': 1,
            'local_rank': 0,
            'device_count': 0
        }

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return config

        device_count = torch.cuda.device_count()
        config['device_count'] = device_count

        if device_count > 1:
            config['use_distributed'] = True
            config['world_size'] = device_count

            if local_rank is not None:
                config['local_rank'] = local_rank

            logger.info(f"Setting up distributed training with {device_count} GPUs")
        else:
            logger.info("Using single GPU training")

        return config

    def get_optimal_batch_size(self, base_batch_size: int, device_count: int) -> int:
        """
        Calculate optimal batch size for multi-GPU training

        Args:
            base_batch_size: Base batch size for single GPU
            device_count: Number of GPUs

        Returns:
            Optimal batch size per device
        """
        if device_count <= 1:
            return base_batch_size

        # For multi-GPU, we typically want to maintain the same effective batch size
        # or slightly increase it for better throughput
        per_device_batch_size = max(1, base_batch_size // device_count)

        logger.info(f"Adjusted batch size from {base_batch_size} to {per_device_batch_size} per device "
                   f"({per_device_batch_size * device_count} total effective batch size)")

        return per_device_batch_size

    def log_device_setup(self):
        """Log current device configuration and status"""
        logger.info("=== GPU Device Configuration ===")

        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")

        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA available: True")
            logger.info(f"PyTorch visible devices: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")

            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                memory_mb = device_props.total_memory / (1024**2)
                logger.info(f"  Device {i}: {device_props.name} ({memory_mb:.0f} MB)")
        else:
            logger.info("CUDA not available")

        logger.info("================================")

    def monitor_gpu_memory(self) -> Dict[int, Dict[str, float]]:
        """
        Monitor GPU memory usage across all visible devices

        Returns:
            Dictionary with memory info for each device
        """
        memory_info = {}

        if not torch.cuda.is_available():
            return memory_info

        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB

                memory_info[i] = {
                    'allocated_gb': allocated,
                    'cached_gb': cached,
                    'total_gb': total,
                    'utilization': allocated / total if total > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Could not get memory info for device {i}: {e}")

        return memory_info

    def clear_gpu_cache(self):
        """Clear GPU cache on all visible devices"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")


def setup_devices_from_config(cfg) -> DeviceManager:
    """
    Setup devices based on configuration

    Args:
        cfg: Configuration object with cuda_visible_devices

    Returns:
        DeviceManager instance
    """
    device_manager = DeviceManager()

    # Set CUDA devices if specified in config
    if hasattr(cfg, 'cuda_visible_devices') and cfg.cuda_visible_devices:
        success = device_manager.set_cuda_devices(cfg.cuda_visible_devices)
        if not success:
            logger.warning("Failed to set CUDA devices from config, using default")

    # Log current setup
    device_manager.log_device_setup()

    return device_manager