#!/usr/bin/env python3
"""
Test script to validate CUDA device configuration and multi-GPU support
"""

import os
import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.device_manager import setup_devices_from_config, DeviceManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_device_manager():
    """Test DeviceManager functionality"""
    logger.info("Testing DeviceManager functionality...")

    device_manager = DeviceManager()

    # Test device logging
    device_manager.log_device_setup()

    # Test memory monitoring
    memory_info = device_manager.monitor_gpu_memory()
    logger.info(f"GPU Memory Info: {memory_info}")

    # Test busy device detection
    busy_devices = device_manager._get_busy_devices()
    logger.info(f"Busy devices: {busy_devices}")

    # Test distributed training setup
    dist_config = device_manager.setup_distributed_training()
    logger.info(f"Distributed config: {dist_config}")

    return device_manager


def test_cuda_visible_devices_setting():
    """Test setting CUDA_VISIBLE_DEVICES"""
    logger.info("Testing CUDA_VISIBLE_DEVICES setting...")

    # Save original value
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    device_manager = DeviceManager()

    # Test setting devices
    test_devices = "0,1,4,7"
    success = device_manager.set_cuda_devices(test_devices)

    if success:
        logger.info(f"Successfully set CUDA_VISIBLE_DEVICES to {test_devices}")
        logger.info(f"Environment variable: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"PyTorch sees {torch.cuda.device_count()} devices")
    else:
        logger.warning("Failed to set CUDA devices")

    # Restore original value
    if original_cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    return success


def test_multi_gpu_batch_size():
    """Test batch size optimization for multi-GPU"""
    logger.info("Testing multi-GPU batch size optimization...")

    device_manager = DeviceManager()

    base_batch_size = 16
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

    optimal_batch_size = device_manager.get_optimal_batch_size(base_batch_size, device_count)

    logger.info(f"Base batch size: {base_batch_size}")
    logger.info(f"Device count: {device_count}")
    logger.info(f"Optimal batch size per device: {optimal_batch_size}")
    logger.info(f"Total effective batch size: {optimal_batch_size * device_count}")

    return optimal_batch_size


@hydra.main(version_base=None, config_path="configs", config_name="config")
def test_config_integration(cfg: DictConfig):
    """Test integration with configuration"""
    logger.info("Testing configuration integration...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Test setup from config
    device_manager = setup_devices_from_config(cfg)

    # Test that CUDA devices were set if specified
    if hasattr(cfg, 'cuda_visible_devices') and cfg.cuda_visible_devices:
        expected_devices = str(cfg.cuda_visible_devices)
        actual_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

        if actual_devices == expected_devices:
            logger.info(f"✓ CUDA devices correctly set to: {actual_devices}")
        else:
            logger.warning(f"✗ CUDA devices mismatch. Expected: {expected_devices}, Got: {actual_devices}")

    # Test multi-GPU configuration
    dist_config = device_manager.setup_distributed_training()
    logger.info(f"Distributed training config: {dist_config}")

    return device_manager


def run_all_tests():
    """Run all device configuration tests"""
    logger.info("=" * 60)
    logger.info("Starting CUDA Device Configuration Tests")
    logger.info("=" * 60)

    try:
        # Test 1: Basic DeviceManager functionality
        logger.info("\n" + "=" * 40)
        logger.info("Test 1: DeviceManager Functionality")
        logger.info("=" * 40)
        device_manager = test_device_manager()

        # Test 2: CUDA_VISIBLE_DEVICES setting
        logger.info("\n" + "=" * 40)
        logger.info("Test 2: CUDA_VISIBLE_DEVICES Setting")
        logger.info("=" * 40)
        cuda_success = test_cuda_visible_devices_setting()

        # Test 3: Multi-GPU batch size optimization
        logger.info("\n" + "=" * 40)
        logger.info("Test 3: Multi-GPU Batch Size Optimization")
        logger.info("=" * 40)
        optimal_batch_size = test_multi_gpu_batch_size()

        # Test 4: Configuration integration (will be called by Hydra)
        logger.info("\n" + "=" * 40)
        logger.info("Test 4: Configuration Integration")
        logger.info("=" * 40)
        # This will be handled by the Hydra decorator

        logger.info("\n" + "=" * 60)
        logger.info("Test Summary:")
        logger.info(f"  DeviceManager functionality: ✓")
        logger.info(f"  CUDA_VISIBLE_DEVICES setting: {'✓' if cuda_success else '✗'}")
        logger.info(f"  Multi-GPU batch optimization: ✓ (optimal size: {optimal_batch_size})")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    # Run basic tests first
    run_all_tests()

    # Then test configuration integration with Hydra
    test_config_integration()