#!/usr/bin/env python3
"""
Advanced printing utilities with Rich console support and GPU status monitoring
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
from loguru import logger

console = Console()


def setup_rich_logging(output_dir: Path, log_level: str = "INFO", disable_vllm_logging: bool = False):
    """Setup rich logging with loguru."""
    # Remove default handler
    logger.remove()

    # Add rich handler for console output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # Add file handler
    log_file = output_dir / "execution.log"
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB"
    )

    if disable_vllm_logging:
        import logging
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.WARNING)


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════════╗
    ║                           ClinicalBERT Evaluation Suite                          ║
    ║                          Advanced Model Evaluation Framework                     ║
    ╚══════════════════════════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def print_config_summary(cfg: Any, output_dir: Path):
    """Print configuration summary in a rich table."""
    table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # Model configuration
    if hasattr(cfg, 'model'):
        table.add_row("Model Name", str(getattr(cfg.model, 'name', 'N/A')))
        table.add_row("Model Path", str(getattr(cfg.model, 'path', 'N/A')))

    # Data configuration
    if hasattr(cfg, 'data'):
        table.add_row("Data Path", str(getattr(cfg.data, 'input_path', 'N/A')))
        table.add_row("Text Key", str(getattr(cfg.data, 'text_key', 'N/A')))
        table.add_row("Label Key", str(getattr(cfg.data, 'label_key', 'N/A')))
        table.add_row("Max Samples", str(getattr(cfg.data, 'max_samples', 'All')))

    # Inference configuration
    if hasattr(cfg, 'inference'):
        table.add_row("Batch Size", str(getattr(cfg.inference, 'batch_size', 'N/A')))
        table.add_row("Max Length", str(getattr(cfg.inference, 'max_length', 'N/A')))
        table.add_row("Temperature", str(getattr(cfg.inference, 'temperature', 'N/A')))

    # Output configuration
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Debug Mode", str(getattr(cfg, 'debug', False)))

    console.print(table)
    console.print()


def print_gpu_info():
    """Print comprehensive GPU information."""
    if not torch.cuda.is_available():
        console.print("[red]CUDA not available[/red]")
        return

    # GPU Overview Table
    gpu_table = Table(title="GPU Configuration", show_header=True, header_style="bold green")
    gpu_table.add_column("Device", style="cyan", no_wrap=True)
    gpu_table.add_column("Name", style="white")
    gpu_table.add_column("Memory (GB)", style="yellow", justify="right")
    gpu_table.add_column("Compute", style="magenta")
    gpu_table.add_column("Status", style="green")

    total_memory = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb

        # Check if device is currently in use
        try:
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            status = f"Used: {allocated:.1f}GB" if allocated > 0.1 else "Available"
            status_style = "yellow" if allocated > 0.1 else "green"
        except:
            status = "Unknown"
            status_style = "red"

        gpu_table.add_row(
            f"GPU {i}",
            props.name,
            f"{memory_gb:.1f}",
            f"{props.major}.{props.minor}",
            f"[{status_style}]{status}[/{status_style}]"
        )

    console.print(gpu_table)

    # Environment info
    env_table = Table(title="Environment", show_header=True, header_style="bold blue")
    env_table.add_column("Variable", style="cyan")
    env_table.add_column("Value", style="white")

    env_table.add_row("CUDA_VISIBLE_DEVICES", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))
    env_table.add_row("PyTorch CUDA Version", torch.version.cuda or 'N/A')
    env_table.add_row("Total GPU Memory", f"{total_memory:.1f} GB")
    env_table.add_row("Available GPUs", str(torch.cuda.device_count()))

    console.print(env_table)
    console.print()


def create_progress_bar() -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return status info."""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }

    if torch.cuda.is_available():
        gpu_info["device_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": props.name,
                "memory_gb": props.total_memory / (1024**3),
                "compute_capability": f"{props.major}.{props.minor}"
            }

            try:
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                device_info["allocated_memory_gb"] = allocated
                device_info["available"] = allocated < 0.1
            except:
                device_info["allocated_memory_gb"] = 0
                device_info["available"] = False

            gpu_info["devices"].append(device_info)

    return gpu_info


def validate_environment() -> Dict[str, bool]:
    """Validate that required packages are available."""
    required_packages = {
        'torch': True,
        'transformers': True,
        'datasets': True,
        'numpy': True,
        'pandas': True,
        'sklearn': True,
        'rich': True,
        'loguru': True
    }

    for package in required_packages:
        try:
            __import__(package)
            required_packages[package] = True
        except ImportError:
            required_packages[package] = False

    return required_packages


def save_run_metadata(output_dir: Path, config: Dict[str, Any], start_time: datetime,
                     end_time: datetime, additional_info: Optional[Dict[str, Any]] = None):
    """Save run metadata to JSON file."""
    import json

    metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "config": config,
        "system_info": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    }

    if additional_info:
        metadata.update(additional_info)

    metadata_file = output_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def create_run_summary(output_dir: Path, metrics: Dict[str, Any], config: Dict[str, Any],
                      sample_count: int, duration: float):
    """Create a human-readable run summary."""
    summary_content = f"""# Evaluation Run Summary

## Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {duration:.2f} seconds
- **Samples Processed**: {sample_count:,}
- **Output Directory**: {output_dir}

## Model Configuration
- **Model**: {config.get('model', {}).get('name', 'N/A')}
- **Batch Size**: {config.get('inference', {}).get('batch_size', 'N/A')}
- **Max Length**: {config.get('inference', {}).get('max_length', 'N/A')}

## Results
"""

    if metrics:
        summary_content += f"""
### Classification Metrics
- **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}
- **F1-Score**: {metrics.get('f1_score', 'N/A'):.4f}
- **Precision**: {metrics.get('precision', 'N/A'):.4f}
- **Recall**: {metrics.get('recall', 'N/A'):.4f}
"""

        if 'auc_roc' in metrics:
            summary_content += f"- **AUC-ROC**: {metrics['auc_roc']:.4f}\n"

    summary_content += f"""
## System Information
- **CUDA Available**: {torch.cuda.is_available()}
- **GPU Count**: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
- **PyTorch Version**: {torch.__version__}

---
*Generated by ClinicalBERT Evaluation Suite*
"""

    summary_file = output_dir / "run_summary.md"
    with open(summary_file, 'w') as f:
        f.write(summary_content)


def print_results_table(metrics: Dict[str, Any], title: str = "Evaluation Results"):
    """Print results in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold green")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="yellow", justify="right")

    # Core metrics
    core_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_roc']
    for metric in core_metrics:
        if metric in metrics:
            value = metrics[metric]
            if isinstance(value, float):
                table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
            else:
                table.add_row(metric.replace('_', ' ').title(), str(value))

    # Additional metrics
    for key, value in metrics.items():
        if key not in core_metrics and not key.startswith('_'):
            if isinstance(value, float):
                table.add_row(key.replace('_', ' ').title(), f"{value:.4f}")
            elif isinstance(value, (int, str)):
                table.add_row(key.replace('_', ' ').title(), str(value))

    console.print(table)


def print_error_summary(errors: List[Dict[str, Any]], max_display: int = 10):
    """Print error summary table."""
    if not errors:
        console.print("[green]No errors detected![/green]")
        return

    table = Table(title=f"Error Summary (showing {min(len(errors), max_display)} of {len(errors)})",
                  show_header=True, header_style="bold red")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Text Preview", style="white")
    table.add_column("True Label", style="green")
    table.add_column("Predicted", style="red")
    table.add_column("Confidence", style="yellow", justify="right")

    for i, error in enumerate(errors[:max_display]):
        text_preview = error.get('text', '')[:50] + '...' if len(error.get('text', '')) > 50 else error.get('text', '')
        table.add_row(
            str(i + 1),
            text_preview,
            str(error.get('true_label', 'N/A')),
            str(error.get('predicted_label', 'N/A')),
            f"{error.get('confidence', 0):.3f}"
        )

    console.print(table)


def print_system_status():
    """Print comprehensive system status."""
    # System info panel
    system_info = f"""
[bold]Python:[/bold] {sys.version.split()[0]}
[bold]PyTorch:[/bold] {torch.__version__}
[bold]CUDA Available:[/bold] {torch.cuda.is_available()}
"""

    if torch.cuda.is_available():
        system_info += f"[bold]GPU Count:[/bold] {torch.cuda.device_count()}\n"
        system_info += f"[bold]Current Device:[/bold] {torch.cuda.current_device()}\n"

    console.print(Panel(system_info, title="System Information", border_style="blue"))

    # Memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_info = f"""
[bold]Total RAM:[/bold] {memory.total / (1024**3):.1f} GB
[bold]Available RAM:[/bold] {memory.available / (1024**3):.1f} GB
[bold]RAM Usage:[/bold] {memory.percent}%
"""
        console.print(Panel(memory_info, title="Memory Information", border_style="green"))
    except ImportError:
        pass