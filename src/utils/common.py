"""
Common Utility Functions

This module provides common utility functions for logging, random seed setting,
and directory management.
"""

import os
import logging
import random
import numpy as np
from datetime import datetime
from typing import Optional


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (optional)
        format_string: Custom format string (optional)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Log initial setup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Level: {level}")
    if log_file:
        logger.info(f"Log file: {log_file}")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set sklearn random state if available
    try:
        import sklearn
        sklearn.set_config(assume_finite=True)  # Optional sklearn optimization
    except ImportError:
        pass
    
    # Set pandas random state if available
    try:
        import pandas as pd
        # Pandas doesn't have a global random state, but we can set numpy
    except ImportError:
        pass
    
    # Log seed setting
    logger = logging.getLogger(__name__)
    logger.info(f"Random seeds set to: {seed}")


def create_output_directory(base_path: str, timestamp: bool = True) -> str:
    """
    Create output directory with optional timestamp.
    
    Args:
        base_path: Base directory path
        timestamp: Whether to add timestamp to directory name
        
    Returns:
        Full path to created directory
    """
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_path, f"run_{timestamp_str}")
    else:
        output_dir = base_path
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'plots', 'tables', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Log directory creation
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory created: {output_dir}")
    
    return output_dir


def save_config_copy(config_path: str, output_dir: str) -> str:
    """
    Save a copy of the configuration file to the output directory.
    
    Args:
        config_path: Path to original config file
        output_dir: Output directory
        
    Returns:
        Path to saved config copy
    """
    import shutil
    
    config_filename = os.path.basename(config_path)
    config_copy_path = os.path.join(output_dir, f"config_used_{config_filename}")
    
    shutil.copy2(config_path, config_copy_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration file copied to: {config_copy_path}")
    
    return config_copy_path


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{seconds:.1f}s"


def get_memory_usage() -> Optional[float]:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB, or None if psutil is not available
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except ImportError:
        return None


def print_system_info() -> None:
    """Print system information for debugging."""
    import platform
    import sys
    
    logger = logging.getLogger(__name__)
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  CPU count: {os.cpu_count()}")
    
    # Memory info if available
    memory_mb = get_memory_usage()
    if memory_mb:
        logger.info(f"  Memory usage: {memory_mb:.1f} MB")
    
    # Package versions
    packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
    for package in packages:
        try:
            module = __import__(package.replace('-', '_'))
            version = getattr(module, '__version__', 'Unknown')
            logger.info(f"  {package}: {version}")
        except ImportError:
            logger.info(f"  {package}: Not installed")


def create_results_summary(results_dir: str, summary_data: dict) -> str:
    """
    Create a summary file for the analysis results.
    
    Args:
        results_dir: Results directory
        summary_data: Dictionary with summary information
        
    Returns:
        Path to summary file
    """
    summary_path = os.path.join(results_dir, 'analysis_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("MACE Prediction Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().isoformat()}\n\n")
        
        for section, data in summary_data.items():
            f.write(f"{section}:\n")
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {data}\n")
            f.write("\n")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Analysis summary saved to: {summary_path}")
    
    return summary_path