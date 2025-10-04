"""
Utility functions for COLMAP pipeline
"""

import os
import sys
import json
import yaml
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import colorlog


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup colorized logging with optional file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger("colmap_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Path, preset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        preset: Optional preset name ("fast" or "high")
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply preset if specified
    if preset and "presets" in config and preset in config["presets"]:
        preset_config = config["presets"][preset]
        config = merge_configs(config, preset_config)
    
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge override config into base config.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def check_colmap_installation() -> Tuple[bool, Optional[str]]:
    """
    Check if COLMAP is installed and accessible.
    
    Returns:
        Tuple of (is_installed, version)
    """
    try:
        result = subprocess.run(
            ["colmap", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # Parse version from output
            version = result.stdout.strip().split('\n')[0]
            return True, version
        else:
            return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available
    """
    try:
        import cv2
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        pass
    
    # Alternative: check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def check_ffmpeg_installation() -> Tuple[bool, Optional[str]]:
    """
    Check if ffmpeg is installed and accessible.
    
    Returns:
        Tuple of (is_installed, version)
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # Parse version from first line
            version = result.stdout.split('\n')[0].split(' ')[2]
            return True, version
        else:
            return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None


def create_directory_structure(base_path: Path) -> Dict[str, Path]:
    """
    Create standard directory structure for reconstruction output.
    
    Args:
        base_path: Base output directory
        
    Returns:
        Dictionary of created paths
    """
    paths = {
        "base": base_path,
        "images": base_path / "images",
        "database": base_path / "database.db",
        "sparse": base_path / "sparse",
        "sparse_0": base_path / "sparse" / "0",
        "dense": base_path / "dense",
        "dense_images": base_path / "dense" / "images",
        "dense_sparse": base_path / "dense" / "sparse",
        "dense_stereo": base_path / "dense" / "stereo",
        "mesh": base_path / "mesh",
        "textured": base_path / "textured",
        "logs": base_path / "logs",
    }
    
    # Create directories
    for key, path in paths.items():
        if key != "database" and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    
    return paths


def count_images(image_dir: Path) -> int:
    """
    Count number of image files in directory.
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        Number of images
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    count = 0
    for file in image_dir.iterdir():
        if file.suffix.lower() in image_extensions:
            count += 1
    
    return count


def get_image_list(image_dir: Path) -> List[Path]:
    """
    Get list of image files in directory.
    
    Args:
        image_dir: Directory containing images
        
    Returns:
        List of image paths
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    images = []
    for file in sorted(image_dir.iterdir()):
        if file.suffix.lower() in image_extensions:
            images.append(file)
    
    return images


def save_checkpoint(checkpoint_path: Path, stage: str, data: Dict[str, Any]):
    """
    Save checkpoint data.
    
    Args:
        checkpoint_path: Path to checkpoint file
        stage: Current stage name
        data: Checkpoint data
    """
    checkpoint = {
        "stage": stage,
        "data": data,
    }
    
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint data.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint data or None if not found
    """
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    except:
        return None


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_file_size(file_path: Path) -> str:
    """
    Get file size in human-readable format.
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size
    """
    size = file_path.stat().st_size
    
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} PB"


def clean_directory(directory: Path, keep_patterns: Optional[List[str]] = None):
    """
    Clean directory, optionally keeping files matching patterns.
    
    Args:
        directory: Directory to clean
        keep_patterns: Optional list of glob patterns to keep
    """
    if not directory.exists():
        return
    
    for item in directory.iterdir():
        # Check if item should be kept
        if keep_patterns:
            should_keep = any(item.match(pattern) for pattern in keep_patterns)
            if should_keep:
                continue
        
        # Remove item
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def copy_with_progress(src: Path, dst: Path, desc: str = "Copying"):
    """
    Copy file with progress display.
    
    Args:
        src: Source file
        dst: Destination file
        desc: Progress description
    """
    from tqdm import tqdm
    
    file_size = src.stat().st_size
    
    with open(src, "rb") as fsrc:
        with open(dst, "wb") as fdst:
            with tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as pbar:
                while True:
                    chunk = fsrc.read(8192)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    pbar.update(len(chunk))

