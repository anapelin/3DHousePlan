#!/usr/bin/env python3
"""
3D House Plan Pipeline - Main Entry Point

A comprehensive pipeline that takes video input and room plans to generate 3D objects for Blender export.

Usage:
    python main.py --video path/to/room_video.mp4 --plan path/to/floor_plan.pdf --output room_model.obj
"""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )
    
    # Run the CLI
    main()
