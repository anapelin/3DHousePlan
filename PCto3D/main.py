#!/usr/bin/env python3
"""
PCto3D Pipeline - Main Entry Point
Process PLY files into 3D environments with surface segmentation.
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import PCto3DPipeline


def setup_logging(log_file=None, verbose=False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (if specified)
    handlers = [console_handler]
    
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=handlers
    )
    
    # Reduce noise from libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PCto3D Pipeline - Convert PLY files to segmented 3D models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings
  python main.py --input input/pointcloud.ply
  
  # Specify output location
  python main.py --input input/pointcloud.ply --output output/model.obj
  
  # Use custom configuration
  python main.py --input input/pointcloud.ply --config my_config.yaml
  
  # Enable verbose logging
  python main.py --input input/pointcloud.ply --verbose
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Path to input PLY file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output OBJ file (optional)'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Save logs to file'
    )
    
    parser.add_argument(
        '--no-intermediate',
        action='store_true',
        help='Do not save intermediate results'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show visualization during processing'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or 'logs/pipeline.log'
    os.makedirs('logs', exist_ok=True)
    setup_logging(log_file, args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.input:
            # Split path into folder and file
            input_path = Path(args.input)
            config['input']['ply_folder'] = str(input_path.parent)
            config['input']['ply_file'] = input_path.name
        
        if args.output:
            output_path = Path(args.output)
            config['output']['folder'] = str(output_path.parent)
            config['output']['obj_name'] = output_path.name
        
        if args.no_intermediate:
            config['output']['save_intermediate'] = False
        
        if args.visualize:
            config['visualization']['enable'] = True
        
        # Create output directory
        os.makedirs(config['output']['folder'], exist_ok=True)
        
        # Initialize and run pipeline
        logger.info("Initializing PCto3D Pipeline...")
        pipeline = PCto3DPipeline(config)
        
        # Run pipeline
        results = pipeline.run(
            ply_path=args.input,
            output_path=args.output
        )
        
        # Check results
        if results['success']:
            logger.info("\n✓ Pipeline completed successfully!")
            logger.info(f"Output: {results['output_path']}")
            
            if 'segment_files' in results:
                logger.info(f"Segments: {len(results['segment_files'])} files")
            
            return 0
        else:
            logger.error("\n✗ Pipeline failed!")
            if 'error' in results:
                logger.error(f"Error: {results['error']}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

