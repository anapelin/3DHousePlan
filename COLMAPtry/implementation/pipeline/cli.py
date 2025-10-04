"""
Command-line interface for COLMAP pipeline
"""

import sys
import argparse
from pathlib import Path
import logging

from .core import COLMAPPipeline
from .utils import (
    setup_logging,
    load_config,
    check_colmap_installation,
    check_cuda_availability,
    check_ffmpeg_installation,
)

logger = logging.getLogger("colmap_pipeline")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="COLMAP SfM/MVS Pipeline - 3D reconstruction from images or video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reconstruct from video
  colmap-reconstruct from-video input.mp4 -o output/

  # Reconstruct from images
  colmap-reconstruct from-images images/ -o output/

  # Reconstruct with known poses
  colmap-reconstruct from-images-with-poses images/ poses.json -o output/

  # Use fast preset
  colmap-reconstruct from-video input.mp4 -o output/ --preset fast

  # Sparse reconstruction only (no dense)
  colmap-reconstruct from-video input.mp4 -o output/ --no-dense
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Reconstruction mode")
    subparsers.required = True
    
    # Common arguments for all subcommands
    def add_common_args(subparser):
        subparser.add_argument(
            "-o", "--output",
            type=Path,
            required=True,
            help="Output directory for reconstruction"
        )
        subparser.add_argument(
            "-c", "--config",
            type=Path,
            default=Path(__file__).parent.parent / "configs" / "default.yaml",
            help="Configuration file (default: configs/default.yaml)"
        )
        subparser.add_argument(
            "--preset",
            choices=["fast", "high"],
            help="Quality preset (overrides config)"
        )
        subparser.add_argument(
            "--no-dense",
            action="store_true",
            help="Skip dense reconstruction (sparse only)"
        )
        subparser.add_argument(
            "--no-mesh",
            action="store_true",
            help="Skip mesh generation"
        )
        subparser.add_argument(
            "--cpu-only",
            action="store_true",
            help="Force CPU-only processing (no GPU)"
        )
        subparser.add_argument(
            "--no-resume",
            action="store_true",
            help="Don't resume from checkpoint, start fresh"
        )
        subparser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Logging level"
        )
    
    # Subcommand: from-video
    parser_video = subparsers.add_parser(
        "from-video",
        help="Reconstruct from video file"
    )
    parser_video.add_argument(
        "video",
        type=Path,
        help="Path to input video file"
    )
    add_common_args(parser_video)
    
    # Subcommand: from-images
    parser_images = subparsers.add_parser(
        "from-images",
        help="Reconstruct from image directory"
    )
    parser_images.add_argument(
        "images",
        type=Path,
        help="Path to directory containing images"
    )
    add_common_args(parser_images)
    
    # Subcommand: from-images-with-poses
    parser_poses = subparsers.add_parser(
        "from-images-with-poses",
        help="Reconstruct from images with known camera poses"
    )
    parser_poses.add_argument(
        "images",
        type=Path,
        help="Path to directory containing images"
    )
    parser_poses.add_argument(
        "poses",
        type=Path,
        help="Path to poses file (JSON or CSV)"
    )
    add_common_args(parser_poses)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.output / "pipeline.log" if args.output else None
    setup_logging(args.log_level, log_file)
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies(args.command):
        return 1
    
    # Load configuration
    try:
        config = load_config(args.config, preset=args.preset)
        logger.info(f"Loaded configuration from: {args.config}")
        
        if args.preset:
            logger.info(f"Using preset: {args.preset}")
        
        # Override config with CLI args
        if args.cpu_only:
            config["general"]["cpu_only"] = True
            config["general"]["use_gpu"] = False
            logger.info("GPU disabled (CPU-only mode)")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline
    pipeline = COLMAPPipeline(
        args.output,
        config,
        resume=not args.no_resume
    )
    
    # Run reconstruction
    try:
        if args.command == "from-video":
            success = run_from_video(pipeline, args)
        elif args.command == "from-images":
            success = run_from_images(pipeline, args)
        elif args.command == "from-images-with-poses":
            success = run_from_images_with_poses(pipeline, args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        if success:
            print_success(args.output)
            return 0
        else:
            logger.error("Reconstruction failed")
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        logger.info("You can resume with the same command (checkpoint saved)")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


def run_from_video(pipeline: COLMAPPipeline, args) -> bool:
    """Run reconstruction from video."""
    # Validate input
    if not args.video.exists():
        logger.error(f"Video file not found: {args.video}")
        return False
    
    logger.info(f"Input video: {args.video}")
    logger.info(f"Output directory: {args.output}")
    
    # Run reconstruction
    return pipeline.reconstruct_from_video(
        args.video,
        dense=not args.no_dense,
        mesh=not args.no_mesh
    )


def run_from_images(pipeline: COLMAPPipeline, args) -> bool:
    """Run reconstruction from images."""
    # Validate input
    if not args.images.exists():
        logger.error(f"Images directory not found: {args.images}")
        return False
    
    if not args.images.is_dir():
        logger.error(f"Not a directory: {args.images}")
        return False
    
    logger.info(f"Input images: {args.images}")
    logger.info(f"Output directory: {args.output}")
    
    # Run reconstruction
    return pipeline.reconstruct_from_images(
        args.images,
        dense=not args.no_dense,
        mesh=not args.no_mesh
    )


def run_from_images_with_poses(pipeline: COLMAPPipeline, args) -> bool:
    """Run reconstruction from images with known poses."""
    # Validate input
    if not args.images.exists():
        logger.error(f"Images directory not found: {args.images}")
        return False
    
    if not args.images.is_dir():
        logger.error(f"Not a directory: {args.images}")
        return False
    
    if not args.poses.exists():
        logger.error(f"Poses file not found: {args.poses}")
        return False
    
    logger.info(f"Input images: {args.images}")
    logger.info(f"Input poses: {args.poses}")
    logger.info(f"Output directory: {args.output}")
    
    # Run reconstruction
    return pipeline.reconstruct_from_images_with_poses(
        args.images,
        args.poses,
        dense=not args.no_dense,
        mesh=not args.no_mesh
    )


def check_dependencies(command: str) -> bool:
    """Check required dependencies."""
    logger.info("Checking dependencies...")
    
    all_ok = True
    
    # Check COLMAP
    colmap_installed, colmap_version = check_colmap_installation()
    if colmap_installed:
        logger.info(f"✓ COLMAP: {colmap_version}")
    else:
        logger.error("✗ COLMAP not found")
        logger.error("Install COLMAP: https://colmap.github.io/install.html")
        all_ok = False
    
    # Check ffmpeg (only for video)
    if command == "from-video":
        ffmpeg_installed, ffmpeg_version = check_ffmpeg_installation()
        if ffmpeg_installed:
            logger.info(f"✓ ffmpeg: {ffmpeg_version}")
        else:
            logger.warning("✗ ffmpeg not found (will use OpenCV fallback)")
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    if cuda_available:
        logger.info("✓ CUDA available (GPU acceleration enabled)")
    else:
        logger.warning("✗ CUDA not available (CPU-only mode)")
    
    return all_ok


def print_banner():
    """Print CLI banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              COLMAP SfM/MVS Pipeline v1.0.0                  ║
║                                                              ║
║           3D Reconstruction from Images & Video              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_success(output_dir: Path):
    """Print success message."""
    message = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                  ✓ RECONSTRUCTION COMPLETE                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Output directory: {output_dir}

Generated files:
  • dense_point_cloud.ply - Dense 3D point cloud
  • poisson_mesh.obj - 3D mesh (textured)
  • sparse/ - Sparse reconstruction
  • report/report.html - Detailed report

View in Blender:
  1. Open Blender
  2. File > Import > Wavefront (.obj)
  3. Select: {output_dir / "poisson_mesh.obj"}

View report:
  Open: {output_dir / "report" / "report.html"}
    """
    print(message)


if __name__ == "__main__":
    sys.exit(main())

