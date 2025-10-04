"""
Main Pipeline Orchestrator for 3D House Plan Pipeline

This module coordinates the entire pipeline from video and plan input
to 3D model export for Blender.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import click
from dataclasses import dataclass

# Import our custom modules
try:
    from .video_processor import VideoProcessor, VideoFrame
    from .plan_parser import PlanParser, FloorPlan
    from .reconstruction import ReconstructionEngine, Room3D
    from .blender_export import BlenderExporter
except ImportError:
    # Fallback for direct execution
    from video_processor import VideoProcessor, VideoFrame
    from plan_parser import PlanParser, FloorPlan
    from reconstruction import ReconstructionEngine, Room3D
    from blender_export import BlenderExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the 3D reconstruction pipeline."""
    # Video processing settings
    video_frame_skip: int = 10
    video_quality_threshold: float = 0.5
    
    # Plan parsing settings
    plan_scale_factor: float = 100.0  # pixels per meter
    plan_min_room_area: float = 5.0   # square meters
    
    # Reconstruction settings
    mesh_optimization: bool = True
    combine_components: bool = True
    
    # Export settings
    export_format: str = 'obj'
    export_include_materials: bool = True
    export_include_normals: bool = True

class Pipeline:
    """Main pipeline class that orchestrates the entire 3D reconstruction process."""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the pipeline with configuration."""
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.video_processor = VideoProcessor({
            'frame_skip': self.config.video_frame_skip,
            'quality_threshold': self.config.video_quality_threshold
        })
        
        self.plan_parser = PlanParser({
            'scale_factor': self.config.plan_scale_factor,
            'min_room_area': self.config.plan_min_room_area
        })
        
        self.reconstruction_engine = ReconstructionEngine({
            'mesh_optimization': self.config.mesh_optimization,
            'combine_components': self.config.combine_components
        })
        
        self.blender_exporter = BlenderExporter({
            'format': self.config.export_format,
            'include_materials': self.config.export_include_materials,
            'include_normals': self.config.export_include_normals
        })
        
        logger.info("Pipeline initialized successfully")
    
    def process(self, video_path: str, plan_path: str, output_path: str, 
                room_name: str = "MainRoom") -> bool:
        """
        Process video and plan to create 3D model.
        
        Args:
            video_path: Path to input video file
            plan_path: Path to floor plan file
            output_path: Path for output 3D model
            room_name: Name of the room to reconstruct
            
        Returns:
            True if processing was successful, False otherwise
        """
        logger.info("Starting 3D reconstruction pipeline")
        logger.info(f"Video: {video_path}")
        logger.info(f"Plan: {plan_path}")
        logger.info(f"Output: {output_path}")
        
        try:
            # Step 1: Process video
            logger.info("Step 1: Processing video...")
            video_frames = self.video_processor.process_video(video_path)
            if not video_frames:
                logger.error("No video frames processed")
                return False
            
            # Step 2: Parse floor plan
            logger.info("Step 2: Parsing floor plan...")
            floor_plan = self.plan_parser.parse_plan(plan_path)
            if not floor_plan.rooms:
                logger.error("No rooms found in floor plan")
                return False
            
            # Step 3: Reconstruct 3D model
            logger.info("Step 3: Reconstructing 3D model...")
            room3d = self.reconstruction_engine.reconstruct_room(
                video_frames, floor_plan, room_name
            )
            
            # Step 4: Optimize mesh
            if self.config.mesh_optimization:
                logger.info("Step 4: Optimizing mesh...")
                room3d.mesh = self.reconstruction_engine.optimize_mesh(room3d.mesh)
            
            # Step 5: Export to Blender format
            logger.info("Step 5: Exporting to Blender format...")
            success = self.blender_exporter.export_room(
                room3d, output_path, self.config.export_format
            )
            
            if success:
                logger.info("Pipeline completed successfully!")
                return True
            else:
                logger.error("Export failed")
                return False
                
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False
    
    def process_batch(self, input_pairs: List[Tuple[str, str]], 
                     output_dir: str) -> Dict[str, bool]:
        """
        Process multiple video-plan pairs in batch.
        
        Args:
            input_pairs: List of (video_path, plan_path) tuples
            output_dir: Output directory for all models
            
        Returns:
            Dictionary mapping input names to success status
        """
        logger.info(f"Processing batch of {len(input_pairs)} input pairs")
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, (video_path, plan_path) in enumerate(input_pairs):
            room_name = f"Room_{i+1}"
            output_file = output_path / f"{room_name}.{self.config.export_format}"
            
            logger.info(f"Processing pair {i+1}/{len(input_pairs)}: {room_name}")
            
            success = self.process(
                video_path, plan_path, str(output_file), room_name
            )
            
            results[room_name] = success
        
        successful_count = sum(1 for success in results.values() if success)
        logger.info(f"Batch processing complete: {successful_count}/{len(input_pairs)} successful")
        
        return results
    
    def validate_inputs(self, video_path: str, plan_path: str) -> List[str]:
        """Validate input files before processing."""
        issues = []
        
        # Check video file
        if not Path(video_path).exists():
            issues.append(f"Video file does not exist: {video_path}")
        else:
            # Check video format
            video_ext = Path(video_path).suffix.lower()
            if video_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
                issues.append(f"Unsupported video format: {video_ext}")
        
        # Check plan file
        if not Path(plan_path).exists():
            issues.append(f"Plan file does not exist: {plan_path}")
        else:
            # Check plan format
            plan_ext = Path(plan_path).suffix.lower()
            if plan_ext not in ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                issues.append(f"Unsupported plan format: {plan_ext}")
        
        return issues
    
    def get_processing_stats(self, video_path: str, plan_path: str) -> Dict:
        """Get statistics about the input files."""
        stats = {}
        
        try:
            # Video stats
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                stats['video'] = {
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                }
                cap.release()
            
            # Plan stats
            plan_file = Path(plan_path)
            stats['plan'] = {
                'file_size': plan_file.stat().st_size,
                'format': plan_file.suffix.lower()
            }
            
        except Exception as e:
            logger.warning(f"Could not get processing stats: {str(e)}")
        
        return stats
    
    def save_config(self, config_path: str) -> None:
        """Save current configuration to file."""
        config_dict = {
            'video_frame_skip': self.config.video_frame_skip,
            'video_quality_threshold': self.config.video_quality_threshold,
            'plan_scale_factor': self.config.plan_scale_factor,
            'plan_min_room_area': self.config.plan_min_room_area,
            'mesh_optimization': self.config.mesh_optimization,
            'combine_components': self.config.combine_components,
            'export_format': self.config.export_format,
            'export_include_materials': self.config.export_include_materials,
            'export_include_normals': self.config.export_include_normals
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to: {config_path}")
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configuration
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"Configuration loaded from: {config_path}")

# CLI Interface
@click.command()
@click.option('--video', '-v', required=True, help='Path to input video file')
@click.option('--plan', '-p', required=True, help='Path to floor plan file')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--room-name', '-r', default='MainRoom', help='Name of the room')
@click.option('--format', '-f', default='obj', 
              type=click.Choice(['obj', 'fbx', 'ply', 'stl', 'blend']),
              help='Export format')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--validate-only', is_flag=True, help='Only validate inputs, do not process')
@click.option('--stats', is_flag=True, help='Show processing statistics')
def main(video, plan, output, room_name, format, config, validate_only, stats):
    """3D House Plan Pipeline - Convert video and floor plan to 3D model."""
    
    # Initialize pipeline
    pipeline_config = PipelineConfig()
    if config:
        pipeline = Pipeline(pipeline_config)
        pipeline.load_config(config)
    else:
        pipeline = Pipeline(pipeline_config)
    
    # Override export format if specified
    pipeline.config.export_format = format
    
    # Validate inputs
    issues = pipeline.validate_inputs(video, plan)
    if issues:
        logger.error("Input validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return
    
    logger.info("Input validation passed")
    
    # Show statistics if requested
    if stats:
        stats_data = pipeline.get_processing_stats(video, plan)
        logger.info("Processing statistics:")
        for category, data in stats_data.items():
            logger.info(f"  {category}:")
            for key, value in data.items():
                logger.info(f"    {key}: {value}")
    
    # Exit if validation only
    if validate_only:
        logger.info("Validation complete - exiting")
        return
    
    # Process the inputs
    success = pipeline.process(video, plan, output, room_name)
    
    if success:
        logger.info(f"3D model successfully created: {output}")
        click.echo(f"Success! 3D model saved to: {output}")
    else:
        logger.error("Processing failed")
        click.echo("Processing failed - check logs for details")

if __name__ == "__main__":
    main()
