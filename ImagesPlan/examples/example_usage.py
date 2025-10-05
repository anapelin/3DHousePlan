#!/usr/bin/env python3
"""
Example usage of the 3D House Plan Pipeline

This script demonstrates how to use the pipeline programmatically
to process video and floor plan data.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import Pipeline, PipelineConfig
from video_processor import VideoProcessor
from plan_parser import PlanParser
from reconstruction import ReconstructionEngine
from blender_export import BlenderExporter

def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Initialize pipeline with default configuration
    config = PipelineConfig()
    pipeline = Pipeline(config)
    
    # Example file paths (replace with your actual files)
    video_path = "examples/sample_room_video.mp4"
    plan_path = "examples/sample_floor_plan.pdf"
    output_path = "examples/output_room.obj"
    
    # Validate inputs
    issues = pipeline.validate_inputs(video_path, plan_path)
    if issues:
        print("Input validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return
    
    # Process the inputs
    success = pipeline.process(video_path, plan_path, output_path, "LivingRoom")
    
    if success:
        print(f"✅ Success! 3D model created: {output_path}")
    else:
        print("❌ Processing failed")

def example_advanced_usage():
    """Advanced usage with custom configuration."""
    print("\n=== Advanced Usage Example ===")
    
    # Create custom configuration
    config = PipelineConfig()
    config.video_frame_skip = 5  # Process more frames
    config.export_format = 'fbx'  # Export as FBX
    config.mesh_optimization = True
    
    # Initialize pipeline
    pipeline = Pipeline(config)
    
    # Get processing statistics
    video_path = "examples/sample_room_video.mp4"
    plan_path = "examples/sample_floor_plan.pdf"
    
    stats = pipeline.get_processing_stats(video_path, plan_path)
    print("Processing statistics:")
    for category, data in stats.items():
        print(f"  {category}:")
        for key, value in data.items():
            print(f"    {key}: {value}")
    
    # Process with custom settings
    output_path = "examples/output_room_advanced.fbx"
    success = pipeline.process(video_path, plan_path, output_path, "Kitchen")
    
    if success:
        print(f"✅ Advanced processing complete: {output_path}")

def example_batch_processing():
    """Example of batch processing multiple rooms."""
    print("\n=== Batch Processing Example ===")
    
    # Define multiple input pairs
    input_pairs = [
        ("examples/room1_video.mp4", "examples/room1_plan.pdf"),
        ("examples/room2_video.mp4", "examples/room2_plan.pdf"),
        ("examples/room3_video.mp4", "examples/room3_plan.pdf")
    ]
    
    # Initialize pipeline
    config = PipelineConfig()
    pipeline = Pipeline(config)
    
    # Process batch
    results = pipeline.process_batch(input_pairs, "examples/batch_output/")
    
    print("Batch processing results:")
    for room_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {room_name}: {status}")

def example_individual_components():
    """Example using individual pipeline components."""
    print("\n=== Individual Components Example ===")
    
    # Initialize individual components
    video_processor = VideoProcessor()
    plan_parser = PlanParser()
    reconstruction_engine = ReconstructionEngine()
    blender_exporter = BlenderExporter()
    
    # Process video
    print("Processing video...")
    video_frames = video_processor.process_video("examples/sample_room_video.mp4")
    print(f"Extracted {len(video_frames)} key frames")
    
    # Parse floor plan
    print("Parsing floor plan...")
    floor_plan = plan_parser.parse_plan("examples/sample_floor_plan.pdf")
    print(f"Found {len(floor_plan.rooms)} rooms")
    
    # Reconstruct 3D model
    print("Reconstructing 3D model...")
    room3d = reconstruction_engine.reconstruct_room(video_frames, floor_plan, "Bedroom")
    print(f"Created 3D model with {len(room3d.mesh.vertices)} vertices")
    
    # Export to Blender
    print("Exporting to Blender...")
    success = blender_exporter.export_room(room3d, "examples/component_output.obj")
    
    if success:
        print("✅ Component-based processing complete")

def example_configuration_management():
    """Example of configuration management."""
    print("\n=== Configuration Management Example ===")
    
    # Create pipeline
    config = PipelineConfig()
    pipeline = Pipeline(config)
    
    # Save configuration
    config_path = "examples/custom_config.yaml"
    pipeline.save_config(config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Load configuration
    pipeline.load_config(config_path)
    print("Configuration loaded successfully")
    
    # Modify configuration
    pipeline.config.export_format = 'ply'
    pipeline.config.video_frame_skip = 15
    
    print("Configuration modified:")
    print(f"  Export format: {pipeline.config.export_format}")
    print(f"  Frame skip: {pipeline.config.video_frame_skip}")

if __name__ == "__main__":
    print("3D House Plan Pipeline - Example Usage")
    print("=" * 50)
    
    # Run examples (comment out the ones you don't want to run)
    try:
        example_basic_usage()
        example_advanced_usage()
        example_batch_processing()
        example_individual_components()
        example_configuration_management()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except FileNotFoundError as e:
        print(f"❌ Example file not found: {e}")
        print("Please ensure you have sample video and plan files in the examples/ directory")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("This is expected if you don't have the required input files")
