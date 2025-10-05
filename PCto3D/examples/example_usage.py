#!/usr/bin/env python3
"""
Example usage of the PCto3D Pipeline
"""

import sys
import os
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import PCto3DPipeline
from src import PLYLoader, MeshRefiner, SurfaceSegmenter, OBJExporter


def example_1_basic_usage():
    """Example 1: Basic pipeline usage with default config"""
    print("="*60)
    print("Example 1: Basic Pipeline Usage")
    print("="*60)
    
    # Load configuration
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for this example
    config['input']['ply_folder'] = 'input'
    config['input']['ply_file'] = 'sample.ply'
    config['output']['folder'] = 'output/example1'
    
    # Create and run pipeline
    pipeline = PCto3DPipeline(config)
    results = pipeline.run()
    
    if results['success']:
        print(f"\n✓ Success! Output: {results['output_path']}")
        print(f"Processing time: {results['processing_time']:.2f}s")
    else:
        print(f"\n✗ Failed: {results.get('error', 'Unknown error')}")


def example_2_custom_processing():
    """Example 2: Manual control over each processing step"""
    print("\n" + "="*60)
    print("Example 2: Custom Processing Steps")
    print("="*60)
    
    # Load configuration
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    loader = PLYLoader(config)
    refiner = MeshRefiner(config)
    segmenter = SurfaceSegmenter(config)
    exporter = OBJExporter(config)
    
    # Step 1: Load
    print("\nStep 1: Loading PLY file...")
    geometry = loader.load('input/sample.ply')
    info = loader.get_info(geometry)
    print(f"Loaded: {info['type']} with {info.get('num_points', 0)} points")
    
    # Step 2: Refine
    print("\nStep 2: Refining mesh...")
    refined = refiner.refine(geometry)
    
    # Step 3: Segment
    print("\nStep 3: Segmenting surfaces...")
    segmented = segmenter.segment(refined)
    
    # Step 4: Export
    print("\nStep 4: Exporting...")
    output_path = exporter.export(segmented, 'output/example2/model.obj')
    print(f"✓ Exported to: {output_path}")


def example_3_region_growing():
    """Example 3: Using region growing segmentation"""
    print("\n" + "="*60)
    print("Example 3: Region Growing Segmentation")
    print("="*60)
    
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for region growing
    config['segmentation']['method'] = 'region_growing'
    config['segmentation']['region_growing']['normal_variance_threshold'] = 0.1
    config['segmentation']['region_growing']['curvature_threshold'] = 1.0
    config['output']['folder'] = 'output/example3'
    
    pipeline = PCto3DPipeline(config)
    results = pipeline.run('input/sample.ply')
    
    if results['success']:
        segments = results['stages']['segment'].get('num_segments', 0)
        print(f"\n✓ Found {segments} regions")


def example_4_clustering():
    """Example 4: Using clustering segmentation"""
    print("\n" + "="*60)
    print("Example 4: Clustering Segmentation")
    print("="*60)
    
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for clustering
    config['segmentation']['method'] = 'clustering'
    config['segmentation']['clustering']['method'] = 'dbscan'
    config['segmentation']['clustering']['eps'] = 0.05
    config['output']['folder'] = 'output/example4'
    
    pipeline = PCto3DPipeline(config)
    results = pipeline.run('input/sample.ply')
    
    if results['success']:
        segments = results['stages']['segment'].get('num_segments', 0)
        print(f"\n✓ Found {segments} clusters")


def example_5_ransac_planes():
    """Example 5: Using RANSAC plane segmentation"""
    print("\n" + "="*60)
    print("Example 5: RANSAC Plane Segmentation")
    print("="*60)
    
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for RANSAC
    config['segmentation']['method'] = 'ransac'
    config['segmentation']['ransac']['distance_threshold'] = 0.01
    config['segmentation']['ransac']['max_planes'] = 10
    config['output']['folder'] = 'output/example5'
    
    pipeline = PCto3DPipeline(config)
    results = pipeline.run('input/sample.ply')
    
    if results['success']:
        segments = results['stages']['segment'].get('num_segments', 0)
        print(f"\n✓ Extracted {segments} planes")


def example_6_separate_segments():
    """Example 6: Export each segment as a separate file"""
    print("\n" + "="*60)
    print("Example 6: Export Separate Segments")
    print("="*60)
    
    with open('../config/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['output']['folder'] = 'output/example6'
    
    # Initialize
    loader = PLYLoader(config)
    refiner = MeshRefiner(config)
    segmenter = SurfaceSegmenter(config)
    exporter = OBJExporter(config)
    
    # Process
    geometry = loader.load('input/sample.ply')
    refined = refiner.refine(geometry)
    segmented = segmenter.segment(refined)
    
    # Export main model
    main_output = exporter.export(segmented)
    print(f"\n✓ Main model: {main_output}")
    
    # Export segments separately
    segment_files = exporter.export_segments_separately(segmented)
    print(f"✓ Exported {len(segment_files)} separate segment files")


if __name__ == '__main__':
    print("PCto3D Pipeline - Example Usage\n")
    
    # Note: These examples assume you have a sample.ply file in the input/ folder
    # Uncomment the examples you want to run
    
    # example_1_basic_usage()
    # example_2_custom_processing()
    # example_3_region_growing()
    # example_4_clustering()
    # example_5_ransac_planes()
    # example_6_separate_segments()
    
    print("\nNote: Place a sample PLY file in input/sample.ply to run these examples")
    print("Uncomment the example functions you want to run in the __main__ block")

