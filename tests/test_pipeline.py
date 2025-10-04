#!/usr/bin/env python3
"""
Unit tests for the 3D House Plan Pipeline

This module contains comprehensive tests for all pipeline components.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import Pipeline, PipelineConfig
from video_processor import VideoProcessor, VideoFrame
from plan_parser import PlanParser, FloorPlan, Room
from reconstruction import ReconstructionEngine, Room3D, Mesh3D
from blender_export import BlenderExporter

class TestVideoProcessor(unittest.TestCase):
    """Test cases for video processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.video_processor = VideoProcessor()
    
    def test_video_processor_initialization(self):
        """Test video processor initialization."""
        self.assertIsNotNone(self.video_processor)
        self.assertIsNotNone(self.video_processor.pose)
    
    def test_video_frame_creation(self):
        """Test video frame creation."""
        frame = VideoFrame(
            frame_id=1,
            timestamp=0.1,
            image=np.zeros((480, 640, 3), dtype=np.uint8)
        )
        
        self.assertEqual(frame.frame_id, 1)
        self.assertEqual(frame.timestamp, 0.1)
        self.assertEqual(frame.image.shape, (480, 640, 3))
    
    def test_depth_estimation(self):
        """Test depth estimation functionality."""
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test depth estimation
        depth_map = self.video_processor._estimate_depth(test_image)
        
        self.assertIsNotNone(depth_map)
        self.assertEqual(depth_map.shape, (480, 640))
        self.assertTrue(np.all(depth_map >= 0))
        self.assertTrue(np.all(depth_map <= 1))
    
    def test_room_feature_extraction(self):
        """Test room feature extraction."""
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test feature extraction
        features = self.video_processor._extract_room_features(test_image, None)
        
        self.assertIsNotNone(features)
        self.assertIn('walls', features)
        self.assertIn('corners', features)
        self.assertIn('room_dimensions', features)

class TestPlanParser(unittest.TestCase):
    """Test cases for plan parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plan_parser = PlanParser()
    
    def test_plan_parser_initialization(self):
        """Test plan parser initialization."""
        self.assertIsNotNone(self.plan_parser)
        self.assertEqual(self.plan_parser.wall_thickness, 0.2)
        self.assertEqual(self.plan_parser.door_width, 0.9)
    
    def test_room_creation(self):
        """Test room object creation."""
        room = Room(
            name="TestRoom",
            area=20.0,
            dimensions={'width': 4.0, 'length': 5.0, 'height': 2.5},
            corners=[(0, 0), (4, 0), (4, 5), (0, 5)],
            walls=[],
            windows=[],
            doors=[],
            furniture=[]
        )
        
        self.assertEqual(room.name, "TestRoom")
        self.assertEqual(room.area, 20.0)
        self.assertEqual(room.dimensions['width'], 4.0)
    
    def test_floor_plan_creation(self):
        """Test floor plan object creation."""
        room = Room(
            name="TestRoom",
            area=20.0,
            dimensions={'width': 4.0, 'length': 5.0, 'height': 2.5},
            corners=[(0, 0), (4, 0), (4, 5), (0, 5)],
            walls=[],
            windows=[],
            doors=[],
            furniture=[]
        )
        
        floor_plan = FloorPlan(
            rooms=[room],
            total_area=20.0,
            scale_factor=100.0,
            image_dimensions=(800, 600)
        )
        
        self.assertEqual(len(floor_plan.rooms), 1)
        self.assertEqual(floor_plan.total_area, 20.0)
        self.assertEqual(floor_plan.scale_factor, 100.0)
    
    def test_wall_merging(self):
        """Test wall merging functionality."""
        walls = [
            {'start': (0, 0), 'end': (100, 0), 'length': 100, 'angle': 0},
            {'start': (0, 5), 'end': (100, 5), 'length': 100, 'angle': 0},  # Similar wall
            {'start': (0, 0), 'end': (0, 100), 'length': 100, 'angle': np.pi/2}  # Different wall
        ]
        
        merged_walls = self.plan_parser._merge_similar_walls(walls)
        
        # Should have fewer walls after merging
        self.assertLessEqual(len(merged_walls), len(walls))

class TestReconstructionEngine(unittest.TestCase):
    """Test cases for 3D reconstruction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reconstruction_engine = ReconstructionEngine()
    
    def test_reconstruction_engine_initialization(self):
        """Test reconstruction engine initialization."""
        self.assertIsNotNone(self.reconstruction_engine)
        self.assertIsNotNone(self.reconstruction_engine.materials)
    
    def test_mesh_creation(self):
        """Test mesh object creation."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 7, 6], [4, 6, 5],  # Top
            [0, 4, 5], [0, 5, 1],  # Front
            [1, 5, 6], [1, 6, 2],  # Right
            [2, 6, 7], [2, 7, 3],  # Back
            [3, 7, 4], [3, 4, 0]   # Left
        ])
        
        mesh = Mesh3D(vertices=vertices, faces=faces)
        
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.faces), 12)
    
    def test_room_geometry_creation(self):
        """Test room geometry creation."""
        vertices, faces = self.reconstruction_engine._create_room_geometry(4.0, 5.0, 2.5)
        
        self.assertIsNotNone(vertices)
        self.assertIsNotNone(faces)
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
    
    def test_normal_calculation(self):
        """Test normal calculation."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3]
        ])
        
        normals = self.reconstruction_engine._calculate_normals(vertices, faces)
        
        self.assertIsNotNone(normals)
        self.assertEqual(normals.shape, vertices.shape)
    
    def test_mesh_optimization(self):
        """Test mesh optimization."""
        # Create a mesh with duplicate vertices
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 0], [1, 0, 0]  # Duplicates
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 2], [4, 2, 3]  # Using duplicate vertices
        ])
        
        mesh = Mesh3D(vertices=vertices, faces=faces)
        optimized_mesh = self.reconstruction_engine.optimize_mesh(mesh)
        
        # Should have fewer vertices after optimization
        self.assertLessEqual(len(optimized_mesh.vertices), len(mesh.vertices))
    
    def test_mesh_validation(self):
        """Test mesh validation."""
        # Valid mesh
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3]
        ])
        
        mesh = Mesh3D(vertices=vertices, faces=faces)
        issues = self.reconstruction_engine.validate_mesh(mesh)
        
        self.assertEqual(len(issues), 0)
        
        # Invalid mesh (empty)
        empty_mesh = Mesh3D(vertices=np.array([]), faces=np.array([]))
        issues = self.reconstruction_engine.validate_mesh(empty_mesh)
        
        self.assertGreater(len(issues), 0)

class TestBlenderExporter(unittest.TestCase):
    """Test cases for Blender export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blender_exporter = BlenderExporter()
    
    def test_blender_exporter_initialization(self):
        """Test Blender exporter initialization."""
        self.assertIsNotNone(self.blender_exporter)
        self.assertIsNotNone(self.blender_exporter.supported_formats)
        self.assertIn('obj', self.blender_exporter.supported_formats)
        self.assertIn('fbx', self.blender_exporter.supported_formats)
    
    def test_export_options_merging(self):
        """Test export options merging."""
        default_options = {'include_materials': True, 'include_normals': True}
        user_options = {'include_materials': False}
        
        merged = self.blender_exporter._merge_export_options('obj', user_options)
        
        self.assertEqual(merged['include_materials'], False)
        self.assertEqual(merged['include_normals'], True)
    
    def test_blender_script_generation(self):
        """Test Blender script generation."""
        # Create a simple room
        vertices = np.array([
            [0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0],
            [0, 0, 2.5], [4, 0, 2.5], [4, 5, 2.5], [0, 5, 2.5]
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Floor
            [4, 7, 6], [4, 6, 5],  # Ceiling
            [0, 4, 5], [0, 5, 1],  # Front wall
            [1, 5, 6], [1, 6, 2],  # Right wall
            [2, 6, 7], [2, 7, 3],  # Back wall
            [3, 7, 4], [3, 4, 0]   # Left wall
        ])
        
        mesh = Mesh3D(vertices=vertices, faces=faces)
        room3d = Room3D(
            name="TestRoom",
            mesh=mesh,
            dimensions={'width': 4.0, 'length': 5.0, 'height': 2.5},
            furniture=[],
            windows=[],
            doors=[]
        )
        
        script = self.blender_exporter._generate_blender_script(room3d)
        
        self.assertIsNotNone(script)
        self.assertIn('bpy', script)
        self.assertIn('TestRoom', script)
        self.assertIn('vertices', script)
        self.assertIn('faces', script)

class TestPipeline(unittest.TestCase):
    """Test cases for the main pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.pipeline = Pipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.video_processor)
        self.assertIsNotNone(self.pipeline.plan_parser)
        self.assertIsNotNone(self.pipeline.reconstruction_engine)
        self.assertIsNotNone(self.pipeline.blender_exporter)
    
    def test_input_validation(self):
        """Test input validation."""
        # Test with non-existent files
        issues = self.pipeline.validate_inputs("nonexistent_video.mp4", "nonexistent_plan.pdf")
        
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("does not exist" in issue for issue in issues))
    
    def test_configuration_management(self):
        """Test configuration save/load."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            self.pipeline.save_config(config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Load configuration
            self.pipeline.load_config(config_path)
            
        finally:
            # Clean up
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_processing_stats(self):
        """Test processing statistics."""
        # Test with non-existent files (should handle gracefully)
        stats = self.pipeline.get_processing_stats("nonexistent_video.mp4", "nonexistent_plan.pdf")
        
        # Should return empty stats without crashing
        self.assertIsInstance(stats, dict)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PipelineConfig()
        self.pipeline = Pipeline(self.config)
    
    def test_pipeline_components_integration(self):
        """Test that all pipeline components work together."""
        # This test would require actual video and plan files
        # For now, we'll just test that components can be initialized together
        
        self.assertIsNotNone(self.pipeline.video_processor)
        self.assertIsNotNone(self.pipeline.plan_parser)
        self.assertIsNotNone(self.pipeline.reconstruction_engine)
        self.assertIsNotNone(self.pipeline.blender_exporter)
        
        # Test that components have the expected interfaces
        self.assertTrue(hasattr(self.pipeline.video_processor, 'process_video'))
        self.assertTrue(hasattr(self.pipeline.plan_parser, 'parse_plan'))
        self.assertTrue(hasattr(self.pipeline.reconstruction_engine, 'reconstruct_room'))
        self.assertTrue(hasattr(self.pipeline.blender_exporter, 'export_room'))

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVideoProcessor,
        TestPlanParser,
        TestReconstructionEngine,
        TestBlenderExporter,
        TestPipeline,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
