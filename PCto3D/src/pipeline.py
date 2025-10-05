"""
Main Pipeline Module
Orchestrates the complete PCto3D processing pipeline.
"""

import os
import logging
import time
from pathlib import Path

from .loader import PLYLoader
from .refinement import MeshRefiner
from .segmentation import SurfaceSegmenter
from .exporter import OBJExporter

logger = logging.getLogger(__name__)


class PCto3DPipeline:
    """Main pipeline for processing PLY files to 3D environments."""
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.loader = PLYLoader(config)
        self.refiner = MeshRefiner(config)
        self.segmenter = SurfaceSegmenter(config)
        self.exporter = OBJExporter(config)
        
        logger.info("PCto3D Pipeline initialized")
    
    def run(self, ply_path=None, output_path=None):
        """
        Run the complete processing pipeline.
        
        Args:
            ply_path: Path to input PLY file (optional)
            output_path: Path to output OBJ file (optional)
            
        Returns:
            dict: Pipeline results and statistics
        """
        start_time = time.time()
        results = {
            'success': False,
            'stages': {},
            'output_path': None,
            'processing_time': 0
        }
        
        try:
            logger.info("="*60)
            logger.info("Starting PCto3D Pipeline")
            logger.info("="*60)
            
            # Stage 1: Load PLY file
            logger.info("\n[Stage 1/4] Loading PLY file...")
            stage_start = time.time()
            
            geometry = self.loader.load(ply_path)
            
            if not self.loader.validate(geometry):
                raise ValueError("Invalid geometry loaded")
            
            info = self.loader.get_info(geometry)
            results['stages']['load'] = {
                'time': time.time() - stage_start,
                'info': info
            }
            
            logger.info(f"✓ Stage 1 complete ({results['stages']['load']['time']:.2f}s)")
            self._log_geometry_info(info)
            
            # Stage 2: Refine mesh and remove outliers
            logger.info("\n[Stage 2/4] Refining mesh and removing outliers...")
            stage_start = time.time()
            
            refined_geometry = self.refiner.refine(geometry)
            self.exporter.save_intermediate(refined_geometry, '01_refined')
            
            results['stages']['refine'] = {
                'time': time.time() - stage_start
            }
            
            logger.info(f"✓ Stage 2 complete ({results['stages']['refine']['time']:.2f}s)")
            
            # Stage 3: Surface segmentation
            logger.info("\n[Stage 3/4] Performing surface segmentation...")
            stage_start = time.time()
            
            segmented_geometry = self.segmenter.segment(refined_geometry)
            self.exporter.save_intermediate(segmented_geometry, '02_segmented')
            
            if hasattr(segmented_geometry, 'segment_labels'):
                import numpy as np
                labels = segmented_geometry.segment_labels
                n_segments = len(np.unique(labels[labels >= 0]))
                results['stages']['segment'] = {
                    'time': time.time() - stage_start,
                    'num_segments': n_segments
                }
                logger.info(f"  Found {n_segments} surface segments")
            else:
                results['stages']['segment'] = {
                    'time': time.time() - stage_start
                }
            
            logger.info(f"✓ Stage 3 complete ({results['stages']['segment']['time']:.2f}s)")
            
            # Stage 4: Export to OBJ
            logger.info("\n[Stage 4/4] Exporting to OBJ format...")
            stage_start = time.time()
            
            output_file = self.exporter.export(segmented_geometry, output_path)
            results['output_path'] = str(output_file)
            
            # Optionally export segments separately
            if hasattr(segmented_geometry, 'segment_labels'):
                segment_files = self.exporter.export_segments_separately(
                    segmented_geometry
                )
                results['segment_files'] = [str(f) for f in segment_files]
            
            results['stages']['export'] = {
                'time': time.time() - stage_start
            }
            
            logger.info(f"✓ Stage 4 complete ({results['stages']['export']['time']:.2f}s)")
            
            # Pipeline complete
            results['success'] = True
            results['processing_time'] = time.time() - start_time
            
            logger.info("\n" + "="*60)
            logger.info("Pipeline Complete!")
            logger.info("="*60)
            logger.info(f"Total processing time: {results['processing_time']:.2f}s")
            logger.info(f"Output saved to: {results['output_path']}")
            
            if 'segment_files' in results:
                logger.info(f"Individual segments: {len(results['segment_files'])} files")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results['error'] = str(e)
        
        return results
    
    def _log_geometry_info(self, info):
        """Log geometry information."""
        logger.info(f"  Type: {info['type']}")
        
        if 'num_points' in info:
            logger.info(f"  Points: {info['num_points']}")
        
        if 'num_vertices' in info:
            logger.info(f"  Vertices: {info['num_vertices']}")
            logger.info(f"  Triangles: {info['num_triangles']}")
        
        if info['bounds']:
            bounds = info['bounds']
            logger.info(f"  Bounds: {bounds['size']}")
            logger.info(f"  Center: {bounds['center']}")

