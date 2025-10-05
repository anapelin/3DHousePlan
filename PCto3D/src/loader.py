"""
PLY File Loader Module
Handles loading and validation of PLY point cloud files.
"""

import os
import open3d as o3d
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PLYLoader:
    """Loads and validates PLY point cloud files."""
    
    def __init__(self, config):
        """
        Initialize PLY loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.input_folder = config['input']['ply_folder']
        self.default_file = config['input']['ply_file']
        
    def load(self, ply_path=None):
        """
        Load a PLY file from disk.
        
        Args:
            ply_path: Path to PLY file. If None, uses default from config.
            
        Returns:
            open3d.geometry.PointCloud or open3d.geometry.TriangleMesh
        """
        if ply_path is None:
            ply_path = os.path.join(self.input_folder, self.default_file)
            
        ply_path = Path(ply_path)
        
        if not ply_path.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")
            
        logger.info(f"Loading PLY file: {ply_path}")
        
        # Try loading as mesh first, then as point cloud
        try:
            geometry = o3d.io.read_triangle_mesh(str(ply_path))
            if len(geometry.triangles) > 0:
                logger.info(f"Loaded triangle mesh with {len(geometry.vertices)} vertices "
                          f"and {len(geometry.triangles)} triangles")
                return geometry
            else:
                # No triangles, treat as point cloud
                geometry = o3d.io.read_point_cloud(str(ply_path))
                logger.info(f"Loaded point cloud with {len(geometry.points)} points")
                return geometry
        except:
            # Fall back to point cloud loading
            geometry = o3d.io.read_point_cloud(str(ply_path))
            logger.info(f"Loaded point cloud with {len(geometry.points)} points")
            return geometry
    
    def validate(self, geometry):
        """
        Validate the loaded geometry.
        
        Args:
            geometry: Open3D geometry object
            
        Returns:
            bool: True if valid
        """
        if isinstance(geometry, o3d.geometry.PointCloud):
            if len(geometry.points) == 0:
                logger.error("Point cloud is empty")
                return False
            logger.info("✓ Valid point cloud")
            return True
            
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            if len(geometry.vertices) == 0:
                logger.error("Mesh has no vertices")
                return False
            logger.info("✓ Valid mesh")
            return True
            
        else:
            logger.error(f"Unsupported geometry type: {type(geometry)}")
            return False
    
    def get_info(self, geometry):
        """
        Get information about the loaded geometry.
        
        Args:
            geometry: Open3D geometry object
            
        Returns:
            dict: Geometry information
        """
        info = {
            'type': type(geometry).__name__,
            'has_normals': False,
            'has_colors': False,
            'bounds': None
        }
        
        if isinstance(geometry, o3d.geometry.PointCloud):
            info['num_points'] = len(geometry.points)
            info['has_normals'] = geometry.has_normals()
            info['has_colors'] = geometry.has_colors()
            points = np.asarray(geometry.points)
            
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            info['num_vertices'] = len(geometry.vertices)
            info['num_triangles'] = len(geometry.triangles)
            info['has_normals'] = geometry.has_vertex_normals()
            info['has_colors'] = geometry.has_vertex_colors()
            points = np.asarray(geometry.vertices)
        
        # Calculate bounds
        if len(points) > 0:
            info['bounds'] = {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist(),
                'center': points.mean(axis=0).tolist(),
                'size': (points.max(axis=0) - points.min(axis=0)).tolist()
            }
        
        return info

