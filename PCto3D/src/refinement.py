"""
Mesh Refinement Module
Handles mesh refinement, outlier removal, and noise filtering.
"""

import open3d as o3d
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MeshRefiner:
    """Refines meshes and point clouds by removing outliers and filtering noise."""
    
    def __init__(self, config):
        """
        Initialize mesh refiner with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.refinement_config = config['refinement']
        
    def refine(self, geometry):
        """
        Apply full refinement pipeline to geometry.
        
        Args:
            geometry: Open3D geometry object
            
        Returns:
            Refined geometry object
        """
        logger.info("Starting mesh refinement...")
        
        # Convert mesh to point cloud if needed
        is_mesh = isinstance(geometry, o3d.geometry.TriangleMesh)
        if is_mesh:
            logger.info("Converting mesh to point cloud for processing...")
            pcd = geometry.sample_points_uniformly(
                number_of_points=max(len(geometry.vertices), 10000)
            )
        else:
            pcd = geometry
            
        original_size = len(pcd.points)
        logger.info(f"Original point cloud size: {original_size}")
        
        # Remove outliers
        pcd = self._remove_outliers(pcd)
        logger.info(f"After outlier removal: {len(pcd.points)} points "
                   f"({100*(1-len(pcd.points)/original_size):.1f}% removed)")
        
        # Downsample and filter noise
        if self.refinement_config['noise_filtering']['enable']:
            pcd = self._filter_noise(pcd)
            logger.info(f"After noise filtering: {len(pcd.points)} points")
        
        # Estimate normals if not present
        if not pcd.has_normals():
            logger.info("Computing normals...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(30)
        
        # If original was a mesh, reconstruct
        if is_mesh and self.refinement_config['smoothing']['enable']:
            logger.info("Reconstructing mesh from refined point cloud...")
            mesh = self._reconstruct_mesh(pcd)
            mesh = self._smooth_mesh(mesh)
            return mesh
        
        return pcd
    
    def _remove_outliers(self, pcd):
        """
        Remove outlier points from point cloud.
        
        Args:
            pcd: Open3D point cloud
            
        Returns:
            Filtered point cloud
        """
        outlier_config = self.refinement_config['outlier_removal']
        method = outlier_config['method']
        
        logger.info(f"Removing outliers using {method} method...")
        
        if method == "statistical":
            pcd_filtered, ind = pcd.remove_statistical_outlier(
                nb_neighbors=outlier_config['nb_neighbors'],
                std_ratio=outlier_config['std_ratio']
            )
        elif method == "radius":
            pcd_filtered, ind = pcd.remove_radius_outlier(
                nb_points=outlier_config['min_nb_points'],
                radius=outlier_config['radius']
            )
        else:
            logger.warning(f"Unknown outlier removal method: {method}, skipping")
            return pcd
            
        return pcd_filtered
    
    def _filter_noise(self, pcd):
        """
        Filter noise using voxel downsampling.
        
        Args:
            pcd: Open3D point cloud
            
        Returns:
            Filtered point cloud
        """
        voxel_size = self.refinement_config['noise_filtering']['voxel_size']
        logger.info(f"Filtering noise with voxel size: {voxel_size}")
        
        pcd_filtered = pcd.voxel_down_sample(voxel_size=voxel_size)
        return pcd_filtered
    
    def _reconstruct_mesh(self, pcd):
        """
        Reconstruct mesh from point cloud using Poisson reconstruction.
        
        Args:
            pcd: Open3D point cloud with normals
            
        Returns:
            Reconstructed triangle mesh
        """
        logger.info("Performing Poisson surface reconstruction...")
        
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        logger.info(f"Reconstructed mesh with {len(mesh.vertices)} vertices "
                   f"and {len(mesh.triangles)} triangles")
        
        return mesh
    
    def _smooth_mesh(self, mesh):
        """
        Smooth mesh surface.
        
        Args:
            mesh: Open3D triangle mesh
            
        Returns:
            Smoothed mesh
        """
        smooth_config = self.refinement_config['smoothing']
        method = smooth_config['method']
        iterations = smooth_config['iterations']
        
        logger.info(f"Smoothing mesh using {method} filter ({iterations} iterations)...")
        
        if method == "laplacian":
            mesh_smooth = mesh.filter_smooth_laplacian(
                number_of_iterations=iterations
            )
        elif method == "taubin":
            mesh_smooth = mesh.filter_smooth_taubin(
                number_of_iterations=iterations
            )
        else:
            logger.warning(f"Unknown smoothing method: {method}, skipping")
            return mesh
            
        return mesh_smooth
    
    def get_largest_cluster(self, pcd, min_points=100):
        """
        Extract the largest cluster from point cloud.
        
        Args:
            pcd: Open3D point cloud
            min_points: Minimum points for a valid cluster
            
        Returns:
            Largest cluster point cloud
        """
        logger.info("Extracting largest cluster...")
        
        labels = np.array(pcd.cluster_dbscan(
            eps=0.05, min_points=min_points, print_progress=False
        ))
        
        if len(labels) == 0 or labels.max() < 0:
            logger.warning("No clusters found, returning original point cloud")
            return pcd
        
        # Find largest cluster
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        
        logger.info(f"Found {len(unique_labels)} clusters, "
                   f"largest has {counts.max()} points")
        
        # Extract largest cluster
        largest_cluster_indices = np.where(labels == largest_label)[0]
        largest_cluster = pcd.select_by_index(largest_cluster_indices)
        
        return largest_cluster

