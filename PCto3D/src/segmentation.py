"""
Surface Segmentation Module
Handles surface segmentation from point clouds using various methods.
"""

import open3d as o3d
import numpy as np
import logging
from sklearn.cluster import DBSCAN, KMeans
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SurfaceSegmenter:
    """Performs surface segmentation on point clouds."""
    
    def __init__(self, config):
        """
        Initialize surface segmenter with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.seg_config = config['segmentation']
        
    def segment(self, geometry):
        """
        Perform surface segmentation on geometry.
        
        Args:
            geometry: Open3D geometry object
            
        Returns:
            Segmented geometry with colored segments
        """
        logger.info("Starting surface segmentation...")
        
        # Convert mesh to point cloud if needed
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            logger.info("Sampling points from mesh for segmentation...")
            pcd = geometry.sample_points_uniformly(
                number_of_points=max(len(geometry.vertices) * 2, 50000)
            )
        else:
            pcd = geometry
        
        # Ensure normals are computed
        if not pcd.has_normals():
            logger.info("Computing normals for segmentation...")
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
        
        # Perform segmentation based on method
        method = self.seg_config['method']
        logger.info(f"Using {method} segmentation method...")
        
        if method == "region_growing":
            labels = self._region_growing_segmentation(pcd)
        elif method == "clustering":
            labels = self._clustering_segmentation(pcd)
        elif method == "ransac":
            labels = self._ransac_segmentation(pcd)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Filter small segments
        labels = self._filter_small_segments(labels)
        
        # Merge similar segments if enabled
        if self.seg_config['merge_similar_segments']:
            labels = self._merge_similar_segments(pcd, labels)
        
        # Apply colors to segments
        if self.config['visualization']['colorize_segments']:
            pcd = self._colorize_segments(pcd, labels)
        
        # Store labels as custom attribute
        pcd.segment_labels = labels
        
        num_segments = len(np.unique(labels[labels >= 0]))
        logger.info(f"Segmentation complete: {num_segments} segments found")
        
        return pcd
    
    def _region_growing_segmentation(self, pcd):
        """
        Perform region growing segmentation based on normals and curvature.
        
        Args:
            pcd: Open3D point cloud with normals
            
        Returns:
            numpy array of segment labels
        """
        config = self.seg_config['region_growing']
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        n_points = len(points)
        
        # Compute curvature (simplified)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        curvatures = np.zeros(n_points)
        
        logger.info("Computing curvature...")
        for i in tqdm(range(n_points), desc="Curvature"):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 20)
            if k > 3:
                neighbor_normals = normals[idx[1:]]
                normal_variance = np.var(neighbor_normals, axis=0).sum()
                curvatures[i] = normal_variance
        
        # Region growing
        labels = -np.ones(n_points, dtype=int)
        current_label = 0
        
        # Sort points by curvature (start with flat regions)
        sorted_indices = np.argsort(curvatures)
        
        logger.info("Growing regions...")
        for seed_idx in tqdm(sorted_indices, desc="Region growing"):
            if labels[seed_idx] >= 0:
                continue
                
            # Start new region
            region = [seed_idx]
            labels[seed_idx] = current_label
            queue = [seed_idx]
            
            while queue:
                current_idx = queue.pop(0)
                [k, idx, _] = pcd_tree.search_knn_vector_3d(
                    pcd.points[current_idx], 20
                )
                
                for neighbor_idx in idx[1:]:
                    if labels[neighbor_idx] >= 0:
                        continue
                    
                    # Check normal similarity
                    normal_diff = np.dot(
                        normals[current_idx],
                        normals[neighbor_idx]
                    )
                    
                    if normal_diff > (1 - config['normal_variance_threshold']):
                        if curvatures[neighbor_idx] < config['curvature_threshold']:
                            labels[neighbor_idx] = current_label
                            region.append(neighbor_idx)
                            queue.append(neighbor_idx)
            
            if len(region) >= config['min_cluster_size']:
                current_label += 1
            else:
                # Mark small regions as unlabeled
                labels[region] = -1
        
        return labels
    
    def _clustering_segmentation(self, pcd):
        """
        Perform clustering-based segmentation.
        
        Args:
            pcd: Open3D point cloud
            
        Returns:
            numpy array of segment labels
        """
        config = self.seg_config['clustering']
        method = config['method']
        
        points = np.asarray(pcd.points)
        
        # Combine points and normals for clustering
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            features = np.hstack([points, normals * 0.5])
        else:
            features = points
        
        logger.info(f"Performing {method} clustering...")
        
        if method == "dbscan":
            clusterer = DBSCAN(
                eps=config['eps'],
                min_samples=config['min_samples'],
                n_jobs=-1
            )
            labels = clusterer.fit_predict(features)
            
        elif method == "kmeans":
            clusterer = KMeans(
                n_clusters=config['n_clusters'],
                random_state=42,
                n_init=10
            )
            labels = clusterer.fit_predict(features)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return labels
    
    def _ransac_segmentation(self, pcd):
        """
        Perform RANSAC-based plane segmentation.
        
        Args:
            pcd: Open3D point cloud
            
        Returns:
            numpy array of segment labels
        """
        config = self.seg_config['ransac']
        
        labels = -np.ones(len(pcd.points), dtype=int)
        remaining_pcd = pcd
        current_label = 0
        
        logger.info("Extracting planes using RANSAC...")
        
        for i in tqdm(range(config['max_planes']), desc="RANSAC planes"):
            if len(remaining_pcd.points) < 100:
                break
            
            # Segment plane
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=config['distance_threshold'],
                ransac_n=config['ransac_n'],
                num_iterations=config['num_iterations']
            )
            
            if len(inliers) < self.seg_config['min_segment_size']:
                break
            
            # Get original indices of inliers
            remaining_points = np.asarray(remaining_pcd.points)
            original_points = np.asarray(pcd.points)
            
            for inlier_idx in inliers:
                inlier_point = remaining_points[inlier_idx]
                # Find in original point cloud
                distances = np.linalg.norm(original_points - inlier_point, axis=1)
                original_idx = np.argmin(distances)
                if distances[original_idx] < 0.001:  # Close enough
                    labels[original_idx] = current_label
            
            current_label += 1
            
            # Remove inliers from remaining point cloud
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        
        logger.info(f"Extracted {current_label} planes")
        
        return labels
    
    def _filter_small_segments(self, labels):
        """
        Remove segments with too few points.
        
        Args:
            labels: Segment labels array
            
        Returns:
            Filtered labels array
        """
        min_size = self.seg_config['min_segment_size']
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        
        filtered_labels = labels.copy()
        for label, count in zip(unique_labels, counts):
            if count < min_size:
                filtered_labels[labels == label] = -1
        
        # Re-label to be consecutive
        unique_labels = np.unique(filtered_labels[filtered_labels >= 0])
        new_labels = filtered_labels.copy()
        for new_label, old_label in enumerate(unique_labels):
            new_labels[filtered_labels == old_label] = new_label
        
        return new_labels
    
    def _merge_similar_segments(self, pcd, labels):
        """
        Merge similar adjacent segments.
        
        Args:
            pcd: Open3D point cloud
            labels: Segment labels array
            
        Returns:
            Merged labels array
        """
        logger.info("Merging similar segments...")
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        unique_labels = np.unique(labels[labels >= 0])
        if len(unique_labels) <= 1:
            return labels
        
        # Compute segment centroids and average normals
        centroids = []
        avg_normals = []
        
        for label in unique_labels:
            mask = labels == label
            centroids.append(points[mask].mean(axis=0))
            if normals is not None:
                avg_normals.append(normals[mask].mean(axis=0))
        
        centroids = np.array(centroids)
        if normals is not None:
            avg_normals = np.array(avg_normals)
            # Normalize
            avg_normals = avg_normals / (np.linalg.norm(avg_normals, axis=1, keepdims=True) + 1e-8)
        
        # Merge similar segments
        merge_threshold = self.seg_config['merge_threshold']
        merged_labels = labels.copy()
        label_mapping = {label: label for label in unique_labels}
        
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                label_i = unique_labels[i]
                label_j = unique_labels[j]
                
                # Check if already merged
                if label_mapping[label_i] != label_i or label_mapping[label_j] != label_j:
                    continue
                
                # Check distance
                distance = np.linalg.norm(centroids[i] - centroids[j])
                
                # Check normal similarity
                normal_similarity = 1.0
                if normals is not None:
                    normal_similarity = np.dot(avg_normals[i], avg_normals[j])
                
                # Merge if similar
                if distance < merge_threshold and normal_similarity > 0.9:
                    # Merge j into i
                    merged_labels[labels == label_j] = label_i
                    label_mapping[label_j] = label_i
        
        return merged_labels
    
    def _colorize_segments(self, pcd, labels):
        """
        Apply colors to segments.
        
        Args:
            pcd: Open3D point cloud
            labels: Segment labels array
            
        Returns:
            Colored point cloud
        """
        unique_labels = np.unique(labels[labels >= 0])
        n_segments = len(unique_labels)
        
        # Generate distinct colors
        colors = np.zeros((len(labels), 3))
        
        # Use HSV color space for better distinction
        for i, label in enumerate(unique_labels):
            hue = i / max(n_segments, 1)
            # Convert HSV to RGB (simplified)
            c = np.array([
                abs(hue * 6 - 3) - 1,
                2 - abs(hue * 6 - 2),
                2 - abs(hue * 6 - 4)
            ])
            c = np.clip(c, 0, 1)
            colors[labels == label] = c
        
        # Unlabeled points in gray
        colors[labels < 0] = [0.5, 0.5, 0.5]
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

