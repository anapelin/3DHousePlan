"""
Wrapper for COLMAP command-line interface
"""

import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger("colmap_pipeline")


class COLMAPWrapper:
    """Wrapper for COLMAP commands with logging and error handling."""
    
    def __init__(self, workspace: Path, config: dict, use_gpu: bool = True):
        """
        Initialize COLMAP wrapper.
        
        Args:
            workspace: Workspace directory
            config: COLMAP configuration
            use_gpu: Whether to use GPU acceleration
        """
        self.workspace = workspace
        self.config = config
        self.use_gpu = use_gpu and not config.get("general", {}).get("cpu_only", False)
        
        self.database_path = workspace / "database.db"
        self.image_path = workspace / "images"
        self.sparse_path = workspace / "sparse"
        self.dense_path = workspace / "dense"
        
        # Create log directory
        self.log_dir = workspace / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def feature_extraction(self) -> bool:
        """
        Run COLMAP feature extraction.
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP feature extraction...")
        
        fe_config = self.config.get("feature_extraction", {})
        sift_config = fe_config.get("sift", {})
        
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_path),
        ]
        
        # Camera model
        camera_model = fe_config.get("camera_model", "OPENCV")
        cmd.extend(["--ImageReader.camera_model", camera_model])
        
        # Single camera
        if fe_config.get("single_camera", True):
            cmd.extend(["--ImageReader.single_camera", "1"])
        
        # SIFT options
        cmd.extend([
            "--SiftExtraction.max_num_features", str(sift_config.get("max_num_features", 8192)),
            "--SiftExtraction.first_octave", str(sift_config.get("first_octave", -1)),
            "--SiftExtraction.num_octaves", str(sift_config.get("num_octaves", 4)),
            "--SiftExtraction.octave_resolution", str(sift_config.get("octave_resolution", 3)),
            "--SiftExtraction.peak_threshold", str(sift_config.get("peak_threshold", 0.0066666666667)),
            "--SiftExtraction.edge_threshold", str(sift_config.get("edge_threshold", 10)),
            "--SiftExtraction.max_num_orientations", str(sift_config.get("max_num_orientations", 2)),
        ])
        
        # GPU
        if self.use_gpu:
            gpu_index = fe_config.get("gpu_index", "0")
            cmd.extend(["--SiftExtraction.use_gpu", "1", "--SiftExtraction.gpu_index", gpu_index])
        else:
            cmd.extend(["--SiftExtraction.use_gpu", "0"])
        
        return self._run_command(cmd, "feature_extraction")
    
    def feature_matching(self) -> bool:
        """
        Run COLMAP feature matching.
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP feature matching...")
        
        fm_config = self.config.get("feature_matching", {})
        method = fm_config.get("method", "sequential")
        
        if method == "exhaustive":
            return self._exhaustive_matching()
        elif method == "sequential":
            return self._sequential_matching()
        elif method == "spatial":
            return self._spatial_matching()
        elif method == "vocab_tree":
            return self._vocab_tree_matching()
        else:
            logger.error(f"Unknown matching method: {method}")
            return False
    
    def _exhaustive_matching(self) -> bool:
        """Run exhaustive feature matching."""
        fm_config = self.config.get("feature_matching", {})
        exhaustive_config = fm_config.get("exhaustive", {})
        
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(self.database_path),
        ]
        
        # Block size
        cmd.extend(["--SiftMatching.block_size", str(exhaustive_config.get("block_size", 50))])
        
        # Add common matching options
        self._add_matching_options(cmd, fm_config)
        
        return self._run_command(cmd, "exhaustive_matching")
    
    def _sequential_matching(self) -> bool:
        """Run sequential feature matching (good for video sequences)."""
        fm_config = self.config.get("feature_matching", {})
        seq_config = fm_config.get("sequential", {})
        
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path", str(self.database_path),
        ]
        
        # Sequential options
        cmd.extend([
            "--SequentialMatching.overlap", str(seq_config.get("overlap", 10)),
            "--SequentialMatching.quadratic_overlap", str(int(seq_config.get("quadratic_overlap", False))),
        ])
        
        # Loop detection
        if seq_config.get("loop_detection", True):
            cmd.extend([
                "--SequentialMatching.loop_detection", "1",
                "--SequentialMatching.loop_detection_num_images", 
                str(seq_config.get("loop_detection_num_images", 50)),
            ])
        
        # Add common matching options
        self._add_matching_options(cmd, fm_config)
        
        return self._run_command(cmd, "sequential_matching")
    
    def _spatial_matching(self) -> bool:
        """Run spatial feature matching."""
        fm_config = self.config.get("feature_matching", {})
        spatial_config = fm_config.get("spatial", {})
        
        cmd = [
            "colmap", "spatial_matcher",
            "--database_path", str(self.database_path),
        ]
        
        # Spatial options
        cmd.extend([
            "--SpatialMatching.max_num_neighbors", str(spatial_config.get("max_num_neighbors", 50)),
            "--SpatialMatching.max_distance", str(spatial_config.get("max_distance", 0.7)),
        ])
        
        # Add common matching options
        self._add_matching_options(cmd, fm_config)
        
        return self._run_command(cmd, "spatial_matching")
    
    def _vocab_tree_matching(self) -> bool:
        """Run vocabulary tree feature matching."""
        logger.warning("Vocabulary tree matching requires pre-built vocabulary tree")
        logger.info("Falling back to sequential matching")
        return self._sequential_matching()
    
    def _add_matching_options(self, cmd: List[str], fm_config: dict):
        """Add common matching options to command."""
        cmd.extend([
            "--SiftMatching.cross_check", str(int(fm_config.get("cross_check", True))),
            "--SiftMatching.max_ratio", str(fm_config.get("max_ratio", 0.8)),
            "--SiftMatching.max_distance", str(fm_config.get("max_distance", 0.7)),
            "--SiftMatching.max_error", str(fm_config.get("max_error", 4.0)),
        ])
        
        # GPU
        if self.use_gpu:
            gpu_index = fm_config.get("gpu_index", "0")
            cmd.extend(["--SiftMatching.use_gpu", "1", "--SiftMatching.gpu_index", gpu_index])
        else:
            cmd.extend(["--SiftMatching.use_gpu", "0"])
    
    def sparse_reconstruction(self) -> bool:
        """
        Run COLMAP sparse reconstruction (mapper).
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP sparse reconstruction (mapper)...")
        
        mapper_config = self.config.get("sparse_reconstruction", {})
        
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_path),
            "--output_path", str(self.sparse_path),
        ]
        
        # Mapper options
        cmd.extend([
            "--Mapper.min_num_matches", str(mapper_config.get("min_num_matches", 15)),
            "--Mapper.ignore_watermarks", str(int(mapper_config.get("ignore_watermarks", False))),
            "--Mapper.multiple_models", str(int(mapper_config.get("multiple_models", False))),
            "--Mapper.max_num_models", str(mapper_config.get("max_num_models", 50)),
            "--Mapper.max_model_overlap", str(mapper_config.get("max_model_overlap", 20)),
            "--Mapper.min_model_size", str(mapper_config.get("min_model_size", 10)),
            "--Mapper.init_num_trials", str(mapper_config.get("init_num_trials", 200)),
            "--Mapper.extract_colors", str(int(mapper_config.get("extract_colors", True))),
            "--Mapper.ba_refine_focal_length", str(int(mapper_config.get("ba_refine_focal_length", True))),
            "--Mapper.ba_refine_principal_point", str(int(mapper_config.get("ba_refine_principal_point", False))),
            "--Mapper.ba_refine_extra_params", str(int(mapper_config.get("ba_refine_extra_params", True))),
        ])
        
        # Number of threads
        num_threads = mapper_config.get("num_threads", -1)
        if num_threads > 0:
            cmd.extend(["--Mapper.num_threads", str(num_threads)])
        
        return self._run_command(cmd, "sparse_reconstruction")
    
    def image_undistortion(self) -> bool:
        """
        Run COLMAP image undistortion.
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP image undistortion...")
        
        undist_config = self.config.get("undistortion", {})
        
        # Find sparse model (usually in sparse/0)
        sparse_model = self.sparse_path / "0"
        if not sparse_model.exists():
            logger.error(f"Sparse model not found at {sparse_model}")
            return False
        
        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(self.image_path),
            "--input_path", str(sparse_model),
            "--output_path", str(self.dense_path),
            "--output_type", "COLMAP",
        ]
        
        # Undistortion options
        cmd.extend([
            "--max_image_size", str(undist_config.get("max_image_size", 2000)),
            "--blank_pixels", str(undist_config.get("blank_pixels", 0.0)),
            "--min_scale", str(undist_config.get("min_scale", 0.2)),
            "--max_scale", str(undist_config.get("max_scale", 2.0)),
        ])
        
        return self._run_command(cmd, "image_undistortion")
    
    def patch_match_stereo(self) -> bool:
        """
        Run COLMAP patch match stereo for dense reconstruction.
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP patch match stereo...")
        
        dense_config = self.config.get("dense_reconstruction", {})
        
        cmd = [
            "colmap", "patch_match_stereo",
            "--workspace_path", str(self.dense_path),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", str(int(dense_config.get("geom_consistency", True))),
        ]
        
        # Dense reconstruction options
        cmd.extend([
            "--PatchMatchStereo.window_radius", str(dense_config.get("window_radius", 5)),
            "--PatchMatchStereo.window_step", str(dense_config.get("window_step", 1)),
            "--PatchMatchStereo.sigma_spatial", str(dense_config.get("sigma_spatial", 3.0)),
            "--PatchMatchStereo.sigma_color", str(dense_config.get("sigma_color", 0.2)),
            "--PatchMatchStereo.num_samples", str(dense_config.get("num_samples", 15)),
            "--PatchMatchStereo.ncc_sigma", str(dense_config.get("ncc_sigma", 0.6)),
            "--PatchMatchStereo.min_triangulation_angle", str(dense_config.get("min_triangulation_angle", 1.0)),
            "--PatchMatchStereo.incident_angle_sigma", str(dense_config.get("incident_angle_sigma", 0.9)),
            "--PatchMatchStereo.num_iterations", str(dense_config.get("num_iterations", 5)),
            "--PatchMatchStereo.geom_consistency_regularizer", str(dense_config.get("geom_consistency_regularizer", 0.3)),
            "--PatchMatchStereo.geom_consistency_max_cost", str(dense_config.get("geom_consistency_max_cost", 3.0)),
            "--PatchMatchStereo.filter", str(int(dense_config.get("filter", True))),
            "--PatchMatchStereo.cache_size", str(dense_config.get("cache_size", 32)),
        ])
        
        # Depth range
        if dense_config.get("depth_min", 0.0) > 0:
            cmd.extend(["--PatchMatchStereo.depth_min", str(dense_config.get("depth_min"))])
        if dense_config.get("depth_max", 100.0) < 100:
            cmd.extend(["--PatchMatchStereo.depth_max", str(dense_config.get("depth_max"))])
        
        # GPU
        if self.use_gpu:
            gpu_index = dense_config.get("gpu_index", "0")
            cmd.extend(["--PatchMatchStereo.gpu_index", gpu_index])
        
        return self._run_command(cmd, "patch_match_stereo")
    
    def stereo_fusion(self) -> bool:
        """
        Run COLMAP stereo fusion to create dense point cloud.
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP stereo fusion...")
        
        fusion_config = self.config.get("stereo_fusion", {})
        
        output_path = self.dense_path / "fused.ply"
        
        cmd = [
            "colmap", "stereo_fusion",
            "--workspace_path", str(self.dense_path),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(output_path),
        ]
        
        # Fusion options
        cmd.extend([
            "--StereoFusion.min_num_pixels", str(fusion_config.get("min_num_pixels", 5)),
            "--StereoFusion.max_num_pixels", str(fusion_config.get("max_num_pixels", 10000)),
            "--StereoFusion.max_traversal_depth", str(fusion_config.get("max_traversal_depth", 100)),
            "--StereoFusion.max_reproj_error", str(fusion_config.get("max_reproj_error", 2.0)),
            "--StereoFusion.max_depth_error", str(fusion_config.get("max_depth_error", 0.01)),
            "--StereoFusion.max_normal_error", str(fusion_config.get("max_normal_error", 10.0)),
            "--StereoFusion.check_num_images", str(fusion_config.get("check_num_images", 50)),
            "--StereoFusion.cache_size", str(fusion_config.get("cache_size", 32)),
        ])
        
        return self._run_command(cmd, "stereo_fusion")
    
    def poisson_meshing(self) -> bool:
        """
        Run COLMAP Poisson mesh reconstruction.
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP Poisson meshing...")
        
        mesh_config = self.config.get("meshing", {})
        poisson_config = mesh_config.get("poisson", {})
        
        input_path = self.dense_path / "fused.ply"
        output_path = self.workspace / "mesh" / "poisson_mesh.ply"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            logger.error(f"Dense point cloud not found at {input_path}")
            return False
        
        cmd = [
            "colmap", "poisson_mesher",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
        ]
        
        # Poisson options
        cmd.extend([
            "--PoissonMeshing.depth", str(poisson_config.get("depth", 11)),
            "--PoissonMeshing.trim", str(poisson_config.get("trim", 7)),
            "--PoissonMeshing.color", str(poisson_config.get("color", 16)),
        ])
        
        return self._run_command(cmd, "poisson_meshing")
    
    def delaunay_meshing(self) -> bool:
        """
        Run COLMAP Delaunay mesh reconstruction.
        
        Returns:
            Success status
        """
        logger.info("Running COLMAP Delaunay meshing...")
        
        mesh_config = self.config.get("meshing", {})
        delaunay_config = mesh_config.get("delaunay", {})
        
        input_path = self.workspace / "sparse" / "0"
        output_path = self.workspace / "mesh" / "delaunay_mesh.ply"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "colmap", "delaunay_mesher",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
        ]
        
        # Delaunay options
        cmd.extend([
            "--DelaunayMeshing.max_proj_dist", str(delaunay_config.get("max_proj_dist", 20.0)),
            "--DelaunayMeshing.max_depth_dist", str(delaunay_config.get("max_depth_dist", 20.0)),
            "--DelaunayMeshing.visibility_sigma", str(delaunay_config.get("visibility_sigma", 2.0)),
            "--DelaunayMeshing.distance_sigma_factor", str(delaunay_config.get("distance_sigma_factor", 1.0)),
            "--DelaunayMeshing.quality_regularization", str(delaunay_config.get("quality_regularization", 1.0)),
        ])
        
        return self._run_command(cmd, "delaunay_meshing")
    
    def model_converter(self, input_path: Path, output_path: Path, output_type: str = "TXT") -> bool:
        """
        Convert COLMAP model between BIN and TXT formats.
        
        Args:
            input_path: Path to input model
            output_path: Path to output model
            output_type: Output type ("BIN" or "TXT")
            
        Returns:
            Success status
        """
        logger.info(f"Converting model to {output_type} format...")
        
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", output_type,
        ]
        
        return self._run_command(cmd, "model_converter")
    
    def _run_command(self, cmd: List[str], stage_name: str) -> bool:
        """
        Run COLMAP command with logging and error handling.
        
        Args:
            cmd: Command to run
            stage_name: Name of the stage (for logging)
            
        Returns:
            Success status
        """
        log_file = self.log_dir / f"{stage_name}.log"
        
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"Logging to: {log_file}")
        
        start_time = time.time()
        
        try:
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream output to both log file and console
                for line in process.stdout:
                    f.write(line)
                    f.flush()
                    
                    # Log important lines to console
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in ["error", "warning", "progress", "registered"]):
                        logger.debug(line.strip())
                
                process.wait()
                
                if process.returncode != 0:
                    logger.error(f"{stage_name} failed with return code {process.returncode}")
                    logger.error(f"Check log file: {log_file}")
                    return False
            
            duration = time.time() - start_time
            logger.info(f"{stage_name} completed successfully in {duration:.2f}s")
            return True
            
        except FileNotFoundError:
            logger.error("COLMAP not found. Please install COLMAP and add to PATH")
            logger.error("Installation: https://colmap.github.io/install.html")
            return False
        except Exception as e:
            logger.error(f"{stage_name} failed: {e}")
            return False

