"""
Core COLMAP pipeline orchestrator
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import numpy as np

from .colmap_wrapper import COLMAPWrapper
from .frame_extractor import FrameExtractor
from .keyframe_selector import KeyframeSelector
from .known_poses import KnownPosesHandler
from .reporting import ReportGenerator
from .utils import (
    create_directory_structure,
    save_checkpoint,
    load_checkpoint,
    count_images,
    get_file_size
)

logger = logging.getLogger("colmap_pipeline")


class COLMAPPipeline:
    """Main COLMAP reconstruction pipeline orchestrator."""
    
    def __init__(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        resume: bool = True
    ):
        """
        Initialize COLMAP pipeline.
        
        Args:
            output_dir: Output directory for reconstruction
            config: Pipeline configuration
            resume: Whether to resume from checkpoint
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.resume = resume
        
        # Create directory structure
        self.paths = create_directory_structure(self.output_dir)
        
        # Initialize components
        self.colmap = COLMAPWrapper(
            self.output_dir,
            config,
            use_gpu=config.get("general", {}).get("use_gpu", True)
        )
        
        self.frame_extractor = FrameExtractor(config.get("frame_extraction", {}))
        self.keyframe_selector = KeyframeSelector(config.get("keyframe_selection", {}))
        self.poses_handler = KnownPosesHandler(self.output_dir)
        self.report_generator = ReportGenerator(self.output_dir)
        
        # Track timings
        self.timings: Dict[str, float] = {}
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        
        # Load checkpoint if resuming
        self.checkpoint = None
        if resume:
            self.checkpoint = load_checkpoint(self.checkpoint_path)
            if self.checkpoint:
                logger.info(f"Resuming from checkpoint: {self.checkpoint['stage']}")
    
    def reconstruct_from_video(
        self,
        video_path: Path,
        dense: bool = True,
        mesh: bool = True
    ) -> bool:
        """
        Run reconstruction from video file.
        
        Args:
            video_path: Path to input video
            dense: Whether to run dense reconstruction
            mesh: Whether to generate mesh
            
        Returns:
            Success status
        """
        logger.info(f"Starting reconstruction from video: {video_path}")
        
        # Stage 1: Extract frames
        if not self._should_skip_stage("frame_extraction"):
            success = self._extract_frames(video_path)
            if not success:
                return False
            self._save_checkpoint("frame_extraction")
        
        # Stage 2: Select keyframes
        if not self._should_skip_stage("keyframe_selection"):
            success = self._select_keyframes()
            if not success:
                return False
            self._save_checkpoint("keyframe_selection")
        
        # Run sparse reconstruction
        success = self._run_sparse_reconstruction()
        if not success:
            return False
        
        # Run dense reconstruction if requested
        if dense:
            success = self._run_dense_reconstruction()
            if not success:
                logger.warning("Dense reconstruction failed, but sparse model available")
        
        # Generate mesh if requested
        if mesh and dense:
            success = self._run_meshing()
            if not success:
                logger.warning("Meshing failed, but dense point cloud available")
        
        # Export results
        self._export_results()
        
        # Generate report
        self._generate_report()
        
        logger.info("Reconstruction complete!")
        return True
    
    def reconstruct_from_images(
        self,
        images_dir: Path,
        dense: bool = True,
        mesh: bool = True
    ) -> bool:
        """
        Run reconstruction from images directory.
        
        Args:
            images_dir: Directory containing input images
            dense: Whether to run dense reconstruction
            mesh: Whether to generate mesh
            
        Returns:
            Success status
        """
        logger.info(f"Starting reconstruction from images: {images_dir}")
        
        # Copy or link images to workspace
        if not self._should_skip_stage("image_import"):
            success = self._import_images(images_dir)
            if not success:
                return False
            self._save_checkpoint("image_import")
        
        # Run sparse reconstruction
        success = self._run_sparse_reconstruction()
        if not success:
            return False
        
        # Run dense reconstruction if requested
        if dense:
            success = self._run_dense_reconstruction()
            if not success:
                logger.warning("Dense reconstruction failed, but sparse model available")
        
        # Generate mesh if requested
        if mesh and dense:
            success = self._run_meshing()
            if not success:
                logger.warning("Meshing failed, but dense point cloud available")
        
        # Export results
        self._export_results()
        
        # Generate report
        self._generate_report()
        
        logger.info("Reconstruction complete!")
        return True
    
    def reconstruct_from_images_with_poses(
        self,
        images_dir: Path,
        poses_file: Path,
        dense: bool = True,
        mesh: bool = True
    ) -> bool:
        """
        Run reconstruction from images with known poses.
        
        Args:
            images_dir: Directory containing input images
            poses_file: File with known camera poses (JSON or CSV)
            dense: Whether to run dense reconstruction
            mesh: Whether to generate mesh
            
        Returns:
            Success status
        """
        logger.info(f"Starting reconstruction with known poses: {poses_file}")
        
        # Copy or link images to workspace
        if not self._should_skip_stage("image_import"):
            success = self._import_images(images_dir)
            if not success:
                return False
            self._save_checkpoint("image_import")
        
        # Import poses
        if not self._should_skip_stage("pose_import"):
            success = self._import_poses(poses_file)
            if not success:
                return False
            self._save_checkpoint("pose_import")
        
        # Triangulate points from known poses
        if not self._should_skip_stage("triangulation"):
            success = self._triangulate_points()
            if not success:
                return False
            self._save_checkpoint("triangulation")
        
        # Run dense reconstruction if requested
        if dense:
            success = self._run_dense_reconstruction()
            if not success:
                logger.warning("Dense reconstruction failed, but sparse model available")
        
        # Generate mesh if requested
        if mesh and dense:
            success = self._run_meshing()
            if not success:
                logger.warning("Meshing failed, but dense point cloud available")
        
        # Export results
        self._export_results()
        
        # Generate report
        self._generate_report()
        
        logger.info("Reconstruction complete!")
        return True
    
    def _extract_frames(self, video_path: Path) -> bool:
        """Extract frames from video."""
        logger.info("Stage: Frame extraction")
        start_time = time.time()
        
        temp_frames_dir = self.output_dir / "temp_frames"
        temp_frames_dir.mkdir(exist_ok=True)
        
        try:
            num_frames = self.frame_extractor.extract_frames(
                video_path,
                temp_frames_dir,
                method="ffmpeg"
            )
            
            if num_frames == 0:
                logger.error("No frames extracted")
                return False
            
            self.timings["frame_extraction"] = time.time() - start_time
            return True
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return False
    
    def _select_keyframes(self) -> bool:
        """Select keyframes from extracted frames."""
        logger.info("Stage: Keyframe selection")
        start_time = time.time()
        
        temp_frames_dir = self.output_dir / "temp_frames"
        images_dir = self.paths["images"]
        
        try:
            num_keyframes = self.keyframe_selector.select_keyframes(
                temp_frames_dir,
                images_dir
            )
            
            if num_keyframes == 0:
                logger.error("No keyframes selected")
                return False
            
            self.timings["keyframe_selection"] = time.time() - start_time
            return True
            
        except Exception as e:
            logger.error(f"Keyframe selection failed: {e}")
            return False
    
    def _import_images(self, images_dir: Path) -> bool:
        """Import images from directory."""
        logger.info("Stage: Image import")
        start_time = time.time()
        
        import shutil
        from tqdm import tqdm
        
        try:
            images = list(images_dir.glob("*"))
            images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            
            if not images:
                logger.error(f"No images found in {images_dir}")
                return False
            
            logger.info(f"Importing {len(images)} images")
            
            for img in tqdm(images, desc="Importing images"):
                dest = self.paths["images"] / img.name
                shutil.copy2(img, dest)
            
            self.timings["image_import"] = time.time() - start_time
            return True
            
        except Exception as e:
            logger.error(f"Image import failed: {e}")
            return False
    
    def _import_poses(self, poses_file: Path) -> bool:
        """Import known camera poses."""
        logger.info("Stage: Pose import")
        start_time = time.time()
        
        try:
            if poses_file.suffix.lower() == ".json":
                success = self.poses_handler.import_poses_from_json(poses_file)
            elif poses_file.suffix.lower() == ".csv":
                success = self.poses_handler.import_poses_from_csv(poses_file)
            else:
                logger.error(f"Unsupported poses file format: {poses_file.suffix}")
                return False
            
            self.timings["pose_import"] = time.time() - start_time
            return success
            
        except Exception as e:
            logger.error(f"Pose import failed: {e}")
            return False
    
    def _triangulate_points(self) -> bool:
        """Triangulate 3D points from known poses."""
        logger.info("Stage: Point triangulation")
        start_time = time.time()
        
        try:
            success = self.poses_handler.triangulate_points()
            self.timings["triangulation"] = time.time() - start_time
            return success
            
        except Exception as e:
            logger.error(f"Triangulation failed: {e}")
            return False
    
    def _run_sparse_reconstruction(self) -> bool:
        """Run sparse reconstruction pipeline."""
        # Feature extraction
        if not self._should_skip_stage("feature_extraction"):
            logger.info("Stage: Feature extraction")
            start_time = time.time()
            
            if not self.colmap.feature_extraction():
                return False
            
            self.timings["feature_extraction"] = time.time() - start_time
            self._save_checkpoint("feature_extraction")
        
        # Feature matching
        if not self._should_skip_stage("feature_matching"):
            logger.info("Stage: Feature matching")
            start_time = time.time()
            
            if not self.colmap.feature_matching():
                return False
            
            self.timings["feature_matching"] = time.time() - start_time
            self._save_checkpoint("feature_matching")
        
        # Sparse reconstruction
        if not self._should_skip_stage("sparse_reconstruction"):
            logger.info("Stage: Sparse reconstruction")
            start_time = time.time()
            
            if not self.colmap.sparse_reconstruction():
                return False
            
            self.timings["sparse_reconstruction"] = time.time() - start_time
            self._save_checkpoint("sparse_reconstruction")
        
        return True
    
    def _run_dense_reconstruction(self) -> bool:
        """Run dense reconstruction pipeline."""
        # Image undistortion
        if not self._should_skip_stage("undistortion"):
            logger.info("Stage: Image undistortion")
            start_time = time.time()
            
            if not self.colmap.image_undistortion():
                return False
            
            self.timings["undistortion"] = time.time() - start_time
            self._save_checkpoint("undistortion")
        
        # Patch match stereo
        if not self._should_skip_stage("stereo_matching"):
            logger.info("Stage: Stereo matching")
            start_time = time.time()
            
            if not self.colmap.patch_match_stereo():
                return False
            
            self.timings["stereo_matching"] = time.time() - start_time
            self._save_checkpoint("stereo_matching")
        
        # Stereo fusion
        if not self._should_skip_stage("stereo_fusion"):
            logger.info("Stage: Stereo fusion")
            start_time = time.time()
            
            if not self.colmap.stereo_fusion():
                return False
            
            self.timings["stereo_fusion"] = time.time() - start_time
            self._save_checkpoint("stereo_fusion")
        
        return True
    
    def _run_meshing(self) -> bool:
        """Run meshing pipeline."""
        if not self._should_skip_stage("meshing"):
            logger.info("Stage: Meshing")
            start_time = time.time()
            
            mesh_config = self.config.get("meshing", {})
            method = mesh_config.get("method", "poisson")
            
            if method == "poisson":
                success = self.colmap.poisson_meshing()
            elif method == "delaunay":
                success = self.colmap.delaunay_meshing()
            else:
                logger.error(f"Unknown meshing method: {method}")
                return False
            
            if success:
                self.timings["meshing"] = time.time() - start_time
                self._save_checkpoint("meshing")
                return True
        
        return True
    
    def _export_results(self):
        """Export reconstruction results."""
        logger.info("Exporting results...")
        
        output_config = self.config.get("output", {})
        
        # Convert sparse model to TXT if requested
        if output_config.get("sparse_format") == "TXT":
            sparse_bin = self.paths["sparse_0"]
            sparse_txt = self.paths["sparse"] / "txt"
            if sparse_bin.exists():
                self.colmap.model_converter(sparse_bin, sparse_txt, "TXT")
        
        # Copy outputs to main output directory
        import shutil
        
        # Copy dense point cloud
        dense_ply = self.paths["dense"] / "fused.ply"
        if dense_ply.exists() and output_config.get("export_ply", True):
            dest = self.output_dir / "dense_point_cloud.ply"
            shutil.copy2(dense_ply, dest)
            logger.info(f"Exported: {dest}")
        
        # Copy mesh
        mesh_dir = self.paths["mesh"]
        if mesh_dir.exists():
            for mesh_file in mesh_dir.glob("*.ply"):
                if output_config.get("export_obj", True):
                    # Convert PLY to OBJ using trimesh
                    try:
                        import trimesh
                        mesh = trimesh.load(mesh_file)
                        obj_path = self.output_dir / mesh_file.with_suffix(".obj").name
                        mesh.export(obj_path)
                        logger.info(f"Exported: {obj_path}")
                    except Exception as e:
                        logger.warning(f"Failed to convert mesh to OBJ: {e}")
                        # Copy PLY as fallback
                        dest = self.output_dir / mesh_file.name
                        shutil.copy2(mesh_file, dest)
                        logger.info(f"Exported: {dest}")
    
    def _generate_report(self):
        """Generate HTML report."""
        logger.info("Generating report...")
        
        # Collect statistics
        stats = self._collect_statistics()
        
        # Generate report
        self.report_generator.generate_report(
            stats,
            self.config,
            self.timings
        )
    
    def _collect_statistics(self) -> Dict[str, Any]:
        """Collect reconstruction statistics."""
        stats = {}
        
        # Count images
        stats["num_images"] = count_images(self.paths["images"])
        
        # Read sparse model stats
        sparse_path = self.paths["sparse_0"]
        if sparse_path.exists():
            try:
                # Read points3D.bin to count points
                points_file = sparse_path / "points3D.bin"
                if points_file.exists():
                    stats["num_sparse_points"] = self._count_sparse_points(points_file)
                
                # Read images.bin to count registered images
                images_file = sparse_path / "images.bin"
                if images_file.exists():
                    stats["num_registered"] = self._count_registered_images(images_file)
            except Exception as e:
                logger.warning(f"Failed to read sparse model stats: {e}")
        
        # Read dense point cloud stats
        dense_ply = self.paths["dense"] / "fused.ply"
        if dense_ply.exists():
            try:
                from plyfile import PlyData
                plydata = PlyData.read(str(dense_ply))
                stats["num_dense_points"] = len(plydata["vertex"])
            except Exception as e:
                logger.warning(f"Failed to read dense point cloud stats: {e}")
        
        # Read mesh stats
        mesh_dir = self.paths["mesh"]
        if mesh_dir.exists():
            for mesh_file in mesh_dir.glob("*.ply"):
                try:
                    from plyfile import PlyData
                    plydata = PlyData.read(str(mesh_file))
                    stats["num_mesh_vertices"] = len(plydata["vertex"])
                    if "face" in plydata:
                        stats["num_mesh_faces"] = len(plydata["face"])
                    break
                except Exception as e:
                    logger.warning(f"Failed to read mesh stats: {e}")
        
        return stats
    
    def _count_sparse_points(self, points_file: Path) -> int:
        """Count points in sparse model."""
        # Simple binary file parsing - count 8-byte entries
        # This is a simplified version
        file_size = points_file.stat().st_size
        # Rough estimate: each point is ~40-50 bytes
        return file_size // 45
    
    def _count_registered_images(self, images_file: Path) -> int:
        """Count registered images in sparse model."""
        # Simple binary file parsing
        file_size = images_file.stat().st_size
        # Rough estimate: each image entry is ~60-80 bytes
        return file_size // 70
    
    def _should_skip_stage(self, stage: str) -> bool:
        """Check if stage should be skipped (already completed)."""
        if not self.resume or not self.checkpoint:
            return False
        
        completed_stage = self.checkpoint.get("stage")
        if not completed_stage:
            return False
        
        # Define stage order
        stage_order = [
            "frame_extraction",
            "keyframe_selection",
            "image_import",
            "pose_import",
            "triangulation",
            "feature_extraction",
            "feature_matching",
            "sparse_reconstruction",
            "undistortion",
            "stereo_matching",
            "stereo_fusion",
            "meshing",
        ]
        
        if stage not in stage_order or completed_stage not in stage_order:
            return False
        
        # Skip if current stage comes before or at completed stage
        return stage_order.index(stage) <= stage_order.index(completed_stage)
    
    def _save_checkpoint(self, stage: str):
        """Save checkpoint."""
        save_checkpoint(self.checkpoint_path, stage, {
            "timings": self.timings,
            "timestamp": time.time()
        })

