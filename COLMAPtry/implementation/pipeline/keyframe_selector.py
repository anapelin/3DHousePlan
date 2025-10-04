"""
Keyframe selection based on image difference and blur detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging
import shutil
from tqdm import tqdm

logger = logging.getLogger("colmap_pipeline")


class KeyframeSelector:
    """Select keyframes from extracted frames."""
    
    def __init__(self, config: dict):
        """
        Initialize keyframe selector.
        
        Args:
            config: Keyframe selection configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.method = config.get("method", "difference")
        self.difference_threshold = config.get("difference_threshold", 25.0)
        self.blur_threshold = config.get("blur_threshold", 100.0)
        self.min_interval = config.get("min_keyframe_interval", 5)
    
    def select_keyframes(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> int:
        """
        Select keyframes from input directory.
        
        Args:
            input_dir: Directory containing all frames
            output_dir: Directory to save selected keyframes
            
        Returns:
            Number of keyframes selected
        """
        if not self.enabled:
            logger.info("Keyframe selection disabled, using all frames")
            return self._copy_all_frames(input_dir, output_dir)
        
        logger.info(f"Selecting keyframes using method: {self.method}")
        
        # Get list of frames
        frames = sorted(input_dir.glob("frame_*.jpg"))
        if not frames:
            logger.warning("No frames found in input directory")
            return 0
        
        # Select keyframes based on method
        if self.method == "difference":
            keyframe_indices = self._select_by_difference(frames)
        elif self.method == "blur":
            keyframe_indices = self._select_by_blur(frames)
        elif self.method == "both":
            keyframe_indices = self._select_by_both(frames)
        else:
            logger.warning(f"Unknown method: {self.method}, using all frames")
            keyframe_indices = list(range(len(frames)))
        
        # Copy selected keyframes
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, frame_idx in enumerate(tqdm(keyframe_indices, desc="Copying keyframes")):
            src = frames[frame_idx]
            dst = output_dir / f"keyframe_{i:06d}.jpg"
            shutil.copy2(src, dst)
        
        logger.info(f"Selected {len(keyframe_indices)} keyframes from {len(frames)} frames")
        return len(keyframe_indices)
    
    def _copy_all_frames(self, input_dir: Path, output_dir: Path) -> int:
        """Copy all frames without selection."""
        frames = sorted(input_dir.glob("frame_*.jpg"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for frame in tqdm(frames, desc="Copying all frames"):
            shutil.copy2(frame, output_dir / frame.name)
        
        return len(frames)
    
    def _select_by_difference(self, frames: List[Path]) -> List[int]:
        """
        Select keyframes based on image difference.
        
        Args:
            frames: List of frame paths
            
        Returns:
            List of selected frame indices
        """
        keyframe_indices = [0]  # Always include first frame
        prev_frame = None
        
        for i in tqdm(range(len(frames)), desc="Computing differences"):
            # Check minimum interval
            if i - keyframe_indices[-1] < self.min_interval:
                continue
            
            # Load current frame
            frame = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
            
            if prev_frame is None:
                prev_frame = frame
                continue
            
            # Compute difference
            diff = self._compute_frame_difference(prev_frame, frame)
            
            # Select if difference exceeds threshold
            if diff > self.difference_threshold:
                keyframe_indices.append(i)
                prev_frame = frame
        
        return keyframe_indices
    
    def _select_by_blur(self, frames: List[Path]) -> List[int]:
        """
        Select keyframes based on blur detection (sharpness).
        
        Args:
            frames: List of frame paths
            
        Returns:
            List of selected frame indices
        """
        # Compute blur scores for all frames
        blur_scores = []
        
        for frame_path in tqdm(frames, desc="Computing blur scores"):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            score = self._compute_blur_score(frame)
            blur_scores.append(score)
        
        # Select frames above threshold
        keyframe_indices = []
        
        for i, score in enumerate(blur_scores):
            # Check minimum interval
            if keyframe_indices and i - keyframe_indices[-1] < self.min_interval:
                continue
            
            # Select if sharp enough
            if score > self.blur_threshold:
                keyframe_indices.append(i)
        
        # Ensure we have at least one keyframe
        if not keyframe_indices:
            # Select frame with highest blur score
            keyframe_indices = [int(np.argmax(blur_scores))]
        
        return keyframe_indices
    
    def _select_by_both(self, frames: List[Path]) -> List[int]:
        """
        Select keyframes based on both difference and blur.
        
        Args:
            frames: List of frame paths
            
        Returns:
            List of selected frame indices
        """
        keyframe_indices = [0]  # Always include first frame
        prev_frame = None
        
        for i in tqdm(range(len(frames)), desc="Analyzing frames"):
            # Check minimum interval
            if i - keyframe_indices[-1] < self.min_interval:
                continue
            
            # Load current frame
            frame = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
            
            # Compute blur score
            blur_score = self._compute_blur_score(frame)
            
            # Skip if too blurry
            if blur_score < self.blur_threshold:
                continue
            
            if prev_frame is None:
                prev_frame = frame
                continue
            
            # Compute difference
            diff = self._compute_frame_difference(prev_frame, frame)
            
            # Select if different enough and sharp enough
            if diff > self.difference_threshold:
                keyframe_indices.append(i)
                prev_frame = frame
        
        return keyframe_indices
    
    def _compute_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute difference between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Difference score (mean absolute difference)
        """
        # Ensure same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Compute mean absolute difference
        diff = np.abs(frame1.astype(float) - frame2.astype(float))
        return np.mean(diff)
    
    def _compute_blur_score(self, frame: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        Higher score = sharper image.
        
        Args:
            frame: Input frame (grayscale)
            
        Returns:
            Blur score (variance of Laplacian)
        """
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        return laplacian.var()
    
    def analyze_frames(self, frames_dir: Path) -> dict:
        """
        Analyze frames and return statistics.
        
        Args:
            frames_dir: Directory containing frames
            
        Returns:
            Dictionary with analysis results
        """
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        
        if not frames:
            return {"error": "No frames found"}
        
        blur_scores = []
        differences = []
        prev_frame = None
        
        logger.info("Analyzing frames...")
        
        for frame_path in tqdm(frames, desc="Analyzing"):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            
            # Compute blur score
            blur_score = self._compute_blur_score(frame)
            blur_scores.append(blur_score)
            
            # Compute difference from previous
            if prev_frame is not None:
                diff = self._compute_frame_difference(prev_frame, frame)
                differences.append(diff)
            
            prev_frame = frame
        
        return {
            "num_frames": len(frames),
            "blur_scores": {
                "mean": float(np.mean(blur_scores)),
                "std": float(np.std(blur_scores)),
                "min": float(np.min(blur_scores)),
                "max": float(np.max(blur_scores)),
            },
            "differences": {
                "mean": float(np.mean(differences)) if differences else 0.0,
                "std": float(np.std(differences)) if differences else 0.0,
                "min": float(np.min(differences)) if differences else 0.0,
                "max": float(np.max(differences)) if differences else 0.0,
            },
        }

