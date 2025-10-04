"""
Frame extraction from video files using ffmpeg and OpenCV
"""

import cv2
import subprocess
from pathlib import Path
from typing import Optional, List
import logging
from tqdm import tqdm

logger = logging.getLogger("colmap_pipeline")


class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(self, config: dict):
        """
        Initialize frame extractor.
        
        Args:
            config: Frame extraction configuration
        """
        self.config = config
        self.fps = config.get("fps", 2)
        self.max_frames = config.get("max_frames", 300)
        self.quality = config.get("quality", 95)
        self.resize_max_dimension = config.get("resize_max_dimension", None)
    
    def extract_frames_ffmpeg(
        self,
        video_path: Path,
        output_dir: Path,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> int:
        """
        Extract frames using ffmpeg (faster, hardware accelerated).
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            
        Returns:
            Number of frames extracted
        """
        logger.info(f"Extracting frames from video using ffmpeg: {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command
        cmd = ["ffmpeg", "-i", str(video_path)]
        
        # Add time range if specified
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        if end_time is not None:
            cmd.extend(["-to", str(end_time)])
        
        # Set frame rate
        cmd.extend(["-vf", f"fps={self.fps}"])
        
        # Add resize filter if specified
        if self.resize_max_dimension:
            scale_filter = f"scale='if(gt(iw,ih),{self.resize_max_dimension},-2)':'if(gt(iw,ih),-2,{self.resize_max_dimension})'"
            cmd[-1] = f"{cmd[-1]},{scale_filter}"
        
        # Set quality and output
        cmd.extend([
            "-q:v", str(100 - self.quality),  # Lower number = higher quality
            "-frames:v", str(self.max_frames),
            str(output_dir / "frame_%06d.jpg")
        ])
        
        # Run ffmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Count extracted frames
            num_frames = len(list(output_dir.glob("frame_*.jpg")))
            logger.info(f"Extracted {num_frames} frames to {output_dir}")
            
            return num_frames
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg extraction failed: {e.stderr}")
            # Fall back to OpenCV method
            logger.info("Falling back to OpenCV extraction method")
            return self.extract_frames_opencv(video_path, output_dir)
    
    def extract_frames_opencv(
        self,
        video_path: Path,
        output_dir: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> int:
        """
        Extract frames using OpenCV (slower but more portable).
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            start_frame: Starting frame number
            end_frame: Optional ending frame number
            
        Returns:
            Number of frames extracted
        """
        logger.info(f"Extracting frames from video using OpenCV: {video_path}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        logger.info(f"Video: {total_frames} frames at {video_fps:.2f} FPS ({duration:.2f}s)")
        
        # Calculate frame interval
        frame_interval = int(video_fps / self.fps)
        if frame_interval < 1:
            frame_interval = 1
        
        # Extract frames
        frame_count = 0
        extracted_count = 0
        
        pbar = tqdm(total=min(self.max_frames, total_frames // frame_interval), 
                   desc="Extracting frames")
        
        while extracted_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip to start frame
            if frame_count < start_frame:
                frame_count += 1
                continue
            
            # Check end frame
            if end_frame and frame_count >= end_frame:
                break
            
            # Extract frame at interval
            if frame_count % frame_interval == 0:
                # Resize if needed
                if self.resize_max_dimension:
                    frame = self._resize_frame(frame, self.resize_max_dimension)
                
                # Save frame
                output_path = output_dir / f"frame_{extracted_count:06d}.jpg"
                cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                
                extracted_count += 1
                pbar.update(1)
            
            frame_count += 1
        
        pbar.close()
        cap.release()
        
        logger.info(f"Extracted {extracted_count} frames to {output_dir}")
        return extracted_count
    
    def _resize_frame(self, frame, max_dimension: int):
        """
        Resize frame to fit within max dimension while preserving aspect ratio.
        
        Args:
            frame: Input frame
            max_dimension: Maximum width or height
            
        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        
        if max(h, w) <= max_dimension:
            return frame
        
        # Calculate new dimensions
        if w > h:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
        else:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        
        # Resize
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def get_video_info(self, video_path: Path) -> dict:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        
        cap.release()
        return info
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        method: str = "ffmpeg"
    ) -> int:
        """
        Extract frames using specified method.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            method: Extraction method ("ffmpeg" or "opencv")
            
        Returns:
            Number of frames extracted
        """
        if method == "ffmpeg":
            return self.extract_frames_ffmpeg(video_path, output_dir)
        elif method == "opencv":
            return self.extract_frames_opencv(video_path, output_dir)
        else:
            raise ValueError(f"Unknown extraction method: {method}")

