"""
Video Frame Splitter - Helper function to split videos into frames at 50 FPS

This script provides a simple helper function to extract frames from videos
at 50 frames per second using both ffmpeg and OpenCV methods.
"""

import cv2
import subprocess
from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def split_video_to_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    fps: int = 50,
    max_frames: Optional[int] = None,
    quality: int = 95,
    resize_max_dimension: Optional[int] = None,
    method: str = "ffmpeg"
) -> int:
    """
    Split a video into individual frames at specified FPS.
    
    This helper function extracts frames from a video file at 50 FPS (default)
    and saves them as JPEG images. Supports both ffmpeg (faster) and OpenCV 
    (more portable) extraction methods.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory where frames will be saved
        fps: Frames per second to extract (default: 50)
        max_frames: Maximum number of frames to extract (default: None = unlimited)
        quality: JPEG quality for saved frames, 0-100 (default: 95)
        resize_max_dimension: Maximum dimension (width/height) for resizing (default: None)
        method: Extraction method - "ffmpeg" or "opencv" (default: "ffmpeg")
    
    Returns:
        Number of frames successfully extracted
    
    Examples:
        >>> # Basic usage - extract at 50 FPS
        >>> num_frames = split_video_to_frames("input.mp4", "output_frames/")
        >>> print(f"Extracted {num_frames} frames")
        
        >>> # Extract first 500 frames at 50 FPS with resizing
        >>> num_frames = split_video_to_frames(
        ...     "input.mp4", 
        ...     "output_frames/",
        ...     max_frames=500,
        ...     resize_max_dimension=1920
        ... )
        
        >>> # Use OpenCV method instead of ffmpeg
        >>> num_frames = split_video_to_frames(
        ...     "input.mp4",
        ...     "output_frames/",
        ...     method="opencv"
        ... )
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    # Validate inputs
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not video_path.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose extraction method
    if method.lower() == "ffmpeg":
        return _extract_with_ffmpeg(
            video_path, output_dir, fps, max_frames, quality, resize_max_dimension
        )
    elif method.lower() == "opencv":
        return _extract_with_opencv(
            video_path, output_dir, fps, max_frames, quality, resize_max_dimension
        )
    else:
        raise ValueError(f"Unknown extraction method: {method}. Use 'ffmpeg' or 'opencv'.")


def _extract_with_ffmpeg(
    video_path: Path,
    output_dir: Path,
    fps: int,
    max_frames: Optional[int],
    quality: int,
    resize_max_dimension: Optional[int]
) -> int:
    """
    Extract frames using ffmpeg (faster, hardware accelerated).
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Target frames per second
        max_frames: Maximum number of frames to extract
        quality: JPEG quality (0-100)
        resize_max_dimension: Maximum dimension for resizing
    
    Returns:
        Number of frames extracted
    """
    print(f"Extracting frames from {video_path.name} using ffmpeg at {fps} FPS...")
    
    # Build ffmpeg command
    cmd = ["ffmpeg", "-i", str(video_path)]
    
    # Build video filter string
    vf_parts = [f"fps={fps}"]
    
    # Add resize filter if specified
    if resize_max_dimension:
        scale_filter = (
            f"scale='if(gt(iw,ih),{resize_max_dimension},-2)':"
            f"'if(gt(iw,ih),-2,{resize_max_dimension})'"
        )
        vf_parts.append(scale_filter)
    
    cmd.extend(["-vf", ",".join(vf_parts)])
    
    # Set quality (ffmpeg q:v scale: 2=high quality, 31=low quality)
    qscale = 2 + int((100 - quality) * 29 / 100)
    cmd.extend(["-q:v", str(qscale)])
    
    # Add max frames limit if specified
    if max_frames:
        cmd.extend(["-frames:v", str(max_frames)])
    
    # Set output pattern
    cmd.append(str(output_dir / "frame_%06d.jpg"))
    
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
        print(f"✓ Successfully extracted {num_frames} frames to {output_dir}")
        
        return num_frames
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"✗ ffmpeg extraction failed: {error_msg}")
        print("Falling back to OpenCV extraction method...")
        
        return _extract_with_opencv(
            video_path, output_dir, fps, max_frames, quality, resize_max_dimension
        )
    
    except FileNotFoundError:
        print("✗ ffmpeg not found in system PATH")
        print("Falling back to OpenCV extraction method...")
        
        return _extract_with_opencv(
            video_path, output_dir, fps, max_frames, quality, resize_max_dimension
        )


def _extract_with_opencv(
    video_path: Path,
    output_dir: Path,
    fps: int,
    max_frames: Optional[int],
    quality: int,
    resize_max_dimension: Optional[int]
) -> int:
    """
    Extract frames using OpenCV (slower but more portable).
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Target frames per second
        max_frames: Maximum number of frames to extract
        quality: JPEG quality (0-100)
        resize_max_dimension: Maximum dimension for resizing
    
    Returns:
        Number of frames extracted
    """
    print(f"Extracting frames from {video_path.name} using OpenCV at {fps} FPS...")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    print(f"Video info: {width}x{height}, {video_fps:.2f} FPS, "
          f"{total_frames} frames, {duration:.2f}s duration")
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    
    # Calculate expected number of frames
    expected_frames = total_frames // frame_interval
    if max_frames:
        expected_frames = min(expected_frames, max_frames)
    
    print(f"Extracting every {frame_interval} frame(s) "
          f"(~{expected_frames} frames expected)...")
    
    # Extract frames
    frame_count = 0
    extracted_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at interval
            if frame_count % frame_interval == 0:
                # Resize if needed
                if resize_max_dimension:
                    frame = _resize_frame(frame, resize_max_dimension)
                
                # Save frame
                output_path = output_dir / f"frame_{extracted_count + 1:06d}.jpg"
                cv2.imwrite(
                    str(output_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                
                extracted_count += 1
                
                # Print progress
                if extracted_count % 50 == 0:
                    print(f"  Extracted {extracted_count} frames...")
                
                # Check max frames limit
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
    
    finally:
        cap.release()
    
    print(f"✓ Successfully extracted {extracted_count} frames to {output_dir}")
    return extracted_count


def _resize_frame(frame, max_dimension: int):
    """
    Resize frame to fit within max dimension while preserving aspect ratio.
    
    Args:
        frame: Input frame (numpy array)
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
    
    # Resize using high-quality interpolation
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary containing video information:
        - fps: Frames per second
        - total_frames: Total number of frames
        - width: Video width in pixels
        - height: Video height in pixels
        - duration: Duration in seconds
    
    Example:
        >>> info = get_video_info("video.mp4")
        >>> print(f"Video is {info['duration']:.2f}s at {info['fps']} FPS")
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    info = {
        "fps": fps,
        "total_frames": total_frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": total_frames / fps if fps > 0 else 0,
    }
    
    cap.release()
    return info


# Main execution example
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Video Frame Splitter - Extract frames at 50 FPS")
    print("=" * 70)
    
    # Example usage
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python video_frame_splitter.py <video_path> [output_dir] [fps]")
        print("\nExamples:")
        print("  python video_frame_splitter.py video.mp4")
        print("  python video_frame_splitter.py video.mp4 output_frames/")
        print("  python video_frame_splitter.py video.mp4 output_frames/ 50")
        print("\nDefault output directory: ./frames/")
        print("Default FPS: ")
        sys.exit(0)
    
    # Parse arguments
    video_file = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "frames"
    target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    
    try:
        # Get video info
        print("\nAnalyzing video...")
        info = get_video_info(video_file)
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  Original FPS: {info['fps']:.2f}")
        print(f"  Total frames: {info['total_frames']}")
        print(f"  Duration: {info['duration']:.2f}s")
        print()
        
        # Extract frames
        num_frames = split_video_to_frames(
            video_path=video_file,
            output_dir=output_folder,
            fps=target_fps
        )
        
        print()
        print("=" * 70)
        print(f"✓ Extraction complete! {num_frames} frames saved to '{output_folder}'")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

