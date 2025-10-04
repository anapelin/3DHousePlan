"""
Smoke tests for COLMAP pipeline
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from pipeline.utils import (
    setup_logging,
    load_config,
    check_colmap_installation,
    create_directory_structure,
    count_images,
)
from pipeline.frame_extractor import FrameExtractor
from pipeline.keyframe_selector import KeyframeSelector


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Load default configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    return load_config(config_path)


@pytest.fixture
def sample_images(temp_dir):
    """Create sample test images."""
    images_dir = temp_dir / "images"
    images_dir.mkdir()
    
    # Create 5 test images
    for i in range(5):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = images_dir / f"test_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)
    
    return images_dir


@pytest.fixture
def sample_video(temp_dir):
    """Create sample test video."""
    video_path = temp_dir / "test_video.mp4"
    
    # Create video with 30 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    for i in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    
    writer.release()
    return video_path


class TestUtils:
    """Test utility functions."""
    
    def test_setup_logging(self, temp_dir):
        """Test logging setup."""
        log_file = temp_dir / "test.log"
        logger = setup_logging("INFO", log_file)
        
        assert logger is not None
        logger.info("Test message")
        
        assert log_file.exists()
    
    def test_load_config(self, sample_config):
        """Test configuration loading."""
        assert "general" in sample_config
        assert "feature_extraction" in sample_config
        assert "dense_reconstruction" in sample_config
    
    def test_load_config_with_preset(self):
        """Test configuration loading with preset."""
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        config = load_config(config_path, preset="fast")
        
        # Check that preset was applied
        assert config["feature_extraction"]["sift"]["max_num_features"] == 4096
    
    def test_check_colmap_installation(self):
        """Test COLMAP installation check."""
        installed, version = check_colmap_installation()
        # Note: This may fail in CI without COLMAP installed
        # That's okay - just testing the function works
        assert isinstance(installed, bool)
    
    def test_create_directory_structure(self, temp_dir):
        """Test directory structure creation."""
        paths = create_directory_structure(temp_dir / "test_workspace")
        
        assert paths["base"].exists()
        assert paths["images"].exists()
        assert paths["sparse"].exists()
        assert paths["dense"].exists()
        assert paths["logs"].exists()
    
    def test_count_images(self, sample_images):
        """Test image counting."""
        count = count_images(sample_images)
        assert count == 5


class TestFrameExtractor:
    """Test frame extraction."""
    
    def test_frame_extractor_init(self, sample_config):
        """Test frame extractor initialization."""
        extractor = FrameExtractor(sample_config["frame_extraction"])
        
        assert extractor.fps == 2
        assert extractor.max_frames == 300
        assert extractor.quality == 95
    
    def test_get_video_info(self, sample_video, sample_config):
        """Test video info extraction."""
        extractor = FrameExtractor(sample_config["frame_extraction"])
        
        info = extractor.get_video_info(sample_video)
        
        assert "fps" in info
        assert "total_frames" in info
        assert "width" in info
        assert "height" in info
        assert info["total_frames"] == 30
    
    def test_extract_frames_opencv(self, sample_video, temp_dir, sample_config):
        """Test frame extraction with OpenCV."""
        extractor = FrameExtractor(sample_config["frame_extraction"])
        
        output_dir = temp_dir / "frames"
        num_frames = extractor.extract_frames_opencv(
            sample_video,
            output_dir,
            start_frame=0
        )
        
        assert num_frames > 0
        assert output_dir.exists()
        assert len(list(output_dir.glob("frame_*.jpg"))) == num_frames


class TestKeyframeSelector:
    """Test keyframe selection."""
    
    def test_keyframe_selector_init(self, sample_config):
        """Test keyframe selector initialization."""
        selector = KeyframeSelector(sample_config["keyframe_selection"])
        
        assert selector.enabled is True
        assert selector.method == "difference"
    
    def test_compute_blur_score(self, sample_config):
        """Test blur score computation."""
        selector = KeyframeSelector(sample_config["keyframe_selection"])
        
        # Create test images
        sharp_img = np.zeros((100, 100), dtype=np.uint8)
        sharp_img[40:60, 40:60] = 255  # Sharp edges
        
        blurry_img = cv2.GaussianBlur(sharp_img, (15, 15), 5)
        
        sharp_score = selector._compute_blur_score(sharp_img)
        blurry_score = selector._compute_blur_score(blurry_img)
        
        # Sharp image should have higher score
        assert sharp_score > blurry_score
    
    def test_compute_frame_difference(self, sample_config):
        """Test frame difference computation."""
        selector = KeyframeSelector(sample_config["keyframe_selection"])
        
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame2 = np.ones((100, 100), dtype=np.uint8) * 255
        frame3 = np.zeros((100, 100), dtype=np.uint8)
        
        diff_1_2 = selector._compute_frame_difference(frame1, frame2)
        diff_1_3 = selector._compute_frame_difference(frame1, frame3)
        
        # Different frames should have higher difference
        assert diff_1_2 > diff_1_3
    
    def test_select_keyframes(self, sample_images, temp_dir, sample_config):
        """Test keyframe selection."""
        # Disable keyframe selection to copy all
        config = sample_config["keyframe_selection"].copy()
        config["enabled"] = False
        
        selector = KeyframeSelector(config)
        
        output_dir = temp_dir / "keyframes"
        num_keyframes = selector.select_keyframes(sample_images, output_dir)
        
        assert num_keyframes == 5  # Should copy all 5 images
        assert output_dir.exists()


class TestPipeline:
    """Integration tests for full pipeline."""
    
    @pytest.mark.slow
    @pytest.mark.requires_colmap
    def test_pipeline_from_images_sparse_only(self, sample_images, temp_dir, sample_config):
        """Test sparse reconstruction pipeline (requires COLMAP)."""
        from pipeline.core import COLMAPPipeline
        
        # Check if COLMAP is installed
        colmap_installed, _ = check_colmap_installation()
        if not colmap_installed:
            pytest.skip("COLMAP not installed")
        
        # Create pipeline
        output_dir = temp_dir / "output"
        pipeline = COLMAPPipeline(output_dir, sample_config, resume=False)
        
        # Note: This will likely fail with too few images
        # But we're just testing that the pipeline runs without errors
        try:
            success = pipeline.reconstruct_from_images(
                sample_images,
                dense=False,  # Skip dense for speed
                mesh=False
            )
            # May or may not succeed with random test images
            assert isinstance(success, bool)
        except Exception as e:
            # Expected to fail with random images
            # Just verify the error is reasonable
            assert "sparse reconstruction" in str(e).lower() or "not enough" in str(e).lower()
    
    def test_directory_structure_creation(self, temp_dir, sample_config):
        """Test that pipeline creates proper directory structure."""
        from pipeline.core import COLMAPPipeline
        
        output_dir = temp_dir / "output"
        pipeline = COLMAPPipeline(output_dir, sample_config, resume=False)
        
        # Check that directories were created
        assert pipeline.paths["base"].exists()
        assert pipeline.paths["images"].exists()
        assert pipeline.paths["sparse"].exists()
        assert pipeline.paths["dense"].exists()
        assert pipeline.paths["logs"].exists()
    
    def test_checkpoint_save_load(self, temp_dir, sample_config):
        """Test checkpoint saving and loading."""
        from pipeline.core import COLMAPPipeline
        from pipeline.utils import save_checkpoint, load_checkpoint
        
        output_dir = temp_dir / "output"
        checkpoint_path = output_dir / "checkpoint.json"
        output_dir.mkdir()
        
        # Save checkpoint
        save_checkpoint(checkpoint_path, "test_stage", {"test": "data"})
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)
        
        assert checkpoint is not None
        assert checkpoint["stage"] == "test_stage"
        assert checkpoint["data"]["test"] == "data"


def test_imports():
    """Test that all modules can be imported."""
    from pipeline import COLMAPPipeline, COLMAPWrapper
    from pipeline.frame_extractor import FrameExtractor
    from pipeline.keyframe_selector import KeyframeSelector
    from pipeline.known_poses import KnownPosesHandler
    from pipeline.reporting import ReportGenerator
    from pipeline.colmap_wrapper import COLMAPWrapper
    from pipeline.utils import setup_logging
    
    assert COLMAPPipeline is not None
    assert FrameExtractor is not None
    assert KeyframeSelector is not None
    assert KnownPosesHandler is not None
    assert ReportGenerator is not None
    assert COLMAPWrapper is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

