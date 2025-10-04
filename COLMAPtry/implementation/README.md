# COLMAP SfM/MVS Pipeline

A reproducible, production-ready pipeline for 3D reconstruction from images or video using COLMAP (Structure-from-Motion and Multi-View Stereo).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎥 **Video Input**: Extract frames and reconstruct from video files
- 📸 **Image Input**: Reconstruct from image collections
- 📍 **Known Poses**: Support for ARKit/ARCore camera poses
- ⚡ **GPU Acceleration**: CUDA-accelerated dense reconstruction
- 🔄 **Resumable**: Checkpoint system to resume interrupted reconstructions
- 📊 **Detailed Reports**: HTML reports with statistics and visualizations
- 🐳 **Docker Support**: Fully containerized for reproducibility
- 🎨 **Blender-Ready**: Direct OBJ export for visualization

## Installation

### Prerequisites

1. **COLMAP** (required)
   ```bash
   # Ubuntu/Debian
   sudo apt install colmap
   
   # macOS
   brew install colmap
   
   # Windows
   # Download from https://colmap.github.io/install.html
   ```

2. **ffmpeg** (optional, for video processing)
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

3. **Python 3.8+**

### Install Pipeline

```bash
# Clone repository
git clone <repository-url>
cd implementation

# Install
pip install -e .

# Or use Make
make install
```

### Docker Installation

```bash
# Build Docker image
make docker-build

# Or manually
docker build -t colmap-pipeline:latest .
```

## Quick Start

### Reconstruct from Video

```bash
colmap-reconstruct from-video input.mp4 -o output/
```

### Reconstruct from Images

```bash
colmap-reconstruct from-images images/ -o output/
```

### Reconstruct with Known Poses

```bash
colmap-reconstruct from-images-with-poses images/ poses.json -o output/
```

### View Results in Blender

```bash
blender --python-expr "import bpy; bpy.ops.import_scene.obj(filepath='output/poisson_mesh.obj')"
```

## Usage

### Command-Line Interface

```
colmap-reconstruct <command> [options]

Commands:
  from-video                  Reconstruct from video file
  from-images                 Reconstruct from image directory
  from-images-with-poses      Reconstruct with known camera poses

Options:
  -o, --output DIR           Output directory (required)
  -c, --config FILE          Configuration file
  --preset {fast,high}       Quality preset
  --no-dense                 Skip dense reconstruction
  --no-mesh                  Skip mesh generation
  --cpu-only                 Force CPU-only processing
  --no-resume                Start fresh (ignore checkpoint)
  --log-level LEVEL          Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Examples

#### Basic Video Reconstruction
```bash
colmap-reconstruct from-video video.mp4 -o output/
```

#### High-Quality Image Reconstruction
```bash
colmap-reconstruct from-images images/ -o output/ --preset high
```

#### Fast Sparse Reconstruction Only
```bash
colmap-reconstruct from-images images/ -o output/ --preset fast --no-dense
```

#### Reconstruction with Custom Config
```bash
colmap-reconstruct from-video video.mp4 -o output/ -c custom_config.yaml
```

#### Resume Interrupted Reconstruction
```bash
# Pipeline automatically resumes from checkpoint
colmap-reconstruct from-video video.mp4 -o output/
```

### Configuration

Configuration is managed through YAML files. See `configs/default.yaml` for all options.

#### Key Configuration Sections

**Frame Extraction (Video Input)**
```yaml
frame_extraction:
  fps: 2                      # Extract 2 frames per second
  max_frames: 300            # Maximum frames to extract
  quality: 95                 # JPEG quality (1-100)
```

**Keyframe Selection**
```yaml
keyframe_selection:
  enabled: true
  method: "difference"        # "difference", "blur", or "both"
  difference_threshold: 25.0
  blur_threshold: 100.0
```

**Feature Extraction**
```yaml
feature_extraction:
  camera_model: "OPENCV"      # Camera model type
  sift:
    max_num_features: 8192   # Number of SIFT features
```

**Dense Reconstruction**
```yaml
dense_reconstruction:
  enabled: true
  window_radius: 5
  num_iterations: 5
  geom_consistency: true
```

**Quality Presets**
```yaml
presets:
  fast:
    feature_extraction:
      sift:
        max_num_features: 4096
    dense_reconstruction:
      num_iterations: 3
  
  high:
    feature_extraction:
      sift:
        max_num_features: 16384
    dense_reconstruction:
      num_iterations: 7
```

## Pipeline Stages

1. **Frame Extraction** (video only)
   - Extract frames at specified FPS
   - Optional keyframe selection based on blur/difference

2. **Feature Extraction**
   - SIFT feature detection in all images
   - Camera calibration (auto or manual)

3. **Feature Matching**
   - Match features between image pairs
   - Methods: exhaustive, sequential, spatial

4. **Sparse Reconstruction (SfM)**
   - Incremental structure-from-motion
   - Camera pose estimation
   - Sparse 3D point cloud

5. **Image Undistortion**
   - Prepare images for dense reconstruction

6. **Dense Reconstruction (MVS)**
   - Patch match stereo
   - Dense depth map computation
   - Stereo fusion

7. **Meshing**
   - Poisson surface reconstruction
   - Triangle mesh generation

8. **Export**
   - PLY point clouds
   - OBJ meshes (Blender-compatible)
   - HTML report

## Output Structure

```
output/
├── images/                    # Processed input images
├── database.db               # COLMAP database
├── sparse/                   # Sparse reconstruction
│   └── 0/                    # Sparse model
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── dense/                    # Dense reconstruction
│   ├── fused.ply            # Dense point cloud
│   └── stereo/              # Depth maps
├── mesh/                     # Generated meshes
│   └── poisson_mesh.ply
├── report/                   # HTML report
│   └── report.html
├── logs/                     # Stage logs
├── dense_point_cloud.ply     # Exported dense cloud
├── poisson_mesh.obj          # Exported mesh (Blender-ready)
└── checkpoint.json           # Resume checkpoint
```

## Known Poses Format

### JSON Format
```json
{
  "images": [
    {
      "image_name": "frame_000001.jpg",
      "camera_intrinsics": {
        "fx": 1000.0,
        "fy": 1000.0,
        "cx": 960.0,
        "cy": 540.0,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0
      },
      "camera_pose": {
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0]
      }
    }
  ]
}
```

### CSV Format
```csv
image_name,fx,fy,cx,cy,qw,qx,qy,qz,tx,ty,tz
frame_000001.jpg,1000.0,1000.0,960.0,540.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0
```

## Docker Usage

### Build Image
```bash
make docker-build
```

### Run Reconstruction
```bash
docker run --gpus all \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  colmap-pipeline:latest \
  from-video /data/input/video.mp4 -o /data/output
```

### Interactive Shell
```bash
docker run --gpus all -it \
  -v $(pwd)/data:/data \
  colmap-pipeline:latest \
  /bin/bash
```

## Troubleshooting

### COLMAP Not Found
```
Error: COLMAP not found. Please install COLMAP and add to PATH
```
**Solution**: Install COLMAP following instructions at https://colmap.github.io/install.html

### CUDA Not Available
```
Warning: CUDA not available (CPU-only mode)
```
**Solution**: Use `--cpu-only` flag or install CUDA drivers. CPU mode works but is slower.

### Out of Memory (GPU)
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce image resolution in config: `undistortion.max_image_size: 1600`
- Reduce PatchMatch cache: `dense_reconstruction.cache_size: 16`
- Use `--cpu-only` flag

### Too Few Images Registered
```
Error: Sparse reconstruction failed
```
**Solutions**:
- Ensure sufficient overlap between images
- Lower `feature_matching.max_ratio` threshold
- Increase `feature_extraction.sift.max_num_features`
- Check image quality (not too blurry)

### Reconstruction Takes Too Long
**Solutions**:
- Use `--preset fast`
- Use `--no-dense` for sparse only
- Reduce number of input images
- Use `sequential` matching instead of `exhaustive`

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is available for 5-10x speedup
2. **Image Count**: 50-300 images is optimal for most scenes
3. **Overlap**: 70-80% overlap between adjacent images works best
4. **Resolution**: 1920x1080 to 2560x1440 is a good balance
5. **Frame Rate**: Extract 2-5 FPS from video for good coverage
6. **Matching**: Use `sequential` for video, `exhaustive` for unordered images

## Development

### Running Tests
```bash
make test
```

### Code Formatting
```bash
make format
```

### Linting
```bash
make lint
```

## Citation

If you use this pipeline in your research, please cite COLMAP:

```bibtex
@inproceedings{schoenberger2016sfm,
  author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
  title={Structure-from-Motion Revisited},
  booktitle={CVPR},
  year={2016},
}

@inproceedings{schoenberger2016mvs,
  author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Frahm, Jan-Michael and Pollefeys, Marc},
  title={Pixelwise View Selection for Unstructured Multi-View Stereo},
  booktitle={ECCV},
  year={2016},
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues and questions:
- Open an issue on GitHub
- Check troubleshooting section above
- Consult COLMAP documentation: https://colmap.github.io/

## Acknowledgments

- COLMAP authors for the excellent reconstruction software
- All contributors to this pipeline project

