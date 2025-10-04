# COLMAP Pipeline - Quick Start Guide

Get up and running with 3D reconstruction in 5 minutes!

## 1. Prerequisites

Install COLMAP:
```bash
# Ubuntu/Debian
sudo apt install colmap

# macOS
brew install colmap

# Windows - download from https://colmap.github.io/install.html
```

## 2. Install Pipeline

```bash
cd implementation
pip install -e .
```

Test installation:
```bash
colmap-reconstruct --help
```

## 3. Prepare Your Data

### Option A: Video

Place your video file anywhere, e.g., `my_video.mp4`

### Option B: Images

Create a directory with images:
```
my_images/
├── img001.jpg
├── img002.jpg
└── ...
```

**Important**: Images should have 70-80% overlap between adjacent views!

## 4. Run Reconstruction

### From Video (Easiest)

```bash
colmap-reconstruct from-video my_video.mp4 -o output/
```

This will:
- Extract frames from video
- Select keyframes
- Run sparse reconstruction
- Run dense reconstruction
- Generate 3D mesh
- Create HTML report

### From Images

```bash
colmap-reconstruct from-images my_images/ -o output/
```

### Fast Mode (Quick Test)

```bash
colmap-reconstruct from-video my_video.mp4 -o output/ --preset fast
```

### Sparse Only (Faster)

```bash
colmap-reconstruct from-images my_images/ -o output/ --no-dense
```

## 5. View Results

### In Blender

```bash
blender
# File > Import > Wavefront (.obj)
# Navigate to: output/poisson_mesh.obj
```

Or use command line:
```bash
blender --python-expr "import bpy; bpy.ops.import_scene.obj(filepath='output/poisson_mesh.obj')"
```

### View HTML Report

Open `output/report/report.html` in your browser to see:
- Reconstruction statistics
- Sample input images
- Processing times
- Configuration used

### View Point Cloud

Use MeshLab, CloudCompare, or Blender:
```bash
# Point cloud at: output/dense_point_cloud.ply
```

## 6. Common Workflows

### High-Quality Reconstruction

```bash
colmap-reconstruct from-images images/ -o output/ --preset high
```
Takes longer but produces better results.

### Resume Interrupted Run

Just run the same command again - it resumes automatically:
```bash
colmap-reconstruct from-video my_video.mp4 -o output/
# Pipeline resumes from last checkpoint
```

### CPU-Only (No GPU)

```bash
colmap-reconstruct from-video my_video.mp4 -o output/ --cpu-only
```
Slower but works without GPU.

## 7. Typical Processing Times

On a modern GPU:
- **Small** (10-30 images): 2-5 minutes
- **Medium** (50-100 images): 10-30 minutes
- **Large** (200+ images): 1-3 hours

On CPU only: 3-10x longer

## 8. Troubleshooting

### "COLMAP not found"
- Install COLMAP: `sudo apt install colmap` (Ubuntu)
- Or download from https://colmap.github.io/install.html

### "No images registered" or reconstruction fails
- Ensure images have good overlap (70-80%)
- Check image quality (not too blurry)
- Try with more images
- Use textured scenes (avoid plain walls)

### Out of memory
```bash
# Reduce resolution
colmap-reconstruct from-video my_video.mp4 -o output/ --preset fast

# Or use CPU only
colmap-reconstruct from-video my_video.mp4 -o output/ --cpu-only
```

### Takes too long
```bash
# Use fast preset
colmap-reconstruct from-images images/ -o output/ --preset fast

# Or sparse only
colmap-reconstruct from-images images/ -o output/ --no-dense
```

## 9. Tips for Best Results

1. **Image Overlap**: 70-80% overlap between adjacent images
2. **Lighting**: Even, diffuse lighting works best
3. **Texture**: Textured surfaces reconstruct better than plain ones
4. **Coverage**: Cover the object/scene from many angles
5. **Image Count**: 20-100 images is usually sufficient
6. **Resolution**: 1920x1080 to 2560x1440 is a good balance
7. **Camera**: Keep camera settings consistent (exposure, focus)

## 10. Next Steps

- Read full README.md for detailed documentation
- Customize config in `configs/default.yaml`
- Check examples in `examples/` directory
- Run tests: `make test`
- Try Docker: `make docker-build && make docker-run`

## Quick Reference

```bash
# Video reconstruction
colmap-reconstruct from-video VIDEO -o OUTPUT/

# Image reconstruction
colmap-reconstruct from-images IMAGES/ -o OUTPUT/

# With known poses
colmap-reconstruct from-images-with-poses IMAGES/ POSES.json -o OUTPUT/

# Options
--preset {fast,high}  # Quality preset
--no-dense           # Sparse only
--no-mesh            # No mesh generation
--cpu-only           # Force CPU
--log-level DEBUG    # Verbose output
```

## Getting Help

- Full docs: See README.md
- Issues: Open GitHub issue
- COLMAP docs: https://colmap.github.io/
- Troubleshooting: See README.md troubleshooting section

Happy reconstructing! 🏗️📸

