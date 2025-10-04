# 🚀 Get Started with COLMAP Pipeline

**Welcome!** You now have a complete, production-ready COLMAP reconstruction pipeline.

## What You Got

A comprehensive 3D reconstruction system that can:
- ✅ Turn videos into 3D models
- ✅ Reconstruct from photo collections  
- ✅ Handle ARKit/ARCore camera poses
- ✅ Export to Blender-ready OBJ files
- ✅ Generate beautiful HTML reports
- ✅ Resume from interruptions
- ✅ Run on GPU or CPU
- ✅ Work in Docker containers

## Quick Start (3 Commands)

```bash
# 1. Install
cd COLMAPtry/implementation
pip install -e .

# 2. Test
python verify_installation.py

# 3. Run
colmap-reconstruct from-video your_video.mp4 -o output/
```

That's it! 🎉

## Your First Reconstruction

### Option 1: From Video (Easiest)

```bash
# Place your video
cp /path/to/your/video.mp4 sample_data/

# Reconstruct
colmap-reconstruct from-video sample_data/video.mp4 -o outputs/my_first_reconstruction/

# View in Blender
blender outputs/my_first_reconstruction/poisson_mesh.obj
```

### Option 2: From Photos

```bash
# Place your photos
cp /path/to/your/photos/*.jpg sample_data/images/

# Reconstruct  
colmap-reconstruct from-images sample_data/images/ -o outputs/my_reconstruction/

# View in Blender
blender outputs/my_reconstruction/poisson_mesh.obj
```

## What Gets Created

After reconstruction, you'll have:

```
outputs/my_reconstruction/
├── poisson_mesh.obj         ← Import this into Blender!
├── dense_point_cloud.ply    ← Dense 3D points
├── sparse/                  ← Camera poses & sparse points
├── report/
│   └── report.html          ← Open in browser for stats
├── logs/                    ← Detailed logs per stage
└── checkpoint.json          ← For resuming
```

## View Your 3D Model

### In Blender

1. **Open Blender**
2. **File → Import → Wavefront (.obj)**
3. **Navigate to:** `outputs/my_reconstruction/poisson_mesh.obj`
4. **Done!** Your 3D model is now in Blender

Or use command line:
```bash
blender --python-expr "import bpy; bpy.ops.import_scene.obj(filepath='outputs/my_reconstruction/poisson_mesh.obj')"
```

### In MeshLab/CloudCompare

```bash
# Open point cloud
meshlab outputs/my_reconstruction/dense_point_cloud.ply

# Or mesh
meshlab outputs/my_reconstruction/poisson_mesh.obj
```

## Essential Commands

```bash
# From video
colmap-reconstruct from-video VIDEO.mp4 -o OUTPUT_DIR/

# From images
colmap-reconstruct from-images IMAGES_DIR/ -o OUTPUT_DIR/

# With known poses (ARKit/ARCore)
colmap-reconstruct from-images-with-poses IMAGES_DIR/ POSES.json -o OUTPUT_DIR/

# Fast test
colmap-reconstruct from-video VIDEO.mp4 -o OUTPUT_DIR/ --preset fast

# High quality
colmap-reconstruct from-images IMAGES_DIR/ -o OUTPUT_DIR/ --preset high

# Sparse only (no dense, faster)
colmap-reconstruct from-images IMAGES_DIR/ -o OUTPUT_DIR/ --no-dense

# CPU only (no GPU)
colmap-reconstruct from-video VIDEO.mp4 -o OUTPUT_DIR/ --cpu-only

# Help
colmap-reconstruct --help
```

## Configuration

Edit `configs/default.yaml` to customize:
- Frame extraction rate (FPS)
- Number of SIFT features
- Dense reconstruction quality
- Mesh detail level
- And much more...

Or create your own:
```bash
cp configs/default.yaml configs/my_config.yaml
# Edit my_config.yaml
colmap-reconstruct from-video video.mp4 -o output/ -c configs/my_config.yaml
```

## Tips for Best Results

### 📸 When Capturing Images/Video

1. **Overlap**: 70-80% overlap between adjacent views
2. **Coverage**: Cover object from all angles (360°)
3. **Lighting**: Even, diffuse lighting (avoid harsh shadows)
4. **Texture**: Textured surfaces work better than plain ones
5. **Focus**: Keep camera in focus, avoid blur
6. **Count**: 20-100 images usually sufficient
7. **Resolution**: 1920x1080 to 2560x1440 is good

### ⚙️ Processing Settings

- **Quick test**: Use `--preset fast --no-dense`
- **Best quality**: Use `--preset high`
- **Large scenes**: Use `sequential` matching (automatic for video)
- **Small objects**: Use `exhaustive` matching
- **GPU**: 5-10x faster than CPU

### 🐛 If Reconstruction Fails

1. **Too few images registered**
   - Add more images
   - Increase overlap
   - Check image quality

2. **Out of memory**
   - Use `--preset fast`
   - Reduce image resolution in config
   - Try `--cpu-only`

3. **Takes too long**
   - Use `--preset fast`
   - Use `--no-dense` for sparse only
   - Reduce number of images

## Example Workflows

### Quick Test (2 minutes)
```bash
colmap-reconstruct from-video video.mp4 -o test/ --preset fast --no-dense
```

### High-Quality Model (30 minutes)
```bash
colmap-reconstruct from-images photos/ -o output/ --preset high
```

### Resume Interrupted Run
```bash
# Just run the same command again - it auto-resumes!
colmap-reconstruct from-video video.mp4 -o output/
```

### Batch Processing
```bash
#!/bin/bash
for video in videos/*.mp4; do
  name=$(basename "$video" .mp4)
  colmap-reconstruct from-video "$video" -o "outputs/$name/"
done
```

## Documentation Files

- **QUICKSTART.md** - 5-minute quick start
- **README.md** - Complete documentation
- **INSTALL.md** - Detailed installation
- **PROJECT_SUMMARY.md** - Technical overview
- **GET_STARTED.md** - This file

## Using Docker

```bash
# Build
make docker-build

# Run
docker run --gpus all \
  -v $(pwd)/data:/data \
  colmap-pipeline:latest \
  from-video /data/video.mp4 -o /data/output
```

## Running Tests

```bash
# Install with tests
pip install -e ".[dev]"

# Run tests
make test

# Or manually
pytest tests/ -v
```

## Using Make Commands

```bash
make install       # Install pipeline
make test          # Run tests
make clean         # Clean temp files
make docker-build  # Build Docker image
make example-video # Run video example
```

## Example Scripts

Pre-made scripts in `examples/`:

```bash
# Video reconstruction
bash examples/run_video.sh

# Image reconstruction  
bash examples/run_images.sh

# With known poses
bash examples/run_with_poses.sh
```

## Getting Help

1. **Verification failed?**
   ```bash
   python verify_installation.py
   ```

2. **Check docs:**
   - QUICKSTART.md for basics
   - README.md for details
   - INSTALL.md for setup issues

3. **Check logs:**
   ```bash
   cat outputs/my_reconstruction/logs/*.log
   ```

4. **View report:**
   Open `outputs/my_reconstruction/report/report.html` in browser

## What's Next?

Now that you have the pipeline installed:

1. **Try it out** with your own data
2. **Experiment** with different presets
3. **Customize** configuration for your needs
4. **Share** your 3D models!

## Performance Expectations

On a modern system with GPU:

| Dataset Size | Processing Time |
|-------------|----------------|
| 10-30 images | 2-5 minutes |
| 50-100 images | 10-30 minutes |
| 200+ images | 1-3 hours |

CPU-only: 3-10x longer

## Common Use Cases

### 🏛️ Architecture & Buildings
```bash
# Exterior: walk around building
colmap-reconstruct from-video building_exterior.mp4 -o output/
```

### 🗿 Objects & Sculptures  
```bash
# 360° turntable video
colmap-reconstruct from-video object_360.mp4 -o output/ --preset high
```

### 🏞️ Landscapes & Scenes
```bash
# Photos from different viewpoints
colmap-reconstruct from-images landscape_photos/ -o output/
```

### 📱 ARKit/ARCore Captures
```bash
# With recorded camera poses
colmap-reconstruct from-images-with-poses arkit_images/ poses.json -o output/
```

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| COLMAP not found | Install COLMAP: `sudo apt install colmap` |
| Command not found | Run: `pip install -e .` |
| Out of memory | Use `--preset fast` or `--cpu-only` |
| Too slow | Use `--preset fast` or `--no-dense` |
| Poor results | More images, better overlap, check lighting |
| Can't resume | Delete `checkpoint.json` and restart |

## Support & Community

- **Issues**: Open GitHub issue
- **Questions**: Check README.md FAQ
- **COLMAP docs**: https://colmap.github.io/
- **Examples**: See `examples/` directory

---

## 🎉 Ready to Reconstruct!

You're all set! Start with a simple test:

```bash
# 1. Get test video
# (use your phone to record an object rotating 360°)

# 2. Reconstruct
colmap-reconstruct from-video test.mp4 -o my_first_model/

# 3. View in Blender
blender my_first_model/poisson_mesh.obj
```

Happy reconstructing! 🏗️📸✨

---

**Quick Links:**
- 📖 [Full Documentation](README.md)
- ⚡ [Quick Start](QUICKSTART.md)
- 💻 [Installation](INSTALL.md)
- 🔧 [Configuration](configs/default.yaml)
- 🧪 [Examples](examples/)

