# PCto3D Pipeline - Quick Start Guide

Get started with the PCto3D pipeline in 5 minutes!

## 1. Installation

### Install Dependencies

```bash
cd PCto3D
python install.py
```

Or manually:

```bash
pip install -r requirements.txt
```

## 2. Prepare Your Data

Place your PLY file in the `input/` directory:

```bash
mkdir -p input
# Copy your PLY file here
cp /path/to/your/pointcloud.ply input/
```

You can also use an existing PLY file from the project, like:
```bash
# Copy from COLMAP output
cp ../COLMAPtry/implementation/frames/fused.ply input/
```

## 3. Run the Pipeline

### Basic Usage

Process your PLY file with default settings:

```bash
python main.py --input input/fused.ply
```

The output will be saved to `output/segmented_model.obj`

### With Custom Output

```bash
python main.py --input input/fused.ply --output output/my_house_model.obj
```

### With Verbose Logging

```bash
python main.py --input input/fused.ply --verbose
```

## 4. View Results

The pipeline generates several files:

- **`output/segmented_model.obj`** - Main 3D model with all segments
- **`output/segmented_model.mtl`** - Material file (colors for segments)
- **`output/intermediate/`** - Processing stages (optional)
- **`output/segments/`** - Individual segment files (optional)
- **`logs/pipeline.log`** - Detailed processing log

Import the `.obj` file into:
- **Blender**: File → Import → Wavefront (.obj)
- **MeshLab**: File → Import Mesh
- **CloudCompare**: File → Open
- **3D Viewer**: Any standard 3D viewer

## 5. Customize Settings (Optional)

Edit `config/settings.yaml` to adjust processing:

### For Architectural/Building Scans

```yaml
segmentation:
  method: "ransac"  # Extract flat surfaces
  ransac:
    distance_threshold: 0.01
    max_planes: 15
```

### For Organic/Natural Scenes

```yaml
segmentation:
  method: "region_growing"  # Follow surface curvature
  region_growing:
    normal_variance_threshold: 0.15
    curvature_threshold: 1.5
```

### For Quick Processing

```yaml
refinement:
  noise_filtering:
    voxel_size: 0.05  # Larger = faster, less detail

segmentation:
  method: "clustering"  # Fastest method
```

## Common Workflows

### From Video to Segmented 3D Model

If you're coming from the COLMAP pipeline:

```bash
# 1. Process video with COLMAP (in COLMAPtry/implementation/)
cd ../COLMAPtry/implementation
python -m pipeline.cli video --video sample_data/video/video.mp4 --output frames

# 2. Process the generated PLY file
cd ../../PCto3D
python main.py --input ../COLMAPtry/implementation/frames/fused.ply --output output/house_model.obj
```

### Process Multiple PLY Files

```bash
# Process each file
for ply in input/*.ply; do
    name=$(basename "$ply" .ply)
    python main.py --input "$ply" --output "output/${name}.obj"
done
```

### Export Segments as Separate Files

This happens automatically! Check `output/segments/` for individual segment OBJ files.

## Troubleshooting

### Issue: "PLY file not found"

Make sure your PLY file path is correct:
```bash
python main.py --input input/your_file.ply  # Use correct path
```

### Issue: Too many/few segments

Adjust segmentation sensitivity in `config/settings.yaml`:

```yaml
segmentation:
  min_segment_size: 100  # Increase to get fewer segments
  
  # For RANSAC
  ransac:
    distance_threshold: 0.02  # Increase for fewer planes
  
  # For Region Growing
  region_growing:
    normal_variance_threshold: 0.2  # Increase for fewer regions
```

### Issue: Output mesh has holes

```yaml
refinement:
  outlier_removal:
    std_ratio: 3.0  # Increase (less aggressive removal)
  
  noise_filtering:
    voxel_size: 0.01  # Decrease (keep more points)
```

### Issue: Processing is too slow

```yaml
refinement:
  noise_filtering:
    voxel_size: 0.05  # Increase for faster processing

segmentation:
  method: "clustering"  # Fastest method
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples/](examples/) for advanced usage patterns
- Experiment with different segmentation methods
- Integrate with your existing 3D workflow

## Command Reference

```bash
# Basic usage
python main.py --input INPUT.ply

# Full options
python main.py \
  --input INPUT.ply \
  --output OUTPUT.obj \
  --config CONFIG.yaml \
  --verbose \
  --no-intermediate \
  --log-file custom.log

# Help
python main.py --help
```

## Support

For issues or questions, check the logs at `logs/pipeline.log` for detailed information about what went wrong.

Happy 3D modeling! 🏗️

