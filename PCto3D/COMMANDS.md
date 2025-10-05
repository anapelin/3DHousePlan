# PCto3D Command Reference

Quick reference for common PCto3D commands.

## Installation

```bash
# Install dependencies
python install.py

# Or manually
pip install -r requirements.txt

# Test installation
python tests/test_installation.py
```

## Basic Usage

```bash
# Process a PLY file (minimal)
python main.py --input input/pointcloud.ply

# With custom output
python main.py --input input/pointcloud.ply --output output/model.obj

# Verbose logging
python main.py --input input/pointcloud.ply --verbose

# Custom config
python main.py --input input/pointcloud.ply --config custom_config.yaml
```

## Common Workflows

### From COLMAP Output
```bash
python main.py --input ../COLMAPtry/implementation/frames/fused.ply
```

### Architectural/Indoor Scenes
```bash
python main.py \
  --input input/room.ply \
  --output output/room.obj \
  --config config/settings.yaml \
  --verbose
```

### Quick Processing (Fast, Less Detail)
```bash
python main.py --input input/large.ply --no-intermediate
```

### Save All Intermediate Steps
```bash
# Edit config/settings.yaml first:
# output:
#   save_intermediate: true

python main.py --input input/scan.ply --verbose
```

## Configuration Presets

### Preset 1: Architectural (Flat Surfaces)

Edit `config/settings.yaml`:
```yaml
segmentation:
  method: "ransac"
  ransac:
    distance_threshold: 0.01
    max_planes: 20
```

### Preset 2: Organic Surfaces

```yaml
segmentation:
  method: "region_growing"
  region_growing:
    normal_variance_threshold: 0.1
    curvature_threshold: 1.0
```

### Preset 3: Fast Processing

```yaml
refinement:
  noise_filtering:
    voxel_size: 0.1

segmentation:
  method: "clustering"
  clustering:
    method: "kmeans"
    n_clusters: 10
```

## Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `-i, --input` | Input PLY file path | `--input input/model.ply` |
| `-o, --output` | Output OBJ file path | `--output output/result.obj` |
| `-c, --config` | Config file path | `--config custom.yaml` |
| `-v, --verbose` | Enable verbose logging | `--verbose` |
| `--log-file` | Custom log file location | `--log-file logs/custom.log` |
| `--no-intermediate` | Don't save intermediate files | `--no-intermediate` |
| `--visualize` | Show visualization (if available) | `--visualize` |
| `--help` | Show help message | `--help` |

## Python API

### Basic Pipeline

```python
from src.pipeline import PCto3DPipeline
import yaml

# Load config
with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

# Run pipeline
pipeline = PCto3DPipeline(config)
results = pipeline.run('input/model.ply', 'output/result.obj')

print(f"Success: {results['success']}")
print(f"Segments: {results['stages']['segment']['num_segments']}")
```

### Custom Processing

```python
from src import PLYLoader, MeshRefiner, SurfaceSegmenter, OBJExporter

# Load
loader = PLYLoader(config)
geometry = loader.load('input/model.ply')

# Refine
refiner = MeshRefiner(config)
refined = refiner.refine(geometry)

# Segment
segmenter = SurfaceSegmenter(config)
segmented = segmenter.segment(refined)

# Export
exporter = OBJExporter(config)
exporter.export(segmented, 'output/result.obj')

# Export segments separately
exporter.export_segments_separately(segmented)
```

## Configuration Parameters

### Input/Output
```yaml
input:
  ply_folder: "input"
  ply_file: "pointcloud.ply"

output:
  folder: "output"
  obj_name: "segmented_model.obj"
  save_intermediate: true
```

### Refinement
```yaml
refinement:
  outlier_removal:
    method: "statistical"  # or "radius"
    nb_neighbors: 20
    std_ratio: 2.0
  
  noise_filtering:
    enable: true
    voxel_size: 0.02
  
  smoothing:
    enable: true
    method: "laplacian"  # or "taubin"
    iterations: 5
```

### Segmentation
```yaml
segmentation:
  method: "region_growing"  # or "clustering", "ransac"
  min_segment_size: 50
  merge_similar_segments: true
  merge_threshold: 0.05
  
  # Method-specific parameters
  region_growing:
    normal_variance_threshold: 0.1
    curvature_threshold: 1.0
    min_cluster_size: 100
  
  clustering:
    method: "dbscan"  # or "kmeans"
    eps: 0.05
    min_samples: 10
  
  ransac:
    distance_threshold: 0.01
    num_iterations: 1000
    max_planes: 10
```

## Troubleshooting Commands

### Check Installation
```bash
python tests/test_installation.py
```

### View Logs
```bash
# Windows
type logs\pipeline.log

# Linux/Mac
cat logs/pipeline.log
tail -f logs/pipeline.log  # Follow in real-time
```

### Debug Mode
```bash
python main.py --input input/model.ply --verbose --log-file logs/debug.log
```

## File Outputs

After running the pipeline, you'll find:

```
output/
├── segmented_model.obj          # Main 3D model
├── segmented_model.mtl          # Materials file
├── intermediate/                 # Processing stages
│   ├── 01_refined.ply           # After refinement
│   └── 02_segmented.ply         # After segmentation
└── segments/                    # Individual segments
    ├── segment_000.obj
    ├── segment_001.obj
    └── ...

logs/
└── pipeline.log                 # Processing log
```

## Quick Fixes

### Too Many Segments
```yaml
segmentation:
  min_segment_size: 200  # Increase
  merge_threshold: 0.1   # Increase
```

### Too Few Segments
```yaml
segmentation:
  min_segment_size: 30   # Decrease
  
  # For RANSAC
  ransac:
    distance_threshold: 0.005  # Decrease

  # For region growing  
  region_growing:
    normal_variance_threshold: 0.05  # Decrease
```

### Noisy Output
```yaml
refinement:
  outlier_removal:
    std_ratio: 2.0  # Decrease (more aggressive)
  noise_filtering:
    voxel_size: 0.05  # Increase
```

### Processing Too Slow
```yaml
refinement:
  noise_filtering:
    voxel_size: 0.1  # Increase (more downsampling)

segmentation:
  method: "clustering"  # Fastest method
```

## Examples

See `examples/example_usage.py` for complete working examples of:
- Basic pipeline usage
- Custom processing steps
- Different segmentation methods
- Exporting separate segments

Run with:
```bash
cd examples
python example_usage.py
```

