# PCto3D Pipeline

A comprehensive pipeline for processing PLY point cloud files into segmented 3D environments exported as OBJ files.

## Features

- **PLY File Loading**: Supports both point clouds and mesh PLY files
- **Mesh Refinement**: Advanced outlier removal and noise filtering
- **Surface Segmentation**: Multiple segmentation methods (region growing, clustering, RANSAC)
- **OBJ Export**: High-quality mesh export with materials
- **Intermediate Results**: Save processing stages for inspection
- **Flexible Configuration**: YAML-based configuration system

## Pipeline Steps

1. **Load PLY File**: Read and validate point cloud or mesh from PLY file
2. **Refine Mesh**: Remove outliers, filter noise, and smooth surfaces
3. **Surface Segmentation**: Identify distinct surfaces using configurable methods
4. **Export OBJ**: Convert to triangle mesh and export with materials

## Installation

### Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Process a PLY file with default settings:

```bash
python main.py --input input/pointcloud.ply
```

### Specify Output Location

```bash
python main.py --input input/pointcloud.ply --output output/my_model.obj
```

### Custom Configuration

```bash
python main.py --input input/pointcloud.ply --config my_config.yaml
```

### Enable Verbose Logging

```bash
python main.py --input input/pointcloud.ply --verbose
```

## Configuration

Edit `config/settings.yaml` to customize the pipeline behavior:

### Input/Output Settings

```yaml
input:
  ply_folder: "input"
  ply_file: "pointcloud.ply"

output:
  folder: "output"
  obj_name: "segmented_model.obj"
  save_intermediate: true
```

### Refinement Settings

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

### Segmentation Settings

```yaml
segmentation:
  method: "region_growing"  # or "clustering", "ransac"
  
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

## Directory Structure

```
PCto3D/
├── config/
│   └── settings.yaml          # Configuration file
├── input/                     # Place your PLY files here
├── output/                    # Processed files output here
│   ├── intermediate/          # Intermediate processing stages
│   └── segments/              # Individual segment files
├── logs/                      # Pipeline logs
├── src/
│   ├── __init__.py
│   ├── loader.py              # PLY file loading
│   ├── refinement.py          # Mesh refinement
│   ├── segmentation.py        # Surface segmentation
│   ├── exporter.py            # OBJ export
│   └── pipeline.py            # Main pipeline orchestrator
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Output Files

The pipeline generates several output files:

- **`segmented_model.obj`**: Main output mesh with all segments
- **`segmented_model.mtl`**: Material file for the OBJ
- **`intermediate/01_refined.ply`**: Refined point cloud (optional)
- **`intermediate/02_segmented.ply`**: Segmented point cloud (optional)
- **`segments/segment_XXX.obj`**: Individual segment meshes (optional)

## Segmentation Methods

### Region Growing

Groups points based on normal similarity and curvature. Best for smooth surfaces.

```yaml
segmentation:
  method: "region_growing"
  region_growing:
    normal_variance_threshold: 0.1  # Lower = stricter
    curvature_threshold: 1.0        # Lower = flatter surfaces
```

### Clustering

Uses DBSCAN or K-means clustering on spatial features.

```yaml
segmentation:
  method: "clustering"
  clustering:
    method: "dbscan"
    eps: 0.05           # Neighborhood size
    min_samples: 10     # Minimum cluster size
```

### RANSAC Plane Segmentation

Extracts planar surfaces using RANSAC. Best for architectural scenes.

```yaml
segmentation:
  method: "ransac"
  ransac:
    distance_threshold: 0.01    # Point-to-plane distance
    max_planes: 10              # Maximum planes to extract
```

## Examples

### Process a building scan

```bash
python main.py \
  --input scans/building.ply \
  --output models/building.obj \
  --config configs/architectural.yaml \
  --verbose
```

### Quick preview without saving intermediate files

```bash
python main.py \
  --input input/test.ply \
  --no-intermediate
```

## Troubleshooting

### "Point cloud is empty"

Check that your PLY file contains valid point data.

### Segmentation produces too many/few segments

Adjust segmentation parameters in config:
- For more segments: increase sensitivity (lower thresholds)
- For fewer segments: decrease sensitivity (higher thresholds)

### Output mesh has holes

- Increase Poisson reconstruction depth
- Adjust outlier removal to be less aggressive
- Ensure input has good point density

### Out of memory

- Increase `voxel_size` to downsample more aggressively
- Reduce `depth` in Poisson reconstruction
- Process smaller regions separately

## Advanced Usage

### Python API

```python
from src.pipeline import PCto3DPipeline
import yaml

# Load config
with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

# Create pipeline
pipeline = PCto3DPipeline(config)

# Run
results = pipeline.run(
    ply_path='input/myfile.ply',
    output_path='output/mymodel.obj'
)

print(f"Success: {results['success']}")
print(f"Output: {results['output_path']}")
print(f"Segments: {results['stages']['segment']['num_segments']}")
```

### Custom Processing

```python
from src import PLYLoader, MeshRefiner, SurfaceSegmenter, OBJExporter

# Manual pipeline control
loader = PLYLoader(config)
refiner = MeshRefiner(config)
segmenter = SurfaceSegmenter(config)
exporter = OBJExporter(config)

# Load
geometry = loader.load('input/model.ply')

# Process
refined = refiner.refine(geometry)
segmented = segmenter.segment(refined)

# Export
exporter.export(segmented, 'output/result.obj')
```

## License

This project is part of the BuildersRetreat 3DHousePlan toolkit.

## Support

For issues and questions, please refer to the main project documentation.

