# PCto3D Pipeline Examples

This directory contains example scripts demonstrating various ways to use the PCto3D pipeline.

## Prerequisites

1. Install the pipeline dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Place a sample PLY file in the `input/` directory (or use a path to an existing PLY file)

## Running Examples

### Example 1: Basic Pipeline Usage

Run the complete pipeline with default settings:

```bash
python example_usage.py
```

Edit the `__main__` block to uncomment `example_1_basic_usage()`.

### Example 2: Custom Processing Steps

Manually control each step of the pipeline:

```python
loader = PLYLoader(config)
refiner = MeshRefiner(config)
segmenter = SurfaceSegmenter(config)
exporter = OBJExporter(config)

geometry = loader.load('input/sample.ply')
refined = refiner.refine(geometry)
segmented = segmenter.segment(refined)
output = exporter.export(segmented)
```

### Example 3: Region Growing Segmentation

Use region growing for smooth surface segmentation:

```python
config['segmentation']['method'] = 'region_growing'
config['segmentation']['region_growing']['normal_variance_threshold'] = 0.1
```

### Example 4: Clustering Segmentation

Use DBSCAN or K-means clustering:

```python
config['segmentation']['method'] = 'clustering'
config['segmentation']['clustering']['method'] = 'dbscan'
config['segmentation']['clustering']['eps'] = 0.05
```

### Example 5: RANSAC Plane Segmentation

Extract planar surfaces (good for buildings):

```python
config['segmentation']['method'] = 'ransac'
config['segmentation']['ransac']['max_planes'] = 10
```

### Example 6: Export Separate Segments

Export each segment as an individual OBJ file:

```python
segment_files = exporter.export_segments_separately(segmented)
```

## Sample Data

For testing, you can use PLY files from:

- The `COLMAPtry/implementation/frames/` directory (existing project PLY files)
- Generate a test PLY file using Open3D
- Download sample point clouds from public datasets

## Tips

- Start with **region growing** for organic/smooth surfaces
- Use **RANSAC** for architectural/man-made structures with flat surfaces
- Use **clustering** for quick segmentation without geometric constraints
- Adjust `min_segment_size` to filter out noise
- Enable `save_intermediate` to inspect processing stages
- Use `--verbose` flag for detailed logging

