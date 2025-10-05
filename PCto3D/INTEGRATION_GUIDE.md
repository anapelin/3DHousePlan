# PCto3D Integration Guide

This guide shows how to integrate the PCto3D pipeline with the existing COLMAP reconstruction pipeline.

## Complete Workflow: Video → Point Cloud → Segmented 3D Model

### Step 1: Generate Point Cloud with COLMAP

First, process your video with the COLMAP pipeline to generate a point cloud:

```bash
cd ../COLMAPtry/implementation

# Extract frames and run COLMAP reconstruction
python -m pipeline.cli video \
  --video sample_data/video/video.mp4 \
  --output frames \
  --keyframe-method optical_flow
```

This generates several files in `frames/`:
- `fused.ply` - Dense point cloud (this is what we need!)
- `house.ply` - Alternative reconstruction
- `sparse/` - Sparse reconstruction data

### Step 2: Process Point Cloud with PCto3D

Now process the generated point cloud:

```bash
cd ../../PCto3D

# Process the COLMAP output
python main.py \
  --input ../COLMAPtry/implementation/frames/fused.ply \
  --output output/house_model.obj \
  --verbose
```

### Step 3: View Results

The output files are ready to use:

```
PCto3D/output/
├── house_model.obj          # Main 3D model
├── house_model.mtl          # Materials
├── intermediate/            # Processing stages
│   ├── 01_refined.ply
│   └── 02_segmented.ply
└── segments/                # Individual surfaces
    ├── segment_000.obj
    ├── segment_001.obj
    └── ...
```

## Automated Pipeline Script

Create a script to automate the entire workflow:

### `process_video_to_model.sh` (Linux/Mac)

```bash
#!/bin/bash

VIDEO=$1
OUTPUT_NAME=$2

if [ -z "$VIDEO" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: ./process_video_to_model.sh VIDEO_PATH OUTPUT_NAME"
    exit 1
fi

echo "=== Processing Video to 3D Model ==="
echo "Video: $VIDEO"
echo "Output: $OUTPUT_NAME"

# Step 1: COLMAP reconstruction
echo -e "\n[1/2] Running COLMAP reconstruction..."
cd ../COLMAPtry/implementation
python -m pipeline.cli video \
  --video "$VIDEO" \
  --output frames_temp \
  --keyframe-method optical_flow

# Step 2: Surface segmentation
echo -e "\n[2/2] Running surface segmentation..."
cd ../../PCto3D
python main.py \
  --input ../COLMAPtry/implementation/frames_temp/fused.ply \
  --output "output/${OUTPUT_NAME}.obj" \
  --verbose

echo -e "\n✓ Complete! Output: PCto3D/output/${OUTPUT_NAME}.obj"
```

### `process_video_to_model.bat` (Windows)

```batch
@echo off
setlocal

set VIDEO=%1
set OUTPUT_NAME=%2

if "%VIDEO%"=="" goto usage
if "%OUTPUT_NAME%"=="" goto usage

echo === Processing Video to 3D Model ===
echo Video: %VIDEO%
echo Output: %OUTPUT_NAME%

REM Step 1: COLMAP reconstruction
echo.
echo [1/2] Running COLMAP reconstruction...
cd ..\COLMAPtry\implementation
python -m pipeline.cli video --video "%VIDEO%" --output frames_temp --keyframe-method optical_flow

REM Step 2: Surface segmentation
echo.
echo [2/2] Running surface segmentation...
cd ..\..\PCto3D
python main.py --input ..\COLMAPtry\implementation\frames_temp\fused.ply --output "output\%OUTPUT_NAME%.obj" --verbose

echo.
echo Complete! Output: PCto3D\output\%OUTPUT_NAME%.obj
goto end

:usage
echo Usage: process_video_to_model.bat VIDEO_PATH OUTPUT_NAME
exit /b 1

:end
endlocal
```

Usage:
```bash
# Linux/Mac
chmod +x process_video_to_model.sh
./process_video_to_model.sh path/to/video.mp4 my_house

# Windows
process_video_to_model.bat path\to\video.mp4 my_house
```

## Processing Existing COLMAP Outputs

If you already have COLMAP reconstructions, process them directly:

```bash
cd PCto3D

# Process existing reconstruction
python main.py \
  --input ../COLMAPtry/implementation/frames/fused.ply \
  --output output/existing_house.obj
```

## Batch Processing Multiple Videos

Process multiple videos in sequence:

```bash
#!/bin/bash

VIDEOS=(
    "sample_data/video/video1.mp4"
    "sample_data/video/video2.mp4"
    "sample_data/video/video3.mp4"
)

for VIDEO in "${VIDEOS[@]}"; do
    NAME=$(basename "$VIDEO" .mp4)
    echo "Processing $NAME..."
    
    # COLMAP
    cd ../COLMAPtry/implementation
    python -m pipeline.cli video --video "$VIDEO" --output "frames_$NAME"
    
    # PCto3D
    cd ../../PCto3D
    python main.py \
      --input "../COLMAPtry/implementation/frames_$NAME/fused.ply" \
      --output "output/${NAME}.obj"
done
```

## Configuration Tips for Different Scenarios

### Indoor Spaces (Rooms, Houses)

Best for architectural features (walls, floors, ceilings):

```yaml
# config/indoor.yaml
segmentation:
  method: "ransac"
  ransac:
    distance_threshold: 0.01
    max_planes: 20
  min_segment_size: 200
```

Usage:
```bash
python main.py \
  --input ../COLMAPtry/implementation/frames/fused.ply \
  --config config/indoor.yaml \
  --output output/room.obj
```

### Outdoor Scenes (Buildings, Terrain)

For larger scale with more variation:

```yaml
# config/outdoor.yaml
refinement:
  noise_filtering:
    voxel_size: 0.05
    
segmentation:
  method: "region_growing"
  region_growing:
    normal_variance_threshold: 0.2
    min_cluster_size: 500
```

### Furniture/Objects

For detailed objects with smooth surfaces:

```yaml
# config/objects.yaml
refinement:
  noise_filtering:
    voxel_size: 0.01
    
segmentation:
  method: "region_growing"
  region_growing:
    normal_variance_threshold: 0.05
    curvature_threshold: 0.5
    min_cluster_size: 50
```

## Troubleshooting Integration Issues

### Issue: COLMAP output has too much noise

Adjust COLMAP parameters or increase PCto3D filtering:

```yaml
refinement:
  outlier_removal:
    std_ratio: 3.0  # More aggressive
  noise_filtering:
    voxel_size: 0.03  # Stronger downsampling
```

### Issue: Segmentation doesn't match room structure

Use RANSAC for flat surfaces:

```yaml
segmentation:
  method: "ransac"
  ransac:
    distance_threshold: 0.015
```

### Issue: Processing is too slow

Quick mode configuration:

```yaml
refinement:
  noise_filtering:
    voxel_size: 0.1  # Aggressive downsampling
    
segmentation:
  method: "clustering"
  clustering:
    method: "kmeans"
    n_clusters: 8
```

## Output Integration

### Import to Blender

```python
# Blender Python script
import bpy

# Import the OBJ
bpy.ops.import_scene.obj(filepath="PCto3D/output/house_model.obj")

# Each segment is a separate object
for obj in bpy.context.selected_objects:
    # Apply materials, modifiers, etc.
    pass
```

### Import to Unity

1. Copy `output/house_model.obj` and `house_model.mtl` to `Assets/Models/`
2. Unity will automatically import and create materials for each segment
3. Segments are accessible as submeshes

### Import to Unreal Engine

1. Import OBJ: Content Browser → Import
2. Select `house_model.obj`
3. Import options:
   - ☑ Combine Meshes: Off (to keep segments separate)
   - ☑ Import Materials: On
   - ☑ Import Textures: On

## Directory Structure After Integration

```
BuildersRetreat/3DHousePlan/
├── COLMAPtry/
│   └── implementation/
│       └── frames/
│           └── fused.ply          # COLMAP output
│
└── PCto3D/
    ├── input/                      # (optional, for other inputs)
    ├── output/
    │   ├── house_model.obj         # Final model
    │   ├── house_model.mtl
    │   ├── intermediate/
    │   └── segments/
    └── logs/
        └── pipeline.log
```

## Performance Considerations

- **COLMAP reconstruction**: 5-30 minutes depending on video length and quality
- **PCto3D processing**: 2-10 minutes depending on point cloud size and segmentation method
- **Total time**: ~10-40 minutes for complete workflow

### Optimization Tips

1. **Reduce COLMAP output density** before processing
2. **Use downsampling** in PCto3D config
3. **Choose faster segmentation** methods (clustering vs. region growing)
4. **Process on GPU-enabled machine** for COLMAP step

## Next Steps

- Experiment with different segmentation methods
- Fine-tune parameters for your specific use case
- Create custom configurations for different room types
- Integrate with your preferred 3D software workflow

For more details on individual pipelines:
- COLMAP: See `../COLMAPtry/implementation/README.md`
- PCto3D: See `README.md` in this directory

