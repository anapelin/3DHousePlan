#!/bin/bash
# Example: Reconstruct 3D model from images with known camera poses

# Input images directory
IMAGES="../sample_data/images"

# Poses file (ARKit/ARCore export)
POSES="../sample_data/poses.json"

# Output directory
OUTPUT="outputs/poses_reconstruction"

# Run reconstruction
colmap-reconstruct from-images-with-poses "$IMAGES" "$POSES" -o "$OUTPUT" \
    --log-level INFO

echo ""
echo "Reconstruction complete!"
echo "Results saved to: $OUTPUT"
echo ""
echo "View in Blender:"
echo "  blender --python-expr 'import bpy; bpy.ops.import_scene.obj(filepath=\"$OUTPUT/poisson_mesh.obj\")'"
echo ""
echo "View report:"
echo "  Open: $OUTPUT/report/report.html"

