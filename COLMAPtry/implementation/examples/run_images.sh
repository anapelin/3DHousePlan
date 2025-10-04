#!/bin/bash
# Example: Reconstruct 3D model from images

# Input images directory
IMAGES="../sample_data/images"

# Output directory
OUTPUT="outputs/images_reconstruction"

# Run reconstruction
colmap-reconstruct from-images "$IMAGES" -o "$OUTPUT" \
    --preset fast \
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

