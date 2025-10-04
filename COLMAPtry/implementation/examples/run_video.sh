#!/bin/bash
# Example: Reconstruct 3D model from video

# Input video
VIDEO="../sample_data/video.mp4"

# Output directory
OUTPUT="outputs/video_reconstruction"

# Run reconstruction
colmap-reconstruct from-video "$VIDEO" -o "$OUTPUT" \
    --preset high \
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

