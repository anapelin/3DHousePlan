# Examples

This directory contains examples and sample data for the 3D House Plan Pipeline.

## Files

- `example_usage.py` - Comprehensive examples showing how to use the pipeline
- `sample_room_video.mp4` - Sample room video (you need to provide this)
- `sample_floor_plan.pdf` - Sample floor plan (you need to provide this)

## Getting Started

1. **Install the pipeline**:
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Prepare sample data**:
   - Record a video of a room (walk around the room, showing walls, corners, etc.)
   - Get a floor plan of the same room (PDF or image format)

3. **Run the examples**:
   ```bash
   python example_usage.py
   ```

## Example Commands

### Basic Usage
```bash
python ../main.py --video sample_room_video.mp4 --plan sample_floor_plan.pdf --output my_room.obj
```

### Advanced Usage
```bash
python ../main.py --video sample_room_video.mp4 --plan sample_floor_plan.pdf --output my_room.fbx --format fbx --room-name "LivingRoom"
```

### Batch Processing
```bash
python ../main.py --video room1_video.mp4 --plan room1_plan.pdf --output batch_output/room1.obj
python ../main.py --video room2_video.mp4 --plan room2_plan.pdf --output batch_output/room2.obj
```

## Tips for Best Results

### Video Recording
- **Lighting**: Ensure good, even lighting throughout the room
- **Movement**: Move slowly and steadily around the room
- **Coverage**: Try to capture all walls, corners, and major features
- **Duration**: 30-60 seconds is usually sufficient
- **Resolution**: Higher resolution videos generally produce better results

### Floor Plans
- **Format**: PDF or high-resolution images work best
- **Quality**: Clear, well-defined lines and text
- **Scale**: Include scale information if possible
- **Completeness**: Ensure all room boundaries are clearly visible

### Output Formats
- **OBJ**: Best for general 3D modeling, includes materials
- **FBX**: Good for game engines and animation software
- **PLY**: Good for point cloud processing
- **STL**: Good for 3D printing
- **Blend**: Direct Blender integration (creates import script)

## Troubleshooting

### Common Issues

1. **"No rooms detected in floor plan"**
   - Ensure the floor plan has clear room boundaries
   - Try adjusting the scale factor in configuration
   - Check that the plan format is supported

2. **"No video frames processed"**
   - Verify the video file is not corrupted
   - Check that the video format is supported
   - Try reducing the frame skip value

3. **"Export failed"**
   - Check that the output directory exists
   - Verify you have write permissions
   - Try a different export format

### Performance Tips

- **Reduce frame skip** for better quality (but slower processing)
- **Use GPU acceleration** if available (requires additional setup)
- **Process smaller rooms** first to test the pipeline
- **Use parallel processing** for batch operations

## Sample Data

To test the pipeline, you can:

1. **Create your own data**:
   - Record a video of any room in your house
   - Draw a simple floor plan or use existing architectural drawings

2. **Use public datasets**:
   - Look for room layout datasets online
   - Use sample videos from computer vision datasets

3. **Generate synthetic data**:
   - Create simple room videos using 3D modeling software
   - Generate floor plans using architectural software

## Next Steps

Once you have the basic pipeline working:

1. **Experiment with different rooms** and layouts
2. **Try different export formats** for your specific use case
3. **Adjust configuration parameters** for better results
4. **Integrate with your existing workflow** using the Python API
5. **Extend the pipeline** with additional features

## Support

If you encounter issues:

1. Check the logs in `pipeline.log`
2. Run the validation: `python ../main.py --validate-only --video your_video.mp4 --plan your_plan.pdf`
3. Check the test suite: `python ../tests/test_pipeline.py`
4. Review the configuration in `../config/settings.yaml`
