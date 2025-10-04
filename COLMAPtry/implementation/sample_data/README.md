# Sample Data

Place your sample data here for testing the pipeline.

## Directory Structure

```
sample_data/
├── video.mp4           # Sample video for testing
├── images/             # Sample images for testing
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── poses.json          # Sample camera poses (optional)
```

## Getting Sample Data

### Option 1: Use Your Own Data

- **Video**: Place any .mp4 video of an object or scene
- **Images**: Place 10-50 images with good overlap in `images/`

### Option 2: Download Public Datasets

Download a small dataset from:
- [ETH3D](https://www.eth3d.net/) - Multi-view datasets
- [Tanks and Temples](https://www.tanksandtemples.org/) - Benchmark datasets
- [COLMAP Tutorial Data](https://demuc.de/colmap/)

### Example: Download Fountain Dataset

```bash
cd sample_data
wget https://demuc.de/colmap/datasets/fountain.zip
unzip fountain.zip
mv fountain/images ./images
```

## Testing

Once you have sample data:

```bash
# Test with video
make example-video

# Test with images
make example-images
```

## Recommended Test Data

For best results, use:
- **Images**: 20-100 images with 70-80% overlap
- **Resolution**: 1920x1080 or similar
- **Scene**: Well-lit, textured objects or environments
- **Coverage**: 360° views or multiple viewing angles

