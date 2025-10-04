# 3D House Plan Pipeline

A comprehensive pipeline that takes video input and room plans to generate 3D objects for Blender export. This system combines computer vision, architectural plan analysis, and 3D reconstruction to create accurate room models.

## 🚀 Features

- **Video Processing**: Extract spatial information, depth, and room features from video input
- **Plan Analysis**: Parse architectural drawings and floor plans (PDF, images)
- **3D Reconstruction**: Combine video and plan data to create accurate 3D models
- **Blender Export**: Export models in multiple formats (OBJ, FBX, PLY, STL, direct Blender integration)
- **Batch Processing**: Process multiple rooms simultaneously
- **Configurable**: Extensive configuration options for different use cases
- **Validated**: Comprehensive input validation and error handling

## 📋 Requirements

- Python 3.8 or higher
- OpenCV for video processing
- MediaPipe for pose detection
- Open3D and Trimesh for 3D processing
- PyMuPDF for PDF processing
- Click for CLI interface

## 🛠️ Installation

### Option 1: Direct Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/3d-house-plan-pipeline.git
cd 3d-house-plan-pipeline

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Using pip
```bash
pip install 3d-house-plan-pipeline
```

### Option 3: Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/yourusername/3d-house-plan-pipeline.git
cd 3d-house-plan-pipeline
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Usage
```bash
python main.py --video path/to/room_video.mp4 --plan path/to/floor_plan.jpg --output room_model.obj
```

**Example with included test data:**
```bash
python main.py --video examples/video.mp4 --plan examples/plan.jpg --output output/my_room.obj --room-name "LivingRoom"
```

### Advanced Usage
```bash
python main.py \
  --video room_video.mp4 \
  --plan floor_plan.jpg \
  --output living_room.obj \
  --room-name "LivingRoom" \
  --format obj
```

### Validate Before Processing
```bash
# Check if your files are valid before processing
python main.py --video video.mp4 --plan plan.jpg --output test.obj --validate-only
```

### Batch Processing
```bash
# Process multiple rooms
python main.py --video room1.mp4 --plan plan1.jpg --output batch/room1.obj
python main.py --video room2.mp4 --plan plan2.jpg --output batch/room2.obj
```

## 📊 Pipeline Overview

The pipeline consists of four main stages:

1. **Video Analysis** 📹
   - Extract keyframes from video input
   - Estimate depth using computer vision techniques
   - Detect room features (walls, corners, furniture)
   - Use pose detection for scale reference

2. **Plan Processing** 📐
   - Parse architectural drawings (PDF, images)
   - Extract room boundaries and dimensions
   - Detect doors, windows, and structural elements
   - Calculate scale factors and measurements

3. **3D Reconstruction** 🏗️
   - Combine video and plan data
   - Generate 3D geometry (walls, floor, ceiling)
   - Add furniture, windows, and doors
   - Optimize mesh topology

4. **Export** 📤
   - Generate Blender-compatible files
   - Support multiple formats (OBJ, FBX, PLY, STL)
   - Create Blender import scripts
   - Include materials and textures

## 🏗️ Project Structure

```
3DHousePlan/
├── src/                          # Source code
│   ├── video_processor.py        # Video analysis and feature extraction
│   ├── plan_parser.py            # Architectural plan processing
│   ├── reconstruction.py         # 3D model generation
│   ├── blender_export.py         # Blender integration
│   └── pipeline.py               # Main orchestration
├── config/
│   └── settings.yaml             # Configuration parameters
├── examples/                     # Sample data and usage examples
│   ├── example_usage.py          # Comprehensive usage examples
│   └── README.md                 # Example documentation
├── tests/                        # Unit tests
│   └── test_pipeline.py          # Test suite
├── main.py                       # CLI entry point
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

The pipeline can be configured through YAML files. See `config/settings.yaml` for all available options:

```yaml
# Video Processing
video:
  frame_skip: 10                    # Process every Nth frame
  quality_threshold: 0.5            # Minimum quality threshold

# Plan Parsing
plan:
  scale_factor: 100.0               # Pixels per meter
  min_room_area: 5.0                # Minimum room area (sqm)

# Export Settings
export:
  default_format: "obj"             # Default export format
  include_materials: true           # Include material information
```

## 🎬 Input Requirements

### Video Input
- **Format**: MP4, AVI, MOV, MKV
- **Quality**: Higher resolution produces better results
- **Content**: Walk around the room, showing walls, corners, and features
- **Duration**: 30-60 seconds is usually sufficient
- **Lighting**: Ensure good, even lighting

### Floor Plans
- **Format**: PDF, JPG, PNG, BMP, TIFF
- **Quality**: Clear, well-defined lines and text
- **Content**: Room boundaries, dimensions, doors, windows
- **Scale**: Include scale information if possible

## 📤 Output Formats

| Format | Description | Best For |
|--------|-------------|----------|
| **OBJ** | Wavefront OBJ with materials | General 3D modeling, Blender |
| **FBX** | Autodesk FBX | Game engines, animation software |
| **PLY** | Stanford PLY | Point cloud processing, research |
| **STL** | Stereolithography | 3D printing, rapid prototyping |
| **Blend** | Blender script | Direct Blender integration |

## 🧪 Testing

Run the test suite to verify installation:

```bash
python tests/test_pipeline.py
```

Run specific test categories:
```bash
# Test video processing
python -m pytest tests/test_pipeline.py::TestVideoProcessor -v

# Test plan parsing
python -m pytest tests/test_pipeline.py::TestPlanParser -v

# Test 3D reconstruction
python -m pytest tests/test_pipeline.py::TestReconstructionEngine -v
```

## 🔧 API Usage

Use the pipeline programmatically:

```python
from src.pipeline import Pipeline, PipelineConfig

# Initialize pipeline
config = PipelineConfig()
pipeline = Pipeline(config)

# Process video and plan
success = pipeline.process(
    video_path="room_video.mp4",
    plan_path="floor_plan.pdf", 
    output_path="room_model.obj",
    room_name="LivingRoom"
)

if success:
    print("3D model created successfully!")
```

## 🐛 Troubleshooting

### Common Issues

1. **"No rooms detected in floor plan"**
   - Ensure clear room boundaries in the plan
   - Try adjusting the scale factor
   - Check plan format compatibility

2. **"No video frames processed"**
   - Verify video file integrity
   - Check supported video formats
   - Reduce frame skip value

3. **"Export failed"**
   - Check output directory permissions
   - Verify export format support
   - Review log files for details

### Performance Optimization

- **Reduce frame skip** for better quality (slower processing)
- **Use GPU acceleration** if available
- **Process smaller rooms** for testing
- **Enable parallel processing** for batch operations

## 📚 Examples

See the `examples/` directory for:
- Comprehensive usage examples
- Sample data preparation
- Batch processing workflows
- Configuration examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- MediaPipe for pose detection
- Open3D for 3D processing
- Trimesh for mesh operations
- The Blender community for 3D modeling tools

## 📞 Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: Check the examples and configuration files
- **Community**: Join discussions in GitHub Discussions

---

**Ready to create 3D room models?** Start with the examples in the `examples/` directory!
