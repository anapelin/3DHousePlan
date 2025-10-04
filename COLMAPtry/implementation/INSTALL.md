# Installation Guide

Complete installation instructions for the COLMAP SfM/MVS Pipeline.

## Quick Installation

```bash
# 1. Install COLMAP
sudo apt install colmap  # Ubuntu/Debian

# 2. Install pipeline
cd implementation
pip install -e .

# 3. Verify
python verify_installation.py
```

## Detailed Installation

### Step 1: Install COLMAP

COLMAP is the core dependency. Install it first:

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install colmap
```

#### macOS
```bash
brew install colmap
```

#### Windows
Download from: https://colmap.github.io/install.html

Or use WSL2 with Ubuntu instructions.

#### From Source (Advanced)
```bash
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### Step 2: Install Python Dependencies

```bash
cd implementation
pip install -r requirements.txt
```

Or install individual packages:
```bash
pip install numpy opencv-python PyYAML colorlog tqdm pillow plyfile trimesh
```

### Step 3: Install Pipeline Package

```bash
pip install -e .
```

This installs the package in editable mode and creates the `colmap-reconstruct` command.

### Step 4: Verify Installation

```bash
python verify_installation.py
```

This checks:
- Python version (3.8+)
- Pipeline package
- Dependencies
- COLMAP installation
- ffmpeg (optional)
- CUDA support (optional)
- CLI command

## Optional Dependencies

### ffmpeg (for video processing)

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from: https://ffmpeg.org/download.html

### CUDA (for GPU acceleration)

Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads

Recommended: CUDA 11.x or 12.x

Verify with:
```bash
nvidia-smi
```

## Docker Installation

If you prefer Docker:

```bash
cd implementation
make docker-build
```

Or manually:
```bash
docker build -t colmap-pipeline:latest .
```

Run with:
```bash
docker run --gpus all -v $(pwd)/data:/data colmap-pipeline:latest --help
```

## Troubleshooting Installation

### "COLMAP not found"

**Issue:** COLMAP is not in PATH

**Fix:**
```bash
# Find COLMAP
which colmap

# Add to PATH if needed
export PATH="/path/to/colmap/bin:$PATH"

# Or create symlink
sudo ln -s /path/to/colmap/bin/colmap /usr/local/bin/colmap
```

### "Module not found" errors

**Issue:** Python dependencies missing

**Fix:**
```bash
pip install -r requirements.txt
```

Or:
```bash
pip install numpy opencv-python PyYAML colorlog tqdm pillow plyfile trimesh
```

### "colmap-reconstruct command not found"

**Issue:** Package not installed or not in PATH

**Fix:**
```bash
# Reinstall
pip uninstall colmap-pipeline
pip install -e .

# Or use full path
python -m pipeline.cli --help
```

### Permission denied on scripts

**Issue:** Shell scripts not executable

**Fix:**
```bash
chmod +x examples/*.sh
chmod +x verify_installation.py
```

### Import errors after installation

**Issue:** Old package cached

**Fix:**
```bash
pip uninstall colmap-pipeline
rm -rf build/ dist/ *.egg-info
pip install -e .
```

## Platform-Specific Notes

### Windows

1. Install COLMAP from official builds
2. Use Anaconda/Miniconda for Python
3. Install Visual Studio Build Tools if building from source
4. Use PowerShell or CMD (not Git Bash for COLMAP commands)

### macOS

1. Install Xcode Command Line Tools: `xcode-select --install`
2. Use Homebrew for dependencies
3. Some features may require Rosetta 2 on Apple Silicon

### Linux

1. Ubuntu 20.04+ recommended
2. Install build essentials: `sudo apt install build-essential`
3. For GPU support, install NVIDIA drivers first

## Minimal Installation (No Dense Reconstruction)

If you only need sparse reconstruction:

```bash
# Install COLMAP (required)
sudo apt install colmap

# Install minimal dependencies
pip install numpy opencv-python PyYAML colorlog tqdm

# Install pipeline
pip install -e .
```

Use with `--no-dense` flag:
```bash
colmap-reconstruct from-images images/ -o output/ --no-dense
```

## Development Installation

For development:

```bash
# Clone repository
git clone <repo-url>
cd implementation

# Install with dev dependencies
pip install -e ".[dev]"

# Or manually
pip install pytest pytest-cov black flake8 mypy isort

# Run tests
make test
```

## Updating

To update to latest version:

```bash
cd implementation
git pull  # if from git
pip install -e . --upgrade
```

## Uninstallation

To remove:

```bash
pip uninstall colmap-pipeline
```

To remove all dependencies:
```bash
pip uninstall -r requirements.txt
```

## Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] COLMAP command works: `colmap --version`
- [ ] Pipeline imports: `python -c "import pipeline"`
- [ ] CLI works: `colmap-reconstruct --help`
- [ ] Config files present: `ls configs/default.yaml`
- [ ] Optional: ffmpeg works: `ffmpeg -version`
- [ ] Optional: CUDA available: `nvidia-smi`

Run verification script:
```bash
python verify_installation.py
```

## Next Steps

After successful installation:

1. Read **QUICKSTART.md** for usage
2. Try example: `make example-video`
3. Read **README.md** for full docs
4. Check `examples/` for sample scripts

## Getting Help

If installation fails:

1. Check this troubleshooting section
2. Run `python verify_installation.py` for diagnostics
3. Check COLMAP docs: https://colmap.github.io/
4. Open GitHub issue with verification output

## System Requirements

**Minimum:**
- Python 3.8+
- 4 GB RAM
- 10 GB disk space

**Recommended:**
- Python 3.10+
- 16 GB RAM
- 50 GB disk space
- NVIDIA GPU with 6+ GB VRAM
- CUDA 11.x or 12.x

**For large reconstructions:**
- 32+ GB RAM
- NVIDIA GPU with 12+ GB VRAM
- SSD storage

---

Need help? Check README.md or open an issue!

