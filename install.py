#!/usr/bin/env python3
"""
Installation script for the 3D House Plan Pipeline

This script handles the installation and setup of the pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Installing Python dependencies")

def verify_installation():
    """Verify that the installation was successful."""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from pipeline import Pipeline, PipelineConfig
        from video_processor import VideoProcessor
        from plan_parser import PlanParser
        from reconstruction import ReconstructionEngine
        from blender_export import BlenderExporter
        
        print("✅ All modules imported successfully")
        
        # Test pipeline initialization
        config = PipelineConfig()
        pipeline = Pipeline(config)
        print("✅ Pipeline initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    test_file = Path("tests/test_pipeline.py")
    if not test_file.exists():
        print("⚠️  Test file not found, skipping tests")
        return True
    
    return run_command(f"{sys.executable} tests/test_pipeline.py", 
                      "Running test suite")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = ["examples", "output", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def main():
    """Main installation function."""
    print("🏗️  3D House Plan Pipeline Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Installation failed at dependency installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed, but installation may still work")
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("\n📚 Next steps:")
    print("1. Check the README.md for usage instructions")
    print("2. Look at examples/example_usage.py for examples")
    print("3. Try running: python main.py --help")
    print("4. Test with your own video and floor plan files")
    print("\n🚀 Happy 3D modeling!")

if __name__ == "__main__":
    main()
