#!/usr/bin/env python3
"""
Installation script for PCto3D Pipeline
Checks dependencies and sets up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required Python packages."""
    print("\nInstalling dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    directories = [
        "input",
        "output",
        "output/intermediate",
        "output/segments",
        "logs"
    ]
    
    for directory in directories:
        path = Path(__file__).parent / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {directory}/")
    
    return True


def verify_installation():
    """Verify that all required packages are installed."""
    print("\nVerifying installation...")
    
    required_packages = [
        "open3d",
        "numpy",
        "scipy",
        "PIL",
        "yaml",
        "tqdm",
        "sklearn"
    ]
    
    failed = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n✗ Missing packages: {', '.join(failed)}")
        return False
    
    return True


def create_sample_config():
    """Ensure config file exists."""
    print("\nChecking configuration...")
    config_file = Path(__file__).parent / "config" / "settings.yaml"
    
    if config_file.exists():
        print(f"✓ Configuration file exists: {config_file}")
        return True
    else:
        print(f"✗ Configuration file not found: {config_file}")
        print("  The default config should have been created during project setup")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your PLY file in the 'input/' directory")
    print("2. (Optional) Edit 'config/settings.yaml' to customize processing")
    print("3. Run the pipeline:")
    print("   python main.py --input input/your_file.ply")
    print("\nFor more options:")
    print("   python main.py --help")
    print("\nFor examples:")
    print("   See examples/README.md")
    print("="*60)


def main():
    """Main installation routine."""
    print("="*60)
    print("PCto3D Pipeline Installation")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Installation failed")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n✗ Failed to create directories")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed")
        sys.exit(1)
    
    # Check config
    create_sample_config()
    
    # Print next steps
    print_next_steps()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

