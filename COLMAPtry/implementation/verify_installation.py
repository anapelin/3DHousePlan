#!/usr/bin/env python3
"""
Verify COLMAP Pipeline installation and dependencies
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (requires 3.8+)")
        return False


def check_package_imports():
    """Check if pipeline package can be imported."""
    try:
        import pipeline
        from pipeline import COLMAPPipeline, COLMAPWrapper
        from pipeline.cli import main
        print("✓ Pipeline package installed")
        return True
    except ImportError as e:
        print(f"✗ Pipeline package not installed: {e}")
        print("  Run: pip install -e .")
        return False


def check_dependencies():
    """Check required Python dependencies."""
    deps = {
        "numpy": "numpy",
        "opencv": "cv2",
        "yaml": "yaml",
        "colorlog": "colorlog",
        "tqdm": "tqdm",
        "PIL": "PIL",
        "plyfile": "plyfile",
        "trimesh": "trimesh",
    }
    
    all_ok = True
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False
    
    return all_ok


def check_colmap():
    """Check COLMAP installation."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["colmap", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print(f"✓ COLMAP: {version}")
            return True
        else:
            print("✗ COLMAP found but error running")
            return False
    except FileNotFoundError:
        print("✗ COLMAP not found")
        print("  Install: https://colmap.github.io/install.html")
        return False
    except Exception as e:
        print(f"✗ Error checking COLMAP: {e}")
        return False


def check_ffmpeg():
    """Check ffmpeg installation."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.split('\n')[0].split(' ')[2]
            print(f"✓ ffmpeg: {version}")
            return True
        else:
            print("✗ ffmpeg found but error running")
            return False
    except FileNotFoundError:
        print("⚠ ffmpeg not found (optional for video)")
        return True  # Not critical
    except Exception as e:
        print(f"⚠ Error checking ffmpeg: {e}")
        return True  # Not critical


def check_cuda():
    """Check CUDA availability."""
    try:
        import cv2
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"✓ CUDA available ({cv2.cuda.getCudaEnabledDeviceCount()} device(s))")
            return True
    except:
        pass
    
    # Try nvidia-smi
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ CUDA available (via nvidia-smi)")
            return True
    except:
        pass
    
    print("⚠ CUDA not available (will use CPU)")
    return True  # Not critical


def check_cli():
    """Check CLI command."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["colmap-reconstruct", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ CLI command available")
            return True
        else:
            print("✗ CLI command error")
            return False
    except FileNotFoundError:
        print("✗ CLI command not found")
        print("  The package may not be installed correctly")
        return False
    except Exception as e:
        print(f"✗ Error checking CLI: {e}")
        return False


def check_config_files():
    """Check configuration files exist."""
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    
    if config_path.exists():
        print("✓ Configuration files present")
        return True
    else:
        print("✗ Configuration files missing")
        return False


def run_quick_test():
    """Run a quick import test."""
    try:
        from pipeline.utils import setup_logging, check_colmap_installation
        from pipeline.frame_extractor import FrameExtractor
        from pipeline.keyframe_selector import KeyframeSelector
        
        print("✓ All modules importable")
        return True
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("COLMAP Pipeline - Installation Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Pipeline Package", check_package_imports),
        ("Python Dependencies", check_dependencies),
        ("COLMAP Installation", check_colmap),
        ("ffmpeg Installation", check_ffmpeg),
        ("CUDA Support", check_cuda),
        ("CLI Command", check_cli),
        ("Configuration Files", check_config_files),
        ("Module Imports", run_quick_test),
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! Installation is complete.")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md")
        print("  2. Run: colmap-reconstruct --help")
        print("  3. Try an example: make example-video")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install package: pip install -e .")
        print("  - Install COLMAP: https://colmap.github.io/install.html")
        print("  - Install dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

