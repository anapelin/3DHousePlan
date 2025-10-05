#!/usr/bin/env python3
"""
Test script to verify PCto3D installation
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import open3d
        print("✓ open3d")
    except ImportError as e:
        print(f"✗ open3d: {e}")
        return False
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import scipy
        print("✓ scipy")
    except ImportError as e:
        print(f"✗ scipy: {e}")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError as e:
        print(f"✗ tqdm: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
        return False
    
    return True


def test_pipeline_imports():
    """Test that pipeline modules can be imported."""
    print("\nTesting pipeline modules...")
    
    try:
        from src import PLYLoader, MeshRefiner, SurfaceSegmenter, OBJExporter
        print("✓ Pipeline modules")
    except ImportError as e:
        print(f"✗ Pipeline modules: {e}")
        return False
    
    try:
        from src.pipeline import PCto3DPipeline
        print("✓ PCto3DPipeline")
    except ImportError as e:
        print(f"✗ PCto3DPipeline: {e}")
        return False
    
    return True


def test_config():
    """Test that config file exists and can be loaded."""
    print("\nTesting configuration...")
    
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration file loaded")
        
        # Check required keys
        required_keys = ['input', 'output', 'refinement', 'segmentation']
        for key in required_keys:
            if key not in config:
                print(f"✗ Missing config key: {key}")
                return False
        
        print("✓ Configuration structure valid")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directories...")
    
    from pathlib import Path
    
    base_dir = Path(__file__).parent.parent
    required_dirs = ['config', 'src', 'input', 'output', 'logs', 'examples']
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ not found")
            return False
    
    return True


def test_open3d_basic():
    """Test basic Open3D functionality."""
    print("\nTesting Open3D functionality...")
    
    try:
        import open3d as o3d
        import numpy as np
        
        # Create a simple point cloud
        points = np.random.rand(100, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        print(f"✓ Created point cloud with {len(pcd.points)} points")
        
        # Test normal estimation
        pcd.estimate_normals()
        if pcd.has_normals():
            print("✓ Normal estimation works")
        else:
            print("✗ Normal estimation failed")
            return False
        
        # Test mesh creation
        mesh = o3d.geometry.TriangleMesh.create_sphere()
        if len(mesh.vertices) > 0:
            print(f"✓ Mesh creation works ({len(mesh.vertices)} vertices)")
        else:
            print("✗ Mesh creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Open3D test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("PCto3D Installation Test")
    print("="*60)
    
    tests = [
        ("Module imports", test_imports),
        ("Pipeline imports", test_pipeline_imports),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Open3D functionality", test_open3d_basic),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} raised exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Installation is complete.")
        print("\nYou can now run the pipeline:")
        print("  python main.py --input input/your_file.ply")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the installation.")
        print("\nTry running:")
        print("  python install.py")
        return 1


if __name__ == '__main__':
    sys.exit(main())

