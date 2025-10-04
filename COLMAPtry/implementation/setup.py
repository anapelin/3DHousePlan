"""
Setup script for COLMAP Pipeline
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="colmap-pipeline",
    version="1.0.0",
    author="BuildersRetreat",
    description="A reproducible COLMAP SfM/MVS pipeline for 3D reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "PyYAML>=5.4.0",
        "colorlog>=6.0.0",
        "tqdm>=4.60.0",
        "pillow>=8.0.0",
        "plyfile>=0.7.4",
        "trimesh>=3.9.0",
    ],
    entry_points={
        "console_scripts": [
            "colmap-reconstruct=pipeline.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

