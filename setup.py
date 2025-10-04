#!/usr/bin/env python3
"""
Setup script for the 3D House Plan Pipeline

This script handles installation and setup of the pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="3d-house-plan-pipeline",
    version="1.0.0",
    author="3D House Plan Pipeline Team",
    author_email="contact@example.com",
    description="A comprehensive pipeline that takes video input and room plans to generate 3D objects for Blender export",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/3d-house-plan-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "3d-house-plan=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="3d reconstruction computer vision blender export video processing floor plans",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/3d-house-plan-pipeline/issues",
        "Source": "https://github.com/yourusername/3d-house-plan-pipeline",
        "Documentation": "https://3d-house-plan-pipeline.readthedocs.io/",
    },
)
