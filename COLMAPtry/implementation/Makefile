# Makefile for COLMAP Pipeline

.PHONY: help install install-dev test lint format clean docker-build docker-run example-video example-images

# Default target
help:
	@echo "COLMAP Pipeline - Available targets:"
	@echo ""
	@echo "  install       - Install pipeline and dependencies"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linters"
	@echo "  format        - Format code"
	@echo "  clean         - Clean temporary files"
	@echo ""
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run pipeline in Docker"
	@echo ""
	@echo "  example-video - Run video reconstruction example"
	@echo "  example-images - Run images reconstruction example"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pytest pytest-cov black flake8 mypy

# Testing
test:
	pytest tests/ -v --cov=pipeline --cov-report=html --cov-report=term

# Linting and formatting
lint:
	flake8 pipeline/ tests/
	mypy pipeline/

format:
	black pipeline/ tests/
	isort pipeline/ tests/

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete

# Docker
docker-build:
	docker build -t colmap-pipeline:latest .

docker-run:
	docker run --gpus all -v $(PWD)/data:/data colmap-pipeline:latest

# Examples
example-video:
	@echo "Running video reconstruction example..."
	@if [ ! -f sample_data/video.mp4 ]; then \
		echo "Error: sample_data/video.mp4 not found"; \
		echo "Please add a sample video to sample_data/"; \
		exit 1; \
	fi
	colmap-reconstruct from-video sample_data/video.mp4 -o outputs/video_example

example-images:
	@echo "Running images reconstruction example..."
	@if [ ! -d sample_data/images ]; then \
		echo "Error: sample_data/images/ not found"; \
		echo "Please add sample images to sample_data/images/"; \
		exit 1; \
	fi
	colmap-reconstruct from-images sample_data/images -o outputs/images_example

# Quick start
quickstart: install example-video
	@echo ""
	@echo "Quickstart complete!"
	@echo "Check outputs/video_example/ for results"

