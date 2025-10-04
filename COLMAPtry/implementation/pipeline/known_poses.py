"""
Handle known camera poses from ARKit, ARCore, or other sources
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sqlite3

logger = logging.getLogger("colmap_pipeline")


class KnownPosesHandler:
    """Handle import of known camera poses into COLMAP."""
    
    def __init__(self, workspace: Path):
        """
        Initialize known poses handler.
        
        Args:
            workspace: Workspace directory
        """
        self.workspace = workspace
        self.database_path = workspace / "database.db"
    
    def import_poses_from_json(self, poses_file: Path) -> bool:
        """
        Import camera poses from JSON file.
        
        Expected JSON format:
        {
            "images": [
                {
                    "image_name": "frame_000001.jpg",
                    "camera_intrinsics": {
                        "fx": 1000.0,
                        "fy": 1000.0,
                        "cx": 960.0,
                        "cy": 540.0,
                        "k1": 0.0,
                        "k2": 0.0,
                        "p1": 0.0,
                        "p2": 0.0
                    },
                    "camera_pose": {
                        "rotation": [qw, qx, qy, qz],  # quaternion
                        "translation": [tx, ty, tz]
                    }
                }
            ]
        }
        
        Args:
            poses_file: Path to JSON file with poses
            
        Returns:
            Success status
        """
        logger.info(f"Importing poses from JSON: {poses_file}")
        
        with open(poses_file, "r") as f:
            data = json.load(f)
        
        images = data.get("images", [])
        if not images:
            logger.error("No images found in poses file")
            return False
        
        logger.info(f"Found {len(images)} images with poses")
        
        # Create or connect to database
        if not self._initialize_database():
            return False
        
        # Import camera intrinsics and poses
        for img_data in images:
            self._add_image_with_pose(
                img_data["image_name"],
                img_data["camera_intrinsics"],
                img_data["camera_pose"]
            )
        
        logger.info("Successfully imported poses")
        return True
    
    def import_poses_from_csv(self, poses_file: Path) -> bool:
        """
        Import camera poses from CSV file.
        
        Expected CSV format:
        image_name,fx,fy,cx,cy,qw,qx,qy,qz,tx,ty,tz
        
        Args:
            poses_file: Path to CSV file with poses
            
        Returns:
            Success status
        """
        logger.info(f"Importing poses from CSV: {poses_file}")
        
        images = []
        
        with open(poses_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_data = {
                    "image_name": row["image_name"],
                    "camera_intrinsics": {
                        "fx": float(row["fx"]),
                        "fy": float(row["fy"]),
                        "cx": float(row["cx"]),
                        "cy": float(row["cy"]),
                        "k1": float(row.get("k1", 0.0)),
                        "k2": float(row.get("k2", 0.0)),
                        "p1": float(row.get("p1", 0.0)),
                        "p2": float(row.get("p2", 0.0)),
                    },
                    "camera_pose": {
                        "rotation": [
                            float(row["qw"]),
                            float(row["qx"]),
                            float(row["qy"]),
                            float(row["qz"])
                        ],
                        "translation": [
                            float(row["tx"]),
                            float(row["ty"]),
                            float(row["tz"])
                        ]
                    }
                }
                images.append(img_data)
        
        logger.info(f"Found {len(images)} images with poses")
        
        # Create or connect to database
        if not self._initialize_database():
            return False
        
        # Import camera intrinsics and poses
        for img_data in images:
            self._add_image_with_pose(
                img_data["image_name"],
                img_data["camera_intrinsics"],
                img_data["camera_pose"]
            )
        
        logger.info("Successfully imported poses")
        return True
    
    def _initialize_database(self) -> bool:
        """
        Initialize COLMAP database with cameras and images tables.
        
        Returns:
            Success status
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create cameras table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    model INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    params BLOB,
                    prior_focal_length INTEGER NOT NULL
                )
            """)
            
            # Create images table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    name TEXT NOT NULL UNIQUE,
                    camera_id INTEGER NOT NULL,
                    prior_qw REAL,
                    prior_qx REAL,
                    prior_qy REAL,
                    prior_qz REAL,
                    prior_tx REAL,
                    prior_ty REAL,
                    prior_tz REAL,
                    CONSTRAINT image_name_unique UNIQUE (name),
                    FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def _add_image_with_pose(
        self,
        image_name: str,
        intrinsics: Dict,
        pose: Dict
    ):
        """
        Add image with known pose to database.
        
        Args:
            image_name: Image filename
            intrinsics: Camera intrinsics dict
            pose: Camera pose dict (rotation quaternion + translation)
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Add camera (or get existing)
            # COLMAP camera model 4 = OPENCV (fx, fy, cx, cy, k1, k2, p1, p2)
            camera_params = np.array([
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["cx"],
                intrinsics["cy"],
                intrinsics.get("k1", 0.0),
                intrinsics.get("k2", 0.0),
                intrinsics.get("p1", 0.0),
                intrinsics.get("p2", 0.0),
            ])
            
            # Assume 1920x1080 if not specified
            width = intrinsics.get("width", 1920)
            height = intrinsics.get("height", 1080)
            
            cursor.execute("""
                INSERT OR IGNORE INTO cameras (model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?)
            """, (4, width, height, camera_params.tobytes(), 1))
            
            camera_id = cursor.lastrowid
            if camera_id == 0:
                # Camera already exists, get its ID
                cursor.execute("SELECT camera_id FROM cameras WHERE params = ?", 
                             (camera_params.tobytes(),))
                camera_id = cursor.fetchone()[0]
            
            # Add image with pose
            rotation = pose["rotation"]
            translation = pose["translation"]
            
            cursor.execute("""
                INSERT OR REPLACE INTO images 
                (name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_name,
                camera_id,
                rotation[0],  # qw
                rotation[1],  # qx
                rotation[2],  # qy
                rotation[3],  # qz
                translation[0],  # tx
                translation[1],  # ty
                translation[2],  # tz
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to add image {image_name}: {e}")
    
    def convert_arkit_poses(self, arkit_file: Path, output_json: Path) -> bool:
        """
        Convert ARKit pose data to standard JSON format.
        
        ARKit provides 4x4 transformation matrices.
        
        Args:
            arkit_file: Path to ARKit data file
            output_json: Path to output JSON file
            
        Returns:
            Success status
        """
        logger.info("Converting ARKit poses...")
        
        # This is a placeholder - actual ARKit format depends on export method
        # Typically ARKit provides 4x4 matrices that need to be decomposed
        
        try:
            with open(arkit_file, "r") as f:
                arkit_data = json.load(f)
            
            output_data = {"images": []}
            
            for frame in arkit_data.get("frames", []):
                # Extract transformation matrix
                matrix = np.array(frame["transform"]).reshape(4, 4)
                
                # Decompose into rotation (quaternion) and translation
                rotation_matrix = matrix[:3, :3]
                translation = matrix[:3, 3]
                
                quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
                
                # Extract intrinsics
                intrinsics = frame.get("intrinsics", {})
                
                output_data["images"].append({
                    "image_name": frame["image_name"],
                    "camera_intrinsics": intrinsics,
                    "camera_pose": {
                        "rotation": quaternion.tolist(),
                        "translation": translation.tolist()
                    }
                })
            
            # Save to JSON
            with open(output_json, "w") as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Converted {len(output_data['images'])} ARKit poses")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert ARKit poses: {e}")
            return False
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion (w, x, y, z).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion as numpy array [w, x, y, z]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def triangulate_points(self) -> bool:
        """
        Run COLMAP point triangulator to fill in 3D structure from known poses.
        
        Returns:
            Success status
        """
        logger.info("Triangulating 3D points from known poses...")
        
        sparse_path = self.workspace / "sparse" / "0"
        sparse_path.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "colmap", "point_triangulator",
            "--database_path", str(self.database_path),
            "--image_path", str(self.workspace / "images"),
            "--input_path", str(sparse_path),
            "--output_path", str(sparse_path),
        ]
        
        try:
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Point triangulation complete")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Point triangulation failed: {e.stderr}")
            return False

