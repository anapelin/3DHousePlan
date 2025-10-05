"""
3D Reconstruction Module for 3D House Plan Pipeline

This module combines video analysis and floor plan data to create accurate 3D models
of rooms and spaces.
"""

import numpy as np
import trimesh
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import cv2

# Import our custom modules
try:
    from .video_processor import VideoFrame, RoomFeatures
    from .plan_parser import FloorPlan, Room
except ImportError:
    from video_processor import VideoFrame, RoomFeatures
    from plan_parser import FloorPlan, Room

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Point3D:
    """Represents a 3D point with coordinates and optional color."""
    x: float
    y: float
    z: float
    color: Optional[Tuple[float, float, float]] = None

@dataclass
class Mesh3D:
    """Represents a 3D mesh with vertices, faces, and materials."""
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    materials: Optional[Dict] = None

@dataclass
class Room3D:
    """Represents a complete 3D room model."""
    name: str
    mesh: Mesh3D
    dimensions: Dict[str, float]
    furniture: List[Mesh3D]
    windows: List[Mesh3D]
    doors: List[Mesh3D]

class ReconstructionEngine:
    """Main class for 3D reconstruction from video and plan data."""
    
    def __init__(self, config: Dict = None):
        """Initialize the reconstruction engine."""
        self.config = config or {}
        
        # Default material properties
        self.materials = {
            'wall': {'color': [0.8, 0.8, 0.8], 'roughness': 0.7},
            'floor': {'color': [0.6, 0.4, 0.2], 'roughness': 0.3},
            'ceiling': {'color': [0.9, 0.9, 0.9], 'roughness': 0.8},
            'window': {'color': [0.7, 0.8, 1.0], 'roughness': 0.1},
            'door': {'color': [0.4, 0.2, 0.1], 'roughness': 0.5}
        }
    
    def reconstruct_room(self, video_frames: List[VideoFrame], 
                        floor_plan: FloorPlan, 
                        room_name: str = "MainRoom") -> Room3D:
        """
        Reconstruct a 3D room model from video frames and floor plan data.
        
        Args:
            video_frames: List of processed video frames
            floor_plan: Parsed floor plan data
            room_name: Name of the room to reconstruct
            
        Returns:
            Complete 3D room model
        """
        logger.info(f"Reconstructing 3D room: {room_name}")
        
        # Extract room features from video
        room_features = self._extract_room_features_from_video(video_frames)
        
        # Get corresponding room from floor plan
        plan_room = self._find_room_in_plan(floor_plan, room_name)
        
        # Create 3D mesh
        room_mesh = self._create_room_mesh(room_features, plan_room)
        
        # Add furniture, windows, and doors
        furniture = self._create_furniture_meshes(room_features, plan_room)
        windows = self._create_window_meshes(plan_room)
        doors = self._create_door_meshes(plan_room)
        
        return Room3D(
            name=room_name,
            mesh=room_mesh,
            dimensions=plan_room.dimensions if plan_room else room_features.room_dimensions,
            furniture=furniture,
            windows=windows,
            doors=doors
        )
    
    def _extract_room_features_from_video(self, video_frames: List[VideoFrame]) -> RoomFeatures:
        """Extract room features from video frames."""
        # This would use the video processor's room feature extraction
        # For now, we'll create a simplified version
        
        all_walls = []
        all_corners = []
        dimensions = {'width': 4.0, 'height': 2.5, 'depth': 4.0}
        
        for frame in video_frames:
            if frame.room_features:
                all_walls.extend(frame.room_features.get('walls', []))
                all_corners.extend(frame.room_features.get('corners', []))
                if 'room_dimensions' in frame.room_features:
                    dims = frame.room_features['room_dimensions']
                    dimensions = {
                        'width': (dimensions['width'] + dims['width']) / 2,
                        'height': (dimensions['height'] + dims['height']) / 2,
                        'depth': (dimensions['depth'] + dims['depth']) / 2
                    }
        
        return RoomFeatures(
            walls=all_walls,
            corners=all_corners,
            floor_area=dimensions['width'] * dimensions['depth'],
            ceiling_height=dimensions['height'],
            windows=[],
            doors=[],
            furniture=[]
        )
    
    def _find_room_in_plan(self, floor_plan: FloorPlan, room_name: str) -> Optional[Room]:
        """Find a specific room in the floor plan."""
        for room in floor_plan.rooms:
            if room.name == room_name:
                return room
        return floor_plan.rooms[0] if floor_plan.rooms else None
    
    def _create_room_mesh(self, room_features: RoomFeatures, plan_room: Optional[Room]) -> Mesh3D:
        """Create the main room mesh (walls, floor, ceiling)."""
        logger.info("Creating room mesh")
        
        # Use plan room dimensions if available, otherwise use video estimates
        if plan_room:
            width = plan_room.dimensions['width']
            length = plan_room.dimensions['length']
            height = plan_room.dimensions['height']
        else:
            width = room_features.room_dimensions['width']
            length = room_features.room_dimensions['depth']
            height = room_features.ceiling_height
        
        # Create room geometry
        vertices, faces = self._create_room_geometry(width, length, height)
        
        # Create normals
        normals = self._calculate_normals(vertices, faces)
        
        # Assign colors based on surface type
        colors = self._assign_surface_colors(vertices, faces)
        
        return Mesh3D(
            vertices=vertices,
            faces=faces,
            normals=normals,
            colors=colors,
            materials=self.materials
        )
    
    def _create_room_geometry(self, width: float, length: float, height: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create the basic room geometry (rectangular room)."""
        # Define vertices for a rectangular room
        # Floor vertices
        floor_vertices = np.array([
            [0, 0, 0],           # Bottom-left
            [width, 0, 0],       # Bottom-right
            [width, length, 0],  # Top-right
            [0, length, 0]       # Top-left
        ])
        
        # Ceiling vertices
        ceiling_vertices = floor_vertices + np.array([0, 0, height])
        
        # Wall vertices
        wall_vertices = np.array([
            # Front wall
            [0, 0, 0], [width, 0, 0], [width, 0, height], [0, 0, height],
            # Right wall
            [width, 0, 0], [width, length, 0], [width, length, height], [width, 0, height],
            # Back wall
            [width, length, 0], [0, length, 0], [0, length, height], [width, length, height],
            # Left wall
            [0, length, 0], [0, 0, 0], [0, 0, height], [0, length, height]
        ])
        
        # Combine all vertices
        vertices = np.vstack([floor_vertices, ceiling_vertices, wall_vertices])
        
        # Define faces (triangles)
        faces = np.array([
            # Floor (2 triangles)
            [0, 1, 2], [0, 2, 3],
            # Ceiling (2 triangles)
            [4, 7, 6], [4, 6, 5],
            # Front wall (2 triangles)
            [8, 9, 10], [8, 10, 11],
            # Right wall (2 triangles)
            [12, 13, 14], [12, 14, 15],
            # Back wall (2 triangles)
            [16, 17, 18], [16, 18, 19],
            # Left wall (2 triangles)
            [20, 21, 22], [20, 22, 23]
        ])
        
        return vertices, faces
    
    def _calculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Calculate vertex normals for the mesh."""
        normals = np.zeros_like(vertices)
        
        for face in faces:
            # Get triangle vertices
            v0, v1, v2 = vertices[face]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            face_normal = face_normal / np.linalg.norm(face_normal)
            
            # Add to vertex normals
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal
        
        # Normalize vertex normals
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] = normals[i] / norm
        
        return normals
    
    def _assign_surface_colors(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Assign colors to vertices based on surface type."""
        colors = np.zeros((len(vertices), 3))
        
        # Floor vertices (first 4)
        colors[0:4] = self.materials['floor']['color']
        
        # Ceiling vertices (next 4)
        colors[4:8] = self.materials['ceiling']['color']
        
        # Wall vertices (remaining)
        colors[8:] = self.materials['wall']['color']
        
        return colors
    
    def _create_furniture_meshes(self, room_features: RoomFeatures, plan_room: Optional[Room]) -> List[Mesh3D]:
        """Create 3D meshes for furniture items."""
        furniture_meshes = []
        
        # This is a simplified implementation
        # In practice, you'd detect furniture from video and create appropriate meshes
        
        if plan_room and plan_room.furniture:
            for furniture_item in plan_room.furniture:
                # Create a simple box mesh for furniture
                mesh = self._create_box_mesh(
                    furniture_item.get('width', 1.0),
                    furniture_item.get('length', 1.0),
                    furniture_item.get('height', 0.8),
                    furniture_item.get('position', [0, 0, 0])
                )
                furniture_meshes.append(mesh)
        
        return furniture_meshes
    
    def _create_window_meshes(self, plan_room: Optional[Room]) -> List[Mesh3D]:
        """Create 3D meshes for windows."""
        window_meshes = []
        
        if plan_room and plan_room.windows:
            for window in plan_room.windows:
                # Create window mesh
                mesh = self._create_window_mesh(window)
                window_meshes.append(mesh)
        
        return window_meshes
    
    def _create_door_meshes(self, plan_room: Optional[Room]) -> List[Mesh3D]:
        """Create 3D meshes for doors."""
        door_meshes = []
        
        if plan_room and plan_room.doors:
            for door in plan_room.doors:
                # Create door mesh
                mesh = self._create_door_mesh(door)
                door_meshes.append(mesh)
        
        return door_meshes
    
    def _create_box_mesh(self, width: float, length: float, height: float, 
                        position: List[float]) -> Mesh3D:
        """Create a simple box mesh."""
        # Create box vertices
        vertices = np.array([
            [0, 0, 0], [width, 0, 0], [width, length, 0], [0, length, 0],  # Bottom
            [0, 0, height], [width, 0, height], [width, length, height], [0, length, height]  # Top
        ])
        
        # Translate to position
        vertices += np.array(position)
        
        # Define faces
        faces = np.array([
            # Bottom
            [0, 1, 2], [0, 2, 3],
            # Top
            [4, 7, 6], [4, 6, 5],
            # Sides
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0]
        ])
        
        normals = self._calculate_normals(vertices, faces)
        colors = np.full((len(vertices), 3), [0.4, 0.2, 0.1])  # Brown color for furniture
        
        return Mesh3D(vertices=vertices, faces=faces, normals=normals, colors=colors)
    
    def _create_window_mesh(self, window_data: Dict) -> Mesh3D:
        """Create a window mesh."""
        # Simplified window creation
        width = window_data.get('width', 1.2)
        height = window_data.get('height', 1.0)
        position = window_data.get('position', [0, 0, 1.0])
        
        return self._create_box_mesh(width, 0.1, height, position)
    
    def _create_door_mesh(self, door_data: Dict) -> Mesh3D:
        """Create a door mesh."""
        # Simplified door creation
        width = door_data.get('width', 0.9)
        height = door_data.get('height', 2.1)
        position = door_data.get('position', [0, 0, 0])
        
        return self._create_box_mesh(width, 0.1, height, position)
    
    def optimize_mesh(self, mesh: Mesh3D) -> Mesh3D:
        """Optimize mesh by removing duplicate vertices and simplifying geometry."""
        logger.info("Optimizing mesh")
        
        # Remove duplicate vertices
        unique_vertices, inverse_indices = np.unique(mesh.vertices, axis=0, return_inverse=True)
        
        # Update face indices
        optimized_faces = inverse_indices[mesh.faces]
        
        # Recalculate normals
        optimized_normals = self._calculate_normals(unique_vertices, optimized_faces)
        
        # Update colors
        if mesh.colors is not None:
            optimized_colors = mesh.colors[inverse_indices]
        else:
            optimized_colors = None
        
        return Mesh3D(
            vertices=unique_vertices,
            faces=optimized_faces,
            normals=optimized_normals,
            colors=optimized_colors,
            materials=mesh.materials
        )
    
    def combine_room_components(self, room3d: Room3D) -> Mesh3D:
        """Combine all room components into a single mesh."""
        logger.info("Combining room components")
        
        all_vertices = [room3d.mesh.vertices]
        all_faces = [room3d.mesh.faces]
        all_colors = [room3d.mesh.colors] if room3d.mesh.colors is not None else []
        
        vertex_offset = len(room3d.mesh.vertices)
        
        # Add furniture
        for furniture in room3d.furniture:
            all_vertices.append(furniture.vertices)
            all_faces.append(furniture.faces + vertex_offset)
            if furniture.colors is not None:
                all_colors.append(furniture.colors)
            vertex_offset += len(furniture.vertices)
        
        # Add windows
        for window in room3d.windows:
            all_vertices.append(window.vertices)
            all_faces.append(window.faces + vertex_offset)
            if window.colors is not None:
                all_colors.append(window.colors)
            vertex_offset += len(window.vertices)
        
        # Add doors
        for door in room3d.doors:
            all_vertices.append(door.vertices)
            all_faces.append(door.faces + vertex_offset)
            if door.colors is not None:
                all_colors.append(door.colors)
            vertex_offset += len(door.vertices)
        
        # Combine all data
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        combined_colors = np.vstack(all_colors) if all_colors else None
        
        # Recalculate normals
        combined_normals = self._calculate_normals(combined_vertices, combined_faces)
        
        return Mesh3D(
            vertices=combined_vertices,
            faces=combined_faces,
            normals=combined_normals,
            colors=combined_colors,
            materials=room3d.mesh.materials
        )
    
    def validate_mesh(self, mesh: Mesh3D) -> List[str]:
        """Validate the 3D mesh for common issues."""
        issues = []
        
        # Check for empty mesh
        if len(mesh.vertices) == 0:
            issues.append("Mesh has no vertices")
        
        if len(mesh.faces) == 0:
            issues.append("Mesh has no faces")
        
        # Check for invalid face indices
        max_vertex_index = len(mesh.vertices) - 1
        for i, face in enumerate(mesh.faces):
            if np.any(face > max_vertex_index) or np.any(face < 0):
                issues.append(f"Face {i} has invalid vertex indices")
        
        # Check for degenerate faces
        for i, face in enumerate(mesh.faces):
            if len(np.unique(face)) < 3:
                issues.append(f"Face {i} is degenerate (has duplicate vertices)")
        
        # Check for non-manifold edges
        # This is a simplified check - in practice, you'd use more sophisticated algorithms
        
        return issues
