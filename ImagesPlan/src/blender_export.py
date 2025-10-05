"""
Blender Export Module for 3D House Plan Pipeline

This module handles exporting 3D models to Blender in various formats
including OBJ, FBX, and direct Blender API integration.
"""

import numpy as np
import trimesh
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import json

# Import our custom modules
try:
    from .reconstruction import Mesh3D, Room3D
except ImportError:
    from reconstruction import Mesh3D, Room3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlenderExporter:
    """Main class for exporting 3D models to Blender-compatible formats."""
    
    def __init__(self, config: Dict = None):
        """Initialize the Blender exporter."""
        self.config = config or {}
        
        # Supported export formats
        self.supported_formats = ['obj', 'fbx', 'ply', 'stl', 'blend']
        
        # Default export settings
        self.export_settings = {
            'obj': {
                'include_materials': True,
                'include_normals': True,
                'include_textures': False
            },
            'fbx': {
                'include_materials': True,
                'include_animations': False,
                'scale_factor': 1.0
            },
            'ply': {
                'include_colors': True,
                'include_normals': True
            },
            'stl': {
                'ascii': False,
                'include_normals': True
            }
        }
    
    def export_room(self, room3d: Room3D, output_path: str, 
                   format: str = 'obj', options: Dict = None) -> bool:
        """
        Export a 3D room model to the specified format.
        
        Args:
            room3d: 3D room model to export
            output_path: Output file path
            format: Export format ('obj', 'fbx', 'ply', 'stl', 'blend')
            options: Additional export options
            
        Returns:
            True if export was successful, False otherwise
        """
        logger.info(f"Exporting room '{room3d.name}' to {format.upper()} format")
        
        if format.lower() not in self.supported_formats:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        # Merge all room components into a single mesh
        combined_mesh = self._combine_room_mesh(room3d)
        
        # Apply export options
        export_options = self._merge_export_options(format, options)
        
        try:
            if format.lower() == 'obj':
                return self._export_obj(combined_mesh, output_path, export_options)
            elif format.lower() == 'fbx':
                return self._export_fbx(combined_mesh, output_path, export_options)
            elif format.lower() == 'ply':
                return self._export_ply(combined_mesh, output_path, export_options)
            elif format.lower() == 'stl':
                return self._export_stl(combined_mesh, output_path, export_options)
            elif format.lower() == 'blend':
                return self._export_blend(room3d, output_path, export_options)
            else:
                logger.error(f"Export format {format} not implemented")
                return False
                
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    def _combine_room_mesh(self, room3d: Room3D) -> Mesh3D:
        """Combine all room components into a single mesh."""
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
        
        return Mesh3D(
            vertices=combined_vertices,
            faces=combined_faces,
            normals=room3d.mesh.normals,
            colors=combined_colors,
            materials=room3d.mesh.materials
        )
    
    def _merge_export_options(self, format: str, options: Dict = None) -> Dict:
        """Merge default export options with user-provided options."""
        default_options = self.export_settings.get(format.lower(), {})
        if options:
            default_options.update(options)
        return default_options
    
    def _export_obj(self, mesh: Mesh3D, output_path: str, options: Dict) -> bool:
        """Export mesh to OBJ format."""
        logger.info("Exporting to OBJ format")
        
        try:
            # Create trimesh object
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.normals,
                vertex_colors=mesh.colors
            )
            
            # Export to OBJ
            trimesh_mesh.export(output_path, file_type='obj')
            
            # Create material file if colors are present
            if options.get('include_materials', True) and mesh.colors is not None:
                self._create_mtl_file(output_path, mesh)
            
            logger.info(f"OBJ export successful: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"OBJ export failed: {str(e)}")
            return False
    
    def _export_fbx(self, mesh: Mesh3D, output_path: str, options: Dict) -> bool:
        """Export mesh to FBX format."""
        logger.info("Exporting to FBX format")
        
        try:
            # Create trimesh object
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.normals,
                vertex_colors=mesh.colors
            )
            
            # Apply scale factor if specified
            scale_factor = options.get('scale_factor', 1.0)
            if scale_factor != 1.0:
                trimesh_mesh.vertices *= scale_factor
            
            # Export to FBX
            trimesh_mesh.export(output_path, file_type='fbx')
            
            logger.info(f"FBX export successful: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"FBX export failed: {str(e)}")
            return False
    
    def _export_ply(self, mesh: Mesh3D, output_path: str, options: Dict) -> bool:
        """Export mesh to PLY format."""
        logger.info("Exporting to PLY format")
        
        try:
            # Create trimesh object
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.normals,
                vertex_colors=mesh.colors
            )
            
            # Export to PLY
            trimesh_mesh.export(output_path, file_type='ply')
            
            logger.info(f"PLY export successful: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"PLY export failed: {str(e)}")
            return False
    
    def _export_stl(self, mesh: Mesh3D, output_path: str, options: Dict) -> bool:
        """Export mesh to STL format."""
        logger.info("Exporting to STL format")
        
        try:
            # Create trimesh object
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_normals=mesh.normals
            )
            
            # Export to STL
            trimesh_mesh.export(output_path, file_type='stl')
            
            logger.info(f"STL export successful: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"STL export failed: {str(e)}")
            return False
    
    def _export_blend(self, room3d: Room3D, output_path: str, options: Dict) -> bool:
        """Export to Blender format using Blender Python API."""
        logger.info("Exporting to Blender format")
        
        try:
            # This would use the Blender Python API (bpy) to create a .blend file
            # For now, we'll create a script that can be run in Blender
            
            script_content = self._generate_blender_script(room3d)
            script_path = output_path.replace('.blend', '_import_script.py')
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            logger.info(f"Blender import script created: {script_path}")
            logger.info("Run this script in Blender to import the room model")
            
            return True
            
        except Exception as e:
            logger.error(f"Blender export failed: {str(e)}")
            return False
    
    def _create_mtl_file(self, obj_path: str, mesh: Mesh3D) -> None:
        """Create a material file (.mtl) for OBJ export."""
        mtl_path = obj_path.replace('.obj', '.mtl')
        
        with open(mtl_path, 'w') as f:
            f.write("# Material file generated by 3D House Plan Pipeline\n")
            f.write("newmtl default_material\n")
            f.write("Ka 0.2 0.2 0.2\n")  # Ambient color
            f.write("Kd 0.8 0.8 0.8\n")  # Diffuse color
            f.write("Ks 0.0 0.0 0.0\n")  # Specular color
            f.write("Ns 0.0\n")          # Specular exponent
            f.write("d 1.0\n")           # Transparency
            f.write("illum 2\n")         # Illumination model
        
        logger.info(f"Material file created: {mtl_path}")
    
    def _generate_blender_script(self, room3d: Room3D) -> str:
        """Generate a Blender Python script to import the room model."""
        script = f'''
import bpy
import bmesh
import mathutils
from mathutils import Vector

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create new mesh
mesh = bpy.data.meshes.new(name="{room3d.name}")
obj = bpy.data.objects.new("{room3d.name}", mesh)

# Link object to scene
bpy.context.collection.objects.link(obj)

# Create bmesh instance
bm = bmesh.new()

# Add vertices
vertices = {room3d.mesh.vertices.tolist()}
for vertex in vertices:
    bm.verts.new(vertex)

# Update bmesh
bm.verts.ensure_lookup_table()

# Add faces
faces = {room3d.mesh.faces.tolist()}
for face_indices in faces:
    try:
        face_verts = [bm.verts[i] for i in face_indices]
        bm.faces.new(face_verts)
    except ValueError:
        # Skip invalid faces
        pass

# Update mesh
bm.to_mesh(mesh)
bm.free()

# Add materials
if mesh.materials:
    for i, material_data in enumerate(mesh.materials.items()):
        mat = bpy.data.materials.new(name=material_data[0])
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
            material_data[1]['color'][0],
            material_data[1]['color'][1], 
            material_data[1]['color'][2],
            1.0
        )
        mesh.materials.append(mat)

# Set object as active
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# Add room dimensions as text
dimensions = {room3d.dimensions}
text_data = bpy.data.curves.new(type="FONT", name="RoomDimensions")
text_obj = bpy.data.objects.new("RoomDimensions", text_data)
text_data.body = f"Room: {{dimensions['width']:.2f}}m x {{dimensions['length']:.2f}}m x {{dimensions['height']:.2f}}m"
text_obj.location = (0, 0, dimensions['height'] + 0.5)
bpy.context.collection.objects.link(text_obj)

print("Room model imported successfully!")
print(f"Room dimensions: {{dimensions['width']:.2f}}m x {{dimensions['length']:.2f}}m x {{dimensions['height']:.2f}}m")
'''
        return script
    
    def export_multiple_rooms(self, rooms: List[Room3D], output_dir: str, 
                            format: str = 'obj', options: Dict = None) -> Dict[str, bool]:
        """
        Export multiple rooms to separate files.
        
        Args:
            rooms: List of 3D room models
            output_dir: Output directory path
            format: Export format
            options: Export options
            
        Returns:
            Dictionary mapping room names to export success status
        """
        logger.info(f"Exporting {len(rooms)} rooms to {output_dir}")
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for room in rooms:
            room_output_path = output_path / f"{room.name}.{format}"
            success = self.export_room(room, str(room_output_path), format, options)
            results[room.name] = success
        
        successful_exports = sum(1 for success in results.values() if success)
        logger.info(f"Successfully exported {successful_exports}/{len(rooms)} rooms")
        
        return results
    
    def create_blender_project(self, rooms: List[Room3D], output_path: str) -> bool:
        """
        Create a complete Blender project file with all rooms.
        
        Args:
            rooms: List of 3D room models
            output_path: Output .blend file path
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating complete Blender project")
        
        try:
            # Generate comprehensive Blender script
            script_content = self._generate_complete_blender_script(rooms)
            script_path = output_path.replace('.blend', '_complete_import.py')
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            logger.info(f"Complete Blender project script created: {script_path}")
            return True
            
        except Exception as e:
            logger.error(f"Blender project creation failed: {str(e)}")
            return False
    
    def _generate_complete_blender_script(self, rooms: List[Room3D]) -> str:
        """Generate a complete Blender script for multiple rooms."""
        script = '''
import bpy
import bmesh
import mathutils
from mathutils import Vector

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Create collection for rooms
room_collection = bpy.data.collections.new("Rooms")
bpy.context.scene.collection.children.link(room_collection)

'''
        
        for i, room in enumerate(rooms):
            script += f'''
# Create room: {room.name}
mesh_{i} = bpy.data.meshes.new(name="{room.name}")
obj_{i} = bpy.data.objects.new("{room.name}", mesh_{i})

# Link object to room collection
room_collection.objects.link(obj_{i})

# Create bmesh instance
bm_{i} = bmesh.new()

# Add vertices
vertices_{i} = {room.mesh.vertices.tolist()}
for vertex in vertices_{i}:
    bm_{i}.verts.new(vertex)

# Update bmesh
bm_{i}.verts.ensure_lookup_table()

# Add faces
faces_{i} = {room.mesh.faces.tolist()}
for face_indices in faces_{i}:
    try:
        face_verts = [bm_{i}.verts[j] for j in face_indices]
        bm_{i}.faces.new(face_verts)
    except ValueError:
        # Skip invalid faces
        pass

# Update mesh
bm_{i}.to_mesh(mesh_{i})
bm_{i}.free()

# Position room (offset each room)
obj_{i}.location = ({i * 10}, 0, 0)

'''
        
        script += '''
# Set up camera and lighting
bpy.ops.object.camera_add(location=(10, -10, 5))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
sun = bpy.context.object
sun.data.energy = 3

# Set camera as active camera
bpy.context.scene.camera = camera

print("Complete room model imported successfully!")
'''
        
        return script
    
    def validate_export(self, output_path: str, format: str) -> List[str]:
        """Validate exported file for common issues."""
        issues = []
        
        if not Path(output_path).exists():
            issues.append(f"Export file does not exist: {output_path}")
            return issues
        
        file_size = Path(output_path).stat().st_size
        if file_size == 0:
            issues.append("Export file is empty")
        
        if format.lower() == 'obj':
            # Check if OBJ file has required sections
            with open(output_path, 'r') as f:
                content = f.read()
                if 'v ' not in content:
                    issues.append("OBJ file missing vertices")
                if 'f ' not in content:
                    issues.append("OBJ file missing faces")
        
        return issues
