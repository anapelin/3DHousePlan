"""
Room Plan Parser Module for 3D House Plan Pipeline

This module handles parsing of architectural drawings, floor plans, and room layouts
to extract dimensional and structural information.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from dataclasses import dataclass
from pathlib import Path
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
from PIL import Image
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Room:
    """Represents a room with its properties and dimensions."""
    name: str
    area: float
    dimensions: Dict[str, float]  # width, length, height
    corners: List[Tuple[float, float]]
    walls: List[Dict]
    windows: List[Dict]
    doors: List[Dict]
    furniture: List[Dict]

@dataclass
class FloorPlan:
    """Represents a complete floor plan with multiple rooms."""
    rooms: List[Room]
    total_area: float
    scale_factor: float  # pixels per meter
    image_dimensions: Tuple[int, int]

class PlanParser:
    """Main class for parsing architectural plans and floor layouts."""
    
    def __init__(self, config: Dict = None):
        """Initialize the plan parser with configuration."""
        self.config = config or {}
        
        # Common architectural symbols and patterns
        self.wall_thickness = 0.2  # meters, standard wall thickness
        self.door_width = 0.9      # meters, standard door width
        self.window_width = 1.2    # meters, standard window width
        
    def parse_plan(self, plan_path: str) -> FloorPlan:
        """
        Parse a floor plan from various formats (PDF, image, etc.).
        
        Args:
            plan_path: Path to the floor plan file
            
        Returns:
            FloorPlan object with extracted room information
        """
        logger.info(f"Parsing floor plan: {plan_path}")
        
        file_extension = Path(plan_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self._parse_pdf_plan(plan_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return self._parse_image_plan(plan_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _parse_pdf_plan(self, pdf_path: str) -> FloorPlan:
        """Parse a PDF floor plan."""
        logger.info("Parsing PDF floor plan")
        
        if fitz is None:
            logger.warning("PyMuPDF not available, treating PDF as image")
            # Fallback: try to convert PDF to image using PIL
            from PIL import Image
            try:
                img = Image.open(pdf_path)
                image = np.array(img.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except:
                raise ValueError("Cannot parse PDF without PyMuPDF. Please install: pip install PyMuPDF")
        else:
            # Convert PDF to image
            doc = fitz.open(pdf_path)
            page = doc[0]  # Get first page
            mat = fitz.Matrix(2.0, 2.0)  # Scale up for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            doc.close()
        
        return self._parse_image_plan_from_array(image)
    
    def _parse_image_plan(self, image_path: str) -> FloorPlan:
        """Parse an image floor plan."""
        logger.info("Parsing image floor plan")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return self._parse_image_plan_from_array(image)
    
    def _parse_image_plan_from_array(self, image: np.ndarray) -> FloorPlan:
        """Parse floor plan from image array."""
        # Preprocess the image
        processed_image = self._preprocess_plan_image(image)
        
        # Extract walls
        walls = self._extract_walls(processed_image)
        
        # Extract rooms
        rooms = self._extract_rooms(processed_image, walls)
        
        # Calculate scale factor
        scale_factor = self._estimate_scale_factor(image, rooms)
        
        # Calculate total area
        total_area = sum(room.area for room in rooms)
        
        return FloorPlan(
            rooms=rooms,
            total_area=total_area,
            scale_factor=scale_factor,
            image_dimensions=(image.shape[1], image.shape[0])
        )
    
    def _preprocess_plan_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the floor plan image for better feature extraction."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_walls(self, image: np.ndarray) -> List[Dict]:
        """Extract wall lines from the floor plan."""
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate wall properties
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle = np.arctan2(y2 - y1, x2 - x1)
                
                walls.append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length,
                    'angle': angle,
                    'thickness': self.wall_thickness
                })
        
        # Merge similar walls
        walls = self._merge_similar_walls(walls)
        
        return walls
    
    def _merge_similar_walls(self, walls: List[Dict]) -> List[Dict]:
        """Merge walls that are similar in position and angle."""
        if not walls:
            return []
        
        merged_walls = []
        angle_threshold = np.pi / 18  # 10 degrees
        distance_threshold = 20  # pixels
        
        for wall in walls:
            is_merged = False
            
            for merged_wall in merged_walls:
                # Check if walls are similar
                angle_diff = abs(wall['angle'] - merged_wall['angle'])
                if angle_diff > np.pi / 2:
                    angle_diff = np.pi - angle_diff
                
                # Check distance between wall centers
                wall_center = ((wall['start'][0] + wall['end'][0]) / 2,
                              (wall['start'][1] + wall['end'][1]) / 2)
                merged_center = ((merged_wall['start'][0] + merged_wall['end'][0]) / 2,
                                (merged_wall['start'][1] + merged_wall['end'][1]) / 2)
                
                distance = np.sqrt((wall_center[0] - merged_center[0])**2 + 
                                 (wall_center[1] - merged_center[1])**2)
                
                if angle_diff < angle_threshold and distance < distance_threshold:
                    # Merge walls
                    merged_wall['start'] = wall['start']
                    merged_wall['end'] = wall['end']
                    merged_wall['length'] = max(wall['length'], merged_wall['length'])
                    is_merged = True
                    break
            
            if not is_merged:
                merged_walls.append(wall)
        
        return merged_walls
    
    def _extract_rooms(self, image: np.ndarray, walls: List[Dict]) -> List[Room]:
        """Extract individual rooms from the floor plan."""
        # Create a mask of wall areas
        wall_mask = np.zeros(image.shape, dtype=np.uint8)
        
        for wall in walls:
            cv2.line(wall_mask, wall['start'], wall['end'], 255, 3)
        
        # Find contours to identify room areas
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rooms = []
        for i, contour in enumerate(contours):
            # Filter out small contours (likely noise)
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum room area in pixels
                continue
            
            # Get room corners
            corners = self._get_room_corners(contour)
            
            # Estimate room dimensions
            dimensions = self._estimate_room_dimensions(corners)
            
            # Create room object
            room = Room(
                name=f"Room_{i+1}",
                area=area,
                dimensions=dimensions,
                corners=corners,
                walls=self._get_room_walls(corners, walls),
                windows=[],  # Would be detected in more sophisticated implementation
                doors=[],    # Would be detected in more sophisticated implementation
                furniture=[] # Would be detected in more sophisticated implementation
            )
            
            rooms.append(room)
        
        return rooms
    
    def _get_room_corners(self, contour: np.ndarray) -> List[Tuple[float, float]]:
        """Extract corners from room contour."""
        # Approximate the contour to get corners
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        corners = []
        for point in approx:
            x, y = point[0]
            corners.append((float(x), float(y)))
        
        return corners
    
    def _estimate_room_dimensions(self, corners: List[Tuple[float, float]]) -> Dict[str, float]:
        """Estimate room dimensions from corner points."""
        if len(corners) < 4:
            return {'width': 3.0, 'length': 3.0, 'height': 2.5}
        
        # Calculate bounding box
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        
        width = max(x_coords) - min(x_coords)
        length = max(y_coords) - min(y_coords)
        
        return {
            'width': width,
            'length': length,
            'height': 2.5  # Standard ceiling height
        }
    
    def _get_room_walls(self, corners: List[Tuple[float, float]], all_walls: List[Dict]) -> List[Dict]:
        """Get walls that belong to a specific room."""
        room_walls = []
        
        for wall in all_walls:
            # Check if wall is close to room corners
            for corner in corners:
                wall_start = wall['start']
                wall_end = wall['end']
                
                # Calculate distance from corner to wall line
                distance = self._point_to_line_distance(corner, wall_start, wall_end)
                
                if distance < 50:  # Threshold in pixels
                    room_walls.append(wall)
                    break
        
        return room_walls
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> float:
        """Calculate distance from a point to a line."""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate distance using point-to-line formula
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        if denominator == 0:
            return float('inf')
        
        return numerator / denominator
    
    def _estimate_scale_factor(self, image: np.ndarray, rooms: List[Room]) -> float:
        """Estimate the scale factor (pixels per meter) from the floor plan."""
        if not rooms:
            return 100.0  # Default scale factor
        
        # Use the largest room to estimate scale
        largest_room = max(rooms, key=lambda r: r.area)
        
        # Assume average room size is 20 square meters
        # This is a heuristic - in practice, you'd use known dimensions
        expected_area_sqm = 20.0
        actual_area_pixels = largest_room.area
        
        # Calculate scale factor
        scale_factor = np.sqrt(actual_area_pixels / expected_area_sqm)
        
        return scale_factor
    
    def detect_doors_and_windows(self, image: np.ndarray, walls: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect doors and windows in the floor plan.
        This is a simplified implementation.
        """
        doors = []
        windows = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangular shapes that could be doors/windows
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to rectangle
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            
            # Filter by size (doors are typically wider than tall)
            if width > height and 20 < width < 100 and 5 < height < 30:
                doors.append({
                    'center': rect[0],
                    'size': (width, height),
                    'angle': rect[2]
                })
            elif height > width and 10 < width < 50 and 10 < height < 50:
                windows.append({
                    'center': rect[0],
                    'size': (width, height),
                    'angle': rect[2]
                })
        
        return doors, windows
    
    def export_plan_data(self, floor_plan: FloorPlan, output_path: str) -> None:
        """Export parsed floor plan data to JSON format."""
        data = {
            'rooms': [],
            'total_area': floor_plan.total_area,
            'scale_factor': floor_plan.scale_factor,
            'image_dimensions': floor_plan.image_dimensions
        }
        
        for room in floor_plan.rooms:
            room_data = {
                'name': room.name,
                'area': room.area,
                'dimensions': room.dimensions,
                'corners': room.corners,
                'walls': room.walls,
                'windows': room.windows,
                'doors': room.doors,
                'furniture': room.furniture
            }
            data['rooms'].append(room_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Floor plan data exported to: {output_path}")
    
    def validate_plan(self, floor_plan: FloorPlan) -> List[str]:
        """Validate the parsed floor plan for common issues."""
        issues = []
        
        # Check for empty rooms
        if not floor_plan.rooms:
            issues.append("No rooms detected in the floor plan")
        
        # Check for unrealistic dimensions
        for room in floor_plan.rooms:
            if room.dimensions['width'] < 1.0 or room.dimensions['length'] < 1.0:
                issues.append(f"Room {room.name} has unrealistic dimensions")
            
            if room.area < 5.0:  # Less than 5 square meters
                issues.append(f"Room {room.name} is too small")
        
        # Check scale factor
        if floor_plan.scale_factor < 10 or floor_plan.scale_factor > 1000:
            issues.append("Scale factor seems unrealistic")
        
        return issues
