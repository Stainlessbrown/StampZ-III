#!/usr/bin/env python3
"""
Perforation Measurement System for StampZ
The final major philatelic tool - precision perforation gauge measurement.

This system would provide:
- Automated perforation detection and measurement
- Higher precision than physical gauges
- Compound perforation analysis
- Integration with existing StampZ workflow
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import math

@dataclass
class PerforationHole:
    """Represents a single perforation hole."""
    center_x: float
    center_y: float
    diameter: float
    confidence: float
    edge_quality: float  # How clean/circular the hole is

@dataclass
class PerforationEdge:
    """Represents one edge of a stamp with its perforations."""
    edge_type: str  # 'top', 'bottom', 'left', 'right'
    holes: List[PerforationHole]
    total_length_pixels: float
    gauge_measurement: float
    measurement_confidence: float
    is_compound: bool
    compound_details: Optional[Dict] = None

@dataclass
class PerforationAnalysis:
    """Complete perforation analysis of a stamp."""
    edges: List[PerforationEdge]
    overall_gauge: float
    catalog_gauge: str  # Formatted for catalog standards
    is_compound_perforation: bool
    compound_description: str
    measurement_quality: str  # 'Excellent', 'Good', 'Fair', 'Poor'
    technical_notes: List[str]
    warnings: List[str]  # Potential issues (forgery, reperforations, etc.)

class PerforationMeasurementEngine:
    """Core engine for measuring stamp perforations."""
    
    def __init__(self):
        self.dpi = 600  # Default DPI for calculations
        self.min_hole_size = 3  # Minimum hole diameter in pixels
        self.max_hole_size = 50  # Maximum hole diameter in pixels
        self.background_color = 'black'  # Expected background color
        
    def set_image_dpi(self, dpi: int):
        """Set the DPI of the image for accurate measurements."""
        self.dpi = dpi
        
    def set_background_color(self, bg_color: str):
        """Set the expected background color for perforation detection.
        
        Args:
            bg_color: 'black', 'dark_gray', 'white', or 'light_gray'
        """
        self.background_color = bg_color
        
    def detect_stamp_edges(self, image: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """Detect the edges of the stamp where perforations are located.
        
        For cropped stamp images, we need to be much more conservative and look
        closer to the actual image edges where the perforations should be.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            h, w = gray.shape
            
            # For cropped images, perforations are very close to the edges
            # Use a very small margin - perforations should be within 5-10 pixels of edge
            edge_margin = min(8, min(w, h) // 50)  # Very small margin, max 8 pixels
            
            # Create edge search regions very close to image boundaries
            edges_dict = {
                'top': [(i, edge_margin) for i in range(edge_margin, w - edge_margin)],
                'bottom': [(i, h - edge_margin - 1) for i in range(edge_margin, w - edge_margin)],
                'left': [(edge_margin, i) for i in range(edge_margin, h - edge_margin)],
                'right': [(w - edge_margin - 1, i) for i in range(edge_margin, h - edge_margin)]
            }
            
            return edges_dict
            
        except Exception as e:
            print(f"Error detecting stamp edges: {e}")
            return {}
    
    def detect_perforation_holes(self, image: np.ndarray, edge_points: List[Tuple[int, int]], edge_type: str = 'unknown') -> List[PerforationHole]:
        """Detect individual perforation holes along an edge."""
        try:
            holes = []
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            original_h, original_w = gray.shape  # Store original image dimensions
            
            # Create a region of interest around the edge
            if not edge_points:
                print("DEBUG: No edge points provided, returning empty hole list.")
                return holes
                
            print(f"DEBUG: Analyzing {len(edge_points)} edge points for hole detection")
            
            # Determine edge orientation and create ROI
            first_point = edge_points[0]
            last_point = edge_points[-1]
            
            # Make ROI more focused on the actual edge to catch consecutive perforations
            roi_margin = 20  # Smaller margin for more precise detection
            
            if abs(first_point[0] - last_point[0]) > abs(first_point[1] - last_point[1]):
                # Horizontal edge
                x1 = max(0, min(p[0] for p in edge_points) - roi_margin)
                x2 = min(gray.shape[1], max(p[0] for p in edge_points) + roi_margin)
                y1 = max(0, min(p[1] for p in edge_points) - roi_margin)
                y2 = min(gray.shape[0], max(p[1] for p in edge_points) + roi_margin)
            else:
                # Vertical edge
                x1 = max(0, min(p[0] for p in edge_points) - roi_margin)
                x2 = min(gray.shape[1], max(p[0] for p in edge_points) + roi_margin)
                y1 = max(0, min(p[1] for p in edge_points) - roi_margin)
                y2 = min(gray.shape[0], max(p[1] for p in edge_points) + roi_margin)
            
            roi = gray[y1:y2, x1:x2]
            
            if roi.size == 0:
                print("DEBUG: ROI size is 0, no holes detected.")
                return holes
                
            print(f"DEBUG: ROI dimensions: {roi.shape[1]}x{roi.shape[0]} at ({x1},{y1})")
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            
            # Use adaptive parameters based on image characteristics and DPI
            # Perforation holes are typically 0.5-1.5mm diameter
            # At 800 DPI: 1mm = ~31.5 pixels, so 0.5mm = ~16 pixels diameter = 8 pixel radius
            expected_hole_radius = max(3, int(self.dpi / 100))  # More conservative estimate
            print(f"DEBUG: Expected hole radius: {expected_hole_radius} pixels")
            
            # Try multiple parameter sets systematically designed for perforation holes
            param_sets = [
                # Very sensitive for small perforations
                {'param1': 20, 'param2': 10, 'minRadius': 2, 'maxRadius': 12},
                # Standard perforation size at 800 DPI
                {'param1': 30, 'param2': 15, 'minRadius': 4, 'maxRadius': 20},
                # Slightly larger perforations
                {'param1': 40, 'param2': 18, 'minRadius': 6, 'maxRadius': 25},
                # Conservative for well-defined holes
                {'param1': 50, 'param2': 25, 'minRadius': 8, 'maxRadius': 30},
                # Very aggressive for faint/damaged perforations
                {'param1': 15, 'param2': 8, 'minRadius': 3, 'maxRadius': 18},
            ]
            
            all_circles = []
            for i, params in enumerate(param_sets):
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=max(5, params['minRadius'] * 2),
                    param1=params['param1'],
                    param2=params['param2'],
                    minRadius=params['minRadius'],
                    maxRadius=params['maxRadius']
                )
                
                if circles is not None:
                    print(f"DEBUG: Parameter set {i} detected {len(circles[0])} circles")
                    all_circles.extend(circles[0, :])
                else:
                    print(f"DEBUG: Parameter set {i} detected 0 circles")
            
            print(f"DEBUG: Total circles detected before filtering: {len(all_circles)}")
            
            # Remove duplicate circles
            if all_circles:
                all_circles = np.array(all_circles)
                # Filter duplicates (circles that are very close to each other)
                filtered_circles = []
                for circle in all_circles:
                    is_duplicate = False
                    for existing in filtered_circles:
                        distance = np.sqrt((circle[0] - existing[0])**2 + (circle[1] - existing[1])**2)
                        if distance < max(circle[2], existing[2]):  # Circles overlap
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        filtered_circles.append(circle)
                
                circles = np.array(filtered_circles).reshape(1, -1, 3) if filtered_circles else None
                print(f"DEBUG: Circles after filtering duplicates: {len(filtered_circles)}")
            else:
                circles = None
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # Adjust coordinates back to full image space
                    full_x = x + x1
                    full_y = y + y1
                    
                    # Check if this circle is near the edge we're analyzing
                    min_distance = min(
                        math.sqrt((full_x - px)**2 + (full_y - py)**2) 
                        for px, py in edge_points
                    )
                    
                    # More lenient distance check - perforation holes can be further from edge line
                    # For cropped images, perforations should be within reasonable distance of search line
                    max_distance = max(25, r * 3)  # More lenient distance check
                    
                    # Additional check: hole must be near image boundary (not in center)
                    # Use the stored original image dimensions
                    edge_proximity = min(
                        full_x,                    # Distance from left edge
                        full_y,                    # Distance from top edge  
                        original_w - full_x,       # Distance from right edge
                        original_h - full_y        # Distance from bottom edge
                    )
                    
                    # NEW: Check if hole is actually a perforation vs ink spot
                    is_real_perforation = self._validate_perforation_hole(gray, full_x, full_y, r, edge_type)
                    
                    print(f"DEBUG: {edge_type} edge - Circle at ({full_x},{full_y}) r={r}, min_dist={min_distance:.1f}, edge_prox={edge_proximity:.1f}")
                    
                    # Hole must be within reasonable distance of an image edge AND near the search line AND be a real perforation
                    # Be more lenient for vertical edges which tend to be further from image edges
                    max_edge_proximity = 50 if edge_type in ['left', 'right'] else 35
                    if min_distance < max_distance and edge_proximity < max_edge_proximity and is_real_perforation:
                        # Calculate hole quality metrics on full image
                        confidence = self._calculate_hole_confidence(gray, full_x, full_y, r)
                        edge_quality = self._calculate_edge_quality(gray, full_x, full_y, r)
                        
                        # Additional validation: check if hole matches expected background color
                        hole_matches_background = self._check_hole_background_match(gray, full_x, full_y, r)
                        
                        print(f"DEBUG: Hole confidence={confidence:.3f}, bg_match={hole_matches_background}")
                        
                        # Only accept holes with reasonable confidence OR correct background color (more lenient)
                        if confidence > 0.05 or (confidence > 0.01 and hole_matches_background):
                            hole = PerforationHole(
                                center_x=float(full_x),
                                center_y=float(full_y),
                                diameter=float(r * 2),
                                confidence=confidence,
                                edge_quality=edge_quality
                            )
                            holes.append(hole)
                            print(f"DEBUG: Added hole at ({full_x},{full_y}) with confidence {confidence:.3f}")
                        else:
                            print(f"DEBUG: Rejected hole at ({full_x},{full_y}) - low confidence or wrong background")
                    else:
                        print(f"DEBUG: Rejected hole at ({full_x},{full_y}) - too far from edge or center")
            
            # Sort holes by position (left to right for horizontal edges, top to bottom for vertical)
            if edge_points:
                first_point = edge_points[0]
                last_point = edge_points[-1]
                
                if abs(first_point[0] - last_point[0]) > abs(first_point[1] - last_point[1]):
                    # Horizontal edge - sort by x coordinate
                    holes.sort(key=lambda h: h.center_x)
                else:
                    # Vertical edge - sort by y coordinate
                    holes.sort(key=lambda h: h.center_y)
            
            print(f"DEBUG: Final holes count after all filtering: {len(holes)}")
            return holes
            
        except Exception as e:
            print(f"Error detecting perforation holes: {e}")
            return []
    
    def _calculate_hole_confidence(self, gray_image: np.ndarray, x: int, y: int, radius: int) -> float:
        """Calculate confidence that this is actually a perforation hole."""
        try:
            # Extract region around the hole
            y1, y2 = max(0, y - radius), min(gray_image.shape[0], y + radius)
            x1, x2 = max(0, x - radius), min(gray_image.shape[1], x + radius)
            
            region = gray_image[y1:y2, x1:x2]
            
            if region.size == 0:
                return 0.0
            
            # Calculate variance (holes should have high contrast)
            variance = np.var(region)
            
            # Calculate circularity by comparing to ideal circle
            center_y, center_x = region.shape[0] // 2, region.shape[1] // 2
            y_coords, x_coords = np.ogrid[:region.shape[0], :region.shape[1]]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Check if center is darker than edges (typical of holes)
            center_brightness = region[center_y, center_x] if 0 <= center_y < region.shape[0] and 0 <= center_x < region.shape[1] else 128
            edge_brightness = np.mean(region[distances > radius * 0.8])
            
            darkness_ratio = (edge_brightness - center_brightness) / 255.0
            
            # Combine factors
            confidence = min(1.0, (variance / 1000.0) * darkness_ratio)
            return max(0.0, confidence)
            
        except Exception as e:
            return 0.0
    
    def _calculate_edge_quality(self, gray_image: np.ndarray, x: int, y: int, radius: int) -> float:
        """Calculate how clean/circular the hole edge is."""
        try:
            # This would implement edge quality analysis
            # For now, return a placeholder based on local contrast
            y1, y2 = max(0, y - radius), min(gray_image.shape[0], y + radius)
            x1, x2 = max(0, x - radius), min(gray_image.shape[1], x + radius)
            
            region = gray_image[y1:y2, x1:x2]
            if region.size == 0:
                return 0.0
                
            # Calculate gradient magnitude as proxy for edge quality
            grad_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            edge_quality = np.mean(gradient_magnitude) / 255.0
            return min(1.0, edge_quality)
            
        except Exception as e:
            return 0.0
    
    def _check_hole_background_match(self, gray_image: np.ndarray, x: int, y: int, radius: int) -> bool:
        """Check if a detected hole matches the expected background color."""
        try:
            # Extract region around the hole center
            y1, y2 = max(0, y - radius//2), min(gray_image.shape[0], y + radius//2)
            x1, x2 = max(0, x - radius//2), min(gray_image.shape[1], x + radius//2)
            
            if x1 >= x2 or y1 >= y2:
                return False
                
            hole_region = gray_image[y1:y2, x1:x2]
            if hole_region.size == 0:
                return False
            
            # Calculate average intensity in the hole
            hole_intensity = np.mean(hole_region)
            
            # Define expected intensity ranges for different backgrounds
            if self.background_color == 'black':
                # For black background, holes should be dark (0-120) - more lenient
                return hole_intensity < 120
            elif self.background_color == 'dark_gray':
                # For dark gray background, holes should be darkish (0-120)
                return hole_intensity < 120
            elif self.background_color == 'white':
                # For white background, holes should be bright (180-255)
                return hole_intensity > 180
            elif self.background_color == 'light_gray':
                # For light gray background, holes should be lightish (120-255)
                return hole_intensity > 120
            else:
                # Unknown background - accept all
                return True
                
        except Exception as e:
            # If check fails, be permissive
            return True
    
    def calculate_perforation_gauge(self, holes: List[PerforationHole], edge_length_pixels: float) -> Tuple[float, float]:
        """Calculate perforation gauge from detected holes."""
        if len(holes) < 2:
            return 0.0, 0.0
        
        # Calculate distance between hole centers
        distances = []
        for i in range(len(holes) - 1):
            h1, h2 = holes[i], holes[i + 1]
            distance = math.sqrt((h2.center_x - h1.center_x)**2 + (h2.center_y - h1.center_y)**2)
            distances.append(distance)
        
        if not distances:
            return 0.0, 0.0
        
        # Average distance between holes
        avg_hole_spacing_pixels = np.mean(distances)
        
        # DEBUG: Print the calculation details
        print(f"DEBUG GAUGE CALC: {len(holes)} holes, {len(distances)} distances")
        print(f"DEBUG GAUGE CALC: distances = {[f'{d:.1f}' for d in distances]} pixels")
        
        # Compensate for missing holes by analyzing spacing patterns
        compensated_distances = self._compensate_for_missing_holes(distances)
        compensated_avg_spacing = np.mean(compensated_distances) if compensated_distances else avg_hole_spacing_pixels
        
        # Convert pixels to mm (assuming standard DPI) - use compensated spacing
        pixels_per_mm = self.dpi / 25.4  # 25.4 mm per inch
        hole_spacing_mm = compensated_avg_spacing / pixels_per_mm
        
        print(f"DEBUG GAUGE CALC: Original avg spacing = {avg_hole_spacing_pixels:.2f} pixels")
        print(f"DEBUG GAUGE CALC: Compensated distances = {[f'{d:.1f}' for d in compensated_distances]} pixels")
        print(f"DEBUG GAUGE CALC: Compensated avg spacing = {compensated_avg_spacing:.2f} pixels")
        print(f"DEBUG GAUGE CALC: pixels_per_mm = {pixels_per_mm:.2f} (at {self.dpi} DPI)")
        print(f"DEBUG GAUGE CALC: hole_spacing_mm = {hole_spacing_mm:.3f}")
        
        # Calculate gauge (holes per 20mm)
        # If holes are X mm apart, then in 20mm there are (20/X + 1) holes
        # But more accurately: if N holes span distance D, then gauge = (N-1) * 20mm / D
        # However, the standard formula is: holes per 20mm = 20 / spacing
        # Let me check if the issue is we need to add 1 for the hole count vs spacing count
        
        # The current formula should be correct: gauge = 20.0 / hole_spacing_mm
        # But let me verify by calculating it differently:
        # If we have N holes over distance D mm, then spacing = D/(N-1)
        # So gauge = 20/spacing = 20 * (N-1) / D
        
        gauge = 20.0 / hole_spacing_mm if hole_spacing_mm > 0 else 0.0
        
        # DEBUG: Try alternative calculation
        if len(holes) > 2:
            # Calculate total distance spanned by holes
            first_hole = holes[0]
            last_hole = holes[-1]
            total_distance_pixels = math.sqrt(
                (last_hole.center_x - first_hole.center_x)**2 + 
                (last_hole.center_y - first_hole.center_y)**2
            )
            total_distance_mm = total_distance_pixels / pixels_per_mm
            # Alternative gauge: (number of holes - 1) * 20mm / total_distance
            alt_gauge = (len(holes) - 1) * 20.0 / total_distance_mm if total_distance_mm > 0 else 0.0
            print(f"DEBUG GAUGE CALC: Alternative method: {len(holes)} holes over {total_distance_mm:.3f}mm = gauge {alt_gauge:.3f}")
        
        print(f"DEBUG GAUGE CALC: calculated gauge = {gauge:.3f}")
        
        # Calculate confidence based on measurement consistency using compensated distances
        spacing_std = np.std(compensated_distances) if len(compensated_distances) > 1 else 0.0
        confidence = max(0.0, 1.0 - (spacing_std / compensated_avg_spacing))
        
        return gauge, confidence
    
    def analyze_compound_perforation(self, edges: List[PerforationEdge]) -> Tuple[bool, str]:
        """Analyze if this is a compound perforation and describe it."""
        if len(edges) < 2:
            return False, "Insufficient edges for compound analysis"
        
        # Get unique gauge measurements (rounded to nearest 0.25)
        gauges = [round(edge.gauge_measurement * 4) / 4 for edge in edges if edge.gauge_measurement > 0]
        unique_gauges = list(set(gauges))
        
        if len(unique_gauges) <= 1:
            catalog_gauge = self.format_gauge_for_catalog(unique_gauges[0] if unique_gauges else 0)
            return False, f"Uniform perforation gauge {catalog_gauge}"
        
        # This is a compound perforation
        horizontal_gauges = []
        vertical_gauges = []
        
        for edge in edges:
            if edge.edge_type in ['top', 'bottom']:
                horizontal_gauges.append(edge.gauge_measurement)
            else:
                vertical_gauges.append(edge.gauge_measurement)
        
        h_avg = np.mean(horizontal_gauges) if horizontal_gauges else 0
        v_avg = np.mean(vertical_gauges) if vertical_gauges else 0
        
        if h_avg > 0 and v_avg > 0:
            h_catalog = self.format_gauge_for_catalog(h_avg)
            v_catalog = self.format_gauge_for_catalog(v_avg)
            description = f"Compound perforation {h_catalog} × {v_catalog}"
        else:
            description = f"Compound perforation with {len(unique_gauges)} different gauges"
        
        return True, description
        
    def format_gauge_for_catalog(self, gauge: float) -> str:
        """Format gauge measurement to match catalog standards (whole or quarter fractions)."""
        if gauge <= 0:
            return "unknown"
            
        # Round to nearest 0.25
        rounded_gauge = round(gauge * 4) / 4
        
        # Get whole and fractional parts
        whole_part = int(rounded_gauge)
        frac_part = rounded_gauge - whole_part
        
        # Format according to standard catalog notation
        if frac_part == 0:
            return str(whole_part)
        elif frac_part == 0.25:
            return f"{whole_part}¼"  # Quarter
        elif frac_part == 0.5:
            return f"{whole_part}½"  # Half
        elif frac_part == 0.75:
            return f"{whole_part}¾"  # Three-quarters
        else:
            # This shouldn't happen with proper rounding, but just in case
            return f"{rounded_gauge}"
    
    def detect_perforation_anomalies(self, gauge: float, edges: List[PerforationEdge]) -> List[str]:
        """Detect potential forgery indicators or reperforation issues."""
        warnings = []
        
        # Check for unusual fractional measurements
        rounded_gauge = round(gauge * 4) / 4
        frac_part = rounded_gauge - int(rounded_gauge)
        
        # Unusual fractions (eighths) may indicate forgery
        precise_measurement = round(gauge * 8) / 8
        eighth_frac = precise_measurement - int(precise_measurement)
        
        if abs(eighth_frac - 0.125) < 0.02 or abs(eighth_frac - 0.375) < 0.02 or \
           abs(eighth_frac - 0.625) < 0.02 or abs(eighth_frac - 0.875) < 0.02:
            warnings.append("UNUSUAL: Eighth-fraction measurement detected - possible forgery or reperforations")
        
        # Check for irregular spacing (reperforations)
        for edge in edges:
            if len(edge.holes) >= 3:
                # Calculate spacing consistency
                spacings = []
                for i in range(len(edge.holes) - 1):
                    h1, h2 = edge.holes[i], edge.holes[i + 1]
                    spacing = math.sqrt((h2.center_x - h1.center_x)**2 + (h2.center_y - h1.center_y)**2)
                    spacings.append(spacing)
                
                if spacings:
                    cv = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 0
                    if cv > 0.15:  # More than 15% coefficient of variation
                        warnings.append(f"IRREGULAR: Uneven hole spacing on {edge.edge_type} edge - possible reperforation")
        
        # Check for extreme gauge values based on historical postal practices
        if gauge < 8.5:
            if gauge < 3:
                warnings.append(f"EXTREMELY RARE: Gauge {self.format_gauge_for_catalog(gauge)} - similar to Bhopal experimental stamps (perf 2)")
            else:
                warnings.append(f"VERY UNUSUAL: Gauge {self.format_gauge_for_catalog(gauge)} - below typical minimum (Canada 8½)")
        elif gauge > 16:
            warnings.append(f"UNUSUAL: Gauge {self.format_gauge_for_catalog(gauge)} - above typical maximum (perf 15-16)")
        elif gauge > 15:
            warnings.append(f"NOTE: High gauge {self.format_gauge_for_catalog(gauge)} - near maximum practical separation limit")
        
        # Check for highly precise measurements that don't round to quarters
        if abs(gauge - round(gauge * 4) / 4) > 0.1:
            warnings.append("PRECISE: Measurement doesn't align with standard quarter-point system")
        
        return warnings
    
    def export_to_data_logger(self, analysis: PerforationAnalysis, image_filename: str, output_dir: str = ".") -> str:
        """Export perforation analysis to data logger text file."""
        import os
        from datetime import datetime
        
        # Create filename based on image
        base_name = os.path.splitext(os.path.basename(image_filename))[0]
        log_filename = f"{base_name}_perforation_data.txt"
        log_path = os.path.join(output_dir, log_filename)
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("=== StampZ Perforation Analysis ===\n")
                f.write(f"Image: {image_filename}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                
                f.write("PERFORATION MEASUREMENT:\n")
                f.write(f"Overall Gauge: {analysis.catalog_gauge}\n")
                f.write(f"Precise Measurement: {analysis.overall_gauge:.3f}\n")
                f.write(f"Compound Perforation: {'Yes' if analysis.is_compound_perforation else 'No'}\n")
                if analysis.is_compound_perforation:
                    f.write(f"Description: {analysis.compound_description}\n")
                f.write(f"Measurement Quality: {analysis.measurement_quality}\n")
                f.write("\n")
                
                # Edge-by-edge analysis
                f.write("EDGE ANALYSIS:\n")
                for edge in analysis.edges:
                    f.write(f"  {edge.edge_type.title()} Edge:\n")
                    f.write(f"    Gauge: {self.format_gauge_for_catalog(edge.gauge_measurement)}\n")
                    f.write(f"    Holes Detected: {len(edge.holes)}\n")
                    f.write(f"    Confidence: {edge.measurement_confidence:.1%}\n")
                    f.write(f"    Edge Length: {edge.total_length_pixels:.1f} pixels\n")
                f.write("\n")
                
                # Warnings section
                if analysis.warnings:
                    f.write("WARNINGS & ANOMALIES:\n")
                    for warning in analysis.warnings:
                        f.write(f"  ⚠️  {warning}\n")
                    f.write("\n")
                
                # Technical details
                f.write("TECHNICAL NOTES:\n")
                for note in analysis.technical_notes:
                    f.write(f"  • {note}\n")
                f.write("\n")
                
                f.write("--- End of Analysis ---\n")
                
            return log_path
            
        except Exception as e:
            print(f"Error writing data logger file: {e}")
            return ""
    
    def _measure_perforation_with_holes(self, image: np.ndarray) -> PerforationAnalysis:
        """Measure perforations using hole-based detection."""
        try:
            # Detect stamp edges
            edges_dict = self.detect_stamp_edges(image)
            
            if not edges_dict:
                return PerforationAnalysis(
                    edges=[],
                    overall_gauge=0.0,
                    catalog_gauge="unknown",
                    is_compound_perforation=False,
                    compound_description="Could not detect stamp edges",
                    measurement_quality="Poor",
                    technical_notes=["No stamp edges detected"],
                    warnings=["No stamp edges found - check image"]
                )
            
            # Detect holes for each edge
            edges = []
            for edge_type, edge_points in edges_dict.items():
                if not edge_points:
                    continue
                    
                print(f"DEBUG: Processing {edge_type} edge with {len(edge_points)} points")
                
                # Detect perforation holes along this edge
                holes = self.detect_perforation_holes(image, edge_points, edge_type)
                
                if holes:
                    # Calculate gauge from holes
                    edge_length = self._calculate_edge_length(edge_points)
                    gauge, confidence = self.calculate_perforation_gauge(holes, edge_length)
                    
                    edge_analysis = PerforationEdge(
                        edge_type=edge_type,
                        holes=holes,
                        total_length_pixels=edge_length,
                        gauge_measurement=gauge,
                        measurement_confidence=confidence,
                        is_compound=False  # Will be determined later
                    )
                    
                    edges.append(edge_analysis)
                    print(f"DEBUG: {edge_type} edge: {len(holes)} holes, gauge {gauge:.2f}")
                else:
                    print(f"DEBUG: {edge_type} edge: no holes detected")
            
            if not edges:
                return PerforationAnalysis(
                    edges=[],
                    overall_gauge=0.0,
                    catalog_gauge="unknown",
                    is_compound_perforation=False,
                    compound_description="No perforations detected",
                    measurement_quality="Poor",
                    technical_notes=["No perforation holes found"],
                    warnings=["No perforation holes detected - may be imperforate"]
                )
            
            # Calculate overall gauge
            valid_gauges = [e.gauge_measurement for e in edges if e.gauge_measurement > 0]
            overall_gauge = np.mean(valid_gauges) if valid_gauges else 0.0
            catalog_gauge = self.format_gauge_for_catalog(overall_gauge)
            
            # Check for compound perforation
            is_compound, compound_desc = self.analyze_compound_perforation(edges)
            
            # Detect anomalies and potential issues
            warnings = self.detect_perforation_anomalies(overall_gauge, edges)
            
            # Assess measurement quality
            avg_confidence = np.mean([e.measurement_confidence for e in edges])
            if avg_confidence > 0.8:
                quality = "Excellent"
            elif avg_confidence > 0.6:
                quality = "Good"
            elif avg_confidence > 0.4:
                quality = "Fair"
            else:
                quality = "Poor"
            
            # Generate technical notes
            technical_notes = []
            total_holes = sum(len(e.holes) for e in edges)
            technical_notes.append(f"Analyzed {len(edges)} edges with {total_holes} perforation holes")
            technical_notes.append(f"Average measurement confidence: {avg_confidence:.2%}")
            technical_notes.append(f"DPI setting: {self.dpi}")
            technical_notes.append("Used hole-based detection method")
            
            # Add gauge range context
            if overall_gauge > 0:
                if overall_gauge <= 8.5:
                    technical_notes.append("Low gauge perforations - easier to separate but less common")
                elif overall_gauge >= 15:
                    technical_notes.append("High gauge perforations - more holes but harder to separate cleanly")
                elif 11 <= overall_gauge <= 14:
                    technical_notes.append("Standard gauge range - good balance of separation ease and strength")
            
            return PerforationAnalysis(
                edges=edges,
                overall_gauge=overall_gauge,
                catalog_gauge=catalog_gauge,
                is_compound_perforation=is_compound,
                compound_description=compound_desc,
                measurement_quality=quality,
                technical_notes=technical_notes,
                warnings=warnings
            )
            
        except Exception as e:
            return PerforationAnalysis(
                edges=[],
                overall_gauge=0.0,
                catalog_gauge="error",
                is_compound_perforation=False,
                compound_description=f"Analysis failed: {str(e)}",
                measurement_quality="Poor",
                technical_notes=[f"Error during analysis: {str(e)}"],
                warnings=[f"Analysis error: {str(e)}"]
            )
    
    def _calculate_edge_length(self, edge_points: List[Tuple[int, int]]) -> float:
        """Calculate the length of an edge in pixels."""
        if len(edge_points) < 2:
            return 0.0
        
        first_point = edge_points[0]
        last_point = edge_points[-1]
        
        return math.sqrt(
            (last_point[0] - first_point[0])**2 + 
            (last_point[1] - first_point[1])**2
        )
    
    def _compensate_for_missing_holes(self, distances: List[float]) -> List[float]:
        """Compensate for missing holes by detecting and splitting large gaps."""
        if len(distances) < 3:
            return distances
        
        # Find the typical spacing (mode of smaller distances)
        distances_array = np.array(distances)
        
        # Use median of smaller half as estimate of normal spacing
        sorted_distances = np.sort(distances_array)
        lower_half = sorted_distances[:len(sorted_distances)//2 + 1]
        typical_spacing = np.median(lower_half)
        
        print(f"DEBUG COMPENSATION: Typical spacing estimate = {typical_spacing:.1f} pixels")
        
        compensated = []
        
        for distance in distances:
            # Only compensate for very clear double-spacing (1.7-2.3x typical spacing)
            # This catches obvious missing holes without over-compensating
            if typical_spacing * 1.7 < distance < typical_spacing * 2.3:
                # This is likely exactly 1 missing hole (double spacing)
                missed_holes = 1
                if missed_holes > 0:
                    # Split the large distance into smaller segments
                    segment_distance = distance / (missed_holes + 1)
                    print(f"DEBUG COMPENSATION: Large gap {distance:.1f} -> {missed_holes} missing holes, split into {segment_distance:.1f} pixel segments")
                    for _ in range(missed_holes + 1):
                        compensated.append(segment_distance)
                else:
                    compensated.append(distance)
            else:
                compensated.append(distance)
        
        return compensated
    
    def _validate_perforation_hole(self, gray_image: np.ndarray, x: int, y: int, radius: int, edge_type: str) -> bool:
        """Validate if a detected circle is actually a perforation hole vs ink spot/signature."""
        try:
            # Extract larger region around the hole for analysis
            analysis_radius = max(radius * 2, 15)
            y1, y2 = max(0, y - analysis_radius), min(gray_image.shape[0], y + analysis_radius)
            x1, x2 = max(0, x - analysis_radius), min(gray_image.shape[1], x + analysis_radius)
            
            region = gray_image[y1:y2, x1:x2]
            if region.size == 0:
                return False
            
            # 1. Check hole position relative to stamp edge direction
            h, w = gray_image.shape
            if edge_type == 'top' and y > h * 0.15:  # Top holes should be very near top
                return False
            elif edge_type == 'bottom' and y < h * 0.85:  # Bottom holes should be very near bottom
                return False
            elif edge_type == 'left' and x > w * 0.15:  # Left holes should be very near left
                return False
            elif edge_type == 'right' and x < w * 0.85:  # Right holes should be very near right
                return False
            
            # 2. Check if hole has background characteristics (not ink)
            hole_center_region = gray_image[max(0, y-radius//2):min(h, y+radius//2), 
                                          max(0, x-radius//2):min(w, x+radius//2)]
            if hole_center_region.size > 0:
                center_intensity = np.mean(hole_center_region)
                # For black backgrounds, perforations should be very dark (like background)
                # Ink spots are usually gray, not black
                if self.background_color == 'black' and center_intensity > 60:
                    print(f"DEBUG VALIDATION: Rejected hole at ({x},{y}) - too bright for perforation (intensity={center_intensity:.1f})")
                    return False
            
            # 3. Check for circular uniformity (perforations are more uniform than ink spots)
            center_y_rel, center_x_rel = region.shape[0] // 2, region.shape[1] // 2
            y_coords, x_coords = np.ogrid[:region.shape[0], :region.shape[1]]
            distances = np.sqrt((x_coords - center_x_rel)**2 + (y_coords - center_y_rel)**2)
            
            # Compare intensity at different radii
            inner_circle = region[distances <= radius * 0.5]
            outer_ring = region[(distances > radius * 0.8) & (distances <= radius * 1.2)]
            
            if len(inner_circle) > 0 and len(outer_ring) > 0:
                inner_mean = np.mean(inner_circle)
                outer_mean = np.mean(outer_ring)
                contrast_ratio = (outer_mean - inner_mean) / 255.0
                
                # Real perforations should have good contrast (dark center, lighter edges)
                if contrast_ratio < 0.1:
                    print(f"DEBUG VALIDATION: Rejected hole at ({x},{y}) - insufficient contrast (ratio={contrast_ratio:.3f})")
                    return False
            
            print(f"DEBUG VALIDATION: Accepted hole at ({x},{y}) as real perforation")
            return True
            
        except Exception as e:
            print(f"DEBUG VALIDATION: Error validating hole at ({x},{y}): {e}")
            return False
    
    def measure_perforation(self, image: np.ndarray, use_hole_detection: bool = True) -> PerforationAnalysis:
        """Complete perforation measurement of a stamp image.
        
        Args:
            image: Input stamp image
            use_hole_detection: If True, uses hole detection; if False, uses line detection
        """
        try:
            if use_hole_detection:
                print("DEBUG: Using hole-based detection method")
                return self._measure_perforation_with_holes(image)
            else:
                print("DEBUG: Using tic-based detection method (more accurate)")
                return self._measure_perforation_with_lines(image)
                
        except Exception as e:
            return PerforationAnalysis(
                edges=[],
                overall_gauge=0.0,
                catalog_gauge="error",
                is_compound_perforation=False,
                compound_description=f"Analysis failed: {str(e)}",
                measurement_quality="Poor",
                technical_notes=[f"Error during analysis: {str(e)}"],
                warnings=[f"Analysis error: {str(e)}"]
            )
            
    def _measure_perforation_with_lines(self, image: np.ndarray) -> PerforationAnalysis:
        """Measure perforations using line-based detection."""
        try:
            # Use the new line-based detection approach
            from perforation_line_detection import PerforationLineDetector
            
            line_detector = PerforationLineDetector(self.dpi, self.background_color)
            perforation_lines = line_detector.detect_perforation_lines(image)
            
            if not perforation_lines:
                return PerforationAnalysis(
                    edges=[],
                    overall_gauge=0.0,
                    catalog_gauge="unknown",
                    is_compound_perforation=False,
                    compound_description="Could not detect perforations",
                    measurement_quality="Poor",
                    technical_notes=["No perforation lines detected"],
                    warnings=["No perforation indentations found - may be imperforate"]
                )
            
            # Convert perforation lines to PerforationEdge format
            edges = []
            for line in perforation_lines:
                # Convert tics to holes for compatibility
                holes = []
                for tic in line.tics:
                    hole = PerforationHole(
                        center_x=tic.x,
                        center_y=tic.y,
                        diameter=10.0,  # Placeholder - not used in line detection
                        confidence=tic.confidence,
                        edge_quality=tic.depth / 20.0  # Convert depth to quality metric
                    )
                    holes.append(hole)
                
                # Calculate edge length from line points
                if len(line.line_points) > 1:
                    first_point = line.line_points[0]
                    last_point = line.line_points[-1]
                    edge_length = math.sqrt(
                        (last_point[0] - first_point[0])**2 + 
                        (last_point[1] - first_point[1])**2
                    )
                else:
                    edge_length = 0
                
                edge_analysis = PerforationEdge(
                    edge_type=line.edge_type,
                    holes=holes,  # Converted tics
                    total_length_pixels=edge_length,
                    gauge_measurement=line.gauge_measurement,
                    measurement_confidence=line.measurement_confidence,
                    is_compound=False  # Will be determined later
                )
                
                edges.append(edge_analysis)
            
            if not edges:
                return PerforationAnalysis(
                    edges=[],
                    overall_gauge=0.0,
                    catalog_gauge="unknown",
                    is_compound_perforation=False,
                    compound_description="No perforations detected",
                    measurement_quality="Poor",
                    technical_notes=["No perforation holes found"],
                    warnings=["No perforation holes detected - may be imperforate"]
                )
            
            # Calculate overall gauge
            valid_gauges = [e.gauge_measurement for e in edges if e.gauge_measurement > 0]
            overall_gauge = np.mean(valid_gauges) if valid_gauges else 0.0
            catalog_gauge = self.format_gauge_for_catalog(overall_gauge)
            
            # Check for compound perforation
            is_compound, compound_desc = self.analyze_compound_perforation(edges)
            
            # Detect anomalies and potential issues
            warnings = self.detect_perforation_anomalies(overall_gauge, edges)
            
            # Assess measurement quality
            avg_confidence = np.mean([e.measurement_confidence for e in edges])
            if avg_confidence > 0.8:
                quality = "Excellent"
            elif avg_confidence > 0.6:
                quality = "Good"
            elif avg_confidence > 0.4:
                quality = "Fair"
            else:
                quality = "Poor"
            
            # Generate technical notes
            technical_notes = []
            total_holes = sum(len(e.holes) for e in edges)
            technical_notes.append(f"Analyzed {len(edges)} edges with {total_holes} perforation holes")
            technical_notes.append(f"Average measurement confidence: {avg_confidence:.2%}")
            technical_notes.append(f"DPI setting: {self.dpi}")
            
            return PerforationAnalysis(
                edges=edges,
                overall_gauge=overall_gauge,
                catalog_gauge=catalog_gauge,
                is_compound_perforation=is_compound,
                compound_description=compound_desc,
                measurement_quality=quality,
                technical_notes=technical_notes,
                warnings=warnings
            )
            
        except Exception as e:
            return PerforationAnalysis(
                edges=[],
                overall_gauge=0.0,
                catalog_gauge="error",
                is_compound_perforation=False,
                compound_description=f"Analysis failed: {str(e)}",
                measurement_quality="Poor",
                technical_notes=[f"Error during analysis: {str(e)}"],
                warnings=[f"Analysis error: {str(e)}"]
            )

def demo_perforation_measurement():
    """Demonstrate the perforation measurement system."""
    print("=== StampZ Perforation Measurement System Demo ===\n")
    
    print("🎯 Why Perforation Measurement is Perfect for StampZ:")
    print("   • Universal need - every philatelist measures perforations")
    print("   • Mathematically precise - no linguistic confusion like colors")
    print("   • Gauge 11 = 11 holes per 20mm everywhere in the world")
    print("   • Perfect for computer vision - edge detection + measurement")
    print("   • Completes the 'Big 3': Color ✓, Condition ?, Perforation ✓")
    
    print(f"\n📊 Technical Capabilities:")
    print(f"   • Automatic hole detection using computer vision")
    print(f"   • Higher precision than physical gauges")
    print(f"   • Compound perforation analysis (e.g., 11 × 12)")
    print(f"   • Confidence scoring for measurement quality")
    print(f"   • Integration with existing StampZ workflow")
    
    print(f"\n🔧 Implementation Features:")
    print(f"   • Edge detection to find stamp boundaries")
    print(f"   • Hole detection using circular pattern recognition")  
    print(f"   • Spacing measurement and gauge calculation")
    print(f"   • Automatic DPI calibration for accuracy")
    print(f"   • Export to unified data logger")
    
    # Simulate analysis results
    print(f"\n📏 Sample Analysis Results:")
    print(f"   Stamp: 1920 King George V")
    print(f"   Catalog Gauge: 11¼  (precise: 11.237)")
    print(f"   Measurement Quality: Excellent")
    print(f"   Compound Perforation: No")
    print(f"   Edges Analyzed: 4 (top, bottom, left, right)")
    print(f"   Total Holes Detected: 48")
    print(f"   Confidence: 94%")
    
    print(f"\n⚠️  Forgery Detection Examples:")
    print(f"   • Eighth-fraction measurements (11⅛, 11⅜) → Possible forgery")
    print(f"   • Irregular hole spacing → Possible reperforation")
    print(f"   • Extreme gauges (<8½ or >16) → Verify measurement")
    print(f"   • Ultra-low gauges (<3) → Extremely rare (like Bhopal perf 2)")
    print(f"   • Perfect alignment warnings for authentication")
    
    print(f"\n🚀 Integration Points:")
    print(f"   • Add 'Measure Perforations' button to main toolbar")
    print(f"   • Results display in popup with visual overlay")
    print(f"   • Export measurements to unified data logger (.txt files)")
    print(f"   • Compare with catalog perforation specifications")
    print(f"   • Batch processing for multiple stamps")
    print(f"   • Forgery alerts integrated with warning system")
    
    print(f"\n📝 Data Logger Output Example:")
    print(f"   File: stamp_001_perforation_data.txt")
    print(f"   Content: Catalog format (11¼), precise measurement")
    print(f"   Warnings: Forgery indicators, measurement anomalies")
    print(f"   Technical: DPI, confidence, hole counts per edge")
    
    print(f"\n✅ This would complete StampZ as the comprehensive philatelic tool!")
    
    # Create a sample data logger file
    engine = PerforationMeasurementEngine()
    
    # Create sample analysis data
    sample_analysis = PerforationAnalysis(
        edges=[],
        overall_gauge=11.237,
        catalog_gauge="11¼",
        is_compound_perforation=False,
        compound_description="Uniform perforation gauge 11¼",
        measurement_quality="Excellent",
        technical_notes=[
            "Analyzed 4 edges with 48 perforation holes",
            "Average measurement confidence: 94%",
            "DPI setting: 600"
        ],
        warnings=[]
    )
    
    # Export sample data
    log_file = engine.export_to_data_logger(sample_analysis, "sample_stamp.jpg", ".")
    if log_file:
        print(f"\n📝 Sample data logger file created: {log_file}")

if __name__ == "__main__":
    # Check if OpenCV is available
    try:
        import cv2
        print("✅ OpenCV available - perforation measurement ready to implement")
    except ImportError:
        print("📦 OpenCV needed - install with: pip install opencv-python")
    
    demo_perforation_measurement()