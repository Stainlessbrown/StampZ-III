#!/usr/bin/env python3
"""
Advanced Perforation Detection using Edge Line Analysis

Instead of detecting circles, this approach:
1. Finds the inner edge line of perforations
2. Detects indentations (tics) along the line
3. Measures spacing between tics for gauge calculation

This works much better with real stamps that have irregular, torn, or aged perforations.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class PerforationTic:
    """Represents a perforation indentation point along an edge."""
    x: float
    y: float
    depth: float  # How deep the indentation is
    confidence: float
    edge_type: str  # 'top', 'bottom', 'left', 'right'


@dataclass
class PerforationLine:
    """Represents the perforation line along one edge."""
    edge_type: str
    line_points: List[Tuple[int, int]]  # The detected edge line
    tics: List[PerforationTic]  # Detected perforation points
    gauge_measurement: float
    measurement_confidence: float


class PerforationLineDetector:
    """Detects perforations using edge line analysis instead of circle detection."""
    
    def __init__(self, dpi: int = 600, background_color: str = 'black'):
        self.dpi = dpi
        self.background_color = background_color
        
    def detect_perforation_lines(self, image: np.ndarray) -> List[PerforationLine]:
        """Detect perforation lines on all edges of the stamp."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        print(f"DEBUG: Starting line-based perforation detection on {w}x{h} image")
        
        perforation_lines = []
        
        # Define search regions for each edge
        edge_regions = {
            'top': {'y_range': (0, min(50, h//10)), 'x_range': (20, w-20), 'direction': 'horizontal'},
            'bottom': {'y_range': (max(h-50, h*9//10), h), 'x_range': (20, w-20), 'direction': 'horizontal'},
            'left': {'y_range': (20, h-20), 'x_range': (0, min(50, w//10)), 'direction': 'vertical'},
            'right': {'y_range': (20, h-20), 'x_range': (max(w-50, w*9//10), w), 'direction': 'vertical'}
        }
        
        print(f"DEBUG: Analyzing {len(edge_regions)} edges")
        
        for edge_type, region in edge_regions.items():
            print(f"DEBUG: Processing {edge_type} edge with region {region}")
            line = self._detect_edge_line(gray, edge_type, region)
            if line:
                print(f"DEBUG: {edge_type} edge detected {len(line.tics)} tics, gauge: {line.gauge_measurement:.2f}")
                if len(line.tics) >= 3:  # Need at least 3 tics for measurement
                    perforation_lines.append(line)
                else:
                    print(f"DEBUG: {edge_type} edge has insufficient tics ({len(line.tics)} < 3)")
            else:
                print(f"DEBUG: {edge_type} edge detection failed")
        
        print(f"DEBUG: Final result: {len(perforation_lines)} perforation lines detected")
        return perforation_lines
    
    def _detect_edge_line(self, gray: np.ndarray, edge_type: str, region: Dict) -> Optional[PerforationLine]:
        """Detect the perforation line for one edge."""
        try:
            y1, y2 = region['y_range']
            x1, x2 = region['x_range']
            direction = region['direction']
            
            print(f"DEBUG: {edge_type} edge - ROI: ({x1},{y1}) to ({x2},{y2})")
            
            # Extract the region of interest
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                print(f"DEBUG: {edge_type} edge - empty ROI")
                return None
            
            print(f"DEBUG: {edge_type} edge - ROI size: {roi.shape}")
            
            # Find the edge line (boundary between stamp and background)
            edge_line = self._find_stamp_edge_line(roi, direction, edge_type)
            
            if not edge_line:
                print(f"DEBUG: {edge_type} edge - no edge line found")
                return None
            
            print(f"DEBUG: {edge_type} edge - found {len(edge_line)} edge line points")
            
            # Adjust coordinates back to full image space
            adjusted_line = []
            for point in edge_line:
                adj_x = point[0] + x1
                adj_y = point[1] + y1
                adjusted_line.append((adj_x, adj_y))
            
            # Detect perforation tics along the line
            tics = self._detect_perforation_tics(gray, adjusted_line, edge_type)
            
            print(f"DEBUG: {edge_type} edge - detected {len(tics)} tics")
            
            # Calculate gauge from tics
            gauge, confidence = self._calculate_gauge_from_tics(tics)
            
            print(f"DEBUG: {edge_type} edge - gauge: {gauge:.2f}, confidence: {confidence:.2f}")
            
            return PerforationLine(
                edge_type=edge_type,
                line_points=adjusted_line,
                tics=tics,
                gauge_measurement=gauge,
                measurement_confidence=confidence
            )
            
        except Exception as e:
            print(f"Error detecting edge line for {edge_type}: {e}")
            return None
    
    def _find_stamp_edge_line(self, roi: np.ndarray, direction: str, edge_type: str) -> List[Tuple[int, int]]:
        """Find the line that represents the inner edge of perforations.
        
        This should find the scalloped line INSIDE the stamp where perforations create indentations.
        """
        h, w = roi.shape
        edge_line = []
        
        try:
            if direction == 'horizontal':
                # For top/bottom edges, scan vertically to find perforation line
                for x in range(0, w, 2):  # Sample every 2 pixels for efficiency
                    if edge_type == 'top':
                        # Scan from top down, but look for the scalloped perforation line
                        # Skip the very edge and look inside where perforations indent
                        stamp_start = None
                        for y in range(h):
                            if self._is_stamp_pixel(roi[y, x]):
                                stamp_start = y
                                break
                        
                        if stamp_start is not None:
                            # Look a few pixels inside the stamp for perforation indentations
                            search_depth = min(15, h - stamp_start - 1)
                            darkest_y = stamp_start
                            darkest_val = roi[stamp_start, x]
                            
                            # Find the darkest point (deepest indentation) within search depth
                            for y in range(stamp_start, min(stamp_start + search_depth, h)):
                                if roi[y, x] < darkest_val:
                                    darkest_val = roi[y, x]
                                    darkest_y = y
                            
                            edge_line.append((x, darkest_y))
                    
                    else:  # bottom
                        # Scan from bottom up, look for perforation line inside stamp
                        stamp_start = None
                        for y in range(h-1, -1, -1):
                            if self._is_stamp_pixel(roi[y, x]):
                                stamp_start = y
                                break
                        
                        if stamp_start is not None:
                            # Look a few pixels inside the stamp for perforation indentations
                            search_depth = min(15, stamp_start)
                            darkest_y = stamp_start
                            darkest_val = roi[stamp_start, x]
                            
                            # Find the darkest point (deepest indentation) within search depth
                            for y in range(stamp_start, max(stamp_start - search_depth, -1), -1):
                                if roi[y, x] < darkest_val:
                                    darkest_val = roi[y, x]
                                    darkest_y = y
                            
                            edge_line.append((x, darkest_y))
                            
            else:  # vertical
                # For left/right edges, scan horizontally to find perforation line
                for y in range(0, h, 2):  # Sample every 2 pixels for efficiency
                    if edge_type == 'left':
                        # Scan from left to right, look for perforation line inside stamp
                        stamp_start = None
                        for x in range(w):
                            if self._is_stamp_pixel(roi[y, x]):
                                stamp_start = x
                                break
                        
                        if stamp_start is not None:
                            # Look a few pixels inside the stamp for perforation indentations
                            search_depth = min(15, w - stamp_start - 1)
                            darkest_x = stamp_start
                            darkest_val = roi[y, stamp_start]
                            
                            # Find the darkest point (deepest indentation) within search depth
                            for x in range(stamp_start, min(stamp_start + search_depth, w)):
                                if roi[y, x] < darkest_val:
                                    darkest_val = roi[y, x]
                                    darkest_x = x
                            
                            edge_line.append((darkest_x, y))
                    
                    else:  # right
                        # Scan from right to left, look for perforation line inside stamp
                        stamp_start = None
                        for x in range(w-1, -1, -1):
                            if self._is_stamp_pixel(roi[y, x]):
                                stamp_start = x
                                break
                        
                        if stamp_start is not None:
                            # Look a few pixels inside the stamp for perforation indentations
                            search_depth = min(15, stamp_start)
                            darkest_x = stamp_start
                            darkest_val = roi[y, stamp_start]
                            
                            # Find the darkest point (deepest indentation) within search depth
                            for x in range(stamp_start, max(stamp_start - search_depth, -1), -1):
                                if roi[y, x] < darkest_val:
                                    darkest_val = roi[y, x]
                                    darkest_x = x
                            
                            edge_line.append((darkest_x, y))
        except Exception as e:
            print(f"Error finding stamp edge line: {e}")
        
        return edge_line
    
    def _is_stamp_pixel(self, pixel_value: int) -> bool:
        """Determine if a pixel belongs to the stamp (not background)."""
        if self.background_color == 'black':
            is_stamp = pixel_value > 100  # Stamp should be lighter than dark background
        elif self.background_color == 'white':
            is_stamp = pixel_value < 200  # Stamp should be darker than light background
        elif self.background_color == 'dark_gray':
            is_stamp = pixel_value > 120
        elif self.background_color == 'light_gray':
            is_stamp = pixel_value < 180
        else:
            is_stamp = pixel_value > 100  # Default assumption
        
        return is_stamp
    
    def _detect_perforation_tics(self, gray: np.ndarray, edge_line: List[Tuple[int, int]], edge_type: str) -> List[PerforationTic]:
        """Detect perforation tics (indentations) along the edge line."""
        if len(edge_line) < 10:  # Need enough points for analysis
            return []
        
        tics = []
        
        # Sort points along the line
        if edge_type in ['top', 'bottom']:
            # Sort by x coordinate for horizontal edges
            edge_line.sort(key=lambda p: p[0])
        else:
            # Sort by y coordinate for vertical edges
            edge_line.sort(key=lambda p: p[1])
        
        # Smooth the line to reduce noise
        smoothed_line = self._smooth_edge_line(edge_line, edge_type)
        
        # Find indentations (tics) along the smoothed line
        print(f"DEBUG TIC DETECTION: {edge_type} edge - smoothed line has {len(smoothed_line)} points")
        tics = self._find_line_indentations(smoothed_line, edge_type)
        print(f"DEBUG TIC DETECTION: {edge_type} edge - found {len(tics)} raw tics before filtering")
        
        return tics
    
    def _smooth_edge_line(self, edge_line: List[Tuple[int, int]], edge_type: str) -> List[Tuple[float, float]]:
        """Smooth the edge line to reduce noise while preserving perforation features."""
        if len(edge_line) < 5:
            return [(float(p[0]), float(p[1])) for p in edge_line]
        
        # Use a small sliding window to smooth while preserving perforation indentations
        window_size = min(7, len(edge_line) // 3)  # Small window to preserve features
        smoothed = []
        
        for i in range(len(edge_line)):
            # Get window around current point
            start = max(0, i - window_size // 2)
            end = min(len(edge_line), i + window_size // 2 + 1)
            window_points = edge_line[start:end]
            
            # Average the coordinates in the window
            avg_x = sum(p[0] for p in window_points) / len(window_points)
            avg_y = sum(p[1] for p in window_points) / len(window_points)
            
            smoothed.append((avg_x, avg_y))
        
        return smoothed
    
    def _find_line_indentations(self, line: List[Tuple[float, float]], edge_type: str) -> List[PerforationTic]:
        """Find indentations along the line that indicate perforation locations."""
        if len(line) < 10:
            return []
        
        tics = []
        
        # Calculate the general trend line
        trend_line = self._calculate_trend_line(line, edge_type)
        
        # Find points where the actual line deviates significantly inward from trend
        for i in range(2, len(line) - 2):  # Skip edges
            point = line[i]
            
            # Get expected position on trend line
            if edge_type in ['top', 'bottom']:
                expected_y = self._interpolate_trend(trend_line, point[0], 'x')
                deviation = point[1] - expected_y
                
                # For top edge, perforations indent downward (positive deviation)
                # For bottom edge, perforations indent upward (negative deviation)  
                if edge_type == 'top' and deviation > 0.5:  # More sensitive threshold
                    depth = deviation
                elif edge_type == 'bottom' and deviation < -0.5:  # More sensitive threshold
                    depth = abs(deviation)
                else:
                    continue
            else:  # left, right edges
                expected_x = self._interpolate_trend(trend_line, point[1], 'y')
                deviation = point[0] - expected_x
                
                # For left edge, perforations indent rightward (positive deviation)
                # For right edge, perforations indent leftward (negative deviation)
                if edge_type == 'left' and deviation > 0.5:  # More sensitive threshold
                    depth = deviation
                elif edge_type == 'right' and deviation < -0.5:  # More sensitive threshold
                    depth = abs(deviation)
                else:
                    continue
            
            # Check if this is a local maximum (peak of indentation)
            if self._is_local_indentation_peak(line, i, edge_type):
                confidence = min(1.0, depth / 20.0)  # Scale depth to confidence
                
                tic = PerforationTic(
                    x=point[0],
                    y=point[1],
                    depth=depth,
                    confidence=confidence,
                    edge_type=edge_type
                )
                tics.append(tic)
        
        # Filter out tics that are too close together (likely noise)
        filtered_tics = self._filter_close_tics(tics)
        
        return filtered_tics
    
    def _calculate_trend_line(self, line: List[Tuple[float, float]], edge_type: str) -> Tuple[float, float]:
        """Calculate a trend line for the edge to identify deviations."""
        if len(line) < 3:
            return (0, 0)
        
        # Use simple linear regression
        if edge_type in ['top', 'bottom']:
            # Trend in y based on x
            xs = [p[0] for p in line]
            ys = [p[1] for p in line]
        else:
            # Trend in x based on y  
            xs = [p[1] for p in line]
            ys = [p[0] for p in line]
        
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_xx = sum(x * x for x in xs)
        
        # Calculate slope and intercept
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return (0, sum_y / n)  # Horizontal line
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        return (slope, intercept)
    
    def _interpolate_trend(self, trend: Tuple[float, float], coord: float, coord_type: str) -> float:
        """Interpolate expected position on trend line."""
        slope, intercept = trend
        return slope * coord + intercept
    
    def _is_local_indentation_peak(self, line: List[Tuple[float, float]], index: int, edge_type: str) -> bool:
        """Check if this point is a local peak of an indentation."""
        if index < 2 or index >= len(line) - 2:
            return False
        
        # Look at neighboring points to see if this is a peak
        window = 2
        center_point = line[index]
        
        # Get deviation values for surrounding points
        deviations = []
        for i in range(max(0, index - window), min(len(line), index + window + 1)):
            point = line[i]
            if edge_type in ['top', 'bottom']:
                deviation = abs(point[1] - center_point[1])
            else:
                deviation = abs(point[0] - center_point[0])
            deviations.append(deviation)
        
        # Check if center point has maximum deviation (is a peak)
        max_dev = max(deviations)
        center_dev = deviations[window]  # Center of the window
        
        return center_dev == max_dev and center_dev > 0.2  # Very sensitive threshold for real perforations
    
    def _filter_close_tics(self, tics: List[PerforationTic]) -> List[PerforationTic]:
        """Filter out tics that are too close together (likely noise)."""
        if len(tics) <= 1:
            return tics
        
        # Sort tics by position
        if tics[0].edge_type in ['top', 'bottom']:
            tics.sort(key=lambda t: t.x)
        else:
            tics.sort(key=lambda t: t.y)
        
        filtered = [tics[0]]  # Always keep first tic
        
        min_distance = max(10, int(self.dpi / 30))  # Minimum distance between tics
        
        for i in range(1, len(tics)):
            current_tic = tics[i]
            last_kept = filtered[-1]
            
            # Calculate distance
            if current_tic.edge_type in ['top', 'bottom']:
                distance = abs(current_tic.x - last_kept.x)
            else:
                distance = abs(current_tic.y - last_kept.y)
            
            if distance >= min_distance:
                # Keep this tic if it's far enough from the last kept one
                # Or if it has higher confidence
                if distance >= min_distance or current_tic.confidence > last_kept.confidence:
                    filtered.append(current_tic)
                elif current_tic.confidence > last_kept.confidence:
                    # Replace last kept tic with this higher confidence one
                    filtered[-1] = current_tic
        
        return filtered
    
    def _calculate_gauge_from_tics(self, tics: List[PerforationTic]) -> Tuple[float, float]:
        """Calculate perforation gauge from tic spacing."""
        if len(tics) < 2:
            return 0.0, 0.0
        
        # Calculate distances between consecutive tics
        distances = []
        for i in range(len(tics) - 1):
            tic1, tic2 = tics[i], tics[i + 1]
            if tic1.edge_type in ['top', 'bottom']:
                distance = abs(tic2.x - tic1.x)
            else:
                distance = abs(tic2.y - tic1.y)
            distances.append(distance)
        
        if not distances:
            return 0.0, 0.0
        
        # Average distance between tics
        avg_spacing_pixels = sum(distances) / len(distances)
        
        # Convert to gauge
        pixels_per_mm = self.dpi / 25.4  # 25.4 mm per inch
        spacing_mm = avg_spacing_pixels / pixels_per_mm
        gauge = 20.0 / spacing_mm if spacing_mm > 0 else 0.0
        
        # Calculate confidence based on consistency
        if len(distances) > 1:
            spacing_std = np.std(distances)
            consistency = max(0.0, 1.0 - (spacing_std / avg_spacing_pixels))
        else:
            consistency = 0.5
        
        # Weight by average tic confidence
        avg_tic_confidence = sum(t.confidence for t in tics) / len(tics)
        overall_confidence = (consistency * 0.7) + (avg_tic_confidence * 0.3)
        
        return gauge, overall_confidence


def test_line_detection():
    """Test the line-based perforation detection."""
    print("🔍 Testing Line-Based Perforation Detection")
    print("=" * 50)
    
    # This would be called with a real stamp image
    print("This approach should work much better with:")
    print("• Irregular perforation holes")
    print("• Torn or damaged perforations")  
    print("• Aged stamps with deformed holes")
    print("• Any perforation shape (not just circular)")
    print("\nInstead of looking for perfect circles, it:")
    print("1. Finds the inner edge line of perforations")
    print("2. Detects indentations along that line")
    print("3. Measures spacing between indentations")


if __name__ == "__main__":
    test_line_detection()