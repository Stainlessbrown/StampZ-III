#!/usr/bin/env python3
"""
Simplified Stamp Analyser-like Perforation Detection

This implements a simpler, more direct approach similar to what Stamp Analyser likely uses:
1. Find edge lines along stamp boundaries
2. Detect local minima/maxima (tics) along these lines without complex trend analysis
3. Measure spacing between tics directly
4. Convert to perforation gauge

This avoids the "fluctuating hole diameter" problem of circle-based detection.
"""

import numpy as np
import cv2
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass 
class SimpleTic:
    """A simple tic/indentation point along an edge."""
    x: float
    y: float
    intensity: float  # Local minimum intensity
    edge_type: str


class SimplePerforationDetector:
    """Simplified perforation detector similar to Stamp Analyser methodology."""
    
    def __init__(self, dpi: int = 800, background_color: str = 'black'):
        self.dpi = dpi
        self.background_color = background_color
    
    def detect_perforation_gauge(self, image: np.ndarray) -> dict:
        """Detect perforation gauge using simplified method."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        results = {}
        
        # Define edge regions (smaller, focused on actual edges)
        edge_regions = {
            'top': {'y_range': (0, min(25, h//20)), 'x_range': (20, w-20)},
            'bottom': {'y_range': (max(h-25, h*19//20), h), 'x_range': (20, w-20)},
            'left': {'y_range': (20, h-20), 'x_range': (0, min(25, w//20))},
            'right': {'y_range': (20, h-20), 'x_range': (max(w-25, w*19//20), w)}
        }
        
        for edge_type, region in edge_regions.items():
            print(f"DEBUG SIMPLE: Processing {edge_type} edge")
            
            # Extract ROI
            y1, y2 = region['y_range'] 
            x1, x2 = region['x_range']
            roi = gray[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
                
            # Find edge line using simple method
            edge_line = self._find_simple_edge_line(roi, edge_type)
            print(f"DEBUG SIMPLE: {edge_type} edge line has {len(edge_line)} points")
            
            if len(edge_line) < 10:
                continue
                
            # Adjust coordinates back to full image
            adjusted_line = [(p[0] + x1, p[1] + y1) for p in edge_line]
            
            # Find tics using simple local minima/maxima detection
            tics = self._find_simple_tics(gray, adjusted_line, edge_type)
            print(f"DEBUG SIMPLE: {edge_type} edge found {len(tics)} tics")
            
            if len(tics) >= 3:
                # Calculate gauge from tic spacing
                gauge = self._calculate_gauge_from_tics(tics)
                results[edge_type] = {
                    'gauge': gauge,
                    'tics': len(tics),
                    'tic_positions': [(t.x, t.y) for t in tics]
                }
                print(f"DEBUG SIMPLE: {edge_type} gauge = {gauge:.2f}")
        
        return results
    
    def _find_simple_edge_line(self, roi: np.ndarray, edge_type: str) -> List[Tuple[int, int]]:
        """Find edge line using simple intensity-based method."""
        h, w = roi.shape
        edge_line = []
        
        if edge_type in ['top', 'bottom']:
            # Horizontal edge - scan vertically
            for x in range(0, w, 2):  # Every 2 pixels
                if edge_type == 'top':
                    # Find the lowest intensity point in top region (perforation indent)
                    col = roi[:, x]
                    min_idx = np.argmin(col[:min(h, 15)])  # Search only first 15 pixels
                    edge_line.append((x, min_idx))
                else:  # bottom
                    # Find the lowest intensity point in bottom region
                    col = roi[:, x]
                    min_idx = np.argmin(col[max(0, h-15):]) + max(0, h-15)
                    edge_line.append((x, min_idx))
        else:
            # Vertical edge - scan horizontally
            for y in range(0, h, 2):  # Every 2 pixels
                if edge_type == 'left':
                    # Find the lowest intensity point in left region
                    row = roi[y, :]
                    min_idx = np.argmin(row[:min(w, 15)])  # Search only first 15 pixels
                    edge_line.append((min_idx, y))
                else:  # right
                    # Find the lowest intensity point in right region
                    row = roi[y, :]
                    min_idx = np.argmin(row[max(0, w-15):]) + max(0, w-15)
                    edge_line.append((min_idx, y))
        
        return edge_line
    
    def _find_simple_tics(self, gray: np.ndarray, edge_line: List[Tuple[int, int]], edge_type: str) -> List[SimpleTic]:
        """Find tics using simple local minima detection along the edge line."""
        if len(edge_line) < 5:
            return []
        
        # Sort edge line points
        if edge_type in ['top', 'bottom']:
            edge_line.sort(key=lambda p: p[0])  # Sort by x for horizontal edges
        else:
            edge_line.sort(key=lambda p: p[1])  # Sort by y for vertical edges
        
        # Smooth the line slightly
        smoothed_positions = []
        smoothed_intensities = []
        
        window = 3
        for i in range(len(edge_line)):
            start = max(0, i - window//2)
            end = min(len(edge_line), i + window//2 + 1)
            
            # Average position and get intensity
            avg_x = sum(edge_line[j][0] for j in range(start, end)) / (end - start)
            avg_y = sum(edge_line[j][1] for j in range(start, end)) / (end - start)
            
            # Get intensity at this position
            x, y = int(avg_x), int(avg_y)
            if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                intensity = gray[y, x]
            else:
                intensity = 128
                
            smoothed_positions.append((avg_x, avg_y))
            smoothed_intensities.append(intensity)
        
        # Find local minima (darker spots = perforation indentations)
        tics = []
        min_distance = max(20, int(self.dpi / 30))  # Larger minimum distance between tics (more selective)
        
        for i in range(2, len(smoothed_intensities) - 2):
            # Check if this is a local minimum
            current = smoothed_intensities[i]
            left = smoothed_intensities[i-1]
            right = smoothed_intensities[i+1]
            left2 = smoothed_intensities[i-2] 
            right2 = smoothed_intensities[i+2]
            
            # Must be significantly darker than neighbors
            if (current < left - 3 and current < right - 3 and 
                current <= left2 and current <= right2 and
                current < np.mean(smoothed_intensities) - 15):  # Much more stringent
                
                x, y = smoothed_positions[i]
                
                # Check minimum distance from existing tics
                too_close = False
                for existing_tic in tics:
                    dist = math.sqrt((x - existing_tic.x)**2 + (y - existing_tic.y)**2)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    tic = SimpleTic(x=x, y=y, intensity=current, edge_type=edge_type)
                    tics.append(tic)
        
        return tics
    
    def _calculate_gauge_from_tics(self, tics: List[SimpleTic]) -> float:
        """Calculate perforation gauge from tic spacing."""
        if len(tics) < 2:
            return 0.0
        
        # Calculate distances between consecutive tics
        distances = []
        for i in range(len(tics) - 1):
            t1, t2 = tics[i], tics[i + 1] 
            distance = math.sqrt((t2.x - t1.x)**2 + (t2.y - t1.y)**2)
            distances.append(distance)
        
        if not distances:
            return 0.0
            
        # Average spacing
        avg_spacing_pixels = np.mean(distances)
        
        # Convert to gauge
        pixels_per_mm = self.dpi / 25.4  # 25.4 mm per inch
        spacing_mm = avg_spacing_pixels / pixels_per_mm
        gauge = 20.0 / spacing_mm if spacing_mm > 0 else 0.0
        
        print(f"DEBUG GAUGE: {len(tics)} tics, avg spacing {avg_spacing_pixels:.1f}px = {spacing_mm:.3f}mm = gauge {gauge:.2f}")
        
        return gauge


def test_simple_method():
    """Test the simplified detection method."""
    print("=== Testing Simplified Stamp Analyser Method ===")
    
    # Test with your stamp image
    import os
    stamp_path = os.path.expanduser("~/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif")
    
    if os.path.exists(stamp_path):
        image = cv2.imread(stamp_path)
        if image is not None:
            detector = SimplePerforationDetector(dpi=800, background_color='black')
            results = detector.detect_perforation_gauge(image)
            
            print("\nResults:")
            for edge_type, data in results.items():
                print(f"  {edge_type.upper()}: {data['gauge']:.2f} ({data['tics']} tics)")
            
            # Calculate compound gauge
            if len(results) >= 2:
                gauges = [data['gauge'] for data in results.values() if data['gauge'] > 0]
                if gauges:
                    overall = np.mean(gauges)
                    print(f"\nOverall gauge: {overall:.2f}")
        else:
            print("Could not load image")
    else:
        print(f"Image not found: {stamp_path}")


if __name__ == "__main__":
    test_simple_method()