#!/usr/bin/env python3
"""
Mask-based Perforation Detection using Black Ink Extraction Results

This approach uses the mask_adaptive.png from black ink extraction, which outlines
perforations well, and attempts to clean up the noise for perforation measurement.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict


class MaskBasedPerforationDetector:
    """Perforation detection using adaptive mask from black ink extraction."""
    
    def __init__(self, dpi: int = 800):
        self.dpi = dpi
    
    def detect_from_mask(self, original_image_path: str) -> Dict:
        """Detect perforations using the corresponding mask_adaptive.png file."""
        
        # Construct mask path
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        mask_dir = "/Users/stanbrown/Desktop/black_ink_extraction"
        
        # Look for corresponding mask file
        mask_path = None
        for file in os.listdir(mask_dir):
            if file.endswith("_mask_adaptive.png"):
                # Try to match based on partial name
                if any(part in file.lower() for part in base_name.lower().split('-')[:2]):
                    mask_path = os.path.join(mask_dir, file)
                    break
        
        if not mask_path:
            # Try the specific file we know exists
            mask_path = os.path.join(mask_dir, "A-Penny black ew red_mask_adaptive.png")
        
        if not os.path.exists(mask_path):
            print(f"Mask file not found: {mask_path}")
            return {}
        
        print(f"Using mask: {mask_path}")
        
        # Load the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Could not load mask image")
            return {}
        
        print(f"Mask shape: {mask.shape}")
        print(f"Mask intensity range: {mask.min()} to {mask.max()}")
        
        # Clean up the mask to reduce noise
        cleaned_mask = self._clean_mask(mask)
        
        # Find perforation contours in cleaned mask
        results = self._find_perforation_patterns(cleaned_mask)
        
        return results
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up the adaptive mask to reduce noise while preserving perforations."""
        
        # 1. Threshold to get binary mask
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 2. Morphological operations to clean noise
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill)
        
        # 3. Focus on edges - perforations should be near boundaries
        h, w = cleaned.shape
        edge_mask = np.zeros_like(cleaned)
        
        # Create edge regions
        edge_width = min(50, min(h, w) // 10)
        edge_mask[:edge_width, :] = 255  # top
        edge_mask[-edge_width:, :] = 255  # bottom  
        edge_mask[:, :edge_width] = 255  # left
        edge_mask[:, -edge_width:] = 255  # right
        
        # Keep only features near edges
        cleaned = cv2.bitwise_and(cleaned, edge_mask)
        
        return cleaned
    
    def _find_perforation_patterns(self, cleaned_mask: np.ndarray) -> Dict:
        """Find perforation patterns in the cleaned mask."""
        
        results = {}
        h, w = cleaned_mask.shape
        
        # Define edge regions for analysis
        edge_regions = {
            'top': cleaned_mask[:min(30, h//15), :],
            'bottom': cleaned_mask[-min(30, h//15):, :],
            'left': cleaned_mask[:, :min(30, w//15)], 
            'right': cleaned_mask[:, -min(30, w//15):]
        }
        
        for edge_type, region in edge_regions.items():
            if region.size == 0:
                continue
                
            print(f"\\nProcessing {edge_type} edge region: {region.shape}")
            
            # Find contours in this region
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            print(f"Found {len(contours)} contours")
            
            # Filter contours that could be perforations
            perforation_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 1000:  # Reasonable perforation hole size
                    # Check if roughly circular
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Somewhat circular
                            perforation_contours.append(contour)
            
            print(f"Filtered to {len(perforation_contours)} potential perforations")
            
            if len(perforation_contours) >= 3:
                # Calculate centers
                centers = []
                for contour in perforation_contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append((cx, cy))
                
                # Sort centers and calculate spacing
                if edge_type in ['top', 'bottom']:
                    centers.sort(key=lambda p: p[0])  # Sort by x
                else:
                    centers.sort(key=lambda p: p[1])  # Sort by y
                
                # Calculate gauge from center spacing
                gauge = self._calculate_gauge_from_centers(centers)
                
                results[edge_type] = {
                    'gauge': gauge,
                    'holes': len(centers),
                    'centers': centers
                }
                
                print(f"{edge_type} gauge: {gauge:.2f} from {len(centers)} holes")
        
        return results
    
    def _calculate_gauge_from_centers(self, centers: List[Tuple[int, int]]) -> float:
        """Calculate gauge from perforation hole centers."""
        if len(centers) < 2:
            return 0.0
        
        # Calculate distances between consecutive centers
        distances = []
        for i in range(len(centers) - 1):
            c1, c2 = centers[i], centers[i + 1]
            dist = np.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
            distances.append(dist)
        
        if not distances:
            return 0.0
        
        avg_spacing_pixels = np.mean(distances)
        
        # Convert to gauge
        pixels_per_mm = self.dpi / 25.4
        spacing_mm = avg_spacing_pixels / pixels_per_mm
        gauge = 20.0 / spacing_mm if spacing_mm > 0 else 0.0
        
        return gauge


def test_mask_method():
    """Test the mask-based detection method."""
    print("=== Testing Mask-based Perforation Detection ===")
    
    stamp_path = "/Users/stanbrown/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif"
    
    detector = MaskBasedPerforationDetector(dpi=800)
    results = detector.detect_from_mask(stamp_path)
    
    print("\\nResults:")
    if results:
        for edge_type, data in results.items():
            print(f"  {edge_type.upper()}: {data['gauge']:.2f} ({data['holes']} holes)")
        
        gauges = [data['gauge'] for data in results.values() if data['gauge'] > 0]
        if gauges:
            overall = np.mean(gauges)
            print(f"\\nOverall gauge: {overall:.2f}")
    else:
        print("No results found")


if __name__ == "__main__":
    test_mask_method()