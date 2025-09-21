#!/usr/bin/env python3
"""
Digital Perforation Gauge Overlay System

Creates traditional gauge overlay with radiating lines and perpendicular gauge markers.
Users can interactively position the overlay to align with perforation centers
for accurate gauge measurement.
"""

import cv2
import numpy as np
import math
import os
from typing import Dict, List, Tuple

class PerforationGaugeOverlay:
    """Digital perforation gauge with interactive positioning."""
    
    def __init__(self, dpi: int = 800):
        self.dpi = dpi
        self.pixels_per_mm = dpi / 25.4  # 25.4mm per inch
        
        # Standard perforation gauges (common ranges)
        self.gauges = [
            7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
            10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 
            13.0, 13.5, 14.0, 14.5, 15.0, 15.5,
            16.0, 16.5, 17.0, 17.5, 18.0
        ]
    
    def create_gauge_overlay(self, width: int, height: int, 
                           overlay_x: int = 0, overlay_y: int = 0,
                           orientation: str = 'horizontal') -> np.ndarray:
        """Create perforation gauge overlay with radiating lines.
        
        Args:
            width, height: Overlay dimensions (should match stamp image)
            overlay_x, overlay_y: Position offset for interactive placement
            orientation: 'horizontal' or 'vertical' for gauge direction
        """
        
        # Create transparent overlay (RGBA)
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Gauge parameters
        line_color = (0, 255, 0, 180)  # Green, semi-transparent
        gauge_marker_color = (255, 0, 0, 255)  # Red, opaque
        text_color = (255, 255, 255, 255)  # White text
        
        if orientation == 'horizontal':
            self._draw_horizontal_gauge(overlay, overlay_x, overlay_y, 
                                      line_color, gauge_marker_color, text_color)
        else:
            self._draw_vertical_gauge(overlay, overlay_x, overlay_y,
                                    line_color, gauge_marker_color, text_color)
        
        return overlay
    
    def _draw_horizontal_gauge(self, overlay: np.ndarray, offset_x: int, offset_y: int,
                             line_color: Tuple, marker_color: Tuple, text_color: Tuple):
        """Draw horizontal perforation gauge (measures vertical perforations)."""
        
        height, width = overlay.shape[:2]
        
        # Center line (baseline for measurement)
        center_y = height // 2 + offset_y
        if 0 <= center_y < height:
            cv2.line(overlay, (0, center_y), (width, center_y), line_color, 2)
        
        # Calculate spacing for each gauge
        gauge_length_mm = 25  # 25mm measurement span
        gauge_length_pixels = int(gauge_length_mm * self.pixels_per_mm)
        
        # Starting x position (can be offset for positioning)
        start_x = max(50 + offset_x, 0)
        end_x = min(start_x + gauge_length_pixels, width)
        
        # Draw radiating lines for each gauge
        for gauge in self.gauges:
            if gauge < 7 or gauge > 18:  # Skip extreme values
                continue
                
            # Calculate spacing for this gauge: 20mm / gauge = spacing in mm
            spacing_mm = 20.0 / gauge
            spacing_pixels = spacing_mm * self.pixels_per_mm
            
            # Draw vertical lines at this spacing
            current_x = start_x
            line_count = 0
            
            while current_x <= end_x and current_x < width:
                # Draw vertical line
                line_start_y = max(0, center_y - 30)
                line_end_y = min(height - 1, center_y + 30)
                
                if line_start_y < height and line_end_y >= 0:
                    cv2.line(overlay, (int(current_x), line_start_y), 
                            (int(current_x), line_end_y), line_color, 1)
                
                current_x += spacing_pixels
                line_count += 1
            
            # Draw perpendicular marker line for this gauge
            marker_x = start_x + (gauge - 7) * 20  # Spread gauges across width
            if 0 <= marker_x < width:
                # Perpendicular line (extends further)
                marker_start_y = max(0, center_y - 50)
                marker_end_y = min(height - 1, center_y + 50)
                
                cv2.line(overlay, (int(marker_x), marker_start_y),
                        (int(marker_x), marker_end_y), marker_color, 3)
                
                # Gauge label
                label = f"{gauge}"
                if gauge.is_integer():
                    label = f"{int(gauge)}"
                
                text_y = max(20, marker_start_y - 10)
                cv2.putText(overlay, label, (int(marker_x - 10), text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    def _draw_vertical_gauge(self, overlay: np.ndarray, offset_x: int, offset_y: int,
                           line_color: Tuple, marker_color: Tuple, text_color: Tuple):
        """Draw vertical perforation gauge (measures horizontal perforations)."""
        
        height, width = overlay.shape[:2]
        
        # Center line (baseline for measurement)
        center_x = width // 2 + offset_x
        if 0 <= center_x < width:
            cv2.line(overlay, (center_x, 0), (center_x, height), line_color, 2)
        
        # Calculate spacing for each gauge
        gauge_length_mm = 25  # 25mm measurement span
        gauge_length_pixels = int(gauge_length_mm * self.pixels_per_mm)
        
        # Starting y position (can be offset for positioning)
        start_y = max(50 + offset_y, 0)
        end_y = min(start_y + gauge_length_pixels, height)
        
        # Draw radiating lines for each gauge
        for gauge in self.gauges:
            if gauge < 7 or gauge > 18:  # Skip extreme values
                continue
                
            # Calculate spacing for this gauge
            spacing_mm = 20.0 / gauge
            spacing_pixels = spacing_mm * self.pixels_per_mm
            
            # Draw horizontal lines at this spacing
            current_y = start_y
            
            while current_y <= end_y and current_y < height:
                # Draw horizontal line
                line_start_x = max(0, center_x - 30)
                line_end_x = min(width - 1, center_x + 30)
                
                if line_start_x < width and line_end_x >= 0:
                    cv2.line(overlay, (line_start_x, int(current_y)), 
                            (line_end_x, int(current_y)), line_color, 1)
                
                current_y += spacing_pixels
            
            # Draw perpendicular marker line for this gauge
            marker_y = start_y + (gauge - 7) * 15  # Spread gauges vertically
            if 0 <= marker_y < height:
                # Perpendicular line (extends further)
                marker_start_x = max(0, center_x - 50)
                marker_end_x = min(width - 1, center_x + 50)
                
                cv2.line(overlay, (marker_start_x, int(marker_y)),
                        (marker_end_x, int(marker_y)), marker_color, 3)
                
                # Gauge label
                label = f"{gauge}"
                if gauge.is_integer():
                    label = f"{int(gauge)}"
                
                text_x = max(marker_end_x + 10, center_x + 60)
                if text_x < width - 30:
                    cv2.putText(overlay, label, (text_x, int(marker_y + 5)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    def apply_overlay_to_image(self, image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Apply semi-transparent gauge overlay to stamp image."""
        
        # Convert image to RGBA if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif len(image.shape) == 2:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        else:
            image_rgba = image.copy()
        
        # Blend overlay with image
        result = image_rgba.copy().astype(np.float32)
        overlay_float = overlay.astype(np.float32)
        
        # Alpha blending where overlay has alpha > 0
        alpha_mask = overlay_float[:, :, 3] > 0
        alpha = overlay_float[:, :, 3] / 255.0
        
        # Blend channels separately
        for c in range(3):  # BGR channels
            result[:, :, c] = np.where(alpha_mask, 
                                     (1 - alpha) * result[:, :, c] + alpha * overlay_float[:, :, c],
                                     result[:, :, c])
        
        return result.astype(np.uint8)[:, :, :3]  # Return BGR
    
    def create_gauge_measurement_kit(self, stamp_image_path: str, 
                                   output_dir: str = ".", 
                                   gauge_positions: List[Tuple[int, int]] = None):
        """Create a set of gauge overlays for measuring a stamp.
        
        Generates multiple overlay positions so user can select best alignment.
        """
        
        # Load stamp image
        image = cv2.imread(stamp_image_path)
        if image is None:
            print(f"Could not load image: {stamp_image_path}")
            return
        
        height, width = image.shape[:2]
        stamp_name = os.path.splitext(os.path.basename(stamp_image_path))[0]
        
        # Default positions if none provided
        if gauge_positions is None:
            # Create grid of positions for user to choose from
            gauge_positions = [
                (0, 0),           # Center
                (-50, 0),         # Left
                (50, 0),          # Right  
                (0, -50),         # Up
                (0, 50),          # Down
                (-25, -25),       # Upper left
                (25, -25),        # Upper right
                (-25, 25),        # Lower left
                (25, 25)          # Lower right
            ]
        
        print(f"Creating perforation gauge kit for {stamp_name}")
        print(f"Image size: {width} × {height}")
        print(f"DPI: {self.dpi}")
        
        # Create horizontal gauge overlays (for measuring vertical perforations)
        print("\\nCreating horizontal gauges (for vertical perforation measurement):")
        for i, (offset_x, offset_y) in enumerate(gauge_positions):
            overlay = self.create_gauge_overlay(width, height, offset_x, offset_y, 'horizontal')
            result = self.apply_overlay_to_image(image, overlay)
            
            output_path = f"{output_dir}/{stamp_name}_horizontal_gauge_{i+1}.jpg"
            cv2.imwrite(output_path, result)
            print(f"  Saved: {output_path} (offset: {offset_x}, {offset_y})")
        
        # Create vertical gauge overlays (for measuring horizontal perforations)
        print("\\nCreating vertical gauges (for horizontal perforation measurement):")
        for i, (offset_x, offset_y) in enumerate(gauge_positions):
            overlay = self.create_gauge_overlay(width, height, offset_x, offset_y, 'vertical')
            result = self.apply_overlay_to_image(image, overlay)
            
            output_path = f"{output_dir}/{stamp_name}_vertical_gauge_{i+1}.jpg"
            cv2.imwrite(output_path, result)
            print(f"  Saved: {output_path} (offset: {offset_x}, {offset_y})")
        
        print(f"\\nGauge measurement kit complete! Created {len(gauge_positions) * 2} overlay images.")
        print("\\nInstructions:")
        print("1. Open the horizontal_gauge images to measure VERTICAL perforations")
        print("2. Open the vertical_gauge images to measure HORIZONTAL perforations") 
        print("3. Find the overlay position where lines best align with perforation centers")
        print("4. Read the gauge value from the red perpendicular line that aligns best")
        print("5. For compound perforations, get separate readings from each direction")


def test_gauge_overlay():
    """Test the gauge overlay system."""
    
    # Test with the same stamp we've been using
    stamp_path = os.path.expanduser("~/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif")
    
    if os.path.exists(stamp_path):
        gauge_system = PerforationGaugeOverlay(dpi=800)
        gauge_system.create_gauge_measurement_kit(
            stamp_path, 
            output_dir="/Users/stanbrown/Desktop/StampZ-III"
        )
    else:
        print(f"Test stamp not found: {stamp_path}")


if __name__ == "__main__":
    test_gauge_overlay()