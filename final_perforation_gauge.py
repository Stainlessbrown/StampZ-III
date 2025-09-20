#!/usr/bin/env python3
"""
Final Perforation Gauge - Simple and Clean

Matches physical traditional gauges exactly:
- White lines and dots
- Black text labels
- Single horizontal design, rotates for vertical use
- DPI scaling support
"""

import cv2
import numpy as np
import os

class FinalPerforationGauge:
    """Final perforation gauge - clean and simple."""
    
    def __init__(self, dpi: int = 800):
        self.dpi = dpi
        self.pixels_per_mm = dpi / 25.4
        
        # Standard gauge range: 8.0 to 16.0 in quarter steps
        self.gauge_values = []
        for whole in range(8, 17):
            for fraction in [0, 0.25, 0.5, 0.75]:
                gauge = whole + fraction
                if gauge <= 16.0:
                    self.gauge_values.append(gauge)
        
    def create_gauge_overlay(self, width: int, height: int) -> np.ndarray:
        """Create gauge overlay with traditional appearance."""
        
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Scale all dimensions with DPI
        scale = self.dpi / 800.0
        reference_x = int(60 * scale)
        start_x = int(90 * scale)
        start_y = int(60 * scale)  # Start higher up
        row_spacing = int(15 * scale)  # Tighter spacing for all ranges
        
        # Reference line
        cv2.line(overlay, (reference_x, 0), (reference_x, height), 
                (255, 255, 255, 255), max(2, int(3 * scale)))
        
        # Calculate measurement area
        measurement_pixels = int(25 * self.pixels_per_mm)  # 25mm standard
        end_x = min(start_x + measurement_pixels, width - int(60 * scale))
        
        dot_positions = {}
        
        # Create each gauge row
        for i, gauge in enumerate(self.gauge_values):
            y = start_y + i * row_spacing
            if y >= height - int(30 * scale):  # Smaller bottom margin
                break
                
            # Calculate dot spacing (20mm perforation standard)
            spacing_pixels = (20.0 / gauge) * self.pixels_per_mm
            
            # Create dots for this gauge
            dots = []
            x = start_x
            dot_count = 0
            
            while x <= end_x and dot_count < 12:
                dots.append((x, y))
                x += spacing_pixels
                dot_count += 1
            
            dot_positions[gauge] = dots
            
            # Format label
            if gauge == int(gauge):
                label = str(int(gauge))
            elif gauge % 0.5 == 0:
                label = f"{int(gauge)} 1/2"
            elif (gauge * 4) % 1 == 0:
                if gauge % 1 == 0.25:
                    label = f"{int(gauge)} 1/4"
                else:  # 0.75
                    label = f"{int(gauge)} 3/4"
            else:
                label = f"{gauge:.2f}"
            
            # Store label info for later drawing (will create uniform column)
            font_scale = max(0.3, 0.4 * scale)  # Smaller fonts
            thickness = max(1, int(1.5 * scale))  # Thinner text
            
            # Store for drawing after background column is created
            if not hasattr(self, '_text_labels'):
                self._text_labels = []
            
            self._text_labels.append({
                'label': label,
                'y': y,
                'font_scale': font_scale,
                'thickness': thickness
            })
        
        # Draw radiating rays
        self._draw_rays(overlay, dot_positions, scale)
        
        # Draw perpendicular lines
        self._draw_perpendicular_lines(overlay, dot_positions, 
                                     start_x, end_x, scale)
        
        # Draw dots on top
        self._draw_dots(overlay, dot_positions, scale)
        
        # Draw uniform text column background and text
        self._draw_text_column(overlay, scale)
        
        return overlay
    
    def _draw_rays(self, overlay, dot_positions, scale):
        """Draw radiating rays connecting corresponding dots."""
        max_dots = max(len(dots) for dots in dot_positions.values())
        
        for dot_index in range(max_dots):
            points = []
            for gauge in sorted(dot_positions.keys()):
                if dot_index < len(dot_positions[gauge]):
                    points.append(dot_positions[gauge][dot_index])
            
            # Connect the points
            for i in range(len(points) - 1):
                cv2.line(overlay, 
                        (int(points[i][0]), int(points[i][1])),
                        (int(points[i+1][0]), int(points[i+1][1])),
                        (255, 255, 255, 180), max(1, int(scale)))
    
    def _draw_perpendicular_lines(self, overlay, dot_positions, 
                                start_x, end_x, scale):
        """Draw horizontal lines at gauge positions."""
        for gauge, dots in dot_positions.items():
            if not dots:
                continue
                
            y = int(dots[0][1])  # Y position from first dot
            
            # Major lines (whole and half) are thicker
            is_major = (gauge % 0.5 == 0)
            thickness = max(2, int(3 * scale)) if is_major else max(1, int(2 * scale))
            alpha = 255 if is_major else 180
            
            # Draw horizontal line
            line_start = start_x - int(15 * scale)
            line_end = end_x + int(15 * scale)
            cv2.line(overlay, (line_start, y), (line_end, y), 
                    (255, 255, 255, alpha), thickness)
    
    def _draw_dots(self, overlay, dot_positions, scale):
        """Draw white dots on top of everything."""
        dot_radius = max(3, int(4 * scale))
        
        for dots in dot_positions.values():
            for x, y in dots:
                cv2.circle(overlay, (int(x), int(y)), dot_radius, 
                          (255, 255, 255, 255), -1)
    
    def _draw_text_column(self, overlay, scale):
        """Draw uniform background column and text labels."""
        if not hasattr(self, '_text_labels') or not self._text_labels:
            return
            
        # Find the widest label to determine column width
        max_width = 0
        for text_info in self._text_labels:
            text_size = cv2.getTextSize(text_info['label'], cv2.FONT_HERSHEY_SIMPLEX, 
                                      text_info['font_scale'], text_info['thickness'])[0]
            max_width = max(max_width, text_size[0])
        
        # Add padding to column width
        column_width = max_width + int(8 * scale)
        column_start_x = 2
        column_end_x = column_start_x + column_width
        
        # Find the range of Y positions
        min_y = min(info['y'] for info in self._text_labels) - int(12 * scale)
        max_y = max(info['y'] for info in self._text_labels) + int(8 * scale)
        
        # Draw uniform background column
        cv2.rectangle(overlay, (column_start_x, min_y), (column_end_x, max_y), 
                     (255, 255, 255, 200), -1)
        
        # Draw all text labels on the uniform background
        for text_info in self._text_labels:
            cv2.putText(overlay, text_info['label'], 
                       (column_start_x + int(4 * scale), text_info['y'] + int(2 * scale)), 
                       cv2.FONT_HERSHEY_SIMPLEX, text_info['font_scale'], 
                       (0, 0, 0, 255), text_info['thickness'])
        
        # Clear the labels for next use
        self._text_labels = []
    
    def rotate_90(self, overlay):
        """Rotate overlay 90 degrees for vertical use."""
        return cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)


def create_gauge_overlays():
    """Create final gauge overlays."""
    
    stamp_path = "/Users/stanbrown/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif"
    
    if not os.path.exists(stamp_path):
        print(f"Stamp not found: {stamp_path}")
        return
        
    image = cv2.imread(stamp_path)
    if image is None:
        print("Could not load image")
        return
        
    height, width = image.shape[:2]
    gauge = FinalPerforationGauge(dpi=800)
    
    # Create horizontal overlay
    h_overlay = gauge.create_gauge_overlay(width, height)
    result_h = image.copy()
    
    # Apply overlay
    for y in range(height):
        for x in range(width):
            if h_overlay[y, x, 3] > 0:
                alpha = h_overlay[y, x, 3] / 255.0
                for c in range(3):
                    result_h[y, x, c] = int((1 - alpha) * result_h[y, x, c] + 
                                          alpha * h_overlay[y, x, c])
    
    cv2.imwrite("/Users/stanbrown/Desktop/StampZ-III/FINAL_horizontal.jpg", result_h)
    print("✓ Created: FINAL_horizontal.jpg")
    
    # Create vertical overlay (rotated)
    v_overlay = gauge.rotate_90(h_overlay)
    result_v = image.copy()
    
    for y in range(height):
        for x in range(width):
            if (x < v_overlay.shape[1] and y < v_overlay.shape[0] and 
                v_overlay[y, x, 3] > 0):
                alpha = v_overlay[y, x, 3] / 255.0
                for c in range(3):
                    result_v[y, x, c] = int((1 - alpha) * result_v[y, x, c] + 
                                          alpha * v_overlay[y, x, c])
    
    cv2.imwrite("/Users/stanbrown/Desktop/StampZ-III/FINAL_vertical.jpg", result_v)
    print("✓ Created: FINAL_vertical.jpg")
    
    print("\nFINAL PERFORATION GAUGE:")
    print("• Traditional white lines and dots, black text")
    print("• ASCII fractions: 8, 8 1/4, 8 1/2, 8 3/4, 9...")
    print("• 3 quarter rows between whole numbers")
    print("• Perfect alignment of lines and dots")
    print("• Single design rotates for vertical use")
    print("• Scales with DPI automatically")


if __name__ == "__main__":
    create_gauge_overlays()