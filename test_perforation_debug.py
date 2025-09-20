#!/usr/bin/env python3
"""
Debug test for perforation measurement improvements
"""

import cv2
import numpy as np
from perforation_measurement_system import PerforationMeasurementEngine

def test_with_debug():
    """Test perforation detection with debug output."""
    print("🔍 Testing Perforation Detection with Debug Output")
    print("=" * 60)
    
    engine = PerforationMeasurementEngine()
    engine.set_image_dpi(800)  # Match user's DPI setting
    
    # Create a test image that simulates a cropped stamp with 14 gauge perforations
    # At 800 DPI: 20mm = ~630 pixels, so 14 holes per 20mm = holes every 45 pixels
    test_image = create_cropped_stamp_image()
    cv2.imwrite("debug_test_stamp.jpg", test_image)
    print("📸 Created test stamp image: debug_test_stamp.jpg")
    
    # Test edge detection
    print("\n🔍 Testing edge detection...")
    edges = engine.detect_stamp_edges(test_image)
    
    # Test hole detection on each edge
    print("\n🕳️ Testing hole detection...")
    total_holes = 0
    
    for edge_name, points in edges.items():
        if points:
            print(f"\n--- {edge_name.upper()} EDGE ---")
            holes = engine.detect_perforation_holes(test_image, points)
            print(f"Final result: {len(holes)} holes detected on {edge_name} edge")
            total_holes += len(holes)
            
            if holes:
                # Show hole spacing
                if len(holes) > 1:
                    spacings = []
                    for i in range(len(holes) - 1):
                        h1, h2 = holes[i], holes[i + 1]
                        distance = ((h2.center_x - h1.center_x)**2 + (h2.center_y - h1.center_y)**2)**0.5
                        spacings.append(distance)
                    avg_spacing = sum(spacings) / len(spacings)
                    print(f"Average hole spacing: {avg_spacing:.1f} pixels")
    
    print(f"\n📊 SUMMARY: {total_holes} total holes detected")
    
    # Full analysis
    print("\n📏 Running full analysis...")
    analysis = engine.measure_perforation(test_image)
    
    print(f"\nResults:")
    print(f"  Catalog Gauge: {analysis.catalog_gauge}")
    print(f"  Precise Measurement: {analysis.overall_gauge:.3f}")
    print(f"  Edges analyzed: {len(analysis.edges)}")
    
    if analysis.warnings:
        print("\n⚠️ Warnings:")
        for warning in analysis.warnings:
            print(f"  • {warning}")

def create_cropped_stamp_image():
    """Create a cropped stamp image with known perforation gauge."""
    # Create image representing a tightly cropped stamp
    # 800 DPI, 14 gauge = holes every ~45 pixels (20mm/14 holes * 800DPI/25.4mm/inch)
    
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add stamp content
    cv2.rectangle(img, (20, 20), (580, 380), (150, 180, 200), -1)
    cv2.putText(img, "TEST 14x13.5", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Calculate spacing for 14 gauge (top/bottom) and 13.5 gauge (left/right)
    # At 800 DPI: 20mm = ~630 pixels
    # 14 gauge: 630/14 = 45 pixels spacing
    # 13.5 gauge: 630/13.5 = 46.7 pixels spacing
    
    hole_radius = 8  # Reasonable for 800 DPI
    
    # Top edge - 14 gauge (45 pixel spacing)
    for x in range(45, 555, 45):  # Start at 45, spacing of 45
        cv2.circle(img, (x, 10), hole_radius, (0, 0, 0), -1)
    
    # Bottom edge - 14 gauge
    for x in range(45, 555, 45):
        cv2.circle(img, (x, 390), hole_radius, (0, 0, 0), -1)
    
    # Left edge - 13.5 gauge (47 pixel spacing)
    for y in range(47, 353, 47):  # Start at 47, spacing of 47
        cv2.circle(img, (10, y), hole_radius, (0, 0, 0), -1)
    
    # Right edge - 13.5 gauge
    for y in range(47, 353, 47):
        cv2.circle(img, (590, y), hole_radius, (0, 0, 0), -1)
    
    return img

if __name__ == "__main__":
    test_with_debug()