#!/usr/bin/env python3
"""
Test the improved perforation measurement system
"""

import cv2
import numpy as np
from perforation_measurement_system import PerforationMeasurementEngine

def test_perforation_detection():
    """Test perforation detection with improved algorithms."""
    print("🧪 Testing Improved Perforation Detection")
    print("=" * 50)
    
    engine = PerforationMeasurementEngine()
    engine.set_image_dpi(600)
    
    # Try to load the test image we created earlier
    try:
        test_image = cv2.imread("test_stamp.jpg")
        if test_image is not None:
            print("📸 Using existing test stamp image...")
        else:
            print("📸 Creating new test stamp image...")
            test_image = create_better_test_image()
            cv2.imwrite("test_stamp_improved.jpg", test_image)
    except:
        print("📸 Creating new test stamp image...")
        test_image = create_better_test_image()
        cv2.imwrite("test_stamp_improved.jpg", test_image)
    
    # Test edge detection
    print("\n🔍 Testing edge detection...")
    edges = engine.detect_stamp_edges(test_image)
    
    for edge_name, points in edges.items():
        print(f"  {edge_name}: {len(points)} edge points")
    
    # Test hole detection on each edge
    print("\n🕳️ Testing hole detection...")
    total_holes = 0
    
    for edge_name, points in edges.items():
        if points:
            holes = engine.detect_perforation_holes(test_image, points)
            print(f"  {edge_name}: {len(holes)} holes detected")
            total_holes += len(holes)
    
    print(f"\n📊 Total holes detected: {total_holes}")
    
    # Full analysis
    print("\n📏 Running full analysis...")
    analysis = engine.measure_perforation(test_image)
    
    print(f"Catalog Gauge: {analysis.catalog_gauge}")
    print(f"Measurement Quality: {analysis.measurement_quality}")
    print(f"Number of edges analyzed: {len(analysis.edges)}")
    
    if analysis.warnings:
        print("\n⚠️ Warnings:")
        for warning in analysis.warnings:
            print(f"  • {warning}")
    
    print("\n✅ Test complete!")

def create_better_test_image():
    """Create a more realistic test stamp image."""
    # Create larger image for better detection
    img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
    
    # Add stamp design with more realistic content
    cv2.rectangle(img, (100, 100), (1100, 700), (180, 140, 120), -1)  # Stamp background
    cv2.rectangle(img, (150, 150), (1050, 650), (200, 160, 140), -1)  # Inner design
    
    # Add text
    cv2.putText(img, "STAMP", (400, 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
    cv2.putText(img, "TEST", (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    
    # Create more realistic perforation holes
    hole_radius = 12
    
    # Top edge - gauge 11 (spacing ~55 pixels at 600 DPI)
    for x in range(150, 1050, 55):
        cv2.circle(img, (x, 50), hole_radius, (0, 0, 0), -1)
        # Add slight irregularity to simulate real perforations
        cv2.circle(img, (x, 50), hole_radius + 2, (50, 50, 50), 2)
    
    # Bottom edge
    for x in range(150, 1050, 55):
        cv2.circle(img, (x, 750), hole_radius, (0, 0, 0), -1)
        cv2.circle(img, (x, 750), hole_radius + 2, (50, 50, 50), 2)
    
    # Left edge
    for y in range(150, 650, 55):
        cv2.circle(img, (50, y), hole_radius, (0, 0, 0), -1)
        cv2.circle(img, (50, y), hole_radius + 2, (50, 50, 50), 2)
    
    # Right edge
    for y in range(150, 650, 55):
        cv2.circle(img, (1150, y), hole_radius, (0, 0, 0), -1)
        cv2.circle(img, (1150, y), hole_radius + 2, (50, 50, 50), 2)
    
    return img

if __name__ == "__main__":
    test_perforation_detection()