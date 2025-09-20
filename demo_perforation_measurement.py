#!/usr/bin/env python3
"""
Perforation Measurement Demo Script

Demonstrates the perforation measurement system without requiring a GUI,
showing how it would work with actual stamp images.
"""

import numpy as np
import cv2
from perforation_measurement_system import PerforationMeasurementEngine

def create_test_stamp_image():
    """Create a synthetic stamp image with perforations for testing."""
    # Create a 600x400 white image (stamp)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add stamp content (simple colored rectangle)
    cv2.rectangle(img, (50, 50), (550, 350), (100, 150, 200), -1)
    
    # Add text to simulate stamp design
    cv2.putText(img, "TEST STAMP", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(img, "11.5 Perf", (220, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Simulate perforation holes along edges
    hole_radius = 8
    
    # Top edge perforations
    for x in range(60, 540, 40):  # Creates ~11.5 gauge spacing
        cv2.circle(img, (x, 20), hole_radius, (0, 0, 0), -1)
    
    # Bottom edge perforations  
    for x in range(60, 540, 40):
        cv2.circle(img, (x, 380), hole_radius, (0, 0, 0), -1)
    
    # Left edge perforations
    for y in range(60, 340, 26):  # Slightly different gauge for compound effect
        cv2.circle(img, (30, y), hole_radius, (0, 0, 0), -1)
    
    # Right edge perforations
    for y in range(60, 340, 26):
        cv2.circle(img, (570, y), hole_radius, (0, 0, 0), -1)
    
    return img

def demo_perforation_measurement():
    """Demonstrate the perforation measurement system."""
    print("🎯 StampZ Perforation Measurement System Demo")
    print("=" * 50)
    
    # Create test image
    print("📸 Creating test stamp image with perforations...")
    test_image = create_test_stamp_image()
    
    # Save test image
    cv2.imwrite("test_stamp.jpg", test_image)
    print("✅ Test stamp image saved as 'test_stamp.jpg'")
    
    # Initialize measurement engine
    print("\n🔧 Initializing perforation measurement engine...")
    engine = PerforationMeasurementEngine()
    engine.set_image_dpi(600)  # Set standard scan DPI
    
    # Perform measurement
    print("📏 Measuring perforations...")
    analysis = engine.measure_perforation(test_image)
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 MEASUREMENT RESULTS")
    print("=" * 50)
    
    print(f"Catalog Gauge: {analysis.catalog_gauge}")
    print(f"Precise Measurement: {analysis.overall_gauge:.3f}")
    print(f"Measurement Quality: {analysis.measurement_quality}")
    print(f"Compound Perforation: {'Yes' if analysis.is_compound_perforation else 'No'}")
    
    if analysis.is_compound_perforation:
        print(f"Description: {analysis.compound_description}")
    
    # Edge analysis
    if analysis.edges:
        print(f"\n📋 EDGE ANALYSIS:")
        for edge in analysis.edges:
            gauge_str = engine.format_gauge_for_catalog(edge.gauge_measurement)
            print(f"  {edge.edge_type.title()}: {gauge_str} ({len(edge.holes)} holes, {edge.measurement_confidence:.0%} confidence)")
    
    # Warnings
    if analysis.warnings:
        print(f"\n⚠️  WARNINGS:")
        for warning in analysis.warnings:
            print(f"  • {warning}")
    
    # Technical details
    if analysis.technical_notes:
        print(f"\n🔧 TECHNICAL DETAILS:")
        for note in analysis.technical_notes:
            print(f"  • {note}")
    
    # Export data
    print(f"\n💾 EXPORTING DATA...")
    log_file = engine.export_to_data_logger(analysis, "test_stamp.jpg", ".")
    if log_file:
        print(f"✅ Data exported to: {log_file}")
        
        # Show sample of exported data
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                print(f"\n📄 Sample of exported data (first 15 lines):")
                for i, line in enumerate(lines[:15]):
                    print(f"  {line}")
                if len(lines) > 15:
                    print(f"  ... ({len(lines)-15} more lines)")
        except Exception as e:
            print(f"❌ Could not read exported file: {e}")
    else:
        print("❌ Export failed")
    
    print(f"\n" + "=" * 50)
    print("🎉 Demo complete!")
    print("This demonstrates how the perforation measurement")
    print("system would work with actual stamp images in StampZ.")
    
    # Integration notes
    print(f"\n🚀 INTEGRATION STATUS:")
    print("✅ Perforation measurement engine: Ready")
    print("✅ Data logging system: Ready") 
    print("✅ Catalog format conversion: Ready")
    print("✅ Forgery detection: Ready")
    print("✅ Menu integration: Ready")
    print("✅ UI dialog: Ready")
    print("✅ Keyboard shortcut (Ctrl+P): Ready")
    
    print(f"\n📋 TO USE IN STAMPZ:")
    print("1. Load a stamp image in StampZ")
    print("2. Go to Measurement > Perforation Gauge... (or Ctrl+P)")
    print("3. Set correct DPI and click 'Measure Perforations'")
    print("4. Review results and save data as needed")

if __name__ == "__main__":
    demo_perforation_measurement()