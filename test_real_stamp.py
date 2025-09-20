#!/usr/bin/env python3
"""
Test perforation detection on a real stamp image.
"""

import cv2
import numpy as np
import os
from perforation_measurement_system import PerforationMeasurementEngine

def test_real_stamp():
    """Test perforation detection with a real stamp image."""
    print("=== Testing Real Stamp Image ===")
    
    # Path to your stamp image
    stamp_path = os.path.expanduser("~/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif")
    
    if not os.path.exists(stamp_path):
        print(f"❌ Image not found: {stamp_path}")
        print("Please check the path and try again.")
        return
    
    print(f"📸 Loading image: {stamp_path}")
    
    # Load the image
    try:
        image = cv2.imread(stamp_path)
        if image is None:
            print(f"❌ Could not load image: {stamp_path}")
            return
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    print(f"✅ Image loaded successfully")
    print(f"   Shape: {image.shape}")
    print(f"   Intensity range: {image.min()} to {image.max()}")
    
    # Create perforation measurement engine
    engine = PerforationMeasurementEngine()
    engine.set_image_dpi(800)  # Actual scan resolution
    engine.set_background_color('black')  # Assume black background
    
    print(f"🔧 Engine settings:")
    print(f"   DPI: {engine.dpi}")
    print(f"   Background: {engine.background_color}")
    
    # Run hole-based detection
    print(f"\n🔍 Running perforation measurement...")
    try:
        # Test tic-based detection (like Stamp Analyser methodology)
        analysis = engine.measure_perforation(image, use_hole_detection=False)
        
        print(f"\n📊 Results:")
        print(f"   Overall Gauge: {analysis.overall_gauge:.3f}")
        print(f"   Catalog Gauge: {analysis.catalog_gauge}")
        print(f"   Edges Detected: {len(analysis.edges)}")
        print(f"   Quality: {analysis.measurement_quality}")
        
        if analysis.edges:
            print(f"\n📏 Edge Details:")
            for edge in analysis.edges:
                print(f"   {edge.edge_type.upper()}: {len(edge.holes)} holes, gauge {edge.gauge_measurement:.2f} ({engine.format_gauge_for_catalog(edge.gauge_measurement)})")
                print(f"      Confidence: {edge.measurement_confidence:.1%}")
                print(f"      Edge length: {edge.total_length_pixels:.1f} pixels")
        
        if analysis.is_compound_perforation:
            print(f"\n🔀 Compound Perforation:")
            print(f"   {analysis.compound_description}")
        
        if analysis.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in analysis.warnings:
                print(f"   • {warning}")
        
        if analysis.technical_notes:
            print(f"\n🔧 Technical Notes:")
            for note in analysis.technical_notes:
                print(f"   • {note}")
        
        # Try to save results
        try:
            log_file = engine.export_to_data_logger(analysis, stamp_path, ".")
            if log_file:
                print(f"\n💾 Results saved to: {log_file}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_stamp()