#!/usr/bin/env python3
"""
Batch test multiple stamps with the monotone perforation detection method.
"""

import os
from monotone_perforation import MonotonePerforationDetector
import cv2
import numpy as np

def test_multiple_stamps():
    """Test the monotone method on multiple stamps."""
    
    # List of stamps to test
    stamp_paths = [
        "~/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif",  # Default test stamp
        "~/Desktop/2025 Color Analysis/138 - 10c red/138-S1.tif",   # Different from same series
        "~/Desktop/2025 Color Analysis/138 - 10c red/138-S2.tif",   # Another from same series
        "~/Desktop/2025 Color Analysis/134 -10c red w mound/134-S1.tif",  # Different series
        "~/Desktop/2025 Color Analysis/135 - 10c red no mound-thin/135-S1.tif",  # Another series
    ]
    
    detector = MonotonePerforationDetector(dpi=800)
    
    print("=== BATCH TESTING MONOTONE PERFORATION DETECTION ===\n")
    
    results_summary = []
    
    for i, stamp_path in enumerate(stamp_paths, 1):
        expanded_path = os.path.expanduser(stamp_path)
        
        print(f"[{i}/{len(stamp_paths)}] Testing: {os.path.basename(expanded_path)}")
        print(f"Full path: {expanded_path}")
        
        if not os.path.exists(expanded_path):
            print("❌ File not found\n")
            continue
            
        image = cv2.imread(expanded_path)
        if image is None:
            print("❌ Could not load image\n")
            continue
            
        # Run detection
        try:
            results = detector.detect_perforations(image)
            
            if results:
                # Get the best horizontal and vertical gauges
                h_gauge = results.get('horizontal_gauge', 0)
                v_gauge = results.get('vertical_gauge', 0)
                
                if h_gauge > 0 and v_gauge > 0:
                    compound_gauge = (h_gauge + v_gauge) / 2
                    
                    # Find which sides were selected
                    selected_sides = [edge for edge in ['top', 'bottom', 'left', 'right'] 
                                    if edge in results]
                    
                    print(f"✅ SUCCESS:")
                    for edge in selected_sides:
                        data = results[edge]
                        print(f"   {edge.upper()}: {data['gauge']:.2f} ({data['tics']} tics)")
                    print(f"   Compound: {compound_gauge:.2f}")
                    print(f"   Notation: {h_gauge:.1f} × {v_gauge:.1f}")
                    
                    results_summary.append({
                        'file': os.path.basename(expanded_path),
                        'compound': compound_gauge,
                        'horizontal': h_gauge,
                        'vertical': v_gauge,
                        'notation': f"{h_gauge:.1f} × {v_gauge:.1f}"
                    })
                else:
                    print("❌ No valid gauges detected")
            else:
                print("❌ No perforations detected")
                
        except Exception as e:
            print(f"❌ Error during detection: {e}")
        
        print("-" * 50)
    
    # Summary
    print("\n=== SUMMARY OF ALL RESULTS ===")
    if results_summary:
        print(f"{'File':<20} {'Compound':<10} {'Notation':<12}")
        print("-" * 45)
        for result in results_summary:
            print(f"{result['file']:<20} {result['compound']:<10.2f} {result['notation']:<12}")
            
        # Overall statistics
        compounds = [r['compound'] for r in results_summary]
        if len(compounds) > 1:
            print(f"\nStatistics across {len(compounds)} stamps:")
            print(f"  Average compound: {np.mean(compounds):.2f}")
            print(f"  Range: {min(compounds):.2f} - {max(compounds):.2f}")
            print(f"  Std deviation: {np.std(compounds):.2f}")
    else:
        print("No successful detections")

if __name__ == "__main__":
    test_multiple_stamps()