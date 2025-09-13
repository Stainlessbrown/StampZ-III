#!/usr/bin/env python3
"""
Diagnostic script to test the actual worksheet loading process for MAN_MODE.db
This simulates the complete _refresh_from_stampz workflow to identify the exact issue.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.color_analysis_db import ColorAnalysisDB

# Set up logging to capture the actual logging output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def simulate_refresh_from_stampz():
    """Simulate the complete _refresh_from_stampz workflow."""
    
    print("🧪 SIMULATING COMPLETE WORKSHEET LOADING WORKFLOW")
    print("=" * 70)
    
    sample_set_name = "MAN_MODE"
    
    # Step 1: Get measurements from database (same as worksheet)
    print(f"\n📊 STEP 1: Loading from Database")
    db = ColorAnalysisDB(sample_set_name)
    measurements = db.get_all_measurements()
    
    print(f"Database query returned: {len(measurements)} measurements")
    
    if not measurements:
        print("❌ No measurements found!")
        return 0, 0
    
    # Step 2: Process measurements (exactly like _refresh_from_stampz)
    print(f"\n🔄 STEP 2: Processing Measurements")
    
    use_normalized = True
    data_rows = []
    skipped_count = 0
    
    for i, measurement in enumerate(measurements):
        try:
            print(f"  Processing measurement {i+1}...")
            
            # Get Lab values
            l_val = measurement.get('l_value', None)
            a_val = measurement.get('a_value', None)  
            b_val = measurement.get('b_value', None)
            
            print(f"    Lab values: L*={l_val}, a*={a_val}, b*={b_val}")
            
            # Apply normalization (exact same logic as worksheet)
            if use_normalized:
                # Check for missing coordinate data (exact same logic as fixed worksheet)
                if l_val is None or a_val is None or b_val is None:
                    print(f"    ❌ SKIPPED: Missing coordinate data")
                    skipped_count += 1
                    continue
                    
                x_norm = l_val / 100.0
                y_norm = (a_val + 128) / 256.0
                z_norm = (b_val + 128) / 256.0
                
                # Constrain to 0-1 range
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
                z_norm = max(0.0, min(1.0, z_norm))
                
            else:
                # Raw Lab values
                if l_val is None or a_val is None or b_val is None:
                    print(f"    ❌ SKIPPED: Missing coordinate data")
                    skipped_count += 1
                    continue
                    
                x_norm = l_val
                y_norm = a_val
                z_norm = b_val
            
            print(f"    Normalized: X={x_norm:.4f}, Y={y_norm:.4f}, Z={z_norm:.4f}")
            
            # Create DataID (exact same logic)
            image_name = measurement.get('image_name', f"{sample_set_name}_Sample_{i+1:03d}")
            coordinate_point = measurement.get('coordinate_point', 1)
            data_id = f"{image_name}_pt{coordinate_point}"
            
            print(f"    DataID: {data_id}")
            
            # Get Plot_3D data (exact same logic as fixed worksheet)
            saved_marker = measurement.get('marker_preference', '.')
            saved_color = measurement.get('color_preference', 'blue')
            saved_cluster = measurement.get('cluster_id', '')
            saved_delta_e = measurement.get('delta_e', '')
            saved_centroid_x = measurement.get('centroid_x', '')
            saved_centroid_y = measurement.get('centroid_y', '')
            saved_centroid_z = measurement.get('centroid_z', '')
            saved_sphere_color = measurement.get('sphere_color', '')
            saved_sphere_radius = measurement.get('sphere_radius', '')
            
            row = [
                round(x_norm, 4),                   # Xnorm  
                round(y_norm, 4),                   # Ynorm
                round(z_norm, 4),                   # Znorm
                data_id,                            # DataID
                str(saved_cluster) if saved_cluster is not None else '',  # Cluster
                str(saved_delta_e) if saved_delta_e is not None else '',  # ∆E
                saved_marker,                       # Marker
                saved_color,                        # Color
                str(saved_centroid_x) if saved_centroid_x is not None else '',  # Centroid_X
                str(saved_centroid_y) if saved_centroid_y is not None else '',  # Centroid_Y
                str(saved_centroid_z) if saved_centroid_z is not None else '',  # Centroid_Z
                str(saved_sphere_color) if saved_sphere_color else '',          # Sphere
                str(saved_sphere_radius) if saved_sphere_radius is not None else ''  # Radius
            ]
            
            data_rows.append(row)
            print(f"    ✅ Row created successfully")
            
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            skipped_count += 1
            continue
    
    print(f"\nProcessing complete: {len(data_rows)} rows created, {skipped_count} skipped")
    
    # Step 3: Simulate sheet insertion process
    print(f"\n📝 STEP 3: Simulating Sheet Insertion")
    
    if data_rows:
        print(f"  Would insert {len(data_rows)} rows starting at sheet row 7 (display row 8)")
        
        # Check if this matches what you see
        print(f"\n🔍 EXPECTED RESULT IN WORKSHEET:")
        print(f"  Display rows 8-{7+len(data_rows)} should contain data")
        print(f"  Total data rows expected: {len(data_rows)}")
        
        # Show first 10 rows that should appear
        print(f"\nFirst 10 rows that should appear in worksheet:")
        for i, row in enumerate(data_rows[:10]):
            display_row = 8 + i  # Row 7 -> display row 8
            print(f"  Display row {display_row:2d}: {row[3]} (Xnorm={row[0]:.4f})")
            
        if len(data_rows) > 10:
            print(f"  ... and {len(data_rows)-10} more rows")
    
    return len(data_rows), len(measurements)

def check_database_consistency():
    """Check if there are any database consistency issues."""
    
    print(f"\n🔬 STEP 4: Database Consistency Check")
    
    db = ColorAnalysisDB("MAN_MODE")
    
    # Check for duplicate measurements
    measurements = db.get_all_measurements()
    
    # Group by image_name and coordinate_point
    seen_combinations = set()
    duplicates = []
    
    for i, measurement in enumerate(measurements):
        image_name = measurement.get('image_name')
        coord_point = measurement.get('coordinate_point')
        combination = (image_name, coord_point)
        
        if combination in seen_combinations:
            duplicates.append((i+1, image_name, coord_point))
        else:
            seen_combinations.add(combination)
    
    if duplicates:
        print(f"  ⚠️ Found {len(duplicates)} duplicate measurements:")
        for idx, img, coord in duplicates:
            print(f"    Row {idx}: {img}_pt{coord}")
    else:
        print(f"  ✅ No duplicate measurements found")
    
    # Check for NULL coordinate values
    null_coordinates = []
    for i, measurement in enumerate(measurements):
        l_val = measurement.get('l_value')
        a_val = measurement.get('a_value')
        b_val = measurement.get('b_value')
        
        if l_val is None or a_val is None or b_val is None:
            image_name = measurement.get('image_name')
            coord_point = measurement.get('coordinate_point')
            null_coordinates.append((i+1, f"{image_name}_pt{coord_point}", l_val, a_val, b_val))
    
    if null_coordinates:
        print(f"  ⚠️ Found {len(null_coordinates)} measurements with NULL coordinates:")
        for idx, data_id, l, a, b in null_coordinates:
            print(f"    Row {idx}: {data_id} L*={l}, a*={a}, b*={b}")
    else:
        print(f"  ✅ All measurements have valid coordinate data")

if __name__ == "__main__":
    try:
        processed_count, total_count = simulate_refresh_from_stampz()
        check_database_consistency()
        
        print("\n" + "=" * 70)
        print("📊 FINAL DIAGNOSIS")
        
        print(f"\nDatabase contains: {total_count} measurements")
        print(f"Should be loaded: {processed_count} measurements")
        print(f"You reported seeing: 10 measurements")
        
        if processed_count != 10:
            print(f"\n🚨 MISMATCH IDENTIFIED!")
            print(f"   Expected: {processed_count} rows")
            print(f"   Actual: 10 rows")
            print(f"   Missing: {processed_count - 10} rows")
            
            if processed_count == total_count:
                print(f"\n💡 The issue is likely in the worksheet insertion process, not data processing.")
                print(f"   All data processes correctly, but something limits the display to 10 rows.")
                print(f"   Check: tksheet row limits, validation setup limits, or insertion loop issues.")
            else:
                print(f"\n💡 The issue is in the data processing - some rows are being skipped.")
        else:
            print(f"\n✅ Expected count matches what you see!")
            print(f"   This suggests the issue might be elsewhere.")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()