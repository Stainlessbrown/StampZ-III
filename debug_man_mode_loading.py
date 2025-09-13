#!/usr/bin/env python3
"""
Diagnostic script to investigate why only 10 rows are loading from MAN_MODE.db
when the database contains all 20 rows.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.color_analysis_db import ColorAnalysisDB

def diagnose_man_mode_loading():
    """Diagnose why MAN_MODE.db is only showing 10 rows instead of 20."""
    
    print("🔍 DIAGNOSING MAN_MODE.DB LOADING ISSUE")
    print("=" * 60)
    
    # Step 1: Check what's in the database directly
    print("\n📊 STEP 1: Database Content Analysis")
    db = ColorAnalysisDB("MAN_MODE")
    measurements = db.get_all_measurements()
    
    print(f"Database contains: {len(measurements)} measurements")
    
    if not measurements:
        print("❌ No measurements found in database!")
        return
    
    # Show details of all measurements
    print(f"\n📝 All measurements in database:")
    for i, measurement in enumerate(measurements):
        l_val = measurement.get('l_value', 'N/A')
        a_val = measurement.get('a_value', 'N/A')
        b_val = measurement.get('b_value', 'N/A')
        image_name = measurement.get('image_name', 'N/A')
        coord_point = measurement.get('coordinate_point', 'N/A')
        data_id = f"{image_name}_pt{coord_point}" if image_name != 'N/A' and coord_point != 'N/A' else 'N/A'
        
        print(f"  {i+1:2d}. DataID: {data_id:<15} L*={l_val:6.2f} a*={a_val:6.2f} b*={b_val:6.2f}")
    
    # Step 2: Simulate the normalization process
    print(f"\n🔄 STEP 2: Simulating Data Processing")
    
    # Simulate the same logic from _refresh_from_stampz
    data_rows = []
    skipped_count = 0
    
    for i, measurement in enumerate(measurements):
        try:
            # Get Lab values
            l_val = measurement.get('l_value', 0.0)
            a_val = measurement.get('a_value', 0.0)
            b_val = measurement.get('b_value', 0.0)
            
            # Apply normalization (same logic as realtime worksheet)
            x_norm = l_val / 100.0 if l_val else 0.0
            y_norm = (a_val + 128) / 256.0 if a_val else 0.5
            z_norm = (b_val + 128) / 256.0 if b_val else 0.5
            
            # Constrain to 0-1 range
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            z_norm = max(0.0, min(1.0, z_norm))
            
            # Create DataID
            image_name = measurement.get('image_name', f"MAN_MODE_Sample_{i+1:03d}")
            coordinate_point = measurement.get('coordinate_point', 1)
            data_id = f"{image_name}_pt{coordinate_point}"
            
            # Get Plot_3D data
            saved_marker = measurement.get('marker_preference', '.')
            saved_color = measurement.get('color_preference', 'blue')
            saved_cluster = measurement.get('cluster_id', '')
            saved_delta_e = measurement.get('delta_e', '')
            
            row = [
                round(x_norm, 4),                   # Xnorm  
                round(y_norm, 4),                   # Ynorm
                round(z_norm, 4),                   # Znorm
                data_id,                            # DataID
                str(saved_cluster) if saved_cluster is not None else '',  # Cluster
                str(saved_delta_e) if saved_delta_e is not None else '',  # ∆E
                saved_marker,                       # Marker
                saved_color,                        # Color
                '',                                 # Centroid_X
                '',                                 # Centroid_Y
                '',                                 # Centroid_Z
                '',                                 # Sphere
                ''                                  # Radius
            ]
            
            data_rows.append(row)
            
            print(f"  ✅ {i+1:2d}. {data_id:<15} → X={x_norm:.4f}, Y={y_norm:.4f}, Z={z_norm:.4f}")
            
        except Exception as e:
            print(f"  ❌ {i+1:2d}. Error processing measurement: {e}")
            skipped_count += 1
    
    print(f"\nProcessing results: {len(data_rows)} rows created, {skipped_count} skipped")
    
    # Step 3: Simulate DataFrame filtering
    print(f"\n🔍 STEP 3: Simulating DataFrame Filtering")
    
    # Plot_3D column structure
    PLOT3D_COLUMNS = [
        'Xnorm', 'Ynorm', 'Znorm', 'DataID', 'Cluster', 
        '∆E', 'Marker', 'Color', 'Centroid_X', 'Centroid_Y', 
        'Centroid_Z', 'Sphere', 'Radius'
    ]
    
    # Create DataFrame (same as get_data_as_dataframe)
    df = pd.DataFrame(data_rows, columns=PLOT3D_COLUMNS)
    print(f"DataFrame created with {len(df)} rows")
    
    # Show first few rows
    print(f"\nFirst 5 DataFrame rows:")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"  {i}: Xnorm={row['Xnorm']}, Ynorm={row['Ynorm']}, Znorm={row['Znorm']}, DataID={row['DataID']}")
    
    # Clean the data - this is where filtering might happen
    df = df.replace('', np.nan)
    print(f"After replacing empty strings with NaN: {len(df)} rows")
    
    # Check coordinate data
    coordinate_cols = ['Xnorm', 'Ynorm', 'Znorm']
    has_coordinate_data = df[coordinate_cols].notna().any(axis=1)
    
    print(f"\nCoordinate data analysis:")
    print(f"  Rows with at least some coordinate data: {has_coordinate_data.sum()}")
    print(f"  Rows without coordinate data: {(~has_coordinate_data).sum()}")
    
    # Show which rows are being filtered out
    filtered_out_indices = df.index[~has_coordinate_data].tolist()
    if filtered_out_indices:
        print(f"\n❌ Rows being filtered out (no coordinate data):")
        for idx in filtered_out_indices:
            row = df.iloc[idx]
            print(f"  Row {idx}: Xnorm={row['Xnorm']}, Ynorm={row['Ynorm']}, Znorm={row['Znorm']}, DataID={row['DataID']}")
    
    # Apply the filter
    df_filtered = df[has_coordinate_data].copy()
    print(f"\nAfter coordinate filtering: {len(df_filtered)} rows remain")
    
    if len(df_filtered) != len(measurements):
        print(f"🚨 FILTERING ISSUE FOUND!")
        print(f"  Database has: {len(measurements)} measurements")
        print(f"  DataFrame after filtering: {len(df_filtered)} rows")
        print(f"  Missing: {len(measurements) - len(df_filtered)} rows")
    else:
        print(f"✅ No filtering issues - all rows preserved")
    
    # Step 4: Check for data type issues
    print(f"\n🔬 STEP 4: Data Type Analysis")
    
    print(f"Coordinate column data types:")
    for col in coordinate_cols:
        dtype = df[col].dtype
        non_null_count = df[col].notna().sum()
        unique_values = df[col].unique()[:10]  # First 10 unique values
        print(f"  {col}: dtype={dtype}, non-null={non_null_count}, sample values={unique_values}")
    
    # Check for zero values specifically
    print(f"\nZero value analysis:")
    for col in coordinate_cols:
        zero_count = (df[col] == 0.0).sum()
        near_zero_count = ((df[col] >= -0.001) & (df[col] <= 0.001)).sum()
        print(f"  {col}: exact zeros={zero_count}, near-zeros={near_zero_count}")
    
    return len(df_filtered), len(measurements)

if __name__ == "__main__":
    try:
        filtered_count, total_count = diagnose_man_mode_loading()
        
        print("\n" + "=" * 60)
        print("📊 DIAGNOSIS SUMMARY")
        
        if filtered_count < total_count:
            print(f"❌ FILTERING ISSUE CONFIRMED!")
            print(f"   Database: {total_count} measurements")
            print(f"   Loaded: {filtered_count} measurements")
            print(f"   Lost: {total_count - filtered_count} measurements")
            print(f"\n💡 The issue is in the data filtering logic in get_data_as_dataframe()")
            print(f"   Rows are being filtered out during coordinate data validation.")
        else:
            print(f"✅ No filtering issues detected")
            print(f"   All {total_count} measurements should load correctly")
            print(f"\n🤔 The issue might be elsewhere in the loading pipeline")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()