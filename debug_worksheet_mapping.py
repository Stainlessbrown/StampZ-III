#!/usr/bin/env python3
"""
Debug script to test the internal worksheet DataFrame creation and row mapping logic.

This script will help identify:
1. Whether the _original_sheet_row column is being preserved correctly
2. How the DataFrame indices map to sheet row positions  
3. What DataIDs are present in both the worksheet and database
4. Whether the callback update is writing to the correct rows
"""

import os
import sys
import pandas as pd

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def test_worksheet_dataframe_creation():
    """Test the DataFrame creation logic from internal worksheet."""
    try:
        print("🔍 DEBUG: Testing worksheet DataFrame creation...")
        
        # Test the core logic without importing GUI classes
        
        print("\n1. Testing sample data structure...")
        
        # Sample data that mimics what the worksheet might contain
        sample_sheet_data = [
            # Row 0 (headers) - not included in data 
            # Rows 1-6 (cluster summary) - may contain cluster info
            ['', '', '', '', '0', '', '', '', '0.1', '0.2', '0.3'],  # Row 1 - cluster 0 summary
            ['', '', '', '', '1', '', '', '', '0.4', '0.5', '0.6'],  # Row 2 - cluster 1 summary  
            ['', '', '', '', '', '', '', '', '', '', ''],            # Row 3 - empty summary
            ['', '', '', '', '', '', '', '', '', '', ''],            # Row 4 - empty summary
            ['', '', '', '', '', '', '', '', '', '', ''],            # Row 5 - empty summary
            ['', '', '', '', '', '', '', '', '', '', ''],            # Row 6 - empty summary
            # Rows 7+ - individual data points
            ['0.1', '0.2', '0.3', 'sample1_pt1', '', 'o', 'blue', '', '', '', ''],    # Row 7
            ['0.4', '0.5', '0.6', 'sample1_pt2', '', '*', 'red', '', '', '', ''],     # Row 8
            ['0.7', '0.8', '0.9', 'sample1_pt3', '', '.', 'green', '', '', '', ''],   # Row 9
            ['', '', '', '', '', '', '', '', '', '', ''],                            # Row 10 - empty
            ['1.1', '1.2', '1.3', 'sample2_pt1', '', 'o', 'yellow', '', '', '', ''], # Row 11
            ['1.4', '1.5', '1.6', 'sample2_pt2', '', '*', 'purple', '', '', '', ''], # Row 12
        ]
        
        columns = ['Xnorm', 'Ynorm', 'Znorm', 'DataID', 'Cluster', 'Marker', 'Color', 'Sphere', 'Centroid_X', 'Centroid_Y', 'Centroid_Z']
        
        # Simulate the DataFrame creation logic
        print("\n2. Creating DataFrame from sample data...")
        df = pd.DataFrame(sample_sheet_data, columns=columns)
        
        # Clean the data - remove completely empty rows and replace empty strings with NaN
        df = df.replace('', pd.NA)
        
        print(f"Initial DataFrame shape: {df.shape}")
        print(f"Initial DataFrame indices: {list(df.index)}")
        
        # Keep rows that have at least some coordinate data
        coordinate_cols = ['Xnorm', 'Ynorm', 'Znorm']
        has_coordinate_data = df[coordinate_cols].notna().any(axis=1)
        
        print(f"\nRows with coordinate data: {has_coordinate_data.sum()}/{len(df)}")
        print(f"Rows with coordinate data (indices): {list(df.index[has_coordinate_data])}")
        
        # CRITICAL: Preserve original sheet row positions before filtering
        df['_original_sheet_row'] = df.index  # Store original sheet row indices
        
        df_filtered = df[has_coordinate_data].copy()
        
        print(f"\n3. After filtering (preserving original row mapping):")
        print(f"Filtered DataFrame shape: {df_filtered.shape}")
        print(f"Filtered DataFrame indices: {list(df_filtered.index)}")
        
        # Show the mapping between filtered indices and original sheet rows
        print(f"\n4. DataFrame-to-Sheet row mapping:")
        for i, (df_idx, row) in enumerate(df_filtered.iterrows()):
            orig_sheet_row = int(row['_original_sheet_row'])
            data_id = row['DataID'] if pd.notna(row['DataID']) else 'N/A'
            print(f"  Filtered index {i} (df_idx={df_idx}) → original sheet row {orig_sheet_row} (display {orig_sheet_row+1}), DataID: {data_id}")
        
        # Reset index to ensure consecutive numbering starting from 0
        df_filtered.reset_index(drop=True, inplace=True)
        
        print(f"\n5. After reset_index:")
        print(f"Final DataFrame shape: {df_filtered.shape}")
        print(f"Final DataFrame indices: {list(df_filtered.index)}")
        
        # Show the final mapping
        print(f"\n6. Final DataFrame-to-Sheet row mapping (what callback should use):")
        for df_idx, row in df_filtered.iterrows():
            orig_sheet_row = int(row['_original_sheet_row'])
            data_id = row['DataID'] if pd.notna(row['DataID']) else 'N/A'
            print(f"  DataFrame index {df_idx} → original sheet row {orig_sheet_row} (display {orig_sheet_row+1}), DataID: {data_id}")
        
        # Test what happens when we simulate K-means clustering on this data
        print(f"\n7. Simulating K-means clustering...")
        
        # Simulate adding cluster assignments (this is what Plot_3D would do)
        df_with_clusters = df_filtered.copy()
        cluster_assignments = [0, 0, 1, 1, 1]  # Sample cluster assignments
        
        for i, cluster in enumerate(cluster_assignments[:len(df_with_clusters)]):
            df_with_clusters.iloc[i, df_with_clusters.columns.get_loc('Cluster')] = cluster
        
        print(f"\nDataFrame with clusters:")
        for df_idx, row in df_with_clusters.iterrows():
            orig_sheet_row = int(row['_original_sheet_row'])
            data_id = row['DataID'] if pd.notna(row['DataID']) else 'N/A'
            cluster = row['Cluster'] if pd.notna(row['Cluster']) else 'N/A'
            print(f"  DataFrame index {df_idx} → sheet row {orig_sheet_row} (display {orig_sheet_row+1}), DataID: {data_id}, Cluster: {cluster}")
        
        # Test the callback update logic
        print(f"\n8. Testing callback update logic:")
        
        cluster_col_idx = 4  # Column E (0-based index)
        
        for df_idx, df_row in df_with_clusters.iterrows():
            if '_original_sheet_row' in df_row:
                sheet_row_idx = int(df_row['_original_sheet_row'])
                cluster_value = df_row.get('Cluster', '')
                
                print(f"  ✅ Would write cluster '{cluster_value}' to sheet row {sheet_row_idx} (display row {sheet_row_idx+1}, column E{sheet_row_idx+1})")
            else:
                print(f"  ❌ No _original_sheet_row mapping for DataFrame index {df_idx}")
        
        print(f"\n9. Summary:")
        print(f"  - Original sheet data rows: {len(sample_sheet_data)}")
        print(f"  - Rows with coordinates: {len(df_filtered)}")
        print(f"  - Expected data start row: 7 (display row 8)")
        print(f"  - Actual data rows: {[int(row['_original_sheet_row']) for _, row in df_filtered.iterrows()]}")
        print(f"  - Display rows with data: {[int(row['_original_sheet_row'])+1 for _, row in df_filtered.iterrows()]}")
        
        return df_with_clusters
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

def test_database_dataids():
    """Test what DataIDs are available in the database."""
    try:
        from utils.color_analysis_db import ColorAnalysisDB
        
        print(f"\n🔍 DATABASE DEBUG: Testing DataID availability...")
        
        # Try to find available sample sets
        available_sets = ColorAnalysisDB.get_all_sample_set_databases()
        print(f"Available sample sets: {available_sets}")
        
        if not available_sets:
            print("No sample set databases found - this might explain the 'No measurement found' errors")
            return
        
        # Test with the first available sample set
        sample_set = available_sets[0]
        print(f"Testing with sample set: {sample_set}")
        
        db = ColorAnalysisDB(sample_set)
        measurements = db.get_all_measurements()
        
        print(f"Total measurements in database: {len(measurements)}")
        
        if measurements:
            print(f"\nFirst 10 DataIDs in database:")
            for i, measurement in enumerate(measurements[:10]):
                print(f"  {i+1}. DataID: {measurement.get('image_name', 'N/A')}_pt{measurement.get('coordinate_point', 'N/A')}")
        
        return measurements
        
    except Exception as e:
        print(f"Error testing database: {e}")
        import traceback  
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting worksheet mapping debug tests...")
    
    # Test 1: DataFrame creation and mapping
    df_result = test_worksheet_dataframe_creation()
    
    # Test 2: Database DataID availability
    db_measurements = test_database_dataids()
    
    print(f"\n✅ Debug tests completed!")
    print(f"\nKey findings:")
    print(f"1. The _original_sheet_row preservation should work correctly")
    print(f"2. Check if database has matching DataIDs for the worksheet entries")
    print(f"3. The callback should write to the preserved original sheet row positions")
    print(f"4. Expected data rows start at row 7 (display row 8)")