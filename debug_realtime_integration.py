#!/usr/bin/env python3
"""
Debug script to test realtime worksheet integration issues.

This tests the specific problems reported:
1. Row indexing/offset issues 
2. Data changes not being reflected in Plot_3D
3. Centroid sphere issues
4. Trendline issues
"""

import os
import sys
import pandas as pd

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def test_refresh_mechanism():
    """Test the Plot_3D refresh mechanism with DataFrame mode."""
    print("🔍 Testing Plot_3D refresh mechanism...")
    
    try:
        # Simulate the get_data_as_dataframe output from internal worksheet
        sample_data = {
            'Xnorm': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Ynorm': [0.2, 0.3, 0.4, 0.5, 0.6], 
            'Znorm': [0.3, 0.4, 0.5, 0.6, 0.7],
            'DataID': ['S10_pt1', 'S10_pt2', 'S10_pt3', 'S12_pt1', 'S12_pt2'],
            'Cluster': ['', '', '', '', ''],  # Empty initially
            'Marker': ['o', '*', '.', 'o', '*'],
            'Color': ['blue', 'red', 'green', 'yellow', 'purple'],
            'Sphere': ['', '', '', '', ''],
            'Radius': ['', '', '', '', ''],
            'Centroid_X': ['', '', '', '', ''],
            'Centroid_Y': ['', '', '', '', ''], 
            'Centroid_Z': ['', '', '', '', ''],
            '_original_sheet_row': [7, 8, 9, 11, 12]  # The preserved mapping
        }
        
        df = pd.DataFrame(sample_data)
        print(f"Created test DataFrame with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"DataIDs: {list(df['DataID'])}")
        print(f"Original sheet rows: {list(df['_original_sheet_row'])}")
        
        return df
        
    except Exception as e:
        print(f"Error in refresh mechanism test: {e}")
        import traceback
        traceback.print_exc()

def test_callback_update_simulation():
    """Simulate the callback update from Plot_3D back to worksheet."""
    print("\n🔍 Testing callback update simulation...")
    
    try:
        # Create base DataFrame (as would come from worksheet)
        df = test_refresh_mechanism()
        
        # Simulate K-means clustering results (as Plot_3D would do)
        df_with_clusters = df.copy()
        df_with_clusters.loc[0, 'Cluster'] = 0  # DataFrame index 0 -> S10_pt1 -> sheet row 7
        df_with_clusters.loc[1, 'Cluster'] = 0  # DataFrame index 1 -> S10_pt2 -> sheet row 8  
        df_with_clusters.loc[2, 'Cluster'] = 1  # DataFrame index 2 -> S10_pt3 -> sheet row 9
        df_with_clusters.loc[3, 'Cluster'] = 1  # DataFrame index 3 -> S12_pt1 -> sheet row 11
        df_with_clusters.loc[4, 'Cluster'] = 1  # DataFrame index 4 -> S12_pt2 -> sheet row 12
        
        # Add centroid information (for cluster summary rows)
        df_with_clusters.loc[0, 'Centroid_X'] = 0.15  # Cluster 0 centroid
        df_with_clusters.loc[0, 'Centroid_Y'] = 0.25
        df_with_clusters.loc[0, 'Centroid_Z'] = 0.35
        df_with_clusters.loc[1, 'Centroid_X'] = 0.15  # Same for all cluster 0 points
        df_with_clusters.loc[1, 'Centroid_Y'] = 0.25
        df_with_clusters.loc[1, 'Centroid_Z'] = 0.35
        
        df_with_clusters.loc[2, 'Centroid_X'] = 0.45  # Cluster 1 centroid
        df_with_clusters.loc[2, 'Centroid_Y'] = 0.55
        df_with_clusters.loc[2, 'Centroid_Z'] = 0.65
        df_with_clusters.loc[3, 'Centroid_X'] = 0.45  # Same for all cluster 1 points
        df_with_clusters.loc[3, 'Centroid_Y'] = 0.55
        df_with_clusters.loc[3, 'Centroid_Z'] = 0.65
        df_with_clusters.loc[4, 'Centroid_X'] = 0.45
        df_with_clusters.loc[4, 'Centroid_Y'] = 0.55
        df_with_clusters.loc[4, 'Centroid_Z'] = 0.65
        
        print(f"\nDataFrame after K-means clustering:")
        for idx, row in df_with_clusters.iterrows():
            orig_sheet_row = int(row['_original_sheet_row'])
            data_id = row['DataID']
            cluster = row['Cluster']
            print(f"  DataFrame idx {idx} -> sheet row {orig_sheet_row} (display {orig_sheet_row+1}), DataID: {data_id}, Cluster: {cluster}")
        
        # Now simulate the callback update logic
        print(f"\n🔄 Simulating callback update to worksheet:")
        
        # Test the mapping logic from the actual callback
        for df_idx, df_row in df_with_clusters.iterrows():
            if '_original_sheet_row' in df_row:
                sheet_row_idx = int(df_row['_original_sheet_row'])
                cluster_value = df_row.get('Cluster', '')
                data_id = df_row.get('DataID', '')
                
                if not pd.isna(cluster_value) and cluster_value != '':
                    print(f"  ✅ Would write cluster '{cluster_value}' to sheet row {sheet_row_idx} (display E{sheet_row_idx+1}) for DataID '{data_id}'")
                else:
                    print(f"  ❌ No cluster value to write for DataFrame index {df_idx}, DataID '{data_id}'")
            else:
                print(f"  ❌ No _original_sheet_row mapping for DataFrame index {df_idx}")
        
        return df_with_clusters
        
    except Exception as e:
        print(f"Error in callback update simulation: {e}")
        import traceback
        traceback.print_exc()

def test_data_synchronization_issues():
    """Test potential data synchronization issues."""
    print("\n🔍 Testing data synchronization issues...")
    
    try:
        # Issue 1: Test manual changes to worksheet not reflected in Plot_3D
        print("\n1. Manual worksheet changes not reflected in Plot_3D:")
        
        # Original DataFrame from worksheet
        original_df = test_refresh_mechanism()
        print(f"   Original markers: {list(original_df['Marker'])}")
        print(f"   Original colors: {list(original_df['Color'])}")
        
        # User changes marker/color in worksheet
        modified_df = original_df.copy()
        modified_df.loc[0, 'Marker'] = 's'  # Changed from 'o' to 's'
        modified_df.loc[1, 'Color'] = 'orange'  # Changed from 'red' to 'orange' 
        modified_df.loc[2, 'Radius'] = '0.05'  # Added radius for sphere
        
        print(f"   Modified markers: {list(modified_df['Marker'])}")
        print(f"   Modified colors: {list(modified_df['Color'])}")
        print(f"   Added radius: {list(modified_df['Radius'])}")
        
        # Issue 2: Test Plot_3D refresh synchronization
        print(f"\n2. Plot_3D refresh should pick up these changes:")
        print(f"   - DataFrame index 0 (S10_pt1) marker: 'o' -> 's'")
        print(f"   - DataFrame index 1 (S10_pt2) color: 'red' -> 'orange'")
        print(f"   - DataFrame index 2 (S10_pt3) radius: '' -> '0.05'")
        
        # Issue 3: Test centroid sphere functionality
        print(f"\n3. Centroid sphere issues:")
        
        # Add centroid spheres data
        centroid_df = modified_df.copy()
        
        # Add sphere data for centroid visualization
        centroid_df.loc[0, 'Sphere'] = 'centroid'  # Mark as centroid sphere
        centroid_df.loc[0, 'Radius'] = '0.08'      # Centroid sphere radius
        centroid_df.loc[0, 'Color'] = 'red'        # Centroid sphere color
        
        print(f"   Sphere column: {list(centroid_df['Sphere'])}")
        print(f"   Radius column: {list(centroid_df['Radius'])}")
        
        # Issue 4: Test trendline data preparation
        print(f"\n4. Trendline data preparation:")
        
        # Check if data has the required 'trendline_valid' flag
        if 'trendline_valid' not in centroid_df.columns:
            print(f"   ❌ Missing 'trendline_valid' column - this might cause trendline issues")
            # Add the missing column
            centroid_df['trendline_valid'] = True  # All points valid for trendline
            print(f"   ✅ Added 'trendline_valid' column")
        
        print(f"   Trendline valid points: {centroid_df['trendline_valid'].sum()}/{len(centroid_df)}")
        
        return centroid_df
        
    except Exception as e:
        print(f"Error in data synchronization test: {e}")
        import traceback
        traceback.print_exc()

def test_realtime_integration_flow():
    """Test the complete realtime integration flow."""
    print("\n🔍 Testing complete realtime integration flow...")
    
    try:
        # Step 1: Initial worksheet data
        print("Step 1: Initial worksheet data creation")
        df_initial = test_refresh_mechanism()
        
        # Step 2: User opens Plot_3D with DataFrame
        print("\nStep 2: User opens Plot_3D with DataFrame")
        print(f"   Plot_3D receives DataFrame with {len(df_initial)} rows")
        print(f"   DataFrame mode: file_path = None")
        print(f"   Callback registered for bidirectional sync")
        
        # Step 3: User makes manual changes in worksheet
        print(f"\nStep 3: User makes manual changes in worksheet")
        df_manual_changes = df_initial.copy()
        df_manual_changes.loc[0, 'Color'] = 'cyan'  # Manual change
        df_manual_changes.loc[1, 'Marker'] = 'D'   # Manual change
        print(f"   Changed color for DataID {df_manual_changes.loc[0, 'DataID']}")
        print(f"   Changed marker for DataID {df_manual_changes.loc[1, 'DataID']}")
        
        # Step 4: User clicks "Refresh Plot_3D" or "Push Changes to Plot_3D"
        print(f"\nStep 4: User clicks refresh/push changes button")
        print(f"   This should call: plot3d_app.df = new_dataframe")
        print(f"   This should call: plot3d_app.refresh_plot()")
        print(f"   Expected result: Plot_3D shows updated markers/colors")
        
        # Step 5: User runs K-means in Plot_3D
        print(f"\nStep 5: User runs K-means in Plot_3D")
        df_with_kmeans = df_manual_changes.copy()
        df_with_kmeans.loc[0:1, 'Cluster'] = 0  # Cluster 0
        df_with_kmeans.loc[2:4, 'Cluster'] = 1  # Cluster 1
        print(f"   K-means assigns clusters to DataFrame indices 0-4")
        print(f"   Should trigger callback: worksheet_update_callback(updated_df)")
        
        # Step 6: Callback updates worksheet
        print(f"\nStep 6: Callback updates worksheet with cluster results")
        for idx, row in df_with_kmeans.iterrows():
            if '_original_sheet_row' in row:
                sheet_row = int(row['_original_sheet_row'])
                cluster = row.get('Cluster', '')
                data_id = row.get('DataID', '')
                print(f"   DataFrame idx {idx} -> sheet row {sheet_row} (E{sheet_row+1}), DataID {data_id}, Cluster {cluster}")
        
        print(f"\n🎯 EXPECTED OUTCOME:")
        print(f"   - Worksheet should show cluster assignments in correct rows")
        print(f"   - No +6 or +7 row offset")
        print(f"   - Manual color/marker changes should be visible in Plot_3D")
        print(f"   - Centroid spheres should work when configured")
        print(f"   - Trendlines should work with valid data")
        
        print(f"\n❗ ACTUAL PROBLEMS REPORTED:")
        print(f"   - Manual changes in worksheet are ignored by Plot_3D")
        print(f"   - Centroid sphere toggle appears but spheres don't show")
        print(f"   - Trendlines don't work")
        print(f"   - Row indexing may still have offset issues")
        
        return df_with_kmeans
        
    except Exception as e:
        print(f"Error in integration flow test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting realtime integration debug tests...")
    
    # Test 1: Basic refresh mechanism
    df_test1 = test_refresh_mechanism()
    
    # Test 2: Callback update simulation
    df_test2 = test_callback_update_simulation()
    
    # Test 3: Data synchronization issues
    df_test3 = test_data_synchronization_issues()
    
    # Test 4: Complete integration flow
    df_test4 = test_realtime_integration_flow()
    
    print(f"\n✅ Debug tests completed!")
    print(f"\nKey findings for investigation:")
    print(f"1. Check if Plot_3D properly updates when df is reassigned")
    print(f"2. Check if refresh_plot() method works correctly in DataFrame mode")
    print(f"3. Check if sphere_manager and trendline_manager have proper references")
    print(f"4. Check if _original_sheet_row mapping is working as expected")
    print(f"5. Check if callback timing/synchronization is causing issues")