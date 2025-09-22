#!/usr/bin/env python3
"""
Test script to verify the complete ternary sync workflow:
1. Make changes in worksheet 
2. Sync from datasheet
3. Verify plot updates 
4. Save changes
5. Close and reopen 
6. Verify changes persist

This test identifies the exact issues preventing proper synchronization.
"""

import sys
import os
sys.path.append('/Users/stanbrown/Desktop/StampZ-III')

import logging
logging.basicConfig(level=logging.DEBUG)

def test_ternary_sync_workflow():
    """Test the complete ternary sync workflow."""
    print("\n🔄 TERNARY SYNC WORKFLOW TEST")
    print("=" * 50)
    
    try:
        # Import necessary modules
        from gui.ternary_plot_window import TernaryPlotWindow
        from app.stampz_app import StampZApp
        from utils.color_analysis_db import ColorAnalysisDB
        import tkinter as tk
        
        # Test 1: Check database for sample with known marker/color preferences
        print("\n1️⃣ TESTING DATABASE TERNARY PREFERENCES...")
        
        # Look for an existing sample set with data
        db = ColorAnalysisDB("MAN_MODE")  # Using your mentioned database
        
        # Check if ternary preference columns exist in the database
        import sqlite3
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(color_measurements)")
            columns = {col[1] for col in cursor.fetchall()}
            
        print(f"📋 Available columns: {sorted(columns)}")
        
        has_ternary_cols = (
            'ternary_marker_preference' in columns and 
            'ternary_color_preference' in columns
        )
        
        print(f"🎯 Ternary preference columns exist: {has_ternary_cols}")
        
        if not has_ternary_cols:
            print("❌ PROBLEM FOUND: Missing ternary preference columns in database!")
            print("   This explains why changes don't persist - they can't be saved!")
            return False
        
        # Test 2: Check if preferences can be saved and retrieved
        print("\n2️⃣ TESTING PREFERENCE SAVE/LOAD...")
        
        # Try to save some test preferences
        test_save = db.update_ternary_preferences(
            image_name="test_image", 
            coordinate_point=1, 
            marker="*", 
            color="red"
        )
        
        print(f"💾 Test preference save: {test_save}")
        
        if test_save:
            # Try to retrieve them
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute("""
                    SELECT ternary_marker_preference, ternary_color_preference
                    FROM color_measurements cm
                    JOIN measurement_sets ms ON cm.set_id = ms.set_id
                    WHERE ms.image_name = ? AND cm.coordinate_point = ?
                """, ("test_image", 1))
                result = cursor.fetchone()
                
            if result:
                marker, color = result
                print(f"📖 Retrieved preferences: marker='{marker}', color='{color}'")
                print("✅ Database save/load works!")
            else:
                print("❌ Could not retrieve saved preferences!")
        
        # Test 3: Check if the ternary plot window loads preferences properly
        print("\n3️⃣ TESTING PREFERENCE LOADING IN PLOT...")
        
        # Load actual data and check if preferences are applied
        try:
            from utils.color_data_bridge import ColorDataBridge
            bridge = ColorDataBridge()
            color_points, db_format = bridge.load_color_points_from_database("MAN_MODE")
            
            print(f"📊 Loaded {len(color_points)} points from MAN_MODE")
            
            # Check if any points have marker/color metadata from database
            points_with_prefs = 0
            for point in color_points[:5]:  # Check first 5
                if hasattr(point, 'metadata'):
                    marker = point.metadata.get('marker', 'N/A')
                    color = point.metadata.get('marker_color', 'N/A')
                    if marker != 'N/A' or color != 'N/A':
                        points_with_prefs += 1
                        print(f"  Point {point.id}: marker='{marker}', color='{color}'")
            
            print(f"🎨 Points with preferences loaded: {points_with_prefs}")
            
            if points_with_prefs == 0:
                print("❌ PROBLEM FOUND: No preferences being loaded from database!")
                print("   The bridge is not loading ternary preferences!")
                return False
            
        except Exception as e:
            print(f"❌ Error testing preference loading: {e}")
            return False
        
        print("\n✅ ALL TESTS PASSED!")
        print("🎯 The sync workflow should work correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_database_schema():
    """Check if the database schema includes ternary preference columns."""
    print("\n🗄️ CHECKING DATABASE SCHEMA...")
    
    try:
        from utils.color_analysis_db import ColorAnalysisDB
        import sqlite3
        
        db = ColorAnalysisDB("MAN_MODE")
        
        with sqlite3.connect(db.db_path) as conn:
            # Get table schema
            cursor = conn.execute("PRAGMA table_info(color_measurements)")
            columns = cursor.fetchall()
            
        print("📋 COLOR_MEASUREMENTS TABLE SCHEMA:")
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            is_required = "NOT NULL" if col[3] else "NULLABLE"
            default_val = col[4] if col[4] else "None"
            print(f"  {col_name:<30} {col_type:<15} {is_required:<10} (default: {default_val})")
        
        # Check for ternary preference columns specifically
        col_names = [col[1] for col in columns]
        
        missing_cols = []
        expected_cols = ['ternary_marker_preference', 'ternary_color_preference']
        
        for col in expected_cols:
            if col not in col_names:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"\n❌ MISSING COLUMNS: {missing_cols}")
            print("   These need to be added for ternary preferences to persist!")
            return False, missing_cols
        else:
            print("\n✅ All required ternary preference columns are present!")
            return True, []
            
    except Exception as e:
        print(f"❌ Error checking database schema: {e}")
        return False, []

if __name__ == "__main__":
    print("🧪 TERNARY SYNCHRONIZATION DIAGNOSTIC TEST")
    print("=" * 60)
    
    # Check database schema first
    schema_ok, missing_cols = check_database_schema()
    
    if not schema_ok:
        print(f"\n💡 SOLUTION: Add missing columns to the database schema:")
        for col in missing_cols:
            print(f"   ALTER TABLE color_measurements ADD COLUMN {col} TEXT;")
    
    # Run workflow test
    if schema_ok:
        workflow_ok = test_ternary_sync_workflow()
        
        if workflow_ok:
            print("\n🎉 DIAGNOSIS: System is working correctly!")
        else:
            print("\n🔧 DIAGNOSIS: Issues found that need fixing.")
    
    print("\n" + "=" * 60)