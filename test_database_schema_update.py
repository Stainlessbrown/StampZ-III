#!/usr/bin/env python3
"""
Test script to verify database schema updates and Plot_3D column support.

This script tests:
1. Database schema migration (adding new Plot_3D columns)
2. Saving Plot_3D extended values to database
3. Retrieving Plot_3D data from database
4. Database data persistence
"""

import sys
import os
import sqlite3
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.color_analysis_db import ColorAnalysisDB

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_database_schema_migration():
    """Test that the database schema correctly migrates to include Plot_3D columns."""
    print("🧪 Testing Database Schema Migration...")
    
    # Create a test database
    test_db = ColorAnalysisDB("TEST_SCHEMA_MIGRATION")
    
    # Check that the database has all the new columns
    with sqlite3.connect(test_db.db_path) as conn:
        cursor = conn.execute("PRAGMA table_info(color_measurements)")
        columns = [row[1] for row in cursor.fetchall()]
        
        print(f"Database columns: {columns}")
        
        required_columns = [
            'marker_preference', 'color_preference', 'cluster_id', 'delta_e',
            'centroid_x', 'centroid_y', 'centroid_z', 'sphere_color', 
            'sphere_radius', 'trendline_valid'
        ]
        
        missing_columns = []
        for col in required_columns:
            if col not in columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        else:
            print(f"✅ All required Plot_3D columns present in database")
            return True

def test_save_and_retrieve_plot3d_data():
    """Test saving and retrieving Plot_3D extended data."""
    print("\n🧪 Testing Save and Retrieve Plot_3D Data...")
    
    # Create a test database
    test_db = ColorAnalysisDB("TEST_PLOT3D_DATA")
    
    # Create a test measurement set
    set_id = test_db.create_measurement_set("TestImage_S1", "Test measurement set")
    if not set_id:
        print("❌ Failed to create measurement set")
        return False
    
    # Save a test measurement with basic Lab data
    success = test_db.save_color_measurement(
        set_id=set_id,
        coordinate_point=1,
        x_pos=100.0,
        y_pos=150.0,
        l_value=75.5,
        a_value=-2.3,
        b_value=8.7,
        rgb_r=180,
        rgb_g=175,
        rgb_b=170,
        sample_type="test",
        sample_size="3x3",
        sample_anchor="center"
    )
    
    if not success:
        print("❌ Failed to save basic measurement")
        return False
    
    # Update with Plot_3D extended values
    success = test_db.update_plot3d_extended_values(
        image_name="TestImage_S1",
        coordinate_point=1,
        cluster_id=2,
        delta_e=4.8,
        centroid_x=0.755,
        centroid_y=0.487,
        centroid_z=0.534,
        sphere_color="red",
        sphere_radius=0.05,
        marker="o",
        color="green",
        trendline_valid=True
    )
    
    if not success:
        print("❌ Failed to save Plot_3D extended values")
        return False
    
    # Retrieve the data and verify
    measurements = test_db.get_all_measurements()
    
    if not measurements:
        print("❌ No measurements retrieved")
        return False
    
    measurement = measurements[0]
    
    # Verify all the data is correct
    expected_values = {
        'cluster_id': 2,
        'delta_e': 4.8,
        'centroid_x': 0.755,
        'centroid_y': 0.487,
        'centroid_z': 0.534,
        'sphere_color': "red",
        'sphere_radius': 0.05,
        'marker_preference': "o",
        'color_preference': "green",
        'trendline_valid': True
    }
    
    all_correct = True
    for key, expected_value in expected_values.items():
        actual_value = measurement.get(key)
        if actual_value != expected_value:
            print(f"❌ Mismatch for {key}: expected {expected_value}, got {actual_value}")
            all_correct = False
    
    if all_correct:
        print(f"✅ All Plot_3D data saved and retrieved correctly")
        print(f"   Cluster: {measurement.get('cluster_id')}")
        print(f"   ΔE: {measurement.get('delta_e')}")
        print(f"   Centroid: ({measurement.get('centroid_x')}, {measurement.get('centroid_y')}, {measurement.get('centroid_z')})")
        print(f"   Sphere: {measurement.get('sphere_color')} (radius: {measurement.get('sphere_radius')})")
        print(f"   Display: {measurement.get('marker_preference')} marker, {measurement.get('color_preference')} color")
        return True
    else:
        return False

def test_multiple_measurements():
    """Test handling multiple measurements with different Plot_3D data."""
    print("\n🧪 Testing Multiple Measurements...")
    
    # Create a test database
    test_db = ColorAnalysisDB("TEST_MULTIPLE_MEASUREMENTS")
    
    # Create a test measurement set
    set_id = test_db.create_measurement_set("TestImage_S2", "Multiple measurements test")
    
    # Test data for multiple coordinate points
    test_data = [
        {
            'coord_point': 1,
            'cluster': 0,
            'delta_e': 2.1,
            'marker': '.',
            'color': 'blue',
            'sphere': 'cyan'
        },
        {
            'coord_point': 2,
            'cluster': 1,
            'delta_e': 5.7,
            'marker': 'o',
            'color': 'red',
            'sphere': 'orange'
        },
        {
            'coord_point': 3,
            'cluster': 0,
            'delta_e': 1.9,
            'marker': '*',
            'color': 'green',
            'sphere': 'lime'
        }
    ]
    
    # Save basic measurements
    for i, data in enumerate(test_data):
        success = test_db.save_color_measurement(
            set_id=set_id,
            coordinate_point=data['coord_point'],
            x_pos=100.0 + i * 10,
            y_pos=150.0 + i * 10,
            l_value=75.0 + i,
            a_value=-2.0 + i * 0.5,
            b_value=8.0 + i * 0.3,
            rgb_r=180 - i * 5,
            rgb_g=175 - i * 3,
            rgb_b=170 - i * 2
        )
        
        if not success:
            print(f"❌ Failed to save basic measurement {data['coord_point']}")
            return False
        
        # Update with Plot_3D data
        success = test_db.update_plot3d_extended_values(
            image_name="TestImage_S2",
            coordinate_point=data['coord_point'],
            cluster_id=data['cluster'],
            delta_e=data['delta_e'],
            marker=data['marker'],
            color=data['color'],
            sphere_color=data['sphere']
        )
        
        if not success:
            print(f"❌ Failed to save Plot_3D data for measurement {data['coord_point']}")
            return False
    
    # Retrieve and verify
    measurements = test_db.get_all_measurements()
    
    if len(measurements) != len(test_data):
        print(f"❌ Expected {len(test_data)} measurements, got {len(measurements)}")
        return False
    
    # Sort measurements by coordinate point for comparison
    measurements.sort(key=lambda x: x['coordinate_point'])
    
    all_correct = True
    for i, (measurement, expected) in enumerate(zip(measurements, test_data)):
        coord_point = measurement.get('coordinate_point')
        cluster = measurement.get('cluster_id')
        delta_e = measurement.get('delta_e')
        
        if coord_point != expected['coord_point']:
            print(f"❌ Coordinate point mismatch: expected {expected['coord_point']}, got {coord_point}")
            all_correct = False
        
        if cluster != expected['cluster']:
            print(f"❌ Cluster mismatch for point {coord_point}: expected {expected['cluster']}, got {cluster}")
            all_correct = False
        
        if abs(delta_e - expected['delta_e']) > 0.01:
            print(f"❌ ΔE mismatch for point {coord_point}: expected {expected['delta_e']}, got {delta_e}")
            all_correct = False
    
    if all_correct:
        print(f"✅ All {len(test_data)} measurements saved and retrieved correctly")
        return True
    else:
        return False

def cleanup_test_databases():
    """Clean up test databases."""
    print("\n🧹 Cleaning up test databases...")
    
    test_names = ["TEST_SCHEMA_MIGRATION", "TEST_PLOT3D_DATA", "TEST_MULTIPLE_MEASUREMENTS"]
    
    for test_name in test_names:
        try:
            test_db = ColorAnalysisDB(test_name)
            if os.path.exists(test_db.db_path):
                os.remove(test_db.db_path)
                print(f"   Removed: {os.path.basename(test_db.db_path)}")
        except Exception as e:
            print(f"   Warning: Could not remove {test_name}.db: {e}")

def main():
    """Run all database tests."""
    print("🚀 Starting Database Schema Update Tests...")
    print("=" * 50)
    
    tests = [
        test_database_schema_migration,
        test_save_and_retrieve_plot3d_data,
        test_multiple_measurements
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {test_func.__name__}: {status}")
    
    print(f"\n🏁 Overall: {passed}/{total} tests passed")
    
    # Cleanup
    cleanup_test_databases()
    
    if passed == total:
        print("\n🎉 All database schema tests PASSED! The database is ready for Plot_3D integration.")
        return True
    else:
        print(f"\n❌ {total - passed} tests FAILED. Database schema updates need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)