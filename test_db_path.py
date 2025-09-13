#!/usr/bin/env python3

import sys
import os

# Add project paths
sys.path.insert(0, '/Users/stanbrown/Desktop/StampZ-III')
sys.path.insert(0, '/Users/stanbrown/Desktop/StampZ-III/utils')

# Initialize environment (no explicit initialization needed)
# The import of initialize_env was sufficient

from utils.color_analysis_db import ColorAnalysisDB

def main():
    print("🔍 TESTING DATABASE PATH RESOLUTION")
    
    # Test what path ColorAnalysisDB actually uses for MAN_MODE
    db = ColorAnalysisDB("MAN_MODE")
    actual_path = db.get_database_path()
    
    print(f"📍 ColorAnalysisDB resolves MAN_MODE to: {actual_path}")
    print(f"📂 File exists: {os.path.exists(actual_path)}")
    
    if os.path.exists(actual_path):
        print(f"📅 File modified: {os.path.getmtime(actual_path)}")
        
        # Check the actual data in this database
        measurements = db.get_all_measurements()
        print(f"📊 Database contains {len(measurements)} measurements")
        
        # Show marker/color data for first few measurements
        for i, m in enumerate(measurements[:10]):
            marker = m.get('marker_preference', 'NOT_FOUND')
            color = m.get('color_preference', 'NOT_FOUND') 
            image = m.get('image_name', 'NO_NAME')
            pt = m.get('coordinate_point', 'NO_PT')
            print(f"  Row {i+1}: {image}_pt{pt} → marker={marker}, color={color}")

if __name__ == "__main__":
    main()