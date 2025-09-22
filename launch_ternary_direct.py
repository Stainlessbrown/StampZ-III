#!/usr/bin/env python3
"""
Direct Ternary Launcher - No complex dialogs

Just launches the ternary viewer directly with the first available database.
Perfect for testing and daily use.
"""

import sys
import os
sys.path.append('/Users/stanbrown/Desktop/StampZ-III')

def main():
    print("🎨 Direct Ternary Launcher")
    print("=" * 30)
    
    try:
        from utils.color_data_bridge import ColorDataBridge
        from gui.ternary_plot_window import TernaryPlotWindow
        import tkinter as tk
        
        # Get available databases
        bridge = ColorDataBridge()
        sample_sets = bridge.get_available_sample_sets()
        
        print(f"Found {len(sample_sets)} databases: {sample_sets}")
        
        if not sample_sets:
            print("❌ No databases found!")
            print("💡 Run color analysis in StampZ-III first to create databases.")
            return
        
        # Use first database (usually MAN_MODE)
        selected_db = sample_sets[0]
        print(f"🚀 Launching ternary plot for: {selected_db}")
        
        # Load data
        points, db_format = bridge.load_color_points_from_database(selected_db)
        print(f"📊 Loaded {len(points)} points (Format: {db_format})")
        
        # Create ternary window
        root = tk.Tk()
        root.withdraw()  # Hide root
        
        ternary_window = TernaryPlotWindow(
            parent=root,
            sample_set_name=selected_db,
            color_points=points,
            db_format=db_format
        )
        
        print("✅ Ternary window launched!")
        print("")
        print("🔧 TEST THE SYNC FUNCTIONALITY:")
        print("1. Click '📊 Open Datasheet' in the ternary window")  
        print("2. Change some markers/colors in the datasheet")
        print("3. Click '↻ Sync from Datasheet' - should update the plot")
        print("4. Click '↻ Reload Database' - changes should persist")
        print("")
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()