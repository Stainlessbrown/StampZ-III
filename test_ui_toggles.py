#!/usr/bin/env python3
"""
Test script to verify UI toggle fixes work correctly
"""

import sys
import os

# Add current directory to path for imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

def main():
    print("🔧 Testing UI Toggle Fixes...")
    print("This script will:")
    print("1. Launch the ternary plot window")
    print("2. You should see debugging messages when clicking toggles")
    print("3. Test each toggle to verify they respond")
    
    try:
        from utils.color_data_bridge import ColorDataBridge
        from gui.ternary_plot_window import TernaryPlotWindow
        import tkinter as tk
        
        # Check for available databases
        bridge = ColorDataBridge()
        sample_sets = bridge.get_available_sample_sets()
        
        if not sample_sets:
            print("❌ No databases found - please run color analysis in StampZ-III first")
            return False
        
        print(f"📊 Using database: {sample_sets[0]}")
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()
        
        # Load data directly
        color_points, db_format = bridge.load_color_points_from_database(sample_sets[0])
        print(f"✅ Loaded {len(color_points)} points, format: {db_format}")
        
        # Create ternary window with data loaded
        ternary_window = TernaryPlotWindow(
            parent=None,
            sample_set_name=sample_sets[0],
            color_points=color_points,
            db_format=db_format
        )
        
        print("\n🔧 UI TOGGLE TESTING:")
        print("✅ Window created with improved event bindings")
        print("📋 Test these controls and watch console for debug messages:")
        print("   1. ☐ Convex Hull checkbox")
        print("   2. ☐ K-means Clusters checkbox") 
        print("   3. 🔢 Clusters spinbox (change from 3 to another value)")
        print("   4. ☐ Spheres checkbox")
        print("")
        print("💡 You should see 'DEBUG: ... checkbox clicked!' messages")
        print("💡 Spinbox changes should trigger after 200ms delay")
        print("💡 All changes should update the plot visually")
        
        # Show the window for testing
        ternary_window.window.deiconify()
        ternary_window.window.lift()
        
        # Keep the window alive for testing
        try:
            ternary_window.window.mainloop()
        except Exception as e:
            print(f"Window closed: {e}")
        finally:
            root.destroy()
            
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ UI toggle test completed!")
        print("If you saw debug messages when clicking toggles, the fix worked!")
    else:
        print("\n❌ UI toggle test failed!")
        sys.exit(1)