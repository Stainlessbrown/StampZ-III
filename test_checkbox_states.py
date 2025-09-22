#!/usr/bin/env python3
"""
Simple test to verify checkbox states are changing correctly
"""

import sys
import os

# Add current directory to path for imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

def main():
    print("🔧 Testing Checkbox State Changes...")
    
    try:
        from utils.color_data_bridge import ColorDataBridge
        from gui.ternary_plot_window import TernaryPlotWindow
        import tkinter as tk
        
        # Check for available databases
        bridge = ColorDataBridge()
        sample_sets = bridge.get_available_sample_sets()
        
        if not sample_sets:
            print("❌ No databases found")
            return False
        
        # Create a root window
        root = tk.Tk()
        root.withdraw()
        
        # Load data directly
        color_points, db_format = bridge.load_color_points_from_database(sample_sets[0])
        print(f"✅ Loaded {len(color_points)} points")
        
        # Create ternary window
        ternary_window = TernaryPlotWindow(
            parent=None,
            sample_set_name=sample_sets[0],
            color_points=color_points,
            db_format=db_format
        )
        
        # Show the window
        ternary_window.window.deiconify()
        ternary_window.window.lift()
        
        print(f"\n🔍 Initial States:")
        print(f"   Hull: {ternary_window.show_hull.get()}")  
        print(f"   Clusters: {ternary_window.show_clusters.get()}")
        print(f"   Spheres: {ternary_window.show_spheres.get()}")
        print(f"   N_clusters: {ternary_window.n_clusters.get()}")
        
        def test_hull_toggle():
            print(f"\n🔄 Testing Hull Toggle:")
            current = ternary_window.show_hull.get()
            print(f"   Before click: {current}")
            # Simulate checkbox click by toggling the variable
            ternary_window.show_hull.set(not current)
            new_state = ternary_window.show_hull.get()
            print(f"   After toggle: {new_state}")
            # Manually call the refresh
            ternary_window._refresh_plot()
            return new_state != current
        
        def test_clusters_toggle():
            print(f"\n🔄 Testing Clusters Toggle:")
            current = ternary_window.show_clusters.get()
            print(f"   Before click: {current}")
            # Simulate checkbox click by toggling the variable
            ternary_window.show_clusters.set(not current)
            new_state = ternary_window.show_clusters.get()
            print(f"   After toggle: {new_state}")
            # Manually call the toggle method
            ternary_window._toggle_clusters()
            return new_state != current
        
        print(f"\n📋 Click any checkbox and watch console output...")
        print(f"Or press Enter to run automated tests...")
        
        # Wait for user input or let them test manually
        try:
            user_input = input("\nPress Enter for automated tests or Ctrl+C to test manually: ")
            
            # Run automated tests
            hull_result = test_hull_toggle()
            clusters_result = test_clusters_toggle()
            
            print(f"\n✅ Test Results:")
            print(f"   Hull toggle worked: {hull_result}")
            print(f"   Clusters toggle worked: {clusters_result}")
            
        except KeyboardInterrupt:
            print(f"\n📋 Manual testing mode - click checkboxes and watch debug output")
        
        # Keep window open for testing
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
    main()