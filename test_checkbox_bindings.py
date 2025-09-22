#!/usr/bin/env python3
"""
Quick test to verify checkbox bindings work
"""

import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')

def main():
    print("🔧 Testing Checkbox Event Bindings...")
    
    try:
        from utils.color_data_bridge import ColorDataBridge
        from gui.ternary_plot_window import TernaryPlotWindow
        import tkinter as tk
        
        # Get database
        bridge = ColorDataBridge()
        sample_sets = bridge.get_available_sample_sets()
        
        if not sample_sets:
            print("❌ No databases found")
            return
        
        # Create window
        root = tk.Tk()
        root.withdraw()
        
        color_points, db_format = bridge.load_color_points_from_database(sample_sets[0])
        
        ternary_window = TernaryPlotWindow(
            parent=None,
            sample_set_name=sample_sets[0],
            color_points=color_points,
            db_format=db_format
        )
        
        # Show window
        ternary_window.window.deiconify()
        ternary_window.window.lift()
        
        print(f"✅ Window created")
        print(f"📋 Initial states:")
        print(f"   Convex Hull: {ternary_window.show_hull.get()}")
        print(f"   Clusters: {ternary_window.show_clusters.get()}")
        print(f"   Spheres: {ternary_window.show_spheres.get()}")
        
        print(f"\n🔄 Click checkboxes and watch for debug messages:")
        print(f"   - Hull checkbox should show: 'DEBUG: Convex Hull checkbox clicked!'")
        print(f"   - Clusters checkbox should show: 'DEBUG: K-means Clusters checkbox clicked!'")
        print(f"   - Spheres checkbox should show: 'DEBUG: Spheres checkbox clicked!'")
        
        print(f"\n💡 If you see the convex hull (red dashed lines + yellow fill), it's working!")
        print(f"💡 If you see large red 'X' markers, clusters are working!")
        
        # Keep window open
        try:
            ternary_window.window.mainloop()
        except:
            pass
        finally:
            root.destroy()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()