#!/usr/bin/env python3
"""
Standalone Plot_3D Application

This module provides a standalone launcher for the Plot_3D Data Manager,
allowing users to run the 3D analysis and visualization features independently
from the main StampZ-III application.

Usage:
    python3 plot3d/standalone_plot3d.py
    
Or from the launch selector dialog.
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add parent directory to path to access plot3d modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_standalone_environment():
    """Set up the environment for standalone Plot_3D operation."""
    # Ensure required directories exist
    required_dirs = [
        'databases',
        'exports'
    ]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")


def main(auto_load_file=None):
    """Launch the standalone Plot_3D Data Manager.
    
    Args:
        auto_load_file: Optional path to automatically load a file on startup
    """
    print("Starting Plot_3D Data Manager in standalone mode...")
    
    try:
        # Set up environment
        setup_standalone_environment()
        
        # Import and launch Plot_3D
        from plot3d.Plot_3D import Plot3DApp
        
        # Create and configure Plot_3D application in standalone mode
        # Note: Plot3DApp creates its own root window when parent=None
        plot3d_app = Plot3DApp(parent=None, data_path=auto_load_file)  # standalone mode with optional auto-load
        
        print("Plot_3D Data Manager initialized successfully")
        print("Available features:")
        print("  • K-means clustering analysis")
        print("  • ΔE calculations and comparisons")  
        print("  • 3D visualization with interactive controls")
        print("  • Trend line and statistical analysis")
        print("  • Data import/export capabilities")
        print()
        print("To get started, use File → Open Database to load existing analysis data")
        print("or File → Import Data to load CSV files for analysis.")
        
        # Plot3DApp handles its own mainloop when parent=None
        
    except ImportError as e:
        error_msg = f"Failed to import Plot_3D module: {e}"
        print(f"Error: {error_msg}")
        messagebox.showerror("Import Error", error_msg)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Failed to launch Plot_3D Data Manager: {e}"
        print(f"Error: {error_msg}")
        messagebox.showerror("Launch Error", error_msg)
        sys.exit(1)


def launch_plot3d_with_database(database_name):
    """Launch Plot_3D directly with a specific database without showing dialog.
    
    Args:
        database_name (str): Name of the database/sample set to load
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n🚀 DIRECT DATABASE LAUNCH: {database_name}")
        
        # Verify database exists
        from utils.color_data_bridge import ColorDataBridge
        bridge = ColorDataBridge()
        
        sample_sets = bridge.get_available_sample_sets()
        if database_name not in sample_sets:
            print(f"❌ Database not found: {database_name}")
            print(f"Available databases: {sample_sets}")
            return False
        
        print(f"✅ Database found: {database_name}")
        
        # Launch Plot_3D with the specific database
        from plot3d.Plot_3D import Plot3DApp
        import threading
        import tkinter as tk
        
        def launch_with_db():
            try:
                # Create root window if needed (for threading)
                if not hasattr(tk, '_default_root') or tk._default_root is None:
                    root = tk.Tk()
                    root.withdraw()  # Hide the root window initially
                else:
                    root = None
                
                # Create Plot_3D app with database
                app = Plot3DApp(
                    sample_set_name=database_name
                )
                
                print(f"✅ Successfully launched Plot_3D with database: {database_name}")
                
                # If we created a root, start mainloop
                if root:
                    root.mainloop()
                    
            except Exception as e:
                print(f"❌ Error launching Plot_3D with database: {e}")
                import traceback
                traceback.print_exc()
        
        # Launch in separate thread to avoid blocking
        thread = threading.Thread(target=launch_with_db, daemon=False)  # Don't use daemon for main app
        thread.start()
        
        print(f"✅ Plot_3D launch thread started for database: {database_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to launch Plot_3D with database {database_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
