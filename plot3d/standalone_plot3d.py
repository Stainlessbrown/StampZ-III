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


def main():
    """Launch the standalone Plot_3D Data Manager."""
    print("Starting Plot_3D Data Manager in standalone mode...")
    
    try:
        # Set up environment
        setup_standalone_environment()
        
        # Import and launch Plot_3D
        from plot3d.Plot_3D import Plot3DApp
        
        # Create and configure Plot_3D application in standalone mode
        # Note: Plot3DApp creates its own root window when parent=None
        plot3d_app = Plot3DApp(parent=None)  # standalone mode
        
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


if __name__ == "__main__":
    main()
