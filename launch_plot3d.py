#!/usr/bin/env python3
"""
Direct Plot_3D Standalone Launcher
Quick launch script for Plot_3D without going through StampZ-III
"""

import os
import sys

def main():
    """Launch Plot_3D directly."""
    # Add the current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Import and run the standalone launcher
    from standalone_plot3d import main as run_standalone
    run_standalone()

if __name__ == "__main__":
    main()
