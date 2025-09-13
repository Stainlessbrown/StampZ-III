#!/usr/bin/env python3
"""
Direct Plot_3D Launcher

This script provides direct command-line access to the Plot_3D Data Manager
without going through the launch selector dialog.

Usage:
    python3 plot3d_launcher.py

This is useful for:
- Command line users who prefer direct access
- Scripts and automation
- Quick testing and development
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add current directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from plot3d.standalone_plot3d import main
        main()
    except Exception as e:
        print(f"Failed to launch Plot_3D: {e}")
        sys.exit(1)
