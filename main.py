#!/usr/bin/env python3
"""StampZ-III - Main Application Entry Point
A image analysis application optimized for philatelic images

This entry point now shows a launch selector dialog allowing users to choose between:
- Full StampZ-III Application (complete image analysis workflow)
- Plot_3D Only Mode (advanced 3D analysis and visualization)
"""

# Import initialize_env first to set up data preservation system
# Handle module loading for both development and bundled PyInstaller environments
import sys
import os

# For bundled PyInstaller apps, ensure current directory is in Python path
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in a PyInstaller bundle
    if '.' not in sys.path:
        sys.path.insert(0, '.')

import initialize_env

import tkinter as tk
import logging

logger = logging.getLogger(__name__)


def launch_full_stampz():
    """Launch the full StampZ-III application."""
    # Import the refactored application
    from app import StampZApp
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting full StampZ-III application...")
    
    # Create main window
    root = tk.Tk()
    
    try:
        # Create and run the application
        app = StampZApp(root)
        logger.info("StampZ-III application initialized successfully")
        
        # Start the main event loop
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Failed to start StampZ-III: {e}")
        import traceback
        traceback.print_exc()
        
        # Show error to user
        try:
            tk.messagebox.showerror(
                "Startup Error",
                f"Failed to start StampZ-III:\n\n{str(e)}\n\n"
                f"Please check the console for detailed error information."
            )
        except:
            print(f"CRITICAL ERROR: {e}")
    
    finally:
        try:
            root.destroy()
        except:
            pass


def launch_ternary_viewer():
    """Launch Ternary Plot Viewer."""
    try:
        print("🎨 Starting Ternary Plot Viewer...")
        
        from utils.color_data_bridge import ColorDataBridge
        from gui.ternary_plot_window import TernaryPlotWindow
        import tkinter as tk
        from tkinter import messagebox
        
        # Check for available databases
        bridge = ColorDataBridge()
        sample_sets = bridge.get_available_sample_sets()
        
        print(f"Found {len(sample_sets)} available databases: {sample_sets}")
        
        if not sample_sets:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(
                "No Databases Found",
                "No color analysis databases were found.\n\n"
                "Run color analysis in StampZ-III first to create databases."
            )
            root.destroy()
            return
        
        # Create ternary window without auto-loading any database
        # User will choose database via "Load Database" button in the window
        print("Creating ternary window - user will select database...")
        
        # Create a root window to avoid Tkinter variable errors
        root = tk.Tk()
        root.withdraw()  # Hide it immediately
        
        ternary_window = TernaryPlotWindow(
            parent=None,  # Orphaned - no parent
            sample_set_name="No Database Selected",  # Will be updated when user selects
            color_points=[],  # Empty initially
            db_format='UNKNOWN'  # Will be detected when database is loaded
        )
        
        print("Ternary window created successfully!")
        print("Use 'Load Database' button to select which database to analyze.")
        print("Starting main event loop to keep window alive...")
        
        # Keep the process alive by running the main event loop
        try:
            # Use the ternary window's mainloop since it's the main window
            ternary_window.window.mainloop()
        except Exception as e:
            print(f"Window closed or error occurred: {e}")
            root.destroy()
        
    except Exception as e:
        print(f"ERROR: Failed to start Ternary Viewer: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Ternary Viewer Launch Error",
                f"Failed to start Ternary Viewer:\n\n{str(e)}\n\n"
                f"Please check the console for detailed error information."
            )
            root.destroy()
        except:
            print(f"CRITICAL ERROR: {e}")


def launch_plot3d_only():
    """Launch Plot_3D only mode."""
    try:
        from plot3d.standalone_plot3d import main as plot3d_main
        plot3d_main()
    except Exception as e:
        logger.error(f"Failed to start Plot_3D mode: {e}")
        try:
            tk.messagebox.showerror(
                "Plot_3D Launch Error",
                f"Failed to start Plot_3D mode:\n\n{str(e)}\n\n"
                f"Please check the console for detailed error information."
            )
        except:
            print(f"CRITICAL ERROR: {e}")


def main():
    """Main entry point - shows launch mode selector."""
    try:
        # Show launch selector
        from launch_selector import LaunchSelector
        
        selector = LaunchSelector()
        selected_mode = selector.show()
        
        # Ensure the selector window is completely cleaned up
        try:
            if hasattr(selector, 'root'):
                selector.root.quit()  # Stop the mainloop
                selector.root.destroy()  # Destroy the window
        except:
            pass  # Ignore errors during cleanup
        
        if selected_mode == "full":
            launch_full_stampz()
            # Force complete process termination when main app closes
            sys.exit(0)
            
        elif selected_mode == "ternary":
            launch_ternary_viewer()
            # Force complete process termination when Ternary Viewer closes
            sys.exit(0)
            
        elif selected_mode == "plot3d":
            launch_plot3d_only()
            # Force complete process termination when Plot_3D closes
            sys.exit(0)
            
        else:
            # User cancelled or closed dialog
            print("Launch cancelled by user")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error during launch: {e}")
        logging.error(f"Launch selector error: {e}")
        # Fallback to full application if selector fails
        print("Falling back to full StampZ-III application...")
        launch_full_stampz()
        # Force complete process termination after fallback
        sys.exit(0)
    
    finally:
        # Final cleanup to ensure no lingering Tk instances
        try:
            import tkinter as tk
            # Get the default root and destroy it if it exists
            root = tk._default_root
            if root:
                root.quit()
                root.destroy()
                tk._default_root = None
        except:
            pass


if __name__ == "__main__":
    main()
