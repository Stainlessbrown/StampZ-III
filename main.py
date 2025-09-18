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


def launch_full_stampz(root):
    """Launch the full StampZ-III application using existing root."""
    # Import the refactored application
    from app import StampZApp
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting full StampZ-III application...")
    
    try:
        # Create and run the application using the provided root
        app = StampZApp(root)
        logger.info("StampZ-III application initialized successfully")
        
        # The mainloop will continue running in main()
        return True
        
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
        
        return False


def launch_plot3d_only(root):
    """Launch Plot_3D only mode using existing root."""
    try:
        from plot3d.standalone_plot3d import main as plot3d_main
        # Note: plot3d_main() creates its own window, so we hide the root
        root.withdraw()
        plot3d_main()
        return True
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
        return False


def main():
    """Main entry point - shows launch mode selector."""
    # Create single root window for entire application lifecycle
    root = tk.Tk()
    root.withdraw()  # Hide initially
    
    try:
        # Show launch selector using the root window
        from launch_selector import LaunchSelector
        
        # Create selector with our root instead of creating its own
        selector = LaunchSelector(root)
        selected_mode = selector.show()
        
        if selected_mode == "full":
            success = launch_full_stampz(root)
            if success:
                # StampZ-III will manage the root window from here
                root.deiconify()  # Show the window
                root.mainloop()  # Run the main event loop
            
        elif selected_mode == "plot3d":
            success = launch_plot3d_only(root)
            if success:
                root.mainloop()  # Keep root alive for Plot_3D
            
        else:
            # User cancelled or closed dialog
            print("Launch cancelled by user")
            
    except Exception as e:
        print(f"Error during launch: {e}")
        logging.error(f"Launch selector error: {e}")
        # Fallback to full application if selector fails
        print("Falling back to full StampZ-III application...")
        try:
            success = launch_full_stampz(root)
            if success:
                root.deiconify()
                root.mainloop()
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
    
    finally:
        # Clean shutdown
        try:
            root.quit()
            root.destroy()
        except:
            pass


if __name__ == "__main__":
    main()
