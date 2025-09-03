#!/usr/bin/env python3
"""StampZ-III - Main Application Entry Point
A image analysis application optimized for philatelic images

This is the entry point for StampZ-III. The main application logic
has been refactored into component managers in the app/ directory.
"""

# Import initialize_env first to set up data preservation system
import initialize_env

import tkinter as tk
import logging

# Import the refactored application
from app import StampZApp

logger = logging.getLogger(__name__)


def main():
    """Main entry point for StampZ-III application."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting StampZ-III...")
    
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


if __name__ == "__main__":
    main()
