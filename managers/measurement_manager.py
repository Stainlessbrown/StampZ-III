"""
Measurement Manager for StampZ Application

Handles stamp measurement operations including perforation gauge measurement,
centering analysis, and condition assessment.
"""

import os
import tkinter as tk
from tkinter import messagebox
import logging
import numpy as np
from typing import TYPE_CHECKING, Optional

# Import cv2 with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

if TYPE_CHECKING:
    from ..app.stampz_app import StampZApp

logger = logging.getLogger(__name__)


class MeasurementManager:
    """Manages measurement operations for the StampZ application."""
    
    def __init__(self, app: 'StampZApp'):
        self.app = app
        self.root = app.root
    
    def measure_perforations(self):
        """Launch gauge-based perforation measurement dialog."""
        # Check cv2 availability first
        if not CV2_AVAILABLE:
            # Add debug info to help troubleshoot
            import sys
            debug_info = (
                f"Python executable: {sys.executable}\n"
                f"Python version: {sys.version}\n"
                f"Working directory: {os.getcwd()}\n"
                f"CV2_AVAILABLE: {CV2_AVAILABLE}\n"
            )
            messagebox.showerror(
                "Perforation measurement requires additional dependencies.",
                f"Please install: pip install opencv-python\n\n"
                f"Error: No module named 'cv2'\n\n"
                f"Debug info:\n{debug_info}"
            )
            return
            
        try:
            # Check if we have an image loaded
            image_array = None
            image_filename = ""
            
            if hasattr(self.app, 'canvas') and self.app.canvas.original_image:
                # Get the image from the canvas
                from PIL import Image
                
                # Convert PIL image to numpy array
                pil_image = self.app.canvas.original_image
                image_array = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV if needed
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Get filename if available
                if hasattr(self.app, 'current_file') and self.app.current_file:
                    image_filename = self.app.current_file
            
            # Import and launch the new gauge-based perforation UI
            from gui.gauge_perforation_ui import GaugePerforationDialog
            
            dialog = GaugePerforationDialog(
                parent=self.root,
                image_array=image_array,
                image_filename=image_filename
            )
            
        except ImportError as e:
            messagebox.showerror(
                "Feature Not Available",
                f"Gauge perforation measurement requires additional dependencies.\n"
                f"Please install: pip install opencv-python pillow\n\n"
                f"Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error launching gauge perforation measurement: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to launch gauge perforation measurement:\n{str(e)}"
            )
    
    def measure_perforations_legacy(self):
        """Launch legacy hole-detection perforation measurement dialog."""
        # Check cv2 availability first
        if not CV2_AVAILABLE:
            messagebox.showerror(
                "Perforation measurement requires additional dependencies.",
                "Please install: pip install opencv-python\n\n"
                "Error: No module named 'cv2'"
            )
            return
            
        try:
            # Check if we have an image loaded
            image_array = None
            image_filename = ""
            
            if hasattr(self.app, 'canvas') and self.app.canvas.original_image:
                # Get the image from the canvas
                from PIL import Image
                
                # Convert PIL image to numpy array
                pil_image = self.app.canvas.original_image
                image_array = np.array(pil_image)
                
                # Convert RGB to BGR for OpenCV if needed
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Get filename if available
                if hasattr(self.app, 'current_file') and self.app.current_file:
                    image_filename = self.app.current_file
            
            # Import and launch the legacy perforation UI
            from gui.perforation_ui import PerforationMeasurementDialog
            
            dialog = PerforationMeasurementDialog(
                parent=self.root,
                image_array=image_array,
                image_filename=image_filename
            )
            
        except ImportError as e:
            messagebox.showerror(
                "Feature Not Available",
                f"Legacy perforation measurement requires additional dependencies.\n"
                f"Please install: pip install opencv-python\n\n"
                f"Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error launching legacy perforation measurement: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to launch legacy perforation measurement:\n{str(e)}"
            )
