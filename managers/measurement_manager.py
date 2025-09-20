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
import cv2
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..app.stampz_app import StampZApp

logger = logging.getLogger(__name__)


class MeasurementManager:
    """Manages measurement operations for the StampZ application."""
    
    def __init__(self, app: 'StampZApp'):
        self.app = app
        self.root = app.root
    
    def measure_perforations(self):
        """Launch perforation measurement dialog."""
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
            
            # Import and launch the perforation UI
            from gui.perforation_ui import PerforationMeasurementDialog
            
            dialog = PerforationMeasurementDialog(
                parent=self.root,
                image_array=image_array,
                image_filename=image_filename
            )
            
        except ImportError as e:
            messagebox.showerror(
                "Feature Not Available",
                f"Perforation measurement feature requires additional dependencies.\n"
                f"Please install: pip install opencv-python\n\n"
                f"Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error launching perforation measurement: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to launch perforation measurement:\n{str(e)}"
            )
