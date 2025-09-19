#!/usr/bin/env python3
"""
Test script to demonstrate the color comparison export dialog functionality.

This script creates a simple test window with the color comparison manager
to verify that the export dialog button works correctly.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.color_comparison_manager import ColorComparisonManager


def create_test_window():
    """Create a test window with the color comparison manager."""
    root = tk.Tk()
    root.title("Color Comparison Export Dialog Test")
    root.geometry("1400x900")
    
    # Create a notebook to hold the comparison manager
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Create the comparison manager tab
    comparison_frame = ttk.Frame(notebook)
    notebook.add(comparison_frame, text="Color Comparison")
    
    # Create the color comparison manager
    comparison_manager = ColorComparisonManager(comparison_frame)
    
    # Create some sample data to test with
    sample_data = [
        {
            'position': (100, 100),
            'type': 'circle',
            'size': (20, 20),
            'anchor': 'center'
        },
        {
            'position': (200, 150),
            'type': 'rectangle',
            'size': (15, 15),
            'anchor': 'center'
        },
        {
            'position': (300, 200),
            'type': 'circle',
            'size': (25, 25),
            'anchor': 'center'
        }
    ]
    
    # You would normally have a real image path here
    # For testing, we'll use a placeholder path
    test_image_path = "/Users/stanbrown/Desktop/test_image.jpg"
    
    # Set up test data (this would normally be called when analyzing an image)
    try:
        # Only set analyzed data if we have the required modules available
        comparison_manager.set_analyzed_data(test_image_path, sample_data)
    except Exception as e:
        print(f"Note: Could not set analyzed data for test (expected without real image): {e}")
        # Set minimal data for testing the export dialog
        comparison_manager.current_file_path = test_image_path
        comparison_manager.filename_label.config(text="test_image.jpg")
    
    # Add instructions label
    instructions = ttk.Label(
        root,
        text="Test the export functionality by clicking the '📊 Export to Unified Data Logger' button.",
        font=("Arial", 12),
        foreground="blue"
    )
    instructions.pack(pady=10)
    
    return root


if __name__ == "__main__":
    try:
        # Create and run the test window
        test_window = create_test_window()
        test_window.mainloop()
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()