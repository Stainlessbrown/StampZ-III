#!/usr/bin/env python3
"""
Test script to simulate the full workflow and verify the export button appears.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_workflow():
    """Test the full workflow: Analysis -> Compare -> Export"""
    root = tk.Tk()
    root.title("StampZ Workflow Test")
    root.geometry("1400x900")
    
    # Create main app instance (simplified)
    from app.analysis_manager import AnalysisManager
    
    class MockApp:
        def __init__(self):
            self.root = root
            self.current_file = "/Users/stanbrown/Desktop/test_image.jpg"
            
            # Create mock canvas with markers
            self.canvas = MockCanvas()
    
    class MockCanvas:
        def __init__(self):
            self._coord_markers = [
                {
                    'image_pos': (100, 100),
                    'sample_type': 'circle',
                    'sample_width': 20,
                    'sample_height': 20,
                    'anchor': 'center',
                    'is_preview': False
                },
                {
                    'image_pos': (200, 150),
                    'sample_type': 'rectangle',
                    'sample_width': 15,
                    'sample_height': 15,
                    'anchor': 'center',
                    'is_preview': False
                }
            ]
    
    # Create the app and analysis manager
    app = MockApp()
    analysis_manager = AnalysisManager(app)
    
    # Show instructions
    instructions = tk.Label(
        root,
        text="This test simulates the workflow:\n\n"
             "1. Click 'Compare Sample to Library' to open compare window\n"
             "2. Look for '📊 Export to Unified Data Logger' button\n"
             "3. Click the export button to test the dialog",
        font=("Arial", 12),
        justify="left",
        bg="lightblue",
        padx=20,
        pady=20
    )
    instructions.pack(pady=20)
    
    # Create test button
    test_button = tk.Button(
        root,
        text="Compare Sample to Library",
        command=analysis_manager.compare_sample_to_library,
        font=("Arial", 14),
        bg="lightgreen",
        padx=20,
        pady=10
    )
    test_button.pack(pady=20)
    
    # Status label
    status = tk.Label(
        root,
        text="Click the button above to open the compare window and look for the export button",
        font=("Arial", 11),
        fg="gray"
    )
    status.pack(pady=10)
    
    return root

if __name__ == "__main__":
    try:
        test_window = test_workflow()
        test_window.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()