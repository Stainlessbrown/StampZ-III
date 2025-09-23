"""
Menu Manager for StampZ Application

Handles all menu creation and management for the main application window.
"""

import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.stampz_app import StampZApp


class MenuManager:
    """Manages the application menu bar and menu items."""
    
    def __init__(self, app: 'StampZApp'):
        self.app = app
        self.root = app.root
        self.menubar = None
        
    def create_menu(self):
        """Create the complete menu bar for the application."""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        self._create_file_menu()
        self._create_edit_menu()
        self._create_color_menu()
        self._create_measurement_menu()
        self._create_help_menu()
        
    def _create_file_menu(self):
        """Create the File menu."""
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        
        self.file_menu.add_command(
            label="Open...", 
            command=self.app.open_image, 
            accelerator="Ctrl+O"
        )
        self.file_menu.add_command(
            label="Clear", 
            command=self.app.clear_image, 
            accelerator="Ctrl+W"
        )
        self.file_menu.add_command(
            label="Save As...", 
            command=self.app.save_image, 
            accelerator="Ctrl+S"
        )
        self.file_menu.add_separator()
        
        self.file_menu.add_command(
            label="Export Color Data to ODS...", 
            command=self.app.export_color_data
        )
        self.file_menu.add_command(
            label="Database Viewer...", 
            command=self.app.open_database_viewer
        )
        self.file_menu.add_separator()
        
        self.file_menu.add_command(
            label="Exit", 
            command=self.app.quit_app, 
            accelerator="Ctrl+Q"
        )
        
    def _create_edit_menu(self):
        """Create the Edit menu."""
        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)
        
        self.edit_menu.add_command(
            label="Reset View", 
            command=self.app.reset_view, 
            accelerator="Ctrl+R"
        )
        self.edit_menu.add_command(
            label="Fit to Window", 
            command=self.app.fit_to_window, 
            accelerator="F11"
        )
        
    def _create_color_menu(self):
        """Create the Color Analysis menu."""
        self.color_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Color Analysis", menu=self.color_menu)
        
        self.color_menu.add_command(
            label="Color Library Manager...", 
            command=self.app.open_color_library
        )
        self.color_menu.add_command(
            label="Compare Sample to Library...", 
            command=self.app.compare_sample_to_library
        )
        self.color_menu.add_separator()
        
        self.color_menu.add_command(
            label="Create Standard Libraries", 
            command=self.app.create_standard_libraries
        )
        self.color_menu.add_separator()
        
        self.color_menu.add_command(
            label="Spectral Analysis...", 
            command=self.app.open_spectral_analysis
        )
        self.color_menu.add_separator()
        
        self.color_menu.add_command(
            label="Plot_3D Data Manager...", 
            command=self.app.analysis_manager.open_plot3d_data_manager
        )
        self.color_menu.add_command(
            label="Export to Plot_3D Format...", 
            command=self.app.analysis_manager.export_plot3d_flexible
        )
        self.color_menu.add_separator()
        
        self.color_menu.add_command(
            label="Black Ink Extractor...",
            command=self.app.open_black_ink_extractor
        )
        self.color_menu.add_separator()
        
        # Note: Export to Unified Data Logger options have been moved to the Compare window
        # for better user control and workflow integration
        
        self.color_menu.add_command(
            label="Export Analysis with Library Matches...", 
            command=self.app.export_with_library_matches
        )
        
        self.color_menu.add_separator()
        
        try:
            print("DEBUG: Adding Advanced Color Visualization menu item...")
            self.color_menu.add_command(
                label="Advanced Color Visualization...",
                command=self.app.open_advanced_color_visualization
            )
            print("DEBUG: Advanced Color Visualization menu item added successfully")
        except Exception as e:
            print(f"ERROR: Failed to add Advanced Color Visualization menu item: {e}")
        
    def _create_measurement_menu(self):
        """Create the Measurement menu."""
        self.measurement_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Measurement", menu=self.measurement_menu)
        
        self.measurement_menu.add_command(
            label="Perforation Gauge...",
            command=self.app.measure_perforations,
            accelerator="Ctrl+P"
        )
        self.measurement_menu.add_separator()
        
        self.measurement_menu.add_command(
            label="Precision Measurements...",
            command=self.app.open_precision_measurements
        )
        
    def _create_help_menu(self):
        """Create the Help menu."""
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        
        self.help_menu.add_command(
            label="About", 
            command=self.app.show_about
        )
        self.help_menu.add_command(
            label="Preferences...", 
            command=self.app.open_preferences
        )
