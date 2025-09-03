"""
Internal Spreadsheet Manager for StampZ-III

Provides an integrated Tkinter-based spreadsheet view for real-time 
color analysis data with automatic Plot_3D formatting and export.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class InternalSpreadsheet:
    """Tkinter-based spreadsheet widget for displaying color analysis data."""
    
    # Plot_3D column structure
    PLOT3D_COLUMNS = [
        'Xnorm', 'Ynorm', 'Znorm', 'DataID', 'Cluster', 
        '∆E', 'Marker', 'Color', 'Centroid_X', 'Centroid_Y', 
        'Centroid_Z', 'Sphere', 'Radius'
    ]
    
    def __init__(self, parent, title="Color Analysis Data"):
        self.parent = parent
        self.title = title
        self.data = []
        self.window = None
        
        self._create_window()
        self._setup_treeview()
        
    def _create_window(self):
        """Create the spreadsheet window."""
        self.window = tk.Toplevel(self.parent)
        self.window.title(self.title)
        self.window.geometry("1200x600")
        
        # Center the window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        self.window.transient(self.parent)
        
    def _setup_treeview(self):
        """Setup the treeview widget that acts as our spreadsheet."""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title label
        title_label = ttk.Label(main_frame, text=self.title, font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Treeview with scrollbars
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview with Plot_3D columns
        self.tree = ttk.Treeview(tree_frame, columns=self.PLOT3D_COLUMNS, show='headings', height=20)
        
        # Configure column headings and widths
        column_widths = {
            'Xnorm': 80, 'Ynorm': 80, 'Znorm': 80, 'DataID': 120,
            'Cluster': 70, '∆E': 70, 'Marker': 60, 'Color': 80,
            'Centroid_X': 90, 'Centroid_Y': 90, 'Centroid_Z': 90,
            'Sphere': 70, 'Radius': 70
        }
        
        for col in self.PLOT3D_COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=column_widths.get(col, 80), minwidth=50)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Buttons
        ttk.Button(button_frame, text="Refresh Data", command=self.refresh_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export to Plot_3D", command=self.export_to_plot3d).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear View", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_sample_data(self, sample_set_name: str):
        """Load color analysis data for a specific sample set."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Get measurements from database
            db = ColorAnalysisDB(sample_set_name)
            measurements = db.get_all_measurements()
            
            if not measurements:
                messagebox.showinfo("No Data", f"No color analysis data found for '{sample_set_name}'")
                return
                
            # Convert to Plot_3D format
            self.data = self._convert_to_plot3d_format(measurements, sample_set_name)
            self._populate_treeview()
            
            # Update window title
            self.window.title(f"Color Analysis Data - {sample_set_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            logger.error(f"Error loading sample data: {e}")
    
    def _convert_to_plot3d_format(self, measurements: List[Dict], sample_set_name: str) -> List[Dict]:
        """Convert StampZ measurements to Plot_3D format."""
        plot3d_data = []
        
        for i, measurement in enumerate(measurements, 1):
            try:
                # Get Lab values (already normalized according to user preferences)
                l_val = measurement.get('l_value', 0.0)
                a_val = measurement.get('a_value', 0.0)  
                b_val = measurement.get('b_value', 0.0)
                
                # Use values as-is since normalization is handled by user preferences
                # Values are already in the format the user has chosen
                row = {
                    'Xnorm': round(l_val, 6),
                    'Ynorm': round(a_val, 6), 
                    'Znorm': round(b_val, 6),
                    'DataID': f"{sample_set_name}_Sample_{i:03d}",
                    'Cluster': '',
                    '∆E': '',
                    'Marker': '.',
                    'Color': 'blue',
                    'Centroid_X': '',
                    'Centroid_Y': '',
                    'Centroid_Z': '',
                    'Sphere': '',
                    'Radius': ''
                }
                plot3d_data.append(row)
                
            except Exception as e:
                logger.warning(f"Error converting measurement {i}: {e}")
                continue
                
        return plot3d_data
    
    def _populate_treeview(self):
        """Populate the treeview with current data."""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add data rows
        for row_data in self.data:
            values = [row_data.get(col, '') for col in self.PLOT3D_COLUMNS]
            self.tree.insert('', tk.END, values=values)
    
    def refresh_data(self):
        """Refresh data from database.""" 
        # This could be connected to auto-refresh functionality
        messagebox.showinfo("Refresh", "Data refresh functionality - to be implemented")
        
    def export_to_plot3d(self):
        """Export current data to Plot_3D format file."""
        # This could use your existing DirectPlot3DExporter
        messagebox.showinfo("Export", "Plot_3D export functionality - to be implemented")
        
    def clear_data(self):
        """Clear the spreadsheet view."""
        self.data = []
        self._populate_treeview()
        
    def add_row(self, row_data: Dict):
        """Add a new row of data (for real-time updates)."""
        self.data.append(row_data)
        self._populate_treeview()
        
    def update_row(self, row_index: int, row_data: Dict):
        """Update an existing row (for real-time updates)."""
        if 0 <= row_index < len(self.data):
            self.data[row_index] = row_data
            self._populate_treeview()


# Example usage (this would be called from your main app)
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main window for demo
    
    # Create spreadsheet
    spreadsheet = InternalSpreadsheet(root, "StampZ-III Color Analysis")
    
    # Test with some sample data
    # spreadsheet.load_sample_data("test_sample_set")
    
    root.mainloop()
