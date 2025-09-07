"""
Real-time Plot_3D Spreadsheet with tksheet

- Cell-level formatting (pink protected areas, gray validation)
- Real dropdown validation for markers, colors, spheres
- Real-time updates as StampZ analyzes new samples
- Direct editing with auto-save to Plot_3D files
- Live sync with Plot_3D for immediate refresh

HARD RULE: This interface ALWAYS uses normalized data (0-1 range) for Plot_3D.
- L* (0-100) → X (0-1)
- a* (-128 to +127) → Y (0-1)  
- b* (-128 to +127) → Z (0-1)

This ensures consistent 3D visualization without negative quadrants.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tksheet
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


# IMPORTANT: This class enforces normalized data (0-1 range) for Plot_3D compatibility
# All Lab values are automatically normalized to eliminate negative quadrants in 3D visualization
class RealtimePlot3DSheet:
    """Excel-like spreadsheet interface for real-time Plot_3D data management.
    
    ALWAYS uses normalized data (0-1 range) for consistent 3D visualization.
    """
    
    # Plot_3D column structure
    PLOT3D_COLUMNS = [
        'Xnorm', 'Ynorm', 'Znorm', 'DataID', 'Cluster', 
        '∆E', 'Marker', 'Color', 'Centroid_X', 'Centroid_Y', 
        'Centroid_Z', 'Sphere', 'Radius'
    ]
    
    # Data validation lists from Plot_3D
    VALID_MARKERS = ['.', 'o', '*', '^', '<', '>', 'v', 's', 'D', '+', 'x']
    VALID_COLORS = [
        'red', 'blue', 'green', 'orange', 'purple', 'yellow', 
        'cyan', 'magenta', 'brown', 'pink', 'lime', 'navy', 'teal', 'gray'
    ]
    VALID_SPHERES = [
        'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 
        'orange', 'purple', 'brown', 'pink', 'lime', 'navy', 'teal', 'gray'
    ]
    
    def __init__(self, parent, sample_set_name="StampZ_Analysis"):
        self.parent = parent
        self.sample_set_name = sample_set_name
        self.current_file_path = None
        self.plot3d_app = None  # Reference to Plot_3D instance
        
        print(f"DEBUG: Initializing RealtimePlot3DSheet for {sample_set_name}")
        
        try:
            self._create_window()
            print("DEBUG: Window created successfully")
            
            self._setup_spreadsheet()
            print("DEBUG: Spreadsheet setup complete")
            
            self._setup_toolbar()
            print("DEBUG: Toolbar setup complete")
            
            self._load_initial_data()
            print("DEBUG: Initial data loading complete")
            
            print("DEBUG: RealtimePlot3DSheet initialization complete")
            
        except Exception as init_error:
            print(f"DEBUG: Error during initialization: {init_error}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            raise
        
    def _create_window(self):
        """Create the main window."""
        print(f"DEBUG: Creating window for {self.sample_set_name}")
        
        self.window = tk.Toplevel(self.parent)
        self.window.title(f"Plot_3D Normalized Data - {self.sample_set_name} (0-1 Range)")
        self.window.geometry("1400x800")
        
        print("DEBUG: Window created, setting geometry...")
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # macOS-specific window management (fix disappearing window issue)
        try:
            # Don't use transient to prevent window disappearing
            # self.window.transient(self.parent)  # Commented out
            self.window.lift()
            self.window.attributes('-topmost', True)
            self.window.after(100, lambda: self.window.attributes('-topmost', False))
            self.window.focus_force()
            
            # Prevent window from being lost off-screen
            self.window.resizable(True, True)
            self.window.minsize(800, 600)
            
            print("DEBUG: Window configured for macOS (independent window)")
        except Exception as window_error:
            print(f"DEBUG: Window configuration error: {window_error}")
        
        # Ensure window stays open
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        print("DEBUG: Window setup complete")
        
    def _setup_spreadsheet(self):
        """Setup the tksheet spreadsheet widget."""
        # Main container
        sheet_frame = ttk.Frame(self.window)
        sheet_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tksheet with proper configuration
        self.sheet = tksheet.Sheet(
            sheet_frame,
            headers=self.PLOT3D_COLUMNS,
            height=600,
            width=1380,
            show_table=True,
            show_top_left=True,
            show_row_index=True,
            show_header=True,
            font=("Arial", 14)  # Larger font size for better marker visibility
        )
        self.sheet.pack(fill=tk.BOTH, expand=True)
        
        # Enable ALL editing capabilities
        self.sheet.enable_bindings(
            "single_select",
            "row_select",
            "column_width_resize", 
            "double_click_column_resize",
            "row_height_resize",
            "column_select",
            "row_drag_and_drop",
            "column_drag_and_drop",
            "edit_cell",
            "delete_key",
            "copy",
            "paste",
            "undo",
            "edit_header"
        )
        
        print("DEBUG: tksheet created with full editing enabled")
        
        # Set up formatting and validation (with error handling)
        try:
            self._apply_formatting()
            self._setup_validation()
            logger.info("Initial formatting applied successfully")
        except Exception as format_error:
            logger.warning(f"Error applying initial formatting: {format_error}")
        
        # Bind data change events
        self.sheet.bind("<<SheetModified>>", self._on_data_changed)
        
    def _apply_formatting(self):
        """Apply visual formatting to cells."""
        try:
            # Ensure sheet has enough rows for formatting
            current_rows = self.sheet.get_total_rows()
            min_rows = 107
            if current_rows < min_rows:
                # Add empty rows
                empty_rows = [[''] * len(self.PLOT3D_COLUMNS)] * (min_rows - current_rows)
                self.sheet.insert_rows(rows=empty_rows, idx=current_rows)
            
            # Pink fill for protected areas (rows 2-7, columns A-H: A2:H7 non-user entry block)
            print("DEBUG: Applying pink formatting to protected areas...")
            try:
                # Use new tksheet API with highlight_cells
                pink_cells = [(row, col) for row in range(1, 7) for col in range(8)]  # A2:H7 (0-indexed: 1-6, 0-7)
                self.sheet.highlight_cells(
                    cells=pink_cells,
                    bg='#FFB6C1',
                    fg='black'
                )
                print("DEBUG: Pink formatting applied successfully to A2:H7")
            except Exception as pink_error:
                print(f"DEBUG: Pink formatting error: {pink_error}")
            
            # Column formatting as specified
            logger.info("Applying column formatting...")
            
            # Column G (index 6): salmon color from row 8-107
            try:
                column_g_cells = [(row, 6) for row in range(7, min(107, self.sheet.get_total_rows()))]  # Rows 8-107 (0-indexed: 7-106)
                self.sheet.highlight_cells(
                    cells=column_g_cells,
                    bg='#FA8072',  # Salmon color
                    fg='black'
                )
                logger.info("Column G formatted with salmon color (rows 8-107)")
            except Exception as g_error:
                logger.debug(f"Error formatting column G: {g_error}")
            
            # Column H (index 7): yellow color from row 8-107
            try:
                column_h_cells = [(row, 7) for row in range(7, min(107, self.sheet.get_total_rows()))]  # Rows 8-107 (0-indexed: 7-106)
                self.sheet.highlight_cells(
                    cells=column_h_cells,
                    bg='#FFFF99',  # Yellow color
                    fg='black'
                )
                logger.info("Column H formatted with yellow color (rows 8-107)")
            except Exception as h_error:
                logger.debug(f"Error formatting column H: {h_error}")
            
            # Column L (Sphere column, index 11): yellow color from row 2-107  
            try:
                column_l_cells = [(row, 11) for row in range(1, min(107, self.sheet.get_total_rows()))]  # Rows 2-107 (0-indexed: 1-106)
                self.sheet.highlight_cells(
                    cells=column_l_cells,
                    bg='#FFFF99',  # Yellow color
                    fg='black'
                )
                logger.info("Column L formatted with yellow color (rows 2-107)")
            except Exception as l_error:
                logger.debug(f"Error formatting column L: {l_error}")
            
            # Center align all cells
            try:
                # Apply center alignment to all cells in the sheet
                total_rows = self.sheet.get_total_rows()
                total_cols = len(self.PLOT3D_COLUMNS)
                all_cells = [(row, col) for row in range(total_rows) for col in range(total_cols)]
                self.sheet.align_cells(cells=all_cells, align='center')
                logger.info("Applied center alignment to all cells")
            except Exception as align_error:
                logger.debug(f"Error applying center alignment: {align_error}")
                
            logger.info("Applied cell formatting successfully")
            
        except Exception as e:
            logger.warning(f"Could not apply formatting: {e}")
            import traceback
            logger.debug(f"Formatting error details: {traceback.format_exc()}")
    
    def _setup_validation(self):
        """Setup dropdown validation for marker, color, and sphere columns."""
        try:
            print("DEBUG: Setting up validation dropdowns...")
            
            # For tksheet, we need to create dropdowns for ranges, not individual cells
            # Marker column validation (column 6, rows 8+ - skip rows 7 and earlier)
            try:
                for row in range(8, min(50, self.sheet.get_total_rows())):  # Start from row 8 = display row 9
                    self.sheet.create_dropdown(
                        r=row, c=6,
                        values=self.VALID_MARKERS,
                        set_value='.',
                        redraw=False
                    )
                print("DEBUG: Marker dropdowns created (starting row 9, skipping G7/H7)")
            except Exception as marker_error:
                print(f"DEBUG: Marker dropdown error: {marker_error}")
            
            # Color column validation (column 7, rows 8+ - skip rows 7 and earlier)
            try:
                for row in range(8, min(50, self.sheet.get_total_rows())):
                    self.sheet.create_dropdown(
                        r=row, c=7,
                        values=self.VALID_COLORS,
                        set_value='blue',
                        redraw=False
                    )
                print("DEBUG: Color dropdowns created (starting row 9, skipping G7/H7)")
            except Exception as color_error:
                print(f"DEBUG: Color dropdown error: {color_error}")
            
            # Sphere column validation (column 11, rows 1-50)
            try:
                for row in range(1, min(50, self.sheet.get_total_rows())):
                    self.sheet.create_dropdown(
                        r=row, c=11,
                        values=self.VALID_SPHERES,
                        set_value='',
                        redraw=False
                    )
                print("DEBUG: Sphere dropdowns created")
            except Exception as sphere_error:
                print(f"DEBUG: Sphere dropdown error: {sphere_error}")
            
            # Redraw once at the end
            self.sheet.refresh()
            
            logger.info("Setup data validation dropdowns successfully")
            
        except Exception as e:
            print(f"DEBUG: Validation setup error: {e}")
            logger.warning(f"Could not setup validation: {e}")
    
    def _setup_toolbar(self):
        """Setup toolbar with action buttons."""
        toolbar = ttk.Frame(self.window)
        toolbar.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Create buttons with explicit references
        self.refresh_btn = ttk.Button(toolbar, text="Refresh from StampZ", command=self._refresh_from_stampz)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(toolbar, text="Save to File", command=self._save_to_file)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_changes_btn = ttk.Button(toolbar, text="Save Changes", command=self._save_changes)
        self.save_changes_btn.pack(side=tk.LEFT, padx=5)
        
        self.plot3d_btn = ttk.Button(toolbar, text="Open in Plot_3D", command=self._open_in_plot3d)
        self.plot3d_btn.pack(side=tk.LEFT, padx=5)
        
        self.refresh_plot3d_btn = ttk.Button(toolbar, text="Refresh Plot_3D", command=self._refresh_plot3d)
        self.refresh_plot3d_btn.pack(side=tk.LEFT, padx=5)
        
        # Separator for different workflow modes
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=10)
        
        self.export_plot3d_btn = ttk.Button(toolbar, text="Export for Standalone Plot_3D", command=self._export_for_plot3d)
        self.export_plot3d_btn.pack(side=tk.LEFT, padx=5)
        
        self.import_plot3d_btn = ttk.Button(toolbar, text="Import from Plot_3D", command=self._import_from_plot3d)
        self.import_plot3d_btn.pack(side=tk.LEFT, padx=5)
        
        self.auto_refresh_btn = ttk.Button(toolbar, text="Auto-Refresh: ON", command=self._toggle_auto_refresh)
        self.auto_refresh_btn.pack(side=tk.LEFT, padx=20)
        
        print("DEBUG: Toolbar buttons created with explicit commands")
        
        # Status labels
        status_frame = ttk.Frame(toolbar)
        status_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(status_frame, text=f"Sample Set: {self.sample_set_name}", font=('Arial', 9, 'bold')).pack(side=tk.TOP, anchor='e')
        ttk.Label(status_frame, text="Data Format: Normalized (0-1 range)", font=('Arial', 8), foreground='blue').pack(side=tk.TOP, anchor='e')
        
        # Auto-save status
        self.auto_save_status = ttk.Label(status_frame, text="Auto-save: Ready", font=('Arial', 8), foreground='green')
        self.auto_save_status.pack(side=tk.TOP, anchor='e')
        
        # Auto-refresh state
        self.auto_refresh_enabled = True
        self.refresh_job = None
        
    def _load_initial_data(self):
        """Load initial data from StampZ database."""
        try:
            self._refresh_from_stampz()
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            messagebox.showwarning("Data Loading", f"Could not load initial data: {e}")
    
    def _refresh_from_stampz(self):
        """Refresh data from StampZ color analysis database."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            from utils.user_preferences import UserPreferences
            
            # HARD RULE: Plot_3D always uses normalized data (0-1 range)
            # This ensures consistent 3D visualization without negative quadrants
            use_normalized = True
            logger.info("\n=== PLOT_3D NORMALIZATION RULE ===")
            logger.info("Plot_3D ALWAYS uses normalized data (0-1 range)")
            logger.info("L* (0-100) → X (0-1)")
            logger.info("a* (-128 to +127) → Y (0-1)")
            logger.info("b* (-128 to +127) → Z (0-1)")
            logger.info("==================================\n")
            
            # Get measurements from database
            db = ColorAnalysisDB(self.sample_set_name)
            measurements = db.get_all_measurements()
            logger.info(f"Found {len(measurements) if measurements else 0} measurements for {self.sample_set_name}")
            
            # Clear existing data safely
            try:
                current_rows = self.sheet.get_total_rows()
                if current_rows > 0:
                    self.sheet.delete_rows(0, current_rows)
            except Exception as clear_error:
                logger.warning(f"Error clearing sheet: {clear_error}")
            
            if not measurements:
                logger.info("No measurements found - spreadsheet is empty")
                return
            
            logger.info(f"Processing {len(measurements)} measurements")
            
            # Convert to Plot_3D format
            data_rows = []
            for i, measurement in enumerate(measurements):
                try:
                    # Debug the measurement structure
                    logger.debug(f"Processing measurement {i}: keys={list(measurement.keys())}")
                    
                    # Get Lab values and apply normalization if needed
                    l_val = measurement.get('l_value', 0.0)
                    a_val = measurement.get('a_value', 0.0)
                    b_val = measurement.get('b_value', 0.0)
                    
                    # Apply normalization based on user preference
                    if use_normalized:
                        # Normalize: L* (0-100) → (0-1), a*b* (-128 to +127) → (0-1)
                        x_norm = l_val / 100.0 if l_val else 0.0
                        y_norm = (a_val + 128) / 256.0 if a_val else 0.5
                        z_norm = (b_val + 128) / 256.0 if b_val else 0.5
                        
                        # Constrain to 0-1 range for safety
                        x_norm = max(0.0, min(1.0, x_norm))
                        y_norm = max(0.0, min(1.0, y_norm))
                        z_norm = max(0.0, min(1.0, z_norm))
                        
                        logger.info(f"NORMALIZED: Measurement {i+1}: L*={l_val:.2f} a*={a_val:.2f} b*={b_val:.2f} → X={x_norm:.6f}, Y={y_norm:.6f}, Z={z_norm:.6f}")
                    else:
                        # Use raw Lab values
                        x_norm = l_val
                        y_norm = a_val
                        z_norm = b_val
                        
                        logger.info(f"RAW LAB: Measurement {i+1}: L*={l_val:.2f} a*={a_val:.2f} b*={b_val:.2f} → X={x_norm:.6f}, Y={y_norm:.6f}, Z={z_norm:.6f}")
                    
                    # Get the actual image name for this measurement
                    # Try to get from measurement data, fallback to sample set name format
                    image_name = measurement.get('image_name', f"{self.sample_set_name}_Sample_{i+1:03d}")
                    
                    # Use saved marker/color preferences if available, otherwise defaults
                    saved_marker = measurement.get('marker_preference', '.')
                    saved_color = measurement.get('color_preference', 'blue')
                    
                    row = [
                        round(x_norm, 4),                   # Xnorm  
                        round(y_norm, 4),                   # Ynorm
                        round(z_norm, 4),                   # Znorm
                        image_name,                          # DataID (actual image name!)
                        '',                                  # Cluster
                        '',                                  # ∆E
                        saved_marker,                        # Marker (restored from DB!)
                        saved_color,                         # Color (restored from DB!)
                        '',                                  # Centroid_X
                        '',                                  # Centroid_Y
                        '',                                  # Centroid_Z
                        '',                                  # Sphere
                        ''                                   # Radius
                    ]
                    data_rows.append(row)
                    
                except Exception as row_error:
                    logger.warning(f"Error processing measurement {i}: {row_error}")
                    continue
            
            # Insert data into sheet safely - start at row 7 (display row 8)
            if data_rows:
                try:
                    logger.info(f"Inserting {len(data_rows)} rows into sheet starting at row 7")
                    
                    # Ensure sheet has enough rows - add empty rows first
                    target_rows = 7 + len(data_rows)
                    current_rows = self.sheet.get_total_rows()
                    if current_rows < target_rows:
                        # Add empty rows to reach target
                        empty_rows = [[''] * len(self.PLOT3D_COLUMNS)] * (target_rows - current_rows)
                        self.sheet.insert_rows(rows=empty_rows, idx=current_rows)
                    
                    # Now insert data starting at row 7 (display row 8)
                    for i, row in enumerate(data_rows):
                        row_idx = 7 + i  # Start at row 7 (display as row 8)
                        try:
                            self.sheet.set_row_data(row_idx, values=row)
                        except Exception as single_row_error:
                            logger.warning(f"Error setting row {row_idx}: {single_row_error}")
                            
                    logger.info(f"Data insertion successful - {len(data_rows)} rows starting at row 8")
                    
                except Exception as insert_error:
                    logger.error(f"Error inserting rows: {insert_error}")
            
            # Reapply formatting after data changes (with error handling)
            try:
                self._apply_formatting()
                self._setup_validation()
                logger.info("Formatting and validation reapplied successfully")
            except Exception as format_error:
                logger.warning(f"Error reapplying formatting: {format_error}")
            
            logger.info(f"Refreshed with {len(measurements)} measurements")
            
            # Auto-sync to file if one is loaded
            if self.current_file_path and self.auto_refresh_enabled:
                self._auto_save_to_file()
                
        except Exception as e:
            logger.error(f"Error refreshing from StampZ: {e}")
            messagebox.showerror("Refresh Error", f"Failed to refresh data: {e}")
    
    def _on_data_changed(self, event):
        """Handle data changes in the spreadsheet."""
        # Auto-save to both file and database
        if self.refresh_job:
            self.window.after_cancel(self.refresh_job)
        self.refresh_job = self.window.after(1000, self._auto_save_changes)  # 1 second delay
    
    def _auto_save_changes(self):
        """Comprehensive auto-save: saves to both internal database and external file if available."""
        try:
            # Update status
            if hasattr(self, 'auto_save_status'):
                self.auto_save_status.config(text="Auto-save: Saving...", foreground='orange')
                self.window.update_idletasks()
            
            # Always save to internal database for persistence
            self._save_to_internal_database()
            
            # Also save to external file if one exists
            if self.current_file_path:
                self._save_data_to_file(self.current_file_path)
                # Trigger Plot_3D refresh if connected
                self._notify_plot3d_refresh()
            
            # Update status - success
            if hasattr(self, 'auto_save_status'):
                self.auto_save_status.config(text="Auto-save: Saved ✓", foreground='green')
                # Reset to "Ready" after 2 seconds
                self.window.after(2000, lambda: self.auto_save_status.config(text="Auto-save: Ready", foreground='green'))
                
        except Exception as e:
            logger.error(f"Auto-save error: {e}")
            # Update status - error
            if hasattr(self, 'auto_save_status'):
                self.auto_save_status.config(text="Auto-save: Error!", foreground='red')
                self.window.after(3000, lambda: self.auto_save_status.config(text="Auto-save: Ready", foreground='green'))
    
    def _auto_save_to_file(self):
        """Legacy auto-save method for backward compatibility."""
        self._auto_save_changes()
    
    def _save_to_internal_database(self):
        """Save current spreadsheet changes back to the StampZ database."""
        try:
            # Get current sheet data
            data = self.sheet.get_sheet_data(get_header=False)
            
            # Process data to update database
            from utils.color_analysis_db import ColorAnalysisDB
            db = ColorAnalysisDB(self.sample_set_name)
            
            # Update database with modified marker/color information
            # This preserves user changes in the internal database
            updated_count = 0
            for i, row_data in enumerate(data):
                if not row_data or len(row_data) < len(self.PLOT3D_COLUMNS):
                    continue
                    
                # Extract key data
                try:
                    data_id = row_data[3] if len(row_data) > 3 else None  # DataID column
                    marker = row_data[6] if len(row_data) > 6 else '.'     # Marker column
                    color = row_data[7] if len(row_data) > 7 else 'blue'   # Color column
                    
                    if data_id and marker and color:
                        # Extract image name and coordinate point from DataID
                        # Format: SampleSet_Sample_001 or actual_image_name
                        if '_Sample_' in data_id:
                            # Parse coordinate point from generated ID
                            parts = data_id.split('_Sample_')
                            if len(parts) == 2:
                                try:
                                    coord_point = int(parts[1])
                                    # Use the original image name or generate one
                                    image_name = data_id  # For now, use full DataID as image name
                                    
                                    # Actually update the database with preferences!
                                    success = db.update_marker_color_preferences(
                                        image_name=image_name,
                                        coordinate_point=coord_point, 
                                        marker=marker,
                                        color=color
                                    )
                                    
                                    if success:
                                        updated_count += 1
                                        logger.debug(f"✅ Saved preferences for {data_id}: marker={marker}, color={color}")
                                    else:
                                        logger.debug(f"❌ Failed to save preferences for {data_id}")
                                        
                                except ValueError:
                                    logger.debug(f"Could not parse coordinate point from {data_id}")
                        else:
                            # Direct image name - use row index as coordinate point
                            coord_point = i + 1
                            success = db.update_marker_color_preferences(
                                image_name=data_id,
                                coordinate_point=coord_point,
                                marker=marker,
                                color=color
                            )
                            
                            if success:
                                updated_count += 1
                                logger.debug(f"✅ Saved preferences for {data_id} (pt {coord_point}): marker={marker}, color={color}")
                        
                except Exception as row_error:
                    logger.debug(f"Error processing row {i}: {row_error}")
                    continue
            
            if updated_count > 0:
                logger.info(f"✅ Saved {updated_count} marker/color preferences to internal database")
            else:
                logger.info("No marker/color preferences to update")
            
            logger.info("Internal database updated with spreadsheet changes")
            
        except Exception as e:
            logger.warning(f"Error saving to internal database: {e}")
    
    def _save_to_file(self):
        """Save spreadsheet data to file."""
        print("DEBUG: Save to file button clicked")
        if not self.current_file_path:
            # Ask for save location
            default_name = f"{self.sample_set_name}_Plot3D_{datetime.now().strftime('%Y%m%d')}.ods"
            
            self.current_file_path = filedialog.asksaveasfilename(
                title="Save Plot_3D Spreadsheet",
                defaultextension=".ods",
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('Excel Workbook', '*.xlsx'),
                    ('All files', '*.*')
                ],
                initialfile=default_name
            )
        
        if self.current_file_path:
            success = self._save_data_to_file(self.current_file_path)
            if success:
                messagebox.showinfo(
                    "Saved",
                    f"Spreadsheet saved to:\\n{os.path.basename(self.current_file_path)}"
                )
    
    def _save_data_to_file(self, file_path):
        """Save current spreadsheet data to specified file."""
        try:
            # Get all data from sheet
            data = self.sheet.get_sheet_data(get_header=False)
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=self.PLOT3D_COLUMNS)
            
            # Remove empty rows
            df = df.replace('', np.nan).dropna(how='all').fillna('')
            
            # Save to appropriate format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.xlsx':
                df.to_excel(file_path, index=False)
            else:
                df.to_excel(file_path, engine='odf', index=False)
            
            logger.info(f"Saved {len(df)} rows to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            messagebox.showerror("Save Error", f"Failed to save file: {e}")
            return False
    
    def get_data_as_dataframe(self):
        """Get current spreadsheet data as a pandas DataFrame for direct Plot_3D integration.
        
        Returns:
            pandas.DataFrame: Current sheet data in Plot_3D format
        """
        try:
            # Get all data from sheet
            data = self.sheet.get_sheet_data(get_header=False)
            
            # Create DataFrame with correct column names
            df = pd.DataFrame(data, columns=self.PLOT3D_COLUMNS)
            
            # Clean the data - remove completely empty rows and replace empty strings with NaN
            df = df.replace('', np.nan)
            
            # Keep rows that have at least some coordinate data
            coordinate_cols = ['Xnorm', 'Ynorm', 'Znorm']
            has_coordinate_data = df[coordinate_cols].notna().any(axis=1)
            df = df[has_coordinate_data].copy()
            
            # Convert coordinate columns to numeric
            numeric_cols = ['Xnorm', 'Ynorm', 'Znorm', 'Centroid_X', 'Centroid_Y', 'Centroid_Z', '∆E', 'Radius']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set default values for missing data
            df['Cluster'] = df['Cluster'].fillna('')
            df['Marker'] = df['Marker'].fillna('.')
            df['Color'] = df['Color'].fillna('blue')
            df['Sphere'] = df['Sphere'].fillna('')
            
            logger.info(f"Converted tksheet data to DataFrame: {len(df)} rows with coordinate data")
            return df
            
        except Exception as e:
            logger.error(f"Error converting sheet data to DataFrame: {e}")
            return None
    
    def _open_in_plot3d(self):
        """Open current data in Plot_3D - now reads directly from internal worksheet!"""
        print("DEBUG: Open in Plot_3D button clicked (direct integration mode)")
        try:
            # Get current data as DataFrame directly from the sheet
            df = self.get_data_as_dataframe()
            
            if df is None or len(df) == 0:
                messagebox.showwarning(
                    "No Data",
                    "No valid coordinate data found in the spreadsheet.\n\n"
                    "Please ensure you have data in the Xnorm, Ynorm, and Znorm columns."
                )
                return
            
            # Import the modified Plot_3D class
            from plot3d.Plot_3D import Plot3DApp
            
            # Launch Plot_3D with DataFrame directly (no file required!)
            self.plot3d_app = Plot3DApp(parent=self.parent, dataframe=df)
            
            messagebox.showinfo(
                "Plot_3D Launched",
                f"Plot_3D opened with current spreadsheet data ({len(df)} data points).\n\n"
                f"✅ No external files needed!\n"
                f"Changes in this spreadsheet will be reflected when you click 'Refresh Plot_3D'."
            )
            
        except Exception as e:
            logger.error(f"Error launching Plot_3D: {e}")
            messagebox.showerror("Launch Error", f"Failed to open Plot_3D: {e}")
    
    def _refresh_plot3d(self):
        """Refresh Plot_3D with current spreadsheet data."""
        print("DEBUG: Refresh Plot_3D button clicked")
        try:
            if not self.plot3d_app:
                messagebox.showinfo(
                    "Plot_3D Not Open",
                    "Please click 'Open in Plot_3D' first to launch the 3D visualization."
                )
                return
            
            # Get current data as DataFrame
            df = self.get_data_as_dataframe()
            
            if df is None or len(df) == 0:
                messagebox.showwarning(
                    "No Data",
                    "No valid coordinate data found in the spreadsheet."
                )
                return
            
            # Update Plot_3D with new data
            self.plot3d_app.df = df
            
            # Refresh the plot
            if hasattr(self.plot3d_app, 'refresh_plot'):
                self.plot3d_app.refresh_plot()
                logger.info(f"Refreshed Plot_3D with {len(df)} data points")
                messagebox.showinfo(
                    "Plot_3D Refreshed",
                    f"✅ Updated Plot_3D with {len(df)} data points from spreadsheet!"
                )
            else:
                messagebox.showwarning(
                    "Refresh Not Available",
                    "Plot_3D refresh method not available. Please restart Plot_3D."
                )
                
        except Exception as e:
            logger.error(f"Error refreshing Plot_3D: {e}")
            messagebox.showerror("Refresh Error", f"Failed to refresh Plot_3D: {e}")
    
    def _export_for_plot3d(self):
        """Export current data to external file for standalone Plot_3D work.
        
        This creates a protected workflow where the original StampZ data remains untouched.
        """
        print("DEBUG: Export for Standalone Plot_3D button clicked")
        try:
            # Get current data as DataFrame
            df = self.get_data_as_dataframe()
            
            if df is None or len(df) == 0:
                messagebox.showwarning(
                    "No Data",
                    "No valid coordinate data found in the spreadsheet to export."
                )
                return
            
            # Ask for save location with meaningful default name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            default_name = f"{self.sample_set_name}_Plot3D_Export_{timestamp}.ods"
            
            file_path = filedialog.asksaveasfilename(
                title="Export for Standalone Plot_3D",
                defaultextension=".ods",
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('Excel Workbook', '*.xlsx'),
                    ('All files', '*.*')
                ],
                initialfile=default_name
            )
            
            if not file_path:
                return  # User cancelled
            
            # Save DataFrame to file
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.xlsx':
                df.to_excel(file_path, index=False)
            else:
                # For .ods files, use openpyxl engine to write Excel format and rename
                temp_xlsx = file_path.rsplit('.', 1)[0] + '_temp.xlsx'
                df.to_excel(temp_xlsx, index=False)
                
                try:
                    # Try to convert to ODS using pandas with odf engine
                    import odf
                    df.to_excel(file_path, engine='odf', index=False)
                    os.remove(temp_xlsx)  # Clean up temp file
                except ImportError:
                    # If odfpy not available, rename xlsx to ods (Plot_3D can handle it)
                    import shutil
                    shutil.move(temp_xlsx, file_path)
                    logger.warning("odfpy not available, saved as Excel format with .ods extension")
            
            # Show success message with workflow guidance
            result = messagebox.showinfo(
                "Export Successful",
                f"✅ Exported {len(df)} data points to:\n{os.path.basename(file_path)}\n\n"
                f"🔒 PROTECTED WORKFLOW:\n"
                f"• Your original StampZ data is safe and unchanged\n"
                f"• Work with Plot_3D using this external file\n"
                f"• Make K-means clusters, ΔE calculations, etc.\n"
                f"• When satisfied, you can import changes back\n\n"
                f"Would you like to open this file in standalone Plot_3D now?"
            )
            
            # Offer to launch standalone Plot_3D
            if messagebox.askyesno("Open in Plot_3D?", "Launch standalone Plot_3D with this exported file?"):
                try:
                    from plot3d.Plot_3D import Plot3DApp
                    
                    # Launch Plot_3D in standalone mode with the exported file
                    standalone_plot3d = Plot3DApp(parent=None, data_path=file_path)
                    
                    messagebox.showinfo(
                        "Standalone Plot_3D Launched",
                        f"✅ Standalone Plot_3D opened with exported data.\n\n"
                        f"This runs independently from StampZ.\n"
                        f"Your original StampZ data remains protected."
                    )
                    
                except Exception as plot_error:
                    logger.error(f"Error launching standalone Plot_3D: {plot_error}")
                    messagebox.showerror("Launch Error", f"Exported file successfully, but failed to open Plot_3D:\n{plot_error}")
            
            logger.info(f"Exported data to {file_path} for standalone Plot_3D workflow")
            
        except Exception as e:
            logger.error(f"Error exporting for Plot_3D: {e}")
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def _import_from_plot3d(self):
        """Import changes back from a Plot_3D external file.
        
        This completes the protected workflow by importing changes back into StampZ.
        """
        print("DEBUG: Import from Plot_3D button clicked")
        try:
            # Warn about data replacement
            if not messagebox.askyesno(
                "Import Confirmation",
                "⚠️ IMPORT WARNING:\n\n"
                "This will replace your current spreadsheet data with data from an external Plot_3D file.\n\n"
                "Your current StampZ analysis data will be overwritten.\n\n"
                "Are you sure you want to proceed?"
            ):
                return
            
            # Ask for file to import
            file_path = filedialog.askopenfilename(
                title="Import from Plot_3D File",
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('Excel Workbook', '*.xlsx'),
                    ('All files', '*.*')
                ]
            )
            
            if not file_path:
                return  # User cancelled
            
            # Load the external file
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == '.xlsx':
                    imported_df = pd.read_excel(file_path)
                else:
                    # Try to read as ODS first, fallback to Excel
                    try:
                        imported_df = pd.read_excel(file_path, engine='odf')
                    except:
                        imported_df = pd.read_excel(file_path)
                
                logger.info(f"Imported DataFrame with {len(imported_df)} rows from {file_path}")
                
            except Exception as read_error:
                logger.error(f"Error reading file: {read_error}")
                messagebox.showerror("Import Error", f"Failed to read file:\n{read_error}")
                return
            
            # Validate that the imported data has the expected columns
            expected_cols = set(self.PLOT3D_COLUMNS)
            imported_cols = set(imported_df.columns)
            
            if not expected_cols.issubset(imported_cols):
                missing_cols = expected_cols - imported_cols
                messagebox.showerror(
                    "Import Error", 
                    f"Import file is missing required columns:\n{', '.join(missing_cols)}\n\n"
                    f"Expected columns: {', '.join(self.PLOT3D_COLUMNS)}"
                )
                return
            
            # Clear current sheet data
            try:
                current_rows = self.sheet.get_total_rows()
                if current_rows > 0:
                    self.sheet.delete_rows(0, current_rows)
            except Exception as clear_error:
                logger.warning(f"Error clearing sheet: {clear_error}")
            
            # Convert DataFrame to list format for tksheet
            import_data = imported_df[self.PLOT3D_COLUMNS].fillna('').values.tolist()
            
            # Insert imported data into sheet
            if import_data:
                try:
                    # Add empty rows first to accommodate data
                    empty_rows = [[''] * len(self.PLOT3D_COLUMNS)] * len(import_data)
                    self.sheet.insert_rows(rows=empty_rows, idx=0)
                    
                    # Set the actual data
                    for i, row in enumerate(import_data):
                        self.sheet.set_row_data(i, values=row)
                    
                    logger.info(f"Imported {len(import_data)} rows into spreadsheet")
                    
                except Exception as insert_error:
                    logger.error(f"Error inserting imported data: {insert_error}")
                    messagebox.showerror("Import Error", f"Failed to insert data into spreadsheet:\n{insert_error}")
                    return
            
            # Reapply formatting after import
            try:
                self._apply_formatting()
                self._setup_validation()
                logger.info("Reapplied formatting after import")
            except Exception as format_error:
                logger.warning(f"Error reapplying formatting after import: {format_error}")
            
            # Success message
            messagebox.showinfo(
                "Import Successful",
                f"✅ Successfully imported {len(import_data)} data points from:\n{os.path.basename(file_path)}\n\n"
                f"🔄 Your spreadsheet now contains the Plot_3D analysis results.\n"
                f"K-means clusters, ΔE values, and other changes have been imported.\n\n"
                f"Remember to save your StampZ project to preserve these changes!"
            )
            
            logger.info(f"Successfully imported Plot_3D data from {file_path}")
            
        except Exception as e:
            logger.error(f"Error importing from Plot_3D: {e}")
            messagebox.showerror("Import Error", f"Failed to import data: {e}")
    
    def _toggle_auto_refresh(self):
        """Toggle auto-refresh functionality."""
        print("DEBUG: Toggling auto-refresh")
        self.auto_refresh_enabled = not self.auto_refresh_enabled
        
        # Update button text using explicit reference
        if hasattr(self, 'auto_refresh_btn'):
            self.auto_refresh_btn.configure(text=f"Auto-Refresh: {'ON' if self.auto_refresh_enabled else 'OFF'}")
            print(f"DEBUG: Auto-refresh set to: {self.auto_refresh_enabled}")
        
        if self.auto_refresh_enabled:
            # Start periodic refresh from StampZ
            self._start_auto_refresh()
        else:
            # Stop auto-refresh
            if hasattr(self, 'auto_refresh_job'):
                self.window.after_cancel(self.auto_refresh_job)
    
    def _start_auto_refresh(self):
        """Start periodic auto-refresh from StampZ database."""
        if self.auto_refresh_enabled:
            self._check_for_new_stampz_data()
            # Schedule next refresh in 5 seconds
            self.auto_refresh_job = self.window.after(5000, self._start_auto_refresh)
    
    def _check_for_new_stampz_data(self):
        """Check for new data in StampZ database and update if found."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            
            db = ColorAnalysisDB(self.sample_set_name)
            current_measurements = db.get_all_measurements()
            
            # Compare with current sheet data count
            current_rows = self.sheet.get_total_rows()
            new_count = len(current_measurements) if current_measurements else 0
            
            if new_count > current_rows:
                logger.info(f"New data detected: {new_count} vs {current_rows} rows")
                self._refresh_from_stampz()
                
                # Auto-save and notify Plot_3D
                if self.current_file_path:
                    self._auto_save_to_file()
                    
        except Exception as e:
            logger.debug(f"Auto-refresh check error: {e}")  # Debug level to avoid spam
    
    def _notify_plot3d_refresh(self):
        """Notify Plot_3D to refresh its data."""
        try:
            if self.plot3d_app and hasattr(self.plot3d_app, 'refresh_plot'):
                logger.info("Triggering Plot_3D refresh")
                self.plot3d_app.refresh_plot()
        except Exception as e:
            logger.debug(f"Plot_3D refresh notification error: {e}")
    
    def add_new_sample_realtime(self, measurement_data):
        """Add new sample data in real-time (called from StampZ analysis)."""
        try:
            # Convert measurement to Plot_3D row format
            current_row_count = self.sheet.get_total_rows()
            
            new_row = [
                measurement_data.get('l_value', ''),
                measurement_data.get('a_value', ''),
                measurement_data.get('b_value', ''),
                f"{self.sample_set_name}_Sample_{current_row_count+1:03d}",
                '', '', '.', 'blue', '', '', '', '', ''
            ]
            
            # Insert new row
            self.sheet.insert_row(values=new_row, idx=current_row_count)
            
            # Reapply formatting to new row
            self._apply_formatting()
            
            # Auto-save if enabled
            if self.auto_refresh_enabled and self.current_file_path:
                self._auto_save_to_file()
            
            logger.info("Added new sample in real-time")
            
        except Exception as e:
            logger.error(f"Error adding real-time sample: {e}")
    
    def _save_changes(self):
        """Save current spreadsheet changes (for edited cells)."""
        print("DEBUG: Save Changes button clicked")
        try:
            if self.current_file_path:
                # Save current state to existing file
                success = self._save_data_to_file(self.current_file_path)
                if success:
                    messagebox.showinfo(
                        "Changes Saved",
                        f"Spreadsheet changes saved to:\\n{os.path.basename(self.current_file_path)}\\n\\n"
                        f"Plot_3D will use updated data on next refresh."
                    )
                    # Trigger Plot_3D refresh if connected
                    self._notify_plot3d_refresh()
            else:
                # No file associated - offer Save As
                self._save_to_file()
                
        except Exception as e:
            logger.error(f"Error saving changes: {e}")
            messagebox.showerror("Save Error", f"Failed to save changes: {e}")
    
    def _on_window_close(self):
        """Handle window close event."""
        try:
            print(f"DEBUG: Closing real-time spreadsheet for {self.sample_set_name}")
            
            # Stop auto-refresh
            if hasattr(self, 'auto_refresh_job'):
                self.window.after_cancel(self.auto_refresh_job)
            
            # Cleanup and destroy
            self.window.destroy()
            
        except Exception as e:
            print(f"DEBUG: Error closing window: {e}")


# Integration helper for StampZ main app
class Plot3DRealtimeManager:
    """Manager to integrate real-time Plot_3D spreadsheet with StampZ."""
    
    def __init__(self, parent):
        self.parent = parent
        self.active_sheets = {}  # Track open spreadsheets by sample set
    
    def open_realtime_sheet(self, sample_set_name):
        """Open or focus real-time spreadsheet for sample set."""
        if sample_set_name in self.active_sheets:
            # Focus existing window
            self.active_sheets[sample_set_name].window.lift()
            self.active_sheets[sample_set_name].window.focus_force()
        else:
            # Create new spreadsheet
            sheet = RealtimePlot3DSheet(self.parent, sample_set_name)
            self.active_sheets[sample_set_name] = sheet
            
            # Cleanup when window closes
            def on_close():
                if sample_set_name in self.active_sheets:
                    del self.active_sheets[sample_set_name]
                sheet.window.destroy()
            
            sheet.window.protocol("WM_DELETE_WINDOW", on_close)
    
    def notify_new_analysis(self, sample_set_name, measurement_data):
        """Notify spreadsheet of new analysis data."""
        if sample_set_name in self.active_sheets:
            self.active_sheets[sample_set_name].add_new_sample_realtime(measurement_data)


if __name__ == "__main__":
    # Test the real-time spreadsheet
    root = tk.Tk()
    root.withdraw()
    
    sheet = RealtimePlot3DSheet(root, "Test_Sample_Set")
    
    root.mainloop()
