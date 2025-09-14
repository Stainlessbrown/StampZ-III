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
import shutil
import time
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
    
    def __init__(self, parent, sample_set_name="StampZ_Analysis", load_initial_data=True):
        self.parent = parent
        self.sample_set_name = sample_set_name
        self.current_file_path = None
        self.plot3d_app = None  # Reference to Plot_3D instance
        
        print(f"DEBUG: Initializing RealtimePlot3DSheet for {sample_set_name} (load_initial_data={load_initial_data})")
        
        try:
            self._create_window()
            print("DEBUG: Window created successfully")
            
            print("DEBUG: About to setup spreadsheet...")
            self._setup_spreadsheet()
            print("DEBUG: Spreadsheet setup complete")
            
            print("DEBUG: About to setup toolbar...")
            self._setup_toolbar()
            print("DEBUG: Toolbar setup complete")
            
            # TEMPORARILY DISABLED: Add simple header after all setup is complete
            # self._add_simple_header()  # Disabled to test freezing issue
            
            # Only load initial data if requested (default True for backward compatibility)
            if load_initial_data:
                print("DEBUG: About to load initial data...")
                self._load_initial_data()
                print("DEBUG: Initial data loading complete")
            else:
                print("DEBUG: Skipping initial data loading as requested")
            
            print("DEBUG: About to complete initialization...")
            
            # Force window to be responsive
            self.window.update()
            self.window.update_idletasks()
            
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
        self.window.title(f"Plot_3D: {self.sample_set_name} - Normalized Data (0-1 Range)")
        self.window.geometry("1400x800")
        
        print("DEBUG: Window created, setting geometry...")
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # macOS-specific window management (simple approach)
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
            
            print("DEBUG: Window configured for macOS (simple approach)")
        except Exception as window_error:
            print(f"DEBUG: Window configuration error: {window_error}")
        
        # Ensure window stays open
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        print("DEBUG: Window setup complete")
    
    def _add_simple_header(self):
        """Add simple header at the top of existing window."""
        try:
            print(f"DEBUG: Adding simple header for {self.sample_set_name}")
            
            # Create header frame at the very top (insert before existing widgets)
            header_frame = tk.Frame(self.window, bg='lightsteelblue', relief='raised', bd=2)
            header_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2, before=self.window.children[list(self.window.children.keys())[0]])
            
            # Sample set label (left side)
            sample_label = tk.Label(header_frame, 
                                   text=f"📊 SAMPLE SET: {self.sample_set_name}", 
                                   font=('Arial', 14, 'bold'), 
                                   fg='darkblue', 
                                   bg='lightsteelblue')
            sample_label.pack(side=tk.LEFT, padx=10, pady=5)
            
            # Data format info (right side)
            format_label = tk.Label(header_frame, 
                                   text="Plot_3D Normalized Data (0-1 Range)", 
                                   font=('Arial', 10, 'italic'), 
                                   fg='darkred',
                                   bg='lightsteelblue')
            format_label.pack(side=tk.RIGHT, padx=10, pady=5)
            
            print(f"DEBUG: Simple header added successfully")
            
        except Exception as e:
            print(f"DEBUG: Simple header creation failed: {e} - continuing without header")
        
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
            font=("Monaco", 16, "normal")  # Monospace font for better marker symbol visibility
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
            # Get current sheet size - don't force a minimum, work with what we have
            current_rows = self.sheet.get_total_rows()
            print(f"DEBUG: Applying formatting to {current_rows} existing rows (no artificial minimum)")
            
            # Clear any existing formatting first to prevent conflicts
            try:
                self.sheet.dehighlight_all()
                print("DEBUG: Cleared all existing highlighting")
            except Exception as clear_error:
                print(f"DEBUG: Could not clear highlighting: {clear_error}")
            
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
            
            # Column formatting - only apply to rows that actually have data
            logger.info("Applying column formatting...")
            
            # Find the last row with actual data to avoid formatting empty rows
            last_data_row = current_rows
            for row in range(current_rows - 1, 6, -1):  # Start from end, stop at row 7
                try:
                    # Check if any cell in this row has data
                    row_data = [self.sheet.get_cell_data(row, col) for col in range(len(self.PLOT3D_COLUMNS))]
                    if any(cell and str(cell).strip() for cell in row_data):
                        last_data_row = row + 1
                        break
                except Exception:
                    continue
            
            print(f"DEBUG: Formatting columns up to row {last_data_row} (data ends around row {last_data_row})")
            
            # Column G (index 6): salmon color from row 8 to end of data
            try:
                if last_data_row > 7:  # Only apply if we have data rows
                    column_g_cells = [(row, 6) for row in range(7, last_data_row)]  # Rows 8+ with data
                    self.sheet.highlight_cells(
                        cells=column_g_cells,
                        bg='#FA8072',  # Salmon color
                        fg='black'
                    )
                    logger.info(f"Column G formatted with salmon color (rows 8-{last_data_row})")
            except Exception as g_error:
                logger.debug(f"Error formatting column G: {g_error}")
            
            # Column H (index 7): yellow color from row 8 to end of data
            try:
                if last_data_row > 7:  # Only apply if we have data rows
                    column_h_cells = [(row, 7) for row in range(7, last_data_row)]  # Rows 8+ with data
                    self.sheet.highlight_cells(
                        cells=column_h_cells,
                        bg='#FFFF99',  # Yellow color
                        fg='black'
                    )
                    logger.info(f"Column H formatted with yellow color (rows 8-{last_data_row})")
            except Exception as h_error:
                logger.debug(f"Error formatting column H: {h_error}")
            
            # Column L (Sphere column, index 11): yellow color from row 2 to end of data  
            try:
                if last_data_row > 1:  # Only apply if we have data
                    column_l_cells = [(row, 11) for row in range(1, last_data_row)]  # Rows 2+ with data
                    self.sheet.highlight_cells(
                        cells=column_l_cells,
                        bg='#FFFF99',  # Yellow color
                        fg='black'
                    )
                    logger.info(f"Column L formatted with yellow color (rows 2-{last_data_row})")
            except Exception as l_error:
                logger.debug(f"Error formatting column L: {l_error}")
            
            # Center align cells with data (not empty rows)
            try:
                # Apply center alignment only to rows with data
                total_cols = len(self.PLOT3D_COLUMNS)
                data_cells = [(row, col) for row in range(last_data_row) for col in range(total_cols)]
                self.sheet.align_cells(cells=data_cells, align='center')
                logger.info(f"Applied center alignment to data cells (rows 0-{last_data_row})")
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
                # Dynamic validation - cover all data rows plus buffer
                current_sheet_rows = self.sheet.get_total_rows()
                max_data_rows = current_sheet_rows  # Use all available rows
                print(f"DEBUG: Setting up marker dropdowns - sheet has {current_sheet_rows} rows, will create dropdowns for rows 8-{max_data_rows}")
                
                for row in range(7, max_data_rows):  # Start from row 7 = display row 8
                    # Get existing value to preserve it, or use default if empty
                    try:
                        current_value = self.sheet.get_cell_data(row, 6)
                        if not current_value or current_value.strip() == '':
                            current_value = '.'
                    except:
                        current_value = '.'
                    
                    self.sheet.create_dropdown(
                        r=row, c=6,
                        values=self.VALID_MARKERS,
                        set_value=current_value,
                        redraw=False
                    )
                print(f"DEBUG: Marker dropdowns created for rows 8-{max_data_rows} (starting at data row 8)")
            except Exception as marker_error:
                print(f"DEBUG: Marker dropdown error: {marker_error}")
            
            # Color column validation (column 7, rows 8+ - skip rows 7 and earlier)
            try:
                max_data_rows = self.sheet.get_total_rows()  # Dynamic - use all rows
                for row in range(7, max_data_rows):
                    # Get existing value to preserve it, or use default if empty
                    try:
                        current_value = self.sheet.get_cell_data(row, 7)
                        if not current_value or current_value.strip() == '':
                            current_value = 'blue'
                    except:
                        current_value = 'blue'
                    
                    self.sheet.create_dropdown(
                        r=row, c=7,
                        values=self.VALID_COLORS,
                        set_value=current_value,
                        redraw=False
                    )
                print(f"DEBUG: Color dropdowns created for rows 8-{max_data_rows} (starting at data row 8)")
            except Exception as color_error:
                print(f"DEBUG: Color dropdown error: {color_error}")
            
            # Sphere column validation (column 11, rows 1+)
            try:
                max_data_rows = self.sheet.get_total_rows()  # Dynamic - use all rows
                for row in range(1, max_data_rows):
                    # Get existing value to preserve it, or use default if empty
                    try:
                        current_value = self.sheet.get_cell_data(row, 11)
                        if not current_value or current_value.strip() == '':
                            current_value = ''
                    except:
                        current_value = ''
                    
                    self.sheet.create_dropdown(
                        r=row, c=11,
                        values=self.VALID_SPHERES,
                        set_value=current_value,
                        redraw=False
                    )
                print(f"DEBUG: Sphere dropdowns created for rows 2-{max_data_rows}")
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
        
        self.save_changes_btn = ttk.Button(toolbar, text="Save Changes to DB", command=self._save_changes)
        self.save_changes_btn.pack(side=tk.LEFT, padx=5)
        
        self.plot3d_btn = ttk.Button(toolbar, text="Open in Plot_3D", command=self._open_in_plot3d)
        self.plot3d_btn.pack(side=tk.LEFT, padx=5)
        
        self.refresh_plot3d_btn = ttk.Button(toolbar, text="Refresh Plot_3D", command=self._refresh_plot3d)
        self.refresh_plot3d_btn.pack(side=tk.LEFT, padx=5)
        
        # Note: Removed redundant "Push Changes to Plot_3D" button - same functionality as "Refresh Plot_3D"
        
        # Separator for different workflow modes
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=10)
        
        self.export_plot3d_btn = ttk.Button(toolbar, text="Export for Standalone Plot_3D", command=self._export_for_plot3d)
        self.export_plot3d_btn.pack(side=tk.LEFT, padx=5)
        
        # Create import menu button
        self.import_menu_btn = ttk.Menubutton(toolbar, text="Import Data")
        self.import_menu_btn.pack(side=tk.LEFT, padx=5)
        
        # Create import menu - simplified to just legacy import
        import_menu = tk.Menu(self.import_menu_btn, tearoff=0)
        import_menu.add_command(label="Import from Plot_3D (Legacy)", command=self._import_from_plot3d)
        self.import_menu_btn.configure(menu=import_menu)
        
        self.auto_refresh_btn = ttk.Button(toolbar, text="Auto-Refresh: ON", command=self._toggle_auto_refresh)
        self.auto_refresh_btn.pack(side=tk.LEFT, padx=20)
        
        print("DEBUG: Toolbar buttons created with explicit commands")
        
        # Status labels
        status_frame = ttk.Frame(toolbar)
        status_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(status_frame, text=f"Sample Set: {self.sample_set_name}", font=('Arial', 12, 'bold'), foreground='darkblue').pack(side=tk.TOP, anchor='e')
        ttk.Label(status_frame, text="Data Format: Normalized (0-1 range)", font=('Arial', 8, 'normal'), foreground='blue').pack(side=tk.TOP, anchor='e')
        
        # Auto-save status
        self.auto_save_status = ttk.Label(status_frame, text="Auto-save: Ready", font=('Arial', 8, 'normal'), foreground='green')
        self.auto_save_status.pack(side=tk.TOP, anchor='e')
        
        # Auto-refresh state
        self.auto_refresh_enabled = True
        self.refresh_job = None
        
    def _load_initial_data(self):
        """Load initial data from StampZ database."""
        print(f"\n🎆 INITIAL DATA LOADING FOR {self.sample_set_name}")
        try:
            self._refresh_from_stampz()
            print(f"\n✅ Initial data loading completed")
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
            print(f"\n❌ INITIAL LOADING ERROR: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            messagebox.showwarning("Data Loading", f"Could not load initial data: {e}\n\nCheck terminal for details.")
    
    def _refresh_from_stampz(self):
        """Refresh data from StampZ color analysis database."""
        print(f"\n🔄 REFRESH FROM STAMPZ BUTTON CLICKED - DEBUG TRACE")
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            from utils.user_preferences import UserPreferences
            
            # PLOT_3D DATA RULE: Data should already be in normalized (0-1 range) format
            # Plot_3D only works with normalized data - no additional normalization needed
            logger.info("\n=== PLOT_3D DATA HANDLING ===")
            logger.info("Plot_3D data should already be normalized (0-1 range)")
            logger.info("Using values as-is with safety constraints only")
            logger.info("================================\n")
            
            # Get measurements from database
            db = ColorAnalysisDB(self.sample_set_name)
            measurements = db.get_all_measurements()
            logger.info(f"Found {len(measurements) if measurements else 0} measurements for {self.sample_set_name}")
            
            # COMPLETE REFRESH: Clear and rebuild entire sheet from database
            try:
                print(f"\n🧨 COMPLETE REFRESH - CLEARING ENTIRE SHEET:")
                current_rows = self.sheet.get_total_rows()
                print(f"  Current sheet has {current_rows} rows - will do complete rebuild")
                
                # Clear ALL rows and start fresh
                if current_rows > 0:
                    self.sheet.delete_rows(0, current_rows)
                    print(f"  ✅ Deleted all {current_rows} rows")
                
                # Clear any existing formatting
                try:
                    self.sheet.dehighlight_all()
                    print(f"  ✅ Cleared all highlighting")
                except Exception:
                    pass  # Not critical if this fails
                
                # Calculate how many rows we need for database data - NO ARTIFICIAL MINIMUM
                regular_measurements_count = len([m for m in measurements if m.get('image_name') != 'CENTROIDS']) if measurements else 0
                # FIXED: Use only what we need: 7 reserved rows + data rows + small buffer
                min_rows = 7 + regular_measurements_count + 10  # 7 reserved (header + centroids) + data + 10 buffer
                
                # Create fresh sheet structure
                empty_rows = [[''] * len(self.PLOT3D_COLUMNS)] * min_rows
                self.sheet.insert_rows(rows=empty_rows, idx=0)
                print(f"  ✅ Created fresh sheet with {min_rows} rows (7 reserved + {regular_measurements_count} data + 10 buffer)")
                
            except Exception as clear_error:
                logger.error(f"Error during complete refresh: {clear_error}")
                print(f"  ❌ Complete refresh failed: {clear_error}")
                return
            
            if not measurements:
                logger.info("No measurements found - spreadsheet is empty")
                return
            
            logger.info(f"Processing {len(measurements)} measurements")
            
            # Separate CENTROIDS from regular measurements
            centroid_measurements = [m for m in measurements if m.get('image_name') == 'CENTROIDS']
            regular_measurements = [m for m in measurements if m.get('image_name') != 'CENTROIDS']
            
            print(f"\n📊 MEASUREMENT SEPARATION:")
            print(f"  CENTROIDS entries: {len(centroid_measurements)}")
            print(f"  Regular measurements: {len(regular_measurements)}")
            
            # Process CENTROIDS first - place them in centroid area (rows 1-6)
            for centroid in centroid_measurements:
                try:
                    cluster_id = centroid.get('cluster_id')
                    if cluster_id is not None and 0 <= cluster_id <= 5:  # Valid centroid area
                        centroid_row_idx = 1 + cluster_id  # Row 1-6 (display 2-7)
                        
                        # Build centroid row with proper data
                        centroid_row = [
                            '',  # Xnorm - empty for centroids
                            '',  # Ynorm - empty for centroids
                            '',  # Znorm - empty for centroids
                            '',  # DataID - empty for centroids (they don't need DataID)
                            str(cluster_id),  # Cluster
                            '',  # ΔE - empty for centroids
                            '',  # Marker - empty for centroids (spheres don't use markers)
                            '',  # Color - empty for centroids (spheres use sphere_color instead)
                            str(centroid.get('centroid_x')) if centroid.get('centroid_x') is not None else '',  # Centroid_X
                            str(centroid.get('centroid_y')) if centroid.get('centroid_y') is not None else '',  # Centroid_Y
                            str(centroid.get('centroid_z')) if centroid.get('centroid_z') is not None else '',  # Centroid_Z
                            centroid.get('sphere_color', ''),  # Sphere
                            str(centroid.get('sphere_radius')) if centroid.get('sphere_radius') is not None else ''  # Radius
                        ]
                        
                        # Set centroid data in worksheet
                        self.sheet.set_row_data(centroid_row_idx, values=centroid_row)
                        print(f"    ✅ CENTROID cluster {cluster_id} → row {centroid_row_idx} (display {centroid_row_idx+1})")
                        print(f"      Centroid: ({centroid.get('centroid_x')}, {centroid.get('centroid_y')}, {centroid.get('centroid_z')})")
                        print(f"      Sphere: {centroid.get('sphere_color')}, radius: {centroid.get('sphere_radius')}")
                    else:
                        print(f"    ⚠️ CENTROID cluster {cluster_id} out of range [0-5], skipping")
                        
                except Exception as centroid_error:
                    print(f"    ❌ Error processing CENTROID: {centroid_error}")
            
            # Convert regular measurements to Plot_3D format for data area
            data_rows = []
            for i, measurement in enumerate(regular_measurements):
                try:
                    # Debug the measurement structure
                    logger.debug(f"Processing measurement {i}: keys={list(measurement.keys())}")
                    
                    # Get Lab values and apply normalization if needed
                    l_val = measurement.get('l_value', 0.0)
                    a_val = measurement.get('a_value', 0.0)
                    b_val = measurement.get('b_value', 0.0)
                    sample_type = measurement.get('sample_type', '')
                    
                    # PLOT_3D RULE: Data should already be normalized (0-1 range)
                    # Plot_3D only works with normalized data, so we should NOT normalize here
                    # Use values as-is but constrain to 0-1 range for safety
                    x_norm = max(0.0, min(1.0, l_val if l_val is not None else 0.0))
                    y_norm = max(0.0, min(1.0, a_val if a_val is not None else 0.0))
                    z_norm = max(0.0, min(1.0, b_val if b_val is not None else 0.0))
                    
                    # Debug output for first few rows
                    if i < 5:
                        print(f"    DEBUG: Row {i+1} PLOT_3D DATA: X={x_norm:.6f}, Y={y_norm:.6f}, Z={z_norm:.6f} (no normalization applied)")
                        logger.info(f"PLOT_3D DATA: Measurement {i+1}: using values as-is X={x_norm:.6f}, Y={y_norm:.6f}, Z={z_norm:.6f}")
                    
                    # CRITICAL FIX: Create proper DataID that matches database format
                    # Database stores image_name + coordinate_point separately
                    # But DataID should combine them for unique identification
                    image_name = measurement.get('image_name', f"{self.sample_set_name}_Sample_{i+1:03d}")
                    coordinate_point = measurement.get('coordinate_point', 1)
                    
                    # Create DataID - only add _pt suffix if coordinate_point > 1 or follows traditional pattern
                    if coordinate_point > 1 or ('_pt' in image_name):
                        data_id = f"{image_name}_pt{coordinate_point}"
                    else:
                        # Simple single-point entries: keep original name clean
                        data_id = image_name
                    
                    # Get saved Plot_3D data (move BEFORE the debug logging)
                    saved_marker = measurement.get('marker_preference', '.')
                    saved_color = measurement.get('color_preference', 'blue')
                    saved_cluster = measurement.get('cluster_id', '')
                    saved_delta_e = measurement.get('delta_e', '')
                    saved_centroid_x = measurement.get('centroid_x', '')
                    saved_centroid_y = measurement.get('centroid_y', '')
                    saved_centroid_z = measurement.get('centroid_z', '')
                    saved_sphere_color = measurement.get('sphere_color', '')
                    saved_sphere_radius = measurement.get('sphere_radius', '')
                    
                    # DEBUG: Show the DataID creation and Plot_3D data restoration for first few measurements
                    if i < 10:  # Only show first 10 to avoid spam
                        logger.info(f"DATAID FIX: Measurement {i+1}: image_name='{image_name}', coord_pt={coordinate_point} → DataID='{data_id}'")
                        logger.info(f"PLOT3D RESTORE: cluster={saved_cluster}, ∆E={saved_delta_e}, marker={saved_marker}, color={saved_color}, sphere={saved_sphere_color}")
                        if saved_sphere_radius is not None and str(saved_sphere_radius).strip():
                            logger.info(f"RADIUS DEBUG: Raw radius from DB: '{saved_sphere_radius}' (type: {type(saved_sphere_radius)})")
                    
                    # Variables already defined above - no need to redefine
                    
                    row = [
                        round(x_norm, 4),                   # Xnorm  
                        round(y_norm, 4),                   # Ynorm
                        round(z_norm, 4),                   # Znorm
                        data_id,                             # DataID (image_name_ptN format!)
                        str(saved_cluster) if saved_cluster is not None else '',  # Cluster (restored from DB!)
                        str(saved_delta_e) if saved_delta_e is not None else '',  # ∆E (restored from DB!)
                        saved_marker,                        # Marker (restored from DB!)
                        saved_color,                         # Color (restored from DB!)
                        str(saved_centroid_x) if saved_centroid_x is not None else '',  # Centroid_X (restored from DB!)
                        str(saved_centroid_y) if saved_centroid_y is not None else '',  # Centroid_Y (restored from DB!)
                        str(saved_centroid_z) if saved_centroid_z is not None else '',  # Centroid_Z (restored from DB!)
                        str(saved_sphere_color) if saved_sphere_color else '',          # Sphere (restored from DB!)
                        str(saved_sphere_radius) if saved_sphere_radius is not None else ''  # Radius (restored from DB!)
                    ]
                    data_rows.append(row)
                    
                except Exception as row_error:
                    logger.warning(f"Error processing measurement {i}: {row_error}")
                    continue
            
            # SIMPLIFIED: Insert data into sheet starting at row 7 (display row 8)
            if data_rows:
                try:
                    print(f"\n📝 INSERTING DATA:")
                    print(f"  Inserting {len(data_rows)} rows starting at sheet row 7 (display row 8)")
                    
                    # Sheet already has proper size from complete refresh above
                    current_rows = self.sheet.get_total_rows()
                    print(f"  Sheet has {current_rows} rows ready for data insertion")
                    
                    # DEBUG: Check actual data insertion
                    print(f"  Inserting {len(data_rows)} rows starting at row 7 (display row 8)...")
                    successful_count = 0
                    
                    for i, row in enumerate(data_rows):
                        row_idx = 7 + i  # Start at row 7 (display as row 8)
                        try:
                            self.sheet.set_row_data(row_idx, values=row)
                            successful_count += 1
                            
                            # Show every 5th row plus first and last few
                            if i < 3 or i >= len(data_rows) - 3 or i % 5 == 0:
                                print(f"    Row {row_idx} (display {row_idx+1}): DataID={row[3]} [{i+1}/{len(data_rows)}]")
                        except Exception as e:
                            logger.warning(f"Error setting row {row_idx}: {e}")
                            print(f"    FAILED Row {row_idx}: {e}")
                    
                    print(f"  Successfully inserted {successful_count}/{len(data_rows)} rows")
                    print(f"  ✅ Data insertion completed - {len(data_rows)} rows")
                    logger.info(f"Data insertion successful - {len(data_rows)} rows starting at display row 8")
                    
                except Exception as insert_error:
                    logger.error(f"Error inserting rows: {insert_error}")
                    print(f"  ❌ Data insertion failed: {insert_error}")
            
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
            print(f"\n❌ REFRESH ERROR: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            messagebox.showerror("Refresh Error", f"Failed to refresh data: {e}\n\nCheck terminal for full error details.")
    
    def _on_data_changed(self, event):
        """Handle data changes in the spreadsheet."""
        print(f"\n📝 DATA CHANGED EVENT TRIGGERED!")
        print(f"  Event: {event}")
        print(f"  Auto-save will trigger in 1 second...")
        
        # Auto-save to both file and database
        if self.refresh_job:
            self.window.after_cancel(self.refresh_job)
        self.refresh_job = self.window.after(1000, self._auto_save_changes)  # 1 second delay
    
    def _auto_save_changes(self):
        """Comprehensive auto-save: saves to both internal database and external file if available."""
        print(f"\n💾 AUTO-SAVE TRIGGERED!")
        try:
            # Update status
            if hasattr(self, 'auto_save_status'):
                print(f"  Updating auto-save status to 'Saving...'")
                self.auto_save_status.config(text="Auto-save: Saving...", foreground='orange')
                self.window.update_idletasks()
            else:
                print(f"  No auto_save_status widget found")
            
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
    
    def _validate_new_data_row(self, data_id: str, x_pos: float, y_pos: float, z_pos: float, 
                              image_name: str, coord_point: int) -> tuple:
        """Validate new data row for insertion.
        
        Args:
            data_id: The DataID string
            x_pos, y_pos, z_pos: Coordinate values
            image_name: Parsed image name
            coord_point: Parsed coordinate point
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        # Check DataID format - accept any non-empty alphanumeric string
        if not data_id or not str(data_id).strip():
            return False, f"DataID cannot be empty"
        
        # Allow any alphanumeric characters, underscores, periods, hyphens
        import re
        if not re.match(r'^[a-zA-Z0-9_.\-]+$', str(data_id).strip()):
            return False, f"Invalid DataID format: '{data_id}' (use letters, numbers, dots, underscores, hyphens only)"
        
        # Check coordinate values are numeric and reasonable
        coord_checks = [
            (x_pos, 'X'),
            (y_pos, 'Y'),
            (z_pos, 'Z')
        ]
        
        for coord_val, coord_name in coord_checks:
            if coord_val is None:
                return False, f"Missing {coord_name} coordinate"
            
            if not isinstance(coord_val, (int, float)):
                return False, f"Invalid {coord_name} coordinate: must be numeric"
            
            # Check for reasonable ranges (typical Lab* color space ranges)
            if coord_name == 'X' and not (0 <= coord_val <= 100):  # L* typically 0-100
                print(f"    ⚠️ Warning: {coord_name} coordinate {coord_val} is outside typical range [0, 100]")
            elif coord_name in ['Y', 'Z'] and not (-128 <= coord_val <= 127):  # a*, b* typically -128 to 127
                print(f"    ⚠️ Warning: {coord_name} coordinate {coord_val} is outside typical range [-128, 127]")
        
        # Check image name is reasonable
        if not image_name or len(image_name.strip()) == 0:
            return False, "Empty image name"
        
        # Check coordinate point is valid
        if not isinstance(coord_point, int) or coord_point <= 0:
            return False, f"Invalid coordinate point: {coord_point} (should be positive integer)"
        
        return True, "Valid"
    
    def _save_to_internal_database(self):
        """Save current spreadsheet changes back to the StampZ database.
        
        Now saves ALL Plot_3D columns: Cluster, ΔE, Centroid, Sphere, Radius, Marker, Color, etc.
        This ensures complete persistence of manual edits and Plot_3D analysis results.
        
        Enhanced to handle NEW DATA INSERTION when measurements don't exist in database.
        """
        try:
            print(f"\n💾 COMPREHENSIVE DATABASE SAVE:")
            
            # Get current sheet data
            data = self.sheet.get_sheet_data(get_header=False)
            print(f"  Sheet has {len(data)} rows to process")
            
            # Process data to update database
            from utils.color_analysis_db import ColorAnalysisDB
            db = ColorAnalysisDB(self.sample_set_name)
            
            # Update database with ALL Plot_3D column values
            updated_count = 0
            skipped_rows = 0
            
            for i, row_data in enumerate(data):
                if not row_data or len(row_data) < len(self.PLOT3D_COLUMNS):
                    skipped_rows += 1
                    continue
                
                # DEBUG: Show what we're processing for each row
                if i < 10:  # Debug first 10 rows
                    print(f"    DEBUG Row {i} (display {i+1}): {row_data[:8] if len(row_data) >= 8 else row_data}...")
                
                # CRITICAL FIX: Handle centroid area (rows 1-6) vs data area (rows 7+) differently
                is_centroid_area = (1 <= i <= 6)  # Rows 2-7 in display (0-indexed rows 1-6)
                is_data_area = (i >= 7)  # Rows 8+ in display (0-indexed rows 7+)
                
                print(f"    Row {i}: is_centroid_area={is_centroid_area}, is_data_area={is_data_area}")
                
                if is_centroid_area:
                    # Handle centroid area - only process if there's centroid data
                    cluster = row_data[4] if len(row_data) > 4 and row_data[4] else None
                    centroid_x = row_data[8] if len(row_data) > 8 and row_data[8] else None
                    centroid_y = row_data[9] if len(row_data) > 9 and row_data[9] else None
                    centroid_z = row_data[10] if len(row_data) > 10 and row_data[10] else None
                    sphere_color = row_data[11] if len(row_data) > 11 and row_data[11] else None
                    sphere_radius = row_data[12] if len(row_data) > 12 and row_data[12] else None
                    marker = row_data[6] if len(row_data) > 6 and row_data[6] else '.'
                    color = row_data[7] if len(row_data) > 7 and row_data[7] else 'blue'
                    
                    # Process if we have ANY meaningful centroid data
                    # Allow partial data - user might be building it up incrementally
                    has_cluster = cluster is not None and str(cluster).strip()
                    has_centroid = (centroid_x is not None and str(centroid_x).strip() and
                                  centroid_y is not None and str(centroid_y).strip() and
                                  centroid_z is not None and str(centroid_z).strip())
                    has_sphere_data = ((sphere_color is not None and str(sphere_color).strip()) or
                                     (sphere_radius is not None and str(sphere_radius).strip()))
                    
                    # Process if we have at least cluster + centroid coordinates, or any sphere data with cluster
                    if (has_cluster and has_centroid) or (has_cluster and has_sphere_data):
                        
                        try:
                            cluster_id = int(float(str(cluster).strip()))
                            
                            # Handle centroid coordinates - use None if not provided
                            centroid_x_val = float(str(centroid_x).strip()) if centroid_x and str(centroid_x).strip() else None
                            centroid_y_val = float(str(centroid_y).strip()) if centroid_y and str(centroid_y).strip() else None
                            centroid_z_val = float(str(centroid_z).strip()) if centroid_z and str(centroid_z).strip() else None
                            
                            # Handle sphere data
                            sphere_radius_val = None
                            if sphere_radius and str(sphere_radius).strip():
                                sphere_radius_val = float(str(sphere_radius).strip())
                            sphere_color_val = str(sphere_color).strip() if sphere_color and str(sphere_color).strip() else None
                            
                            print(f"    🎯 Row {i} (CENTROID AREA): Processing cluster {cluster_id} centroid data")
                            print(f"      Centroid coords: ({centroid_x_val}, {centroid_y_val}, {centroid_z_val})")
                            print(f"      Sphere data: color={sphere_color_val}, radius={sphere_radius_val}")
                            print(f"      Marker/color: {marker}/{color}")
                            
                            # Call insertion - Plot_3D will ignore NaN values appropriately
                            centroid_success = db.insert_or_update_centroid_data(
                                cluster_id=cluster_id,
                                centroid_x=centroid_x_val,
                                centroid_y=centroid_y_val,
                                centroid_z=centroid_z_val,
                                sphere_color=sphere_color_val,
                                sphere_radius=sphere_radius_val,
                                marker=marker,
                                color=color
                            )
                            
                            print(f"    🔍 Row {i}: CENTROID save result = {centroid_success} for cluster {cluster_id}")
                            
                            if centroid_success:
                                updated_count += 1
                                coord_str = f"({centroid_x_val:.3f}, {centroid_y_val:.3f}, {centroid_z_val:.3f})" if all(v is not None for v in [centroid_x_val, centroid_y_val, centroid_z_val]) else "(partial)"
                                sphere_str = f", sphere={sphere_color_val}, radius={sphere_radius_val}" if sphere_color_val or sphere_radius_val else ""
                                print(f"    ✅ Row {i}: CENTROID saved for cluster {cluster_id} - {coord_str}{sphere_str}")
                            else:
                                print(f"    ❌ Row {i}: CENTROID save failed for cluster {cluster_id}")
                                
                        except (ValueError, TypeError) as e:
                            print(f"    ❌ Row {i}: Invalid centroid data - {e}")
                            skipped_rows += 1
                    else:
                        # Empty centroid row, skip silently
                        skipped_rows += 1
                    continue  # Skip to next row, don't process as regular data
                    
                elif not is_data_area:
                    # Skip rows 0 (header) and any other non-data/non-centroid rows
                    skipped_rows += 1
                    continue
                
                # DATA AREA PROCESSING (rows 7+ only)
                # Extract ALL data columns from the worksheet
                try:
                    # Column indices based on self.PLOT3D_COLUMNS order
                    data_id = row_data[3] if len(row_data) > 3 and row_data[3] else None      # DataID
                    cluster = row_data[4] if len(row_data) > 4 and row_data[4] else None      # Cluster 
                    delta_e = row_data[5] if len(row_data) > 5 and row_data[5] else None      # ΔE
                    marker = row_data[6] if len(row_data) > 6 and row_data[6] else '.'        # Marker
                    color = row_data[7] if len(row_data) > 7 and row_data[7] else 'blue'      # Color
                    centroid_x = row_data[8] if len(row_data) > 8 and row_data[8] else None   # Centroid_X
                    centroid_y = row_data[9] if len(row_data) > 9 and row_data[9] else None   # Centroid_Y
                    centroid_z = row_data[10] if len(row_data) > 10 and row_data[10] else None # Centroid_Z
                    sphere_color = row_data[11] if len(row_data) > 11 and row_data[11] else None  # Sphere
                    sphere_radius = row_data[12] if len(row_data) > 12 and row_data[12] else None # Radius
                    
                    # Skip rows without valid DataID
                    if not data_id or not str(data_id).strip():
                        if i < 10:  # Show first 10 for debugging
                            logger.debug(f"Row {i}: Skipping - no DataID")
                        skipped_rows += 1
                        continue
                    
                    # Use flexible DataID handling - support any alphanumeric name
                    data_id = str(data_id).strip()
                    
                    # Try to parse traditional _pt format first, fallback to flexible format
                    if '_pt' in data_id and data_id.count('_pt') == 1:
                        # Traditional format: "S10_pt1", "S12_pt3", etc.
                        parts = data_id.split('_pt')
                        try:
                            image_name = parts[0]
                            coord_point = int(parts[1])
                        except (ValueError, IndexError):
                            # Fallback to flexible format
                            image_name = data_id
                            coord_point = 1  # Default coordinate point
                    else:
                        # Flexible format: "Moe", "Larry", "King_Louis_IX", etc.
                        image_name = data_id
                        coord_point = 1  # Default coordinate point
                    
                    # Convert values to proper types
                    cluster_id = None
                    if cluster and str(cluster).strip():
                        try:
                            cluster_id = int(float(str(cluster).strip()))
                        except (ValueError, TypeError):
                            pass
                    
                    delta_e_val = None
                    if delta_e and str(delta_e).strip():
                        try:
                            delta_e_val = float(str(delta_e).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    centroid_x_val = None
                    if centroid_x and str(centroid_x).strip():
                        try:
                            centroid_x_val = float(str(centroid_x).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    centroid_y_val = None
                    if centroid_y and str(centroid_y).strip():
                        try:
                            centroid_y_val = float(str(centroid_y).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    centroid_z_val = None
                    if centroid_z and str(centroid_z).strip():
                        try:
                            centroid_z_val = float(str(centroid_z).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    sphere_radius_val = None
                    if sphere_radius and str(sphere_radius).strip():
                        try:
                            sphere_radius_val = float(str(sphere_radius).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    sphere_color_val = None
                    if sphere_color and str(sphere_color).strip():
                        sphere_color_val = str(sphere_color).strip()
                    
                    # Extract X/Y/Z coordinates from the worksheet for potential new data insertion
                    x_pos = row_data[0] if len(row_data) > 0 and row_data[0] else None
                    y_pos = row_data[1] if len(row_data) > 1 and row_data[1] else None
                    z_pos = row_data[2] if len(row_data) > 2 and row_data[2] else None
                    
                    # Convert coordinate values to proper types
                    x_pos_val = None
                    if x_pos and str(x_pos).strip():
                        try:
                            x_pos_val = float(str(x_pos).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    y_pos_val = None
                    if y_pos and str(y_pos).strip():
                        try:
                            y_pos_val = float(str(y_pos).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    z_pos_val = None
                    if z_pos and str(z_pos).strip():
                        try:
                            z_pos_val = float(str(z_pos).strip())
                        except (ValueError, TypeError):
                            pass
                    
                    # First, try to update existing measurement
                    print(f"    🔄 Row {i}: Attempting UPDATE for {image_name} pt{coord_point}")
                    print(f"      Data: cluster={cluster_id}, ∆E={delta_e_val}, marker={marker}, color={color}")
                    print(f"      Centroid: ({centroid_x_val}, {centroid_y_val}, {centroid_z_val})")
                    print(f"      Sphere: color={sphere_color_val}, radius={sphere_radius_val}")
                    
                    success = db.update_plot3d_extended_values(
                        image_name=image_name,
                        coordinate_point=coord_point,
                        cluster_id=cluster_id,
                        delta_e=delta_e_val,
                        centroid_x=centroid_x_val,
                        centroid_y=centroid_y_val,
                        centroid_z=centroid_z_val,
                        sphere_color=sphere_color_val,
                        sphere_radius=sphere_radius_val,
                        marker=marker,
                        color=color,
                        trendline_valid=True  # All valid data points are trendline-valid
                    )
                    
                    print(f"    🔍 Row {i}: UPDATE result = {success} for {image_name} pt{coord_point}")
                    
                    if success:
                        updated_count += 1
                        print(f"    ✅ Row {i}: UPDATED {image_name} pt{coord_point} - cluster={cluster_id}, ∆E={delta_e_val}, marker={marker}, color={color}")
                    else:
                        # Update failed - this might be NEW DATA that needs INSERTION
                        print(f"    🔄 Row {i}: UPDATE FAILED for {image_name} pt{coord_point} - attempting INSERTION of new data")
                        
                        # Validate that we have minimum required data for insertion
                        if x_pos_val is not None and y_pos_val is not None and z_pos_val is not None:
                            # Validate the new data before insertion
                            is_valid, validation_msg = self._validate_new_data_row(
                                data_id, x_pos_val, y_pos_val, z_pos_val, image_name, coord_point
                            )
                            
                            if not is_valid:
                                print(f"    ❌ Row {i}: DATA VALIDATION FAILED - {validation_msg}")
                                skipped_rows += 1
                                continue
                            
                            # We have valid coordinate data - attempt to insert new measurement
                            print(f"    ✅ Row {i}: Data validation passed - proceeding with insertion")
                            insert_success = db.insert_new_measurement(
                                image_name=image_name,
                                coordinate_point=coord_point,
                                x_pos=x_pos_val or 0.0,
                                y_pos=y_pos_val or 0.0,
                                l_value=x_pos_val or 0.0,  # Use X as L for Plot_3D compatibility
                                a_value=y_pos_val or 0.0,  # Use Y as A
                                b_value=z_pos_val or 0.0,  # Use Z as B
                                rgb_r=0.0, rgb_g=0.0, rgb_b=0.0,  # Default RGB values
                                cluster_id=cluster_id,
                                delta_e=delta_e_val,
                                centroid_x=centroid_x_val,
                                centroid_y=centroid_y_val,
                                centroid_z=centroid_z_val,
                                sphere_color=sphere_color_val,
                                sphere_radius=sphere_radius_val,
                                marker=marker,
                                color=color,
                                sample_type='manual_entry',
                                notes=f'Added via internal worksheet row {i+1}'
                            )
                            
                            if insert_success:
                                updated_count += 1
                                print(f"    ✅ Row {i}: INSERTED NEW {image_name} pt{coord_point} - X={x_pos_val}, Y={y_pos_val}, Z={z_pos_val}, cluster={cluster_id}")
                            else:
                                print(f"    ❌ Row {i}: INSERTION ALSO FAILED for {image_name} pt{coord_point}")
                                print(f"      DataID: {data_id}")
                                print(f"      Coordinates: X={x_pos_val}, Y={y_pos_val}, Z={z_pos_val}")
                                print(f"      Extended: marker={marker}, color={color}, cluster={cluster_id}, ∆E={delta_e_val}")
                        else:
                            # Check if this might be centroid data (has centroid coordinates but no sample coordinates)
                            if centroid_x_val is not None and centroid_y_val is not None and centroid_z_val is not None and cluster_id is not None:
                                print(f"    🎯 Row {i}: Detected CENTROID DATA for cluster {cluster_id}")
                                centroid_success = db.insert_or_update_centroid_data(
                                    cluster_id=cluster_id,
                                    centroid_x=centroid_x_val,
                                    centroid_y=centroid_y_val,
                                    centroid_z=centroid_z_val,
                                    sphere_color=sphere_color_val,
                                    sphere_radius=sphere_radius_val,
                                    marker=marker,
                                    color=color
                                )
                                
                                if centroid_success:
                                    updated_count += 1
                                    print(f"    ✅ Row {i}: CENTROID DATA saved for cluster {cluster_id} - ({centroid_x_val:.3f}, {centroid_y_val:.3f}, {centroid_z_val:.3f})")
                                else:
                                    print(f"    ❌ Row {i}: CENTROID INSERTION FAILED for cluster {cluster_id}")
                            else:
                                print(f"    ⚠️ Row {i}: INSUFFICIENT DATA for insertion - need X/Y/Z coordinates OR centroid data")
                                print(f"      DataID: {data_id}")
                                print(f"      Coordinates: X={x_pos_val}, Y={y_pos_val}, Z={z_pos_val}")
                                print(f"      Centroid: ({centroid_x_val}, {centroid_y_val}, {centroid_z_val}), cluster={cluster_id}")
                        
                except Exception as row_error:
                    logger.debug(f"Row {i}: Error processing - {row_error}")
                    skipped_rows += 1
                    continue
            
            print(f"  ✅ Updated {updated_count} measurements in database")
            print(f"  ⚠️ Skipped {skipped_rows} rows (no valid data/DataID)")
            
            logger.info(f"✅ COMPREHENSIVE DATABASE SAVE COMPLETE: {updated_count} measurements updated, {skipped_rows} skipped")
            
        except Exception as e:
            logger.error(f"Error saving to internal database: {e}")
            print(f"  ❌ Database save error: {e}")
    
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
            
            # Reset index after filtering to ensure consecutive numbering
            df.reset_index(drop=True, inplace=True)
            
            # Save to ODS format (Plot_3D compatible)
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
            print(f"\n📝 READING WORKSHEET DATA:")
            data = self.sheet.get_sheet_data(get_header=False)
            print(f"  Raw sheet data rows: {len(data)}")
            
            # DEBUG: Show first few raw rows to verify manual edits are captured
            print(f"\n🔍 FIRST 3 RAW SHEET ROWS:")
            for i in range(min(3, len(data))):
                if i < len(data) and len(data[i]) >= 8:  # Make sure row has enough columns
                    row = data[i]
                    print(f"  Sheet row {i}: DataID='{row[3] if len(row) > 3 else 'N/A'}', Marker='{row[6] if len(row) > 6 else 'N/A'}', Color='{row[7] if len(row) > 7 else 'N/A'}', Sphere='{row[11] if len(row) > 11 else 'N/A'}'")
            
            # Create DataFrame with correct column names
            df = pd.DataFrame(data, columns=self.PLOT3D_COLUMNS)
            
            # Clean the data - remove completely empty rows and replace empty strings with NaN
            df = df.replace('', np.nan)
            
            # Keep rows that have coordinate data OR centroid data
            coordinate_cols = ['Xnorm', 'Ynorm', 'Znorm']
            centroid_cols = ['Centroid_X', 'Centroid_Y', 'Centroid_Z']
            
            has_coordinate_data = df[coordinate_cols].notna().any(axis=1)
            has_centroid_data = df[centroid_cols].notna().all(axis=1)  # All 3 centroid coords must be present
            
            # Keep rows with either coordinate data OR complete centroid data
            has_valid_data = has_coordinate_data | has_centroid_data
            
            print(f"  Filtering: coordinate_data={has_coordinate_data.sum()}, centroid_data={has_centroid_data.sum()}, total_kept={has_valid_data.sum()}")
            
            # CRITICAL FIX: Preserve original sheet row positions before filtering
            df['_original_sheet_row'] = df.index  # Store original sheet row indices
            
            # Filter using the new logic that includes centroid data
            df = df[has_valid_data].copy()
            
            # Reset index to ensure consecutive numbering starting from 0
            # This prevents "positional indexers are out-of-bounds" errors in K-means clustering
            df.reset_index(drop=True, inplace=True)
            
            # DEBUG: Show the mapping between DataFrame indices and original sheet rows
            if len(df) > 0:
                print(f"\n📝 DATAFRAME-TO-SHEET MAPPING:")
                for i in range(min(10, len(df))):
                    orig_row = df.iloc[i]['_original_sheet_row']
                    data_id = df.iloc[i]['DataID']
                    print(f"  DataFrame index {i} → original sheet row {orig_row} (display {orig_row+1}), DataID: {data_id}")
            
            # Convert coordinate columns to numeric
            numeric_cols = ['Xnorm', 'Ynorm', 'Znorm', 'Centroid_X', 'Centroid_Y', 'Centroid_Z', '∆E', 'Radius']
            for col in numeric_cols:
                if col in df.columns:
                    # For Radius column, preserve valid numeric values and convert empty strings to NaN properly
                    if col == 'Radius':
                        # Debug: Check what we're converting
                        print(f"DEBUG: Converting {col} column - sample values: {df[col].head().tolist()}")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set default values for missing data
            df['Cluster'] = df['Cluster'].fillna('')
            df['Marker'] = df['Marker'].fillna('.')
            df['Color'] = df['Color'].fillna('blue')
            df['Sphere'] = df['Sphere'].fillna('')
            
            # CRITICAL FIX: Add trendline_valid column for Plot_3D trendline functionality
            # Mark all data points with valid coordinates as trendline-valid
            df['trendline_valid'] = True  # All filtered points already have valid coordinates
            
            logger.info(f"Converted tksheet data to DataFrame: {len(df)} rows with coordinate data")
            logger.debug(f"DataFrame shape: {df.shape}, Index: {df.index.min()}-{df.index.max() if len(df) > 0 else 'empty'}")
            logger.debug(f"Sample coordinate data (first 3 rows): {df[coordinate_cols].head(3).to_dict('records') if len(df) > 0 else 'No data'}")
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
            # Pass the worksheet update callback to enable bidirectional data flow
            self.plot3d_app = Plot3DApp(
                parent=self.parent, 
                dataframe=df,
                worksheet_update_callback=self.update_worksheet_from_plot3d
            )
            
            messagebox.showinfo(
                "Plot_3D Launched",
                f"Plot_3D opened with current spreadsheet data ({len(df)} data points).\n\n"
                f"✅ No external files needed!\n"
                f"✅ Bidirectional sync enabled - K-means and ΔE results will automatically update the worksheet!\n"
                f"Changes in this spreadsheet will be reflected when you click 'Refresh Plot_3D'."
            )
            
        except Exception as e:
            logger.error(f"Error launching Plot_3D: {e}")
            messagebox.showerror("Launch Error", f"Failed to open Plot_3D: {e}")
    
    def update_worksheet_from_plot3d(self, updated_df, kmeans_start_row=None, kmeans_end_row=None):
        """Callback method to receive updates from Plot_3D and update internal worksheet.
        
        This method is called by Plot_3D when data changes (e.g., after K-means clustering)
        to push those changes back to the internal worksheet UI.
        
        Template structure:
        - Rows 1-6: Sequential cluster summary (0,1,2,3,4,5) with centroid coordinates  
        - Rows 7+: Individual data points with cluster assignments only
        
        Args:
            updated_df (pd.DataFrame): Updated DataFrame from Plot_3D with new cluster/centroid data
            kmeans_start_row (int): Original start row from K-means selection (display row numbers)
            kmeans_end_row (int): Original end row from K-means selection (display row numbers)
        """
        try:
            logger.info(f"Received Plot_3D update with {len(updated_df)} rows")
            logger.debug(f"DataFrame index range: {updated_df.index.min()}-{updated_df.index.max()}")
            
            # ENHANCED DEBUG: Show DataFrame content
            print(f"\n🔍 CALLBACK DEBUG:")
            print(f"DataFrame shape: {updated_df.shape}")
            print(f"DataFrame columns: {list(updated_df.columns)}")
            print(f"DataFrame indices: {list(updated_df.index[:10])}...")  # First 10 indices
            print(f"Sample DataIDs: {list(updated_df['DataID'].head(5))}")
            print(f"Has _original_sheet_row column: {'_original_sheet_row' in updated_df.columns}")
            
            # Show first few DataFrame rows with their mapping
            if '_original_sheet_row' in updated_df.columns:
                print(f"\n📝 FIRST 5 DATAFRAME ROWS WITH MAPPING:")
                for i in range(min(5, len(updated_df))):
                    row = updated_df.iloc[i]
                    df_idx = updated_df.index[i]
                    orig_sheet_row = row['_original_sheet_row'] if pd.notna(row['_original_sheet_row']) else 'N/A'
                    data_id = row['DataID'] if pd.notna(row['DataID']) else 'N/A'
                    cluster = row['Cluster'] if pd.notna(row['Cluster']) else 'N/A'
                    print(f"  Row {i}: DataFrame idx={df_idx} → sheet row {orig_sheet_row} (display {int(orig_sheet_row)+1 if orig_sheet_row != 'N/A' else 'N/A'}), DataID='{data_id}', Cluster='{cluster}'")
            else:
                print(f"\n❌ NO _original_sheet_row COLUMN FOUND!")
            
            if kmeans_start_row is not None and kmeans_end_row is not None:
                logger.info(f"K-means row selection: display rows {kmeans_start_row}-{kmeans_end_row}")
                print(f"K-means selection: display rows {kmeans_start_row}-{kmeans_end_row}")
            else:
                print("No K-means row selection info available")
            
            # CRITICAL DEBUG: Show DataFrame index range vs expected
            df_min_idx = updated_df.index.min()
            df_max_idx = updated_df.index.max()
            print(f"\n🚨 SELECTION MISMATCH DEBUG:")
            print(f"  - You selected: rows 8-27 (20 rows)")
            print(f"  - DataFrame received: indices {df_min_idx}-{df_max_idx} ({len(updated_df)} rows)")
            print(f"  - Expected mapping: DataFrame index {df_min_idx} should → sheet row 8 (display E9)")
            
            # Column indices based on self.PLOT3D_COLUMNS
            cluster_col_idx = 4  # Cluster column (E)
            delta_e_col_idx = 5  # ∆E column (F) 
            centroid_x_col_idx = 8  # Centroid_X column (I)
            centroid_y_col_idx = 9  # Centroid_Y column (J)
            centroid_z_col_idx = 10 # Centroid_Z column (K)
            
            # STEP 1: Update cluster summary section (rows 1-6) with sequential cluster info  
            cluster_summary_start_row = 1  # 0-based row 1 (headers are in row 0)
            cluster_summary_updated = 0
            
            # Get unique clusters and their centroids from the updated DataFrame
            clusters_with_data = updated_df[updated_df['Cluster'].notna()]
            if not clusters_with_data.empty:
                unique_clusters = sorted(clusters_with_data['Cluster'].unique())
                logger.info(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")
                
                for i, cluster_num in enumerate(unique_clusters):
                    summary_row_idx = cluster_summary_start_row + i  # Rows 1,2,3,4,5,6 (display 2,3,4,5,6,7)
                    
                    # Skip if we exceed the reserved summary area (rows 1-6)
                    if summary_row_idx > 6:  # 0-based row 6 is the last summary row
                        logger.warning(f"Cluster {cluster_num} exceeds summary area, skipping")
                        break
                        
                    try:
                        # Get current row data
                        try:
                            current_row = list(self.sheet.get_row_data(summary_row_idx))
                        except:
                            current_row = [''] * len(self.PLOT3D_COLUMNS)
                        
                        # Ensure row has enough columns
                        while len(current_row) < len(self.PLOT3D_COLUMNS):
                            current_row.append('')
                        
                        # Set sequential cluster number (0, 1, 2, 3...)
                        current_row[cluster_col_idx] = str(int(cluster_num))
                        
                        # Calculate and set centroid coordinates for this cluster
                        cluster_data = clusters_with_data[clusters_with_data['Cluster'] == cluster_num]
                        if not cluster_data.empty:
                            centroid_x = cluster_data['Centroid_X'].iloc[0]  # All rows in cluster have same centroid
                            centroid_y = cluster_data['Centroid_Y'].iloc[0]
                            centroid_z = cluster_data['Centroid_Z'].iloc[0]
                            
                            if not pd.isna(centroid_x):
                                current_row[centroid_x_col_idx] = str(round(float(centroid_x), 4))
                            if not pd.isna(centroid_y):
                                current_row[centroid_y_col_idx] = str(round(float(centroid_y), 4))
                            if not pd.isna(centroid_z):
                                current_row[centroid_z_col_idx] = str(round(float(centroid_z), 4))
                        
                        # Update the summary row
                        self.sheet.set_row_data(summary_row_idx, values=current_row)
                        cluster_summary_updated += 1
                        logger.debug(f"Updated cluster summary row {summary_row_idx + 1} (display {summary_row_idx + 2}) for cluster {int(cluster_num)}")
                        
                    except Exception as summary_error:
                        logger.warning(f"Error updating cluster summary for cluster {cluster_num}: {summary_error}")
                        continue
            
            # STEP 2: Update individual data points with cluster assignments only
            data_points_updated = 0
            
            # DEBUG: Check actual worksheet structure to find where data really starts
            print(f"\n🔍 WORKSHEET STRUCTURE DEBUG:")
            sheet_data = self.sheet.get_sheet_data(get_header=False)
            print(f"Total sheet rows: {len(sheet_data)}")
            
            # Check rows 8-30 to see the actual data structure where we expect to write
            print(f"\n🔍 SHEET ROWS 8-30 (where individual data should be):")
            for i in range(8, min(31, len(sheet_data))):
                if i < len(sheet_data):
                    row_data = sheet_data[i]
                    has_coords = False
                    if len(row_data) >= 3:
                        has_coords = any(str(row_data[j]).strip() not in ['', 'None', 'nan'] for j in [0,1,2])
                    
                    data_id = row_data[3] if len(row_data) > 3 else ''
                    print(f"  Sheet row {i} (display {i+1}): coords={has_coords}, DataID='{data_id}'")
            
            # DEBUG: Check cluster values and DataFrame structure
            print(f"\n🔍 DATAFRAME STRUCTURE DEBUG:")
            cluster_counts = updated_df['Cluster'].value_counts(dropna=False)
            print(f"Cluster value counts: {cluster_counts.to_dict()}")
            print(f"Non-null cluster count: {updated_df['Cluster'].notna().sum()}/{len(updated_df)}")
            
            # Show first 10 DataFrame rows with their DataIDs and clusters
            print(f"\n🔍 FIRST 10 DATAFRAME ROWS:")
            for idx in range(min(10, len(updated_df))):
                df_idx = updated_df.index[idx]
                row = updated_df.iloc[idx]
                data_id = row.get('DataID', 'N/A')
                cluster = row.get('Cluster', 'N/A')
                print(f"  DataFrame index {df_idx}: DataID='{data_id}', Cluster='{cluster}'")
            
            # Now update the data points with cluster assignments
            for df_idx, df_row in updated_df.iterrows():
                try:
                    # FINAL FIX: Use original sheet row positions from DataFrame
                    # Each DataFrame row stores its original sheet position in '_original_sheet_row'
                    if '_original_sheet_row' in df_row:
                        sheet_row_idx = int(df_row['_original_sheet_row'])
                        print(f"  🔄 USING PRESERVED MAPPING: DataFrame index {df_idx} → original sheet row {sheet_row_idx}")
                    else:
                        # Fallback to old calculation if mapping not available
                        min_df_index = updated_df.index.min()
                        relative_index = df_idx - min_df_index
                        sheet_row_idx = 8 + relative_index
                        print(f"  ⚠️ FALLBACK MAPPING: DataFrame index {df_idx} → calculated sheet row {sheet_row_idx}")
                        
                    # ENHANCED DEBUG LOGGING
                    logger.info(f"🔍 PRESERVED MAPPING: DataFrame index {df_idx} → sheet row {sheet_row_idx} (display E{sheet_row_idx + 1})")
                    print(f"🔍 PRESERVED MAPPING: DataFrame index {df_idx} → sheet row {sheet_row_idx} (display E{sheet_row_idx + 1})")
                    
                    # Get current row data
                    try:
                        current_row = list(self.sheet.get_row_data(sheet_row_idx))
                    except:
                        current_row = [''] * len(self.PLOT3D_COLUMNS)
                    
                    # Ensure row has enough columns
                    while len(current_row) < len(self.PLOT3D_COLUMNS):
                        current_row.append('')
                    
                    # Update cluster assignment for individual data point
                    cluster_value = df_row.get('Cluster', '')
                    
                    # ENHANCED DEBUG: Show cluster value for each DataFrame index
                    if df_idx < 10:  # Show first 10 for clarity
                        print(f"  DataFrame index {df_idx}: cluster_value = '{cluster_value}', type = {type(cluster_value)}")
                    
                    if not pd.isna(cluster_value) and cluster_value != '':
                        current_row[cluster_col_idx] = str(int(cluster_value))
                        print(f"✅ WRITING CLUSTER {int(cluster_value)} to sheet row {sheet_row_idx} (display E{sheet_row_idx + 1})")
                    else:
                        if df_idx < 10:  # Show first 10 for clarity
                            print(f"  ❌ NO CLUSTER VALUE to write for DataFrame index {df_idx}")
                    
                    # Update ΔE value if present
                    if not pd.isna(df_row.get('∆E', '')) and df_row.get('∆E', '') != '':
                        current_row[delta_e_col_idx] = str(df_row['∆E'])
                    
                    # NOTE: Individual data points do NOT get centroid coordinates 
                    # (those go only in the cluster summary section)
                    
                    # Set the updated row data back to the sheet
                    self.sheet.set_row_data(sheet_row_idx, values=current_row)
                    data_points_updated += 1
                    
                    # CRITICAL DEBUG: Show exactly where we're writing vs where it should appear
                    if df_idx < 3:  # Only show first 3 for clarity
                        print(f"\n📝 WRITE DEBUG for DataFrame index {df_idx}:")
                        print(f"  - Writing to internal sheet row index: {sheet_row_idx}")
                        print(f"  - Expected display column: E{sheet_row_idx + 1} (corrected for row 8 start)")
                        print(f"  - You see it in: (please check and report)")
                        print(f"  - Cluster value written: {cluster_value if 'cluster_value' in locals() else 'None'}")
                    
                    logger.debug(f"Updated data point row {sheet_row_idx + 1} (display {sheet_row_idx + 2}) with cluster assignment")
                    
                except Exception as row_error:
                    logger.warning(f"Error updating data point row {df_idx}: {row_error}")
                    continue
            
            logger.info(f"Successfully updated worksheet: {cluster_summary_updated} cluster summaries + {data_points_updated} data points")
            
            # Trigger auto-save to preserve changes
            self._auto_save_changes()
            
            # Show a brief status update
            if hasattr(self, 'auto_save_status'):
                original_text = self.auto_save_status.cget('text')
                self.auto_save_status.config(text=f"Updated from Plot_3D ({cluster_summary_updated} clusters, {data_points_updated} points) ✓", foreground='blue')
                # Reset after 3 seconds
                self.window.after(3000, lambda: self.auto_save_status.config(text=original_text, foreground='green'))
                
        except Exception as e:
            logger.error(f"Error updating worksheet from Plot_3D: {e}")
            messagebox.showerror("Update Error", f"Failed to update worksheet from Plot_3D: {e}")
    
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
            
            # ENHANCED DEBUG: Show DataFrame details being sent to Plot_3D
            print(f"\n🔄 PLOT_3D REFRESH DEBUG:")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {list(df.columns)}")
            print(f"Has trendline_valid: {'trendline_valid' in df.columns}")
            
            # Show first few rows to verify data
            print(f"\nFirst 3 rows being sent to Plot_3D:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                print(f"  Row {i}: DataID='{row.get('DataID', 'N/A')}', Marker='{row.get('Marker', 'N/A')}', Color='{row.get('Color', 'N/A')}', Radius='{row.get('Radius', 'N/A')}'")
                print(f"    Sphere='{row.get('Sphere', 'N/A')}', Centroid_X='{row.get('Centroid_X', 'N/A')}', Centroid_Y='{row.get('Centroid_Y', 'N/A')}', Centroid_Z='{row.get('Centroid_Z', 'N/A')}'")
            
            # Update Plot_3D with new data
            print(f"\n⚙️ Updating plot3d_app.df with new DataFrame...")
            self.plot3d_app.df = df
            
            # Refresh the plot
            if hasattr(self.plot3d_app, 'refresh_plot'):
                print(f"🔄 Calling plot3d_app.refresh_plot()...")
                self.plot3d_app.refresh_plot()
                logger.info(f"Refreshed Plot_3D with {len(df)} data points")
                print(f"✅ Plot_3D refresh completed successfully")
                messagebox.showinfo(
                    "Plot_3D Refreshed",
                    f"✅ Updated Plot_3D with {len(df)} data points from spreadsheet!\n\n"
                    f"Check Plot_3D window to verify changes are visible."
                )
            else:
                print(f"❌ Plot_3D refresh method not available")
                messagebox.showwarning(
                    "Refresh Not Available",
                    "Plot_3D refresh method not available. Please restart Plot_3D."
                )
                
        except Exception as e:
            logger.error(f"Error refreshing Plot_3D: {e}")
            messagebox.showerror("Refresh Error", f"Failed to refresh Plot_3D: {e}")
    
    def _push_changes_to_plot3d(self):
        """Push current worksheet changes to Plot_3D without requiring external file save.
        
        This allows users to edit markers, colors, radius values, etc. in the worksheet
        and see those changes reflected immediately in Plot_3D.
        """
        print("DEBUG: Push Changes to Plot_3D button clicked")
        try:
            if not self.plot3d_app:
                messagebox.showinfo(
                    "Plot_3D Not Open",
                    "Please click 'Open in Plot_3D' first to launch the 3D visualization."
                )
                return
            
            # Get current data as DataFrame from worksheet
            df = self.get_data_as_dataframe()
            
            if df is None or len(df) == 0:
                messagebox.showwarning(
                    "No Data",
                    "No valid coordinate data found in the spreadsheet."
                )
                return
            
            # ENHANCED DEBUG: Show changes being pushed to Plot_3D
            print(f"\n🚀 PUSH CHANGES DEBUG:")
            print(f"DataFrame shape: {df.shape}")
            print(f"Has trendline_valid: {'trendline_valid' in df.columns}")
            
            # Show changes in first few rows
            print(f"\nChanges being pushed to Plot_3D (first 3 rows):")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                print(f"  Row {i}: DataID='{row.get('DataID', 'N/A')}', Marker='{row.get('Marker', 'N/A')}', Color='{row.get('Color', 'N/A')}', Cluster='{row.get('Cluster', 'N/A')}'")
            
            # Update Plot_3D's internal DataFrame with current worksheet data
            print(f"\n💾 Updating plot3d_app.df...")
            self.plot3d_app.df = df
            
            # Refresh the plot to show changes
            if hasattr(self.plot3d_app, 'refresh_plot'):
                print(f"🔄 Calling plot3d_app.refresh_plot()...")
                self.plot3d_app.refresh_plot()
                logger.info(f"Pushed worksheet changes to Plot_3D: {len(df)} data points")
                print(f"✅ Push changes completed successfully")
                
                # Also save changes to internal database for persistence
                self._auto_save_changes()
                
                messagebox.showinfo(
                    "Changes Pushed",
                    f"✅ Successfully pushed worksheet changes to Plot_3D!\n\n"
                    f"Updated {len(df)} data points with current:\n"
                    f"• Cluster assignments\n"
                    f"• Marker preferences\n"
                    f"• Color preferences\n"
                    f"• Radius values\n"
                    f"• ΔE values\n\n"
                    f"Plot_3D visualization has been refreshed."
                )
            else:
                messagebox.showwarning(
                    "Refresh Not Available",
                    "Plot_3D refresh method not available. Please restart Plot_3D."
                )
                
        except Exception as e:
            logger.error(f"Error pushing changes to Plot_3D: {e}")
            messagebox.showerror("Push Error", f"Failed to push changes to Plot_3D: {e}")
    
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
                    ('All files', '*.*')
                ],
                initialfile=default_name
            )
            
            if not file_path:
                return  # User cancelled
            
            # Check if file exists and offer to append/merge instead of overwriting
            if os.path.exists(file_path):
                # Check if it's a Plot_3D compatible file by looking for existing data
                try:
                    existing_df = pd.read_excel(file_path, engine='odf')
                    has_plot3d_data = ('Xnorm' in existing_df.columns and 
                                     'DataID' in existing_df.columns and 
                                     len(existing_df) > 0)
                    
                    if has_plot3d_data:
                        # Ask user if they want to merge/append or replace
                        from tkinter import messagebox
                        choice = messagebox.askyesnocancel(
                            "File Already Exists",
                            f"The file '{os.path.basename(file_path)}' already exists and contains Plot_3D data.\n\n"
                            f"• YES: Merge/Append new data (preserves existing K-means, ΔE results)\n"
                            f"• NO: Replace all data (loses existing analysis results)\n"
                            f"• Cancel: Choose different filename\n\n"
                            f"Recommended: Choose YES to preserve your analysis results."
                        )
                        
                        if choice is None:  # Cancel
                            return
                        elif choice:  # Yes - merge/append
                            success = self._merge_with_existing_file(df, file_path)
                            if success:
                                logger.info(f"Successfully merged data with existing file: {file_path}")
                            else:
                                logger.warning("Merge failed, falling back to template export")
                                success = self._export_using_template(df, file_path)
                        else:  # No - replace
                            success = self._export_using_template(df, file_path)
                    else:
                        # File exists but no Plot_3D data, use template export
                        success = self._export_using_template(df, file_path)
                        
                except Exception as e:
                    logger.warning(f"Could not read existing file for merge check: {e}")
                    success = self._export_using_template(df, file_path)
            else:
                # File doesn't exist, create new
                success = self._export_using_template(df, file_path)
            
            if not success:
                # Fallback to basic export if template method fails
                logger.warning("Template export failed, using basic export method")
                
                # Check if file exists and handle permissions for fallback method too
                if os.path.exists(file_path):
                    try:
                        import stat
                        file_stat = os.stat(file_path)
                        if not (file_stat.st_mode & stat.S_IWRITE):
                            os.chmod(file_path, file_stat.st_mode | stat.S_IWRITE)
                            logger.info(f"Made existing file writable for fallback export: {file_path}")
                    except Exception as perm_error:
                        logger.warning(f"Could not modify file permissions for fallback: {perm_error}")
                        raise PermissionError(f"Cannot write to existing file: {file_path}. Please close the file if it's open in another application, or choose a different filename.")
                
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
            from tkinter import messagebox
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
            
        except PermissionError as e:
            logger.error(f"Permission error during export: {e}")
            from tkinter import messagebox
            messagebox.showerror(
                "Permission Error", 
                f"Cannot write to the selected file location.\n\n"
                f"This usually means:\n"
                f"• The file is currently open in another application\n"
                f"• The file is read-only\n"
                f"• You don't have write permissions to that location\n\n"
                f"Solutions:\n"
                f"• Close the file if it's open elsewhere\n"
                f"• Choose a different filename\n"
                f"• Save to a different location (like Documents folder)\n\n"
                f"Technical details: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error exporting for Plot_3D: {e}")
            from tkinter import messagebox
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def _merge_with_existing_file(self, new_df: pd.DataFrame, file_path: str) -> bool:
        """Merge new data with existing Plot_3D file, preserving analysis results.
        
        This function:
        1. Reads existing file data
        2. Identifies which DataIDs are new vs existing
        3. Updates existing rows with new coordinate data (preserves Cluster, ΔE, etc.)
        4. Appends completely new DataIDs
        5. Maintains Plot_3D format structure
        
        Args:
            new_df: DataFrame with new data from internal worksheet
            file_path: Path to existing Plot_3D file
            
        Returns:
            bool: True if merge was successful
        """
        try:
            import ezodf
            import pandas as pd
            
            logger.info(f"Starting merge operation with existing file: {file_path}")
            
            # Read existing file
            existing_df = pd.read_excel(file_path, engine='odf')
            logger.info(f"Existing file has {len(existing_df)} rows")
            
            # Get existing DataIDs
            existing_dataids = set()
            if 'DataID' in existing_df.columns:
                existing_dataids = set(existing_df['DataID'].dropna().astype(str))
            
            # Get new DataIDs
            new_dataids = set()
            if 'DataID' in new_df.columns:
                new_dataids = set(new_df['DataID'].dropna().astype(str))
            
            # Identify what needs to be updated vs added
            dataids_to_update = existing_dataids.intersection(new_dataids)
            dataids_to_add = new_dataids - existing_dataids
            
            logger.info(f"DataIDs to update (preserve analysis): {len(dataids_to_update)}")
            logger.info(f"DataIDs to add (new data): {len(dataids_to_add)}")
            
            # Create backup
            backup_path = f"{file_path}.backup_merge_{int(time.time())}"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            # Open file for editing
            doc = ezodf.opendoc(file_path)
            sheet = doc.sheets[0]
            
            # Map column names to indices
            coord_columns = {'Xnorm': None, 'Ynorm': None, 'Znorm': None, 'DataID': None}
            
            # Find column indices (assuming row 8 contains headers in Plot_3D format)
            header_row = 7  # Row 8, 0-based
            for col_idx in range(min(15, sheet.ncols())):  # Check first 15 columns
                cell_value = str(sheet[header_row, col_idx].value or '').strip()
                if cell_value in coord_columns:
                    coord_columns[cell_value] = col_idx
            
            # Verify we found the essential columns
            missing_columns = [col for col, idx in coord_columns.items() if idx is None]
            if missing_columns:
                logger.error(f"Could not find columns in existing file: {missing_columns}")
                return False
            
            logger.info(f"Found columns: {coord_columns}")
            
            # Update existing rows (preserve analysis columns)
            update_count = 0
            data_start_row = 8  # Row 9, 0-based (data starts after header)
            
            for row_idx in range(data_start_row, sheet.nrows()):
                existing_dataid = sheet[row_idx, coord_columns['DataID']].value
                if existing_dataid and str(existing_dataid).strip() in dataids_to_update:
                    # Find corresponding row in new data
                    new_row = new_df[new_df['DataID'] == str(existing_dataid).strip()]
                    if not new_row.empty:
                        new_row = new_row.iloc[0]
                        
                        # Update only coordinate columns, preserve analysis results
                        sheet[row_idx, coord_columns['Xnorm']].set_value(float(new_row['Xnorm']))
                        sheet[row_idx, coord_columns['Ynorm']].set_value(float(new_row['Ynorm']))
                        sheet[row_idx, coord_columns['Znorm']].set_value(float(new_row['Znorm']))
                        # DataID stays the same
                        
                        update_count += 1
                        logger.debug(f"Updated existing DataID: {existing_dataid}")
            
            # Add new rows
            add_count = 0
            if dataids_to_add:
                # Find next empty row
                next_empty_row = sheet.nrows()
                for row_idx in range(data_start_row, sheet.nrows()):
                    # Check if all coordinate columns are empty
                    if all(not sheet[row_idx, coord_columns[col]].value 
                          for col in ['Xnorm', 'Ynorm', 'Znorm', 'DataID']):
                        next_empty_row = row_idx
                        break
                
                # Add new DataIDs
                current_row = next_empty_row
                for dataid in dataids_to_add:
                    new_row = new_df[new_df['DataID'] == dataid]
                    if not new_row.empty:
                        new_row = new_row.iloc[0]
                        
                        # Set coordinate data
                        sheet[current_row, coord_columns['Xnorm']].set_value(float(new_row['Xnorm']))
                        sheet[current_row, coord_columns['Ynorm']].set_value(float(new_row['Ynorm']))
                        sheet[current_row, coord_columns['Znorm']].set_value(float(new_row['Znorm']))
                        sheet[current_row, coord_columns['DataID']].set_value(str(new_row['DataID']))
                        
                        # Set default values for analysis columns (will be empty for Plot_3D to fill)
                        # Don't overwrite if they already have values
                        
                        current_row += 1
                        add_count += 1
                        logger.debug(f"Added new DataID: {dataid}")
            
            # Save the merged file
            temp_path = f"{file_path}.temp_merge"
            doc.saveas(temp_path)
            os.replace(temp_path, file_path)
            
            # Clean up backup on success
            try:
                os.remove(backup_path)
            except Exception:
                logger.warning(f"Could not remove backup file: {backup_path}")
            
            logger.info(f"Merge completed successfully: {update_count} updated, {add_count} added")
            
            # Show user feedback
            from tkinter import messagebox
            messagebox.showinfo(
                "Merge Complete",
                f"Successfully merged data with existing file!\n\n"
                f"• Updated coordinates for {update_count} existing entries\n"
                f"• Added {add_count} new entries\n"
                f"• Preserved all existing K-means clusters and ΔE values\n\n"
                f"Your analysis results are intact!"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error during merge operation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _export_using_template(self, df, output_path):
        """Export data using Plot3D template to preserve formatting.
        
        Args:
            df: DataFrame with Plot3D data
            output_path: Path where to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"DEBUG: Starting template export to {output_path}")
            import shutil
            
            # Determine file extension
            file_ext = os.path.splitext(output_path)[1].lower()
            
            # Get ODS template path (Plot_3D only supports .ods format)
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_dir = os.path.join(current_dir, 'data', 'templates', 'plot3d')
            template_path = os.path.join(template_dir, 'Plot3D_Rigid_Template.ods')
            
            # Check if template exists
            print(f"DEBUG: Looking for template at: {template_path}")
            if not os.path.exists(template_path):
                print(f"DEBUG: Template not found: {template_path}")
                return False
            print(f"DEBUG: Template found successfully")
            
            # Check if output file exists and handle permissions
            if os.path.exists(output_path):
                try:
                    # Try to make the file writable if it's read-only
                    import stat
                    file_stat = os.stat(output_path)
                    if not (file_stat.st_mode & stat.S_IWRITE):
                        os.chmod(output_path, file_stat.st_mode | stat.S_IWRITE)
                        logger.info(f"Made existing file writable: {output_path}")
                except Exception as perm_error:
                    logger.warning(f"Could not modify file permissions: {perm_error}")
                    raise PermissionError(f"Cannot write to existing file: {output_path}. Please close the file if it's open in another application, or choose a different filename.")
            
            # Copy template to output location
            shutil.copy2(template_path, output_path)
            logger.info(f"Copied template from {template_path} to {output_path}")
            
            # Process ODS template (Plot_3D only supports .ods format)
            try:
                    # For ODS rigid template, we need to preserve the structure
                    print(f"DEBUG: Processing ODS format for {output_path}")
                    try:
                        print(f"DEBUG: Importing ezodf")
                        import ezodf
                        
                        # Open the copied rigid template
                        doc = ezodf.opendoc(output_path)
                        sheet = doc.sheets[0]
                        
                        # Clear all existing data and rebuild with correct rigid format
                        # Headers in row 1, reserved area rows 2-7, data starts row 8
                        data_start_row = 7  # Row 8 in 1-based, 7 in 0-based
                        
                        # Clear entire sheet and rebuild with correct rigid format
                        print(f"DEBUG: Clearing entire sheet and rebuilding rigid format")
                        
                        # Clear all existing content
                        for row_idx in range(sheet.nrows()):
                            for col_idx in range(sheet.ncols()):
                                sheet[row_idx, col_idx].set_value('')
                        
                        # Set headers in row 1 (0-based index 0) - ensure M1 has "Radius"
                        print(f"DEBUG: Setting headers in row 1")
                        for col_idx, column_name in enumerate(self.PLOT3D_COLUMNS):
                            if col_idx < sheet.ncols():  # Make sure we don't exceed sheet width
                                sheet[0, col_idx].set_value(column_name)
                        print(f"DEBUG: Headers set, including Radius in column M")
                        
                        # Create column mapping for data writing
                        coord_columns = {col: idx for idx, col in enumerate(self.PLOT3D_COLUMNS) if idx < sheet.ncols()}
                        
                        print(f"DEBUG: Created column mapping: {coord_columns}")
                        print(f"DEBUG: A2:H7 restricted area will remain blank (no imported data)")
                        print(f"DEBUG: I2:K7 available for K-means centroids, L2:M7 for Sphere/Radius input")
                        
                        # Ensure sheet has enough rows (minimum 107 rows for Plot_3D)
                        min_rows = max(107, data_start_row + len(df))
                        current_sheet_rows = sheet.nrows()
                        print(f"DEBUG: Sheet has {current_sheet_rows} rows, need {min_rows} rows")
                        
                        if current_sheet_rows < min_rows:
                            rows_to_add = min_rows - current_sheet_rows
                            print(f"DEBUG: Adding {rows_to_add} empty rows to sheet")
                            # Add empty rows using ezodf
                            for _ in range(rows_to_add):
                                sheet.append_rows(1)
                            print(f"DEBUG: Sheet now has {sheet.nrows()} rows")
                        
                        # Rows 2-8 are already blank (reserved for Plot_3D)
                        # No need to clear since we cleared the entire sheet above
                        
                        # Write our data starting from row 8 (0-based index 7)
                        # A2:H7 remains blank (restricted), I2:M7 available for Plot_3D functions
                        current_row = data_start_row  # Should be 7 (row 8 in 1-based)
                        print(f"DEBUG: data_start_row = {data_start_row}")
                        print(f"DEBUG: Writing {len(df)} rows of data starting from row {current_row + 1} (1-based)")
                        print(f"DEBUG: First data will go to 0-based row {current_row}, which is 1-based row {current_row + 1}")
                        
                        for row_idx, (_, row_data) in enumerate(df.iterrows()):
                            actual_sheet_row = current_row + row_idx  # This should start at 7 (row 8 in 1-based)
                            print(f"DEBUG: Data row {row_idx + 1} -> 0-based sheet row {actual_sheet_row} (1-based row {actual_sheet_row + 1})")
                            for column_name, col_idx in coord_columns.items():
                                value = row_data.get(column_name, '')
                                if pd.isna(value):
                                    value = ''
                                try:
                                    if isinstance(value, (int, float)):
                                        sheet[actual_sheet_row, col_idx].set_value(float(value))
                                    else:
                                        sheet[actual_sheet_row, col_idx].set_value(str(value))
                                except IndexError as idx_error:
                                    print(f"DEBUG: IndexError at row {actual_sheet_row}, col {col_idx}: {idx_error}")
                                    print(f"DEBUG: Sheet dimensions: {sheet.nrows()} x {sheet.ncols()}")
                                    print(f"DEBUG: Trying to access row {actual_sheet_row}, col {col_idx}")
                                    raise
                        
                        # Add comprehensive validation reference for ODS format
                        print(f"DEBUG: Adding validation reference data for columns G (Marker), H (Color), L (Sphere)")
                        try:
                            # Since ODS doesn't support dropdown validation, we'll create:
                            # 1. Validation reference lists in columns O, P, Q (clearly labeled)
                            # 2. Instructions in specific cells
                            # 3. Sample valid values in the reserved area
                            
                            validation_col_start = 14  # Column O (0-based)
                            
                            # Add clear headers for validation columns
                            sheet[0, validation_col_start].set_value('Valid_Markers')     # Column O
                            sheet[0, validation_col_start + 1].set_value('Valid_Colors')  # Column P  
                            sheet[0, validation_col_start + 2].set_value('Valid_Spheres') # Column Q
                            
                            # Add validation lists with clear organization
                            max_items = max(len(self.VALID_MARKERS), len(self.VALID_COLORS), len(self.VALID_SPHERES))
                            
                            for i in range(max_items):
                                # Marker validation list (column O)
                                if i < len(self.VALID_MARKERS):
                                    sheet[i + 1, validation_col_start].set_value(self.VALID_MARKERS[i])
                                
                                # Color validation list (column P)
                                if i < len(self.VALID_COLORS):
                                    sheet[i + 1, validation_col_start + 1].set_value(self.VALID_COLORS[i])
                                
                                # Sphere validation list (column Q)
                                if i < len(self.VALID_SPHERES):
                                    sheet[i + 1, validation_col_start + 2].set_value(self.VALID_SPHERES[i])
                            
                            # IMPORTANT: Don't put validation instructions in main data columns A-M
                            # because pandas will interpret them as data rows!
                            # Keep A2:M7 completely empty for pandas compatibility
                            
                            # Add validation instructions in far-right columns only
                            # This won't interfere with pandas reading the main data
                            sheet[1, validation_col_start + 3].set_value('VALIDATION HELP:')  # Column R
                            sheet[2, validation_col_start + 3].set_value('G = Markers: . o * ^ < > v s D + x')
                            sheet[3, validation_col_start + 3].set_value('H = Colors: red blue green orange...')
                            sheet[4, validation_col_start + 3].set_value('L = Spheres: red green blue yellow...')
                            
                            print(f"DEBUG: Added comprehensive validation reference data")
                            print(f"DEBUG: Validation lists in columns O, P, Q")
                            print(f"DEBUG: Instructions in row 2: G2, H2, L2")
                            print(f"DEBUG: Sample values in row 3: G3, H3, L3")
                            print(f"DEBUG: Markers ({len(self.VALID_MARKERS)}): {self.VALID_MARKERS}")
                            print(f"DEBUG: Colors ({len(self.VALID_COLORS)}): {self.VALID_COLORS}")
                            print(f"DEBUG: Spheres ({len(self.VALID_SPHERES)}): {self.VALID_SPHERES}")
                            
                        except Exception as validation_error:
                            print(f"DEBUG: Validation setup error: {validation_error}")
                            import traceback
                            traceback.print_exc()
                            # Don't fail the export if validation setup fails
                        
                        # Save the document
                        doc.save()
                        print(f"DEBUG: Successfully saved ODS file with rigid format")
                        logger.info(f"Successfully exported {len(df)} rows to ODS rigid template")
                        
                    except Exception as ods_error:
                        logger.warning(f"ODS rigid template processing failed: {ods_error}")
                        import traceback
                        logger.warning(f"Full error: {traceback.format_exc()}")
                        return False
                    
                    return True
                    
            except Exception as template_error:
                logger.error(f"Error processing template: {template_error}")
                return False
                
        except Exception as e:
            logger.error(f"Error in template export: {e}")
            return False
    
    
    
    def _populate_new_worksheet_with_data(self, worksheet, import_result):
        """Populate a new worksheet with imported data.
        
        Args:
            worksheet: RealTimePlot3DSheet instance to populate
            import_result: ImportResult with data to populate
        """
        try:
            # Calculate rows needed
            imported_data_rows = len(import_result.data) if import_result.data else 0
            min_rows = 7 + imported_data_rows + 10  # 7 reserved rows + data + 10 buffer
            
            # Clear the new sheet and set up structure
            current_rows = worksheet.sheet.get_total_rows()
            if current_rows > 0:
                worksheet.sheet.delete_rows(0, current_rows)
            
            # Create structure
            empty_rows = [[''] * len(self.PLOT3D_COLUMNS)] * min_rows
            worksheet.sheet.insert_rows(rows=empty_rows, idx=0)
            
            # Insert centroid data first (rows 1-6)
            if import_result.centroid_data:
                for cluster_id, centroid_row in import_result.centroid_data:
                    if 0 <= cluster_id <= 5:  # Valid centroid area
                        centroid_row_idx = 1 + cluster_id  # Rows 1-6 for clusters 0-5
                        worksheet.sheet.set_row_data(centroid_row_idx, values=centroid_row)
                        logger.info(f"Populated centroid for cluster {cluster_id} in new worksheet")
            
            # Insert imported data starting at row 7 (data area)
            if import_result.data:
                for i, row in enumerate(import_result.data):
                    worksheet.sheet.set_row_data(7 + i, values=row)
            
            # Apply formatting to the new worksheet
            worksheet._apply_formatting()
            worksheet._setup_validation()
            
            logger.info(f"Successfully populated new worksheet with {imported_data_rows} rows")
            
        except Exception as e:
            logger.error(f"Error populating new worksheet: {e}")
            raise
    
    def _import_external_data(self, importer, file_path):
        """Helper method to import external data using the importer.
        
        Args:
            importer: ExternalDataImporter instance
            file_path: Path to the file to import
            
        Returns:
            ImportResult object
        """
        try:
            # Import the file
            result = importer.import_file(file_path)
            
            # Show warnings/errors if any
            if result.warnings or result.errors:
                warning_text = ""
                if result.warnings:
                    warning_text += "Warnings:\n" + "\n".join([f"• {w}" for w in result.warnings[:10]])  # Limit to first 10
                    if len(result.warnings) > 10:
                        warning_text += f"\n... and {len(result.warnings) - 10} more warnings"
                
                if result.errors:
                    if warning_text:
                        warning_text += "\n\n"
                    warning_text += "Errors:\n" + "\n".join([f"• {e}" for e in result.errors[:5]])  # Limit to first 5
                
                if result.errors:
                    messagebox.showerror("Import Errors", warning_text)
                    return result
                elif result.warnings:
                    messagebox.showwarning("Import Warnings", warning_text)
            
            return result
            
        except Exception as e:
            logger.error(f"Error importing external data: {e}")
            messagebox.showerror("Import Error", f"Failed to import external data: {e}")
            from utils.external_data_importer import ImportResult
            return ImportResult(success=False, errors=[str(e)])
    
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
        """Save current spreadsheet changes to internal database (manual save).
        
        This is the explicit "Save Changes to DB" button that saves ALL Plot_3D data
        (clusters, ∆E, centroids, spheres, etc.) to the internal StampZ database.
        """
        print("\n💾 MANUAL SAVE TO DATABASE TRIGGERED - DEV VERSION 2025-01-13")
        try:
            # Always save to internal database first (this is the primary action)
            self._save_to_internal_database()
            
            # Also save to external file if one exists (for export compatibility)
            file_saved = False
            if self.current_file_path:
                success = self._save_data_to_file(self.current_file_path)
                if success:
                    file_saved = True
                    # Trigger Plot_3D refresh if connected
                    self._notify_plot3d_refresh()
            
            # Show comprehensive success message
            message_parts = [
                "✅ All spreadsheet changes have been saved to the StampZ database!",
                "",
                "Saved data includes:",
                "• K-means cluster assignments",
                "• ΔE calculation results", 
                "• Cluster centroid coordinates",
                "• Sphere colors and radius values",
                "• Marker and color preferences",
                "• Manual edits to any Plot_3D columns",
                "",
                "Your changes are now permanently stored and will persist when you:",
                "• Click 'Refresh from StampZ'",
                "• Restart the application",
                "• Open this sample set again"
            ]
            
            if file_saved:
                message_parts.extend([
                    "",
                    f"✅ Also saved to external file: {os.path.basename(self.current_file_path)}",
                    "Plot_3D will use the updated data on next refresh."
                ])
            elif self.current_file_path:
                message_parts.extend([
                    "",
                    "⚠️ External file save failed, but database save was successful."
                ])
                
            messagebox.showinfo(
                "Database Save Complete",
                "\n".join(message_parts)
            )
            
            print("✅ Manual database save completed successfully")
                
        except Exception as e:
            logger.error(f"Error saving changes to database: {e}")
            messagebox.showerror(
                "Database Save Error", 
                f"Failed to save changes to database:\n\n{e}\n\n"
                f"This means your manual edits may be lost when you refresh or restart.\n"
                f"Please try again or check the terminal for detailed error messages."
            )
            print(f"❌ Manual database save failed: {e}")
    
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
