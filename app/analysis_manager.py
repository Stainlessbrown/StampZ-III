"""
Analysis Manager for StampZ Application

Handles all color analysis operations including spectral analysis, 
color library operations, and data export functionality.
"""

import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import logging
import pandas as pd
from datetime import datetime
from typing import TYPE_CHECKING
from PIL import Image
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from .stampz_app import StampZApp

logger = logging.getLogger(__name__)


class AnalysisManager:
    """Manages color analysis operations for the StampZ application."""
    
    def __init__(self, app: 'StampZApp'):
        self.app = app
        self.root = app.root
        
        # Initialize specialized managers
        self._init_managers()
        
    def _init_managers(self):
        """Initialize specialized manager classes for better code organization."""
        try:
            # Data Export Manager
            from managers.data_export_manager import DataExportManager
            self.data_export_manager = DataExportManager(self.app)
        except ImportError:
            logger.warning("DataExportManager not available, using legacy methods")
            self.data_export_manager = None
            
        try:
            # Black Ink Manager
            from managers.black_ink_manager import BlackInkManager
            self.black_ink_manager = BlackInkManager(self.app)
        except ImportError:
            logger.warning("BlackInkManager not available, using legacy methods")
            self.black_ink_manager = None
            
        try:
            # Database Manager
            from managers.database_manager import DatabaseManager
            self.database_manager = DatabaseManager(self.app, self.root)
        except ImportError:
            logger.warning("DatabaseManager not available, using legacy methods")
            self.database_manager = None
            
        try:
            # Measurement Manager (for perforation measurement, etc.)
            from managers.measurement_manager import MeasurementManager
            self.measurement_manager = MeasurementManager(self.app)
        except ImportError:
            logger.warning("MeasurementManager not available, using direct methods")
            self.measurement_manager = None
        
    def analyze_colors(self):
        """Analyze colors from sample markers on the canvas."""
        if not hasattr(self.app.canvas, '_coord_markers') or not self.app.canvas._coord_markers:
            messagebox.showwarning(
                "No Samples", 
                "No sample points found. Please place some sample markers using the Sample tool first."
            )
            return
        
        if not self.app.canvas.original_image:
            messagebox.showwarning(
                "No Image", 
                "Please open an image before analyzing colors."
            )
            return
        
        sample_set_name = self.app.control_panel.sample_set_name.get().strip()
        if not sample_set_name:
            messagebox.showwarning(
                "No Sample Set Name", 
                "Please enter a sample set name in the Template field before analyzing."
            )
            return
        
        try:
            from utils.color_analyzer import ColorAnalyzer
            # Create analyzer
            analyzer = ColorAnalyzer()
            
            if not self.app.current_file:
                messagebox.showerror("Error", "No image loaded. Please open an image first.")
                return
            
            actual_sample_set = sample_set_name
            if '_' in sample_set_name:
                parts = sample_set_name.split('_')
                if len(parts) >= 2:
                    potential_sample_set = '_'.join(parts[1:])
                    
                    try:
                        from utils.coordinate_db import CoordinateDB
                        coord_db = CoordinateDB()
                        available_sets = coord_db.get_all_set_names()
                        
                        if potential_sample_set in available_sets:
                            actual_sample_set = potential_sample_set
                    except:
                        pass
            
            print(f"DEBUG: About to call analyze_image_colors_from_canvas with:")
            print(f"  - image_path: {self.app.current_file}")
            print(f"  - sample_set_name: {actual_sample_set}")
            print(f"  - number of markers: {len(self.app.canvas._coord_markers)}")
            
            measurements = analyzer.analyze_image_colors_from_canvas(
                self.app.current_file, actual_sample_set, self.app.canvas._coord_markers
            )
            
            print(f"DEBUG: analyze_image_colors_from_canvas returned: {measurements is not None}")
            if measurements:
                print(f"DEBUG: Number of measurements: {len(measurements)}")
            
            if measurements:
                self._show_analysis_complete_dialog(measurements, actual_sample_set)
            else:
                messagebox.showwarning(
                    "Analysis Failed", 
                    "No color samples could be analyzed. Please check your sample markers."
                )
                
        except Exception as e:
            import traceback
            messagebox.showerror(
                "Analysis Error", 
                f"Failed to analyze color samples:\\n\\n{str(e)}"
            )
    
    def _show_analysis_complete_dialog(self, measurements, sample_set_name):
        """Show simple analysis complete dialog without export options."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Color Analysis Complete")
        
        dialog_width = 480
        dialog_height = 280
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        x = screen_width - dialog_width - 50
        y = (screen_height - dialog_height) // 2
        
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        
        # Header
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ttk.Label(
            header_frame,
            text="✅ Color Analysis Complete",
            font=("Arial", 14, "bold")
        ).pack()
        
        # Results summary
        summary_frame = ttk.Frame(dialog)
        summary_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        summary_text = (
            f"Successfully analyzed {len(measurements)} color samples from set '{sample_set_name}'.\n"
            f"Color data has been saved to the database.\n\n"
            f"Individual measurements: {len(measurements)} samples\n\n"
            f"Next steps:\n"
            f"• Use the Compare tab to review and compare colors\n"
            f"• Export data using the Compare window's export button\n"
            f"• View spreadsheet or export for Plot3D analysis"
        )
        
        ttk.Label(
            summary_frame,
            text=summary_text,
            wraplength=440,
            justify="left",
            font=("Arial", 10)
        ).pack()
        
        # Action buttons frame
        action_frame = ttk.Frame(dialog)
        action_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # View spreadsheet button
        ttk.Button(
            action_frame,
            text="📊 View Spreadsheet",
            command=lambda: [dialog.destroy(), self.view_spreadsheet()]
        ).pack(side="left", padx=(0, 10))
        
        # Export for Plot3D button
        ttk.Button(
            action_frame,
            text="📈 Export for Plot3D",
            command=lambda: [dialog.destroy(), self.export_plot3d_flexible()]
        ).pack(side="left", padx=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ttk.Button(
            button_frame,
            text="Close",
            command=dialog.destroy
        ).pack(side="right")
        
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center on screen
        dialog.update_idletasks()
        dialog.focus_force()
        
        self.root.wait_window(dialog)
    
    def _export_individual_to_logger(self, measurements, sample_set_name, dialog):
        """Export individual measurements to unified data logger."""
        try:
            if not self.app.current_file:
                messagebox.showerror("Error", "No current image file available.")
                return
                
            from utils.unified_data_logger import UnifiedDataLogger
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Get measurements data from database filtered by current image
            db = ColorAnalysisDB(sample_set_name)
            current_image_name = os.path.basename(self.app.current_file)
            
            # Debug: Check what's in the database vs what we're looking for
            all_measurements = db.get_all_measurements()
            if all_measurements:
                unique_image_names = set(m['image_name'] for m in all_measurements)
                print(f"DEBUG: Looking for image: '{current_image_name}'")
                print(f"DEBUG: Available images in database: {list(unique_image_names)}")
                
                # Try to find a match using the pattern extraction logic
                from utils.color_analyzer import ColorAnalyzer
                analyzer = ColorAnalyzer()
                sample_identifier = analyzer._extract_sample_identifier_from_filename(self.app.current_file)
                print(f"DEBUG: Sample identifier from filename: '{sample_identifier}'")
                
                # Try both full filename and extracted identifier
                db_measurements = db.get_measurements_for_image(current_image_name)
                if not db_measurements:
                    print(f"DEBUG: No measurements found for full filename, trying sample identifier...")
                    db_measurements = db.get_measurements_for_image(sample_identifier)
            else:
                print(f"DEBUG: No measurements found in database at all")
                db_measurements = []
            
            # Create data logger for current image
            logger = UnifiedDataLogger(self.app.current_file)
            
            # Log individual measurements
            log_file = logger.log_individual_color_measurements(
                db_measurements, sample_set_name, self.app.current_file
            )
            
            if log_file:
                messagebox.showinfo(
                    "Export Complete",
                    f"Individual color measurements exported to unified data logger.\n\n"
                    f"File: {log_file.name}\n"
                    f"Measurements: {len(db_measurements)} samples"
                )
                dialog.destroy()
            else:
                messagebox.showerror("Export Failed", "Failed to export individual measurements to data logger.")
                
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export individual measurements:\n\n{str(e)}"
            )
    
    def _export_averaged_to_logger(self, measurements, sample_set_name, dialog):
        """Export averaged measurement to unified data logger."""
        try:
            if not self.app.current_file:
                messagebox.showerror("Error", "No current image file available.")
                return
                
            from utils.unified_data_logger import UnifiedDataLogger
            from utils.color_analyzer import ColorAnalyzer
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Calculate averaged measurement from individual measurements
            analyzer = ColorAnalyzer()
            
            # Get individual measurements from database filtered by current image
            db = ColorAnalysisDB(sample_set_name)
            current_image_name = os.path.basename(self.app.current_file)
            
            # Debug: Check what's in the database vs what we're looking for
            all_measurements = db.get_all_measurements()
            if all_measurements:
                unique_image_names = set(m['image_name'] for m in all_measurements)
                print(f"DEBUG: Looking for image: '{current_image_name}'")
                print(f"DEBUG: Available images in database: {list(unique_image_names)}")
                
                # Try to find a match using the pattern extraction logic
                from utils.color_analyzer import ColorAnalyzer
                analyzer = ColorAnalyzer()
                sample_identifier = analyzer._extract_sample_identifier_from_filename(self.app.current_file)
                print(f"DEBUG: Sample identifier from filename: '{sample_identifier}'")
                
                # Try both full filename and extracted identifier
                db_measurements = db.get_measurements_for_image(current_image_name)
                if not db_measurements:
                    print(f"DEBUG: No measurements found for full filename, trying sample identifier...")
                    db_measurements = db.get_measurements_for_image(sample_identifier)
            else:
                print(f"DEBUG: No measurements found in database at all")
                db_measurements = []
            
            if not db_measurements:
                messagebox.showerror("Error", "No measurements found in database.")
                return
            
            # Calculate quality-controlled average
            lab_values = [(m['l_value'], m['a_value'], m['b_value']) for m in db_measurements]
            rgb_values = [(m['rgb_r'], m['rgb_g'], m['rgb_b']) for m in db_measurements]
            
            averaging_result = analyzer._calculate_quality_controlled_average(lab_values, rgb_values)
            
            # Create averaged data dictionary
            averaged_data = {
                'l_value': averaging_result['avg_lab'][0],
                'a_value': averaging_result['avg_lab'][1], 
                'b_value': averaging_result['avg_lab'][2],
                'rgb_r': averaging_result['avg_rgb'][0],
                'rgb_g': averaging_result['avg_rgb'][1],
                'rgb_b': averaging_result['avg_rgb'][2],
                'notes': f"ΔE max: {averaging_result['max_delta_e']:.2f}, "
                        f"used {averaging_result['samples_used']}/{len(db_measurements)} samples"
            }
            
            # Create data logger for current image
            logger = UnifiedDataLogger(self.app.current_file)
            
            # Log averaged measurement
            log_file = logger.log_averaged_color_measurement(
                averaged_data, sample_set_name, self.app.current_file, len(db_measurements)
            )
            
            if log_file:
                messagebox.showinfo(
                    "Export Complete",
                    f"Averaged color measurement exported to unified data logger.\n\n"
                    f"File: {log_file.name}\n"
                    f"Averaged from: {len(db_measurements)} individual samples\n"
                    f"Quality: ΔE max = {averaging_result['max_delta_e']:.2f}"
                )
                dialog.destroy()
            else:
                messagebox.showerror("Export Failed", "Failed to export averaged measurement to data logger.")
                
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export averaged measurement:\\n\\n{str(e)}"
            )
    
    def _export_both_to_logger(self, measurements, sample_set_name, dialog):
        """Export both individual and averaged measurements to unified data logger."""
        try:
            if not self.app.current_file:
                messagebox.showerror("Error", "No current image file available.")
                return
                
            from utils.unified_data_logger import UnifiedDataLogger
            from utils.color_analyzer import ColorAnalyzer
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Get measurements from database filtered by current image
            db = ColorAnalysisDB(sample_set_name)
            current_image_name = os.path.basename(self.app.current_file)
            
            # Debug: Check what's in the database vs what we're looking for
            all_measurements = db.get_all_measurements()
            if all_measurements:
                unique_image_names = set(m['image_name'] for m in all_measurements)
                print(f"DEBUG: Looking for image: '{current_image_name}'")
                print(f"DEBUG: Available images in database: {list(unique_image_names)}")
                
                # Try to find a match using the pattern extraction logic
                from utils.color_analyzer import ColorAnalyzer
                analyzer = ColorAnalyzer()
                sample_identifier = analyzer._extract_sample_identifier_from_filename(self.app.current_file)
                print(f"DEBUG: Sample identifier from filename: '{sample_identifier}'")
                
                # Try both full filename and extracted identifier
                db_measurements = db.get_measurements_for_image(current_image_name)
                if not db_measurements:
                    print(f"DEBUG: No measurements found for full filename, trying sample identifier...")
                    db_measurements = db.get_measurements_for_image(sample_identifier)
            else:
                print(f"DEBUG: No measurements found in database at all")
                db_measurements = []
            
            if not db_measurements:
                messagebox.showerror("Error", "No measurements found in database.")
                return
            
            # Create data logger for current image
            logger = UnifiedDataLogger(self.app.current_file)
            
            # First export individual measurements
            individual_log_file = logger.log_individual_color_measurements(
                db_measurements, sample_set_name, self.app.current_file
            )
            
            if not individual_log_file:
                messagebox.showerror("Export Failed", "Failed to export individual measurements to data logger.")
                return
            
            # Calculate and export averaged measurement
            analyzer = ColorAnalyzer()
            
            # Calculate quality-controlled average
            lab_values = [(m['l_value'], m['a_value'], m['b_value']) for m in db_measurements]
            rgb_values = [(m['rgb_r'], m['rgb_g'], m['rgb_b']) for m in db_measurements]
            
            averaging_result = analyzer._calculate_quality_controlled_average(lab_values, rgb_values)
            
            # Create averaged data dictionary
            averaged_data = {
                'l_value': averaging_result['avg_lab'][0],
                'a_value': averaging_result['avg_lab'][1], 
                'b_value': averaging_result['avg_lab'][2],
                'rgb_r': averaging_result['avg_rgb'][0],
                'rgb_g': averaging_result['avg_rgb'][1],
                'rgb_b': averaging_result['avg_rgb'][2],
                'notes': f"ΔE max: {averaging_result['max_delta_e']:.2f}, "
                        f"used {averaging_result['samples_used']}/{len(db_measurements)} samples"
            }
            
            # Export averaged measurement
            averaged_log_file = logger.log_averaged_color_measurement(
                averaged_data, sample_set_name, self.app.current_file, len(db_measurements)
            )
            
            if averaged_log_file:
                messagebox.showinfo(
                    "Export Complete",
                    f"BOTH individual and averaged measurements exported to unified data logger!\n\n"
                    f"Sample Set: {sample_set_name}\n"
                    f"File: {averaged_log_file.name}\n\n"
                    f"INDIVIDUAL DATA:\n"
                    f"  • {len(db_measurements)} sample measurements\n"
                    f"  • Complete L*a*b* and RGB values\n"
                    f"  • Position and sample details\n\n"
                    f"AVERAGED DATA:\n"
                    f"  • Quality-controlled average\n"
                    f"  • ΔE max = {averaging_result['max_delta_e']:.2f}\n"
                    f"  • Used {averaging_result['samples_used']}/{len(db_measurements)} samples\n\n"
                    f"Your comprehensive analysis log now contains both detailed individual \n"
                    f"measurements AND high-quality averaged data for complete documentation."
                )
                dialog.destroy()
            else:
                messagebox.showerror("Export Failed", "Failed to export averaged measurement to data logger.")
                
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export both measurements:\\n\\n{str(e)}"
            )
    
    def export_color_data(self):
        """Export color analysis data - delegate to DataExportManager."""
        if self.data_export_manager:
            return self.data_export_manager.export_color_data()
        else:
            return self._legacy_export_color_data()
            
    def create_plot3d_worksheet(self):
        """Create a formatted Excel worksheet for Plot_3D integration."""
        if self.data_export_manager:
            return self.data_export_manager.create_plot3d_worksheet()
        else:
            return self._legacy_create_plot3d_worksheet()
            
    def export_plot3d_flexible(self):
        """Export data in Plot_3D flexible format."""
        if self.data_export_manager:
            return self.data_export_manager.export_plot3d_flexible()
        else:
            return self._legacy_export_plot3d_flexible()
            
    def export_with_library_matches(self):
        """Export analysis data with library matches."""
        if self.data_export_manager:
            return self.data_export_manager.export_with_library_matches()
        else:
            return self._legacy_export_with_library_matches()
            
    def open_plot3d_data_manager(self):
        """Open unified Plot_3D Data Manager."""
        if self.data_export_manager:
            # Could delegate specific Plot_3D management to data export manager
            return self._legacy_open_plot3d_data_manager()
        else:
            return self._legacy_open_plot3d_data_manager()
            
    # Legacy methods (fallbacks)
    def _legacy_create_plot3d_worksheet(self):
        """Legacy implementation of Plot_3D worksheet creation."""
        try:
            from utils.worksheet_manager import WorksheetManager
            
            # Get current sample set name
            sample_set_name = "StampZ_Analysis"  # Default
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                sample_set_name = self.app.control_panel.sample_set_name.get().strip()
            
            # Get save location
            default_filename = f"{sample_set_name}_Plot3D_{datetime.now().strftime('%Y%m%d')}.ods"
            
            # macOS-friendly file dialog approach
            initial_filename = sample_set_name + "_Plot3D_" + datetime.now().strftime('%Y%m%d')
            
            # Try primary dialog first
            try:
                filepath = filedialog.asksaveasfilename(
                    title="Create Plot_3D Worksheet",
                    defaultextension=".ods",
                    filetypes=[
                        ('OpenDocument Spreadsheet', '*.ods'),
                        ('All files', '*.*')
                    ],
                    initialfile=initial_filename,
                    initialdir=os.path.expanduser("~/Desktop")
                )
            except Exception as dialog_error:
                # Fallback: simpler dialog
                logger.warning(f"Primary dialog failed: {dialog_error}")
                filepath = filedialog.asksaveasfilename(
                    title="Save Plot_3D Worksheet",
                    filetypes=[('OpenDocument Spreadsheet', '*.ods'), ('All files', '*.*')]
                )
            
            if filepath:
                # Create worksheet manager and template (ODS format only)
                manager = WorksheetManager()
                
                # Create ODS template using the formatted template file
                success = self._create_clean_template(filepath, sample_set_name)
                
                if success:
                    # Ask if user wants to populate with existing data
                    populate = messagebox.askyesno(
                        "Populate Data",
                        f"Template created successfully!\\n\\n"
                        f"Would you like to populate it with existing data from sample set '{sample_set_name}'?"
                    )
                    
                    if populate:
                        # Populate ODS template with existing data
                        self._populate_ods_template(filepath, sample_set_name)
                        messagebox.showinfo(
                            "Success",
                            f"Plot_3D template created and populated with data from '{sample_set_name}'.\\n\\n"
                            f"File saved: {os.path.basename(filepath)}\\n\\n"
                            f"Format: OpenDocument Spreadsheet (.ods) - Plot_3D compatible\\n"
                            f"Ready for 3D analysis in Plot_3D standalone mode."
                        )
                    else:
                        messagebox.showinfo(
                            "Template Created",
                            f"Plot_3D template created successfully.\\n\\n"
                            f"File saved: {os.path.basename(filepath)}\\n\\n"
                            f"Format: OpenDocument Spreadsheet (.ods) - Plot_3D compatible\\n"
                            f"Ready for data entry - columns match Plot_3D format."
                        )
                else:
                    messagebox.showerror(
                        "Creation Failed",
                        f"Failed to create Plot_3D worksheet.\\n\\nPlease check file permissions and try again."
                    )
                    
        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "The worksheet manager requires the 'openpyxl' library.\\n\\n"
                "Please install it using: pip install openpyxl"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to create Plot_3D worksheet:\\n\\n{str(e)}"
            )
    
    def import_external_plot3d_data(self):
        """Import external Plot_3D data from CSV files (for backup/legacy data)."""
        try:
            # File dialog for CSV import (consistent with library import/export)
            filepath = filedialog.askopenfilename(
                title="Import Plot_3D Data from CSV",
                filetypes=[
                    ('CSV files', '*.csv'),
                    ('All files', '*.*')
                ]
            )
            
            if filepath:
                from utils.worksheet_manager import WorksheetManager
                import pandas as pd
                
                # Load CSV data
                df = pd.read_csv(filepath)
                
                # Validate that it has Plot_3D structure
                required_cols = ['Xnorm', 'Ynorm', 'Znorm', 'DataID']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    messagebox.showerror(
                        "Invalid Format",
                        f"CSV file is missing required Plot_3D columns:\\n\\n"
                        f"Missing: {', '.join(missing_cols)}\\n\\n"
                        f"Please ensure the CSV has the correct Plot_3D format."
                    )
                    return
                
                # Ask for sample set name for integration
                sample_set_name = simpledialog.askstring(
                    "Sample Set Name",
                    "Enter a name for this imported data set:",
                    initialvalue=os.path.splitext(os.path.basename(filepath))[0]
                )
                
                if not sample_set_name:
                    return
                
                # Create Excel worksheet from the CSV data
                output_file = filepath.replace('.csv', '_formatted.xlsx')
                
                manager = WorksheetManager()
                success = manager.create_plot3d_worksheet(output_file, sample_set_name)
                
                if success:
                    # Populate with imported data (manual population since it's external)
                    self._populate_worksheet_from_csv(manager, df, sample_set_name)
                    manager.save_worksheet(output_file)
                    
                    messagebox.showinfo(
                        "Import Successful",
                        f"External Plot_3D data imported and formatted.\\n\\n"
                        f"Original: {os.path.basename(filepath)}\\n"
                        f"Formatted: {os.path.basename(output_file)}\\n\\n"
                        f"The formatted file includes proper validation and formatting."
                    )
                else:
                    messagebox.showerror(
                        "Import Failed",
                        "Failed to create formatted worksheet from CSV data."
                    )
                    
        except Exception as e:
            messagebox.showerror(
                "Import Error",
                f"Failed to import external Plot_3D data:\\n\\n{str(e)}"
            )
    
    def _populate_worksheet_from_csv(self, manager: 'WorksheetManager', df: pd.DataFrame, sample_set_name: str):
        """Populate worksheet from CSV DataFrame."""
        try:
            # Start data at row 8 (after protected area)
            start_row = 8
            
            for i, (_, row) in enumerate(df.iterrows()):
                worksheet_row = start_row + i
                
                # Core coordinate data
                manager.worksheet.cell(row=worksheet_row, column=1).value = row.get('Xnorm', '')
                manager.worksheet.cell(row=worksheet_row, column=2).value = row.get('Ynorm', '')
                manager.worksheet.cell(row=worksheet_row, column=3).value = row.get('Znorm', '')
                manager.worksheet.cell(row=worksheet_row, column=4).value = row.get('DataID', f"{sample_set_name}_Sample_{i+1:03d}")
                
                # Optional data (preserve if present)
                manager.worksheet.cell(row=worksheet_row, column=5).value = row.get('Cluster', '')
                manager.worksheet.cell(row=worksheet_row, column=6).value = row.get('∆E', '')
                manager.worksheet.cell(row=worksheet_row, column=7).value = row.get('Marker', '.')
                manager.worksheet.cell(row=worksheet_row, column=8).value = row.get('Color', 'blue')
                manager.worksheet.cell(row=worksheet_row, column=9).value = row.get('Centroid_X', '')
                manager.worksheet.cell(row=worksheet_row, column=10).value = row.get('Centroid_Y', '')
                manager.worksheet.cell(row=worksheet_row, column=11).value = row.get('Centroid_Z', '')
                manager.worksheet.cell(row=worksheet_row, column=12).value = row.get('Sphere', '')
                manager.worksheet.cell(row=worksheet_row, column=13).value = row.get('Radius', '')
                
        except Exception as e:
            logger.error(f"Error populating worksheet from CSV: {e}")
    
    def _populate_ods_template(self, file_path: str, sample_set_name: str):
        """Populate ODS template with StampZ data using rigid Plot_3D layout."""
        try:
            from utils.worksheet_manager import WorksheetManager
            
            # Create formatted ODS template and populate with data
            success = self._create_clean_template(file_path, sample_set_name)
            
            # Check if template was created successfully
            if success:
                # Now populate it with actual data using the rigid format
                self._populate_rigid_ods_with_data(file_path, sample_set_name)
            
        except Exception as e:
            logger.error(f"Error populating ODS template: {e}")
    
    def _populate_rigid_ods_with_data(self, file_path: str, sample_set_name: str):
        """Populate the rigid ODS template with actual measurement data."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            from utils.worksheet_manager import ODF_AVAILABLE
            
            if not ODF_AVAILABLE:
                logger.warning("ODF not available, cannot populate rigid ODS template")
                return
            
            from odf.opendocument import load
            from odf.table import Table, TableRow, TableCell
            from odf.text import P
            
            # Get measurements from database
            db = ColorAnalysisDB(sample_set_name)
            measurements = db.get_all_measurements()
            
            if not measurements:
                logger.warning(f"No measurements found for sample set: {sample_set_name}")
                return
            
            # Load existing ODS file
            doc = load(file_path)
            table = doc.spreadsheet.getElementsByType(Table)[0]
            
            # Remove existing example rows (rows 9-11) if any
            rows = table.getElementsByType(TableRow)
            while len(rows) > 8:  # Keep only metadata + header rows
                table.removeChild(rows[-1])
                rows = table.getElementsByType(TableRow)
            
            # Add data rows starting at row 9
            plot3d_columns = [
                'Xnorm', 'Ynorm', 'Znorm', 'DataID', 'Cluster', 
                '∆E', 'Marker', 'Color', 'Centroid_X', 'Centroid_Y', 
                'Centroid_Z', 'Sphere', 'Radius'
            ]
            
            for i, measurement in enumerate(measurements):
                tr = TableRow()
                
                # CRITICAL FIX: Normalize raw L*a*b* values from database for Plot_3D
                l_val = measurement.get('l_value', 0.0)
                a_val = measurement.get('a_value', 0.0) 
                b_val = measurement.get('b_value', 0.0)
                
                # Apply proper normalization:
                # L*: 0-100 → 0-1
                # a*: -128 to +127 → 0-1
                # b*: -128 to +127 → 0-1
                x_norm = max(0.0, min(1.0, (l_val if l_val is not None else 0.0) / 100.0))
                y_norm = max(0.0, min(1.0, ((a_val if a_val is not None else 0.0) + 128.0) / 255.0))
                z_norm = max(0.0, min(1.0, ((b_val if b_val is not None else 0.0) + 128.0) / 255.0))
                
                row_data = {
                    'Xnorm': round(x_norm, 4),  # Normalized L* value
                    'Ynorm': round(y_norm, 4),  # Normalized a* value
                    'Znorm': round(z_norm, 4),  # Normalized b* value
                    'DataID': f"{sample_set_name}_Sample_{i+1:03d}",
                    'Cluster': '',
                    '∆E': '',
                    'Marker': measurement.get('marker_preference', '.') or '.',
                    'Color': measurement.get('color_preference', 'blue') or 'blue',
                    'Centroid_X': '',
                    'Centroid_Y': '',
                    'Centroid_Z': '',
                    'Sphere': '',
                    'Radius': ''
                }
                
                for col in plot3d_columns:
                    tc = TableCell()
                    tc.addElement(P(text=str(row_data.get(col, ''))))
                    tr.addElement(tc)
                table.addElement(tr)
            
            # Save the updated document
            doc.save(file_path)
            logger.info(f"Populated rigid ODS with {len(measurements)} measurements → {file_path}")
            
        except Exception as e:
            logger.error(f"Error populating rigid ODS with data: {e}")
    
    def open_plot3d_data_manager(self):
        """Open unified Plot_3D Data Manager with all data source options."""
        try:
            from tkinter import Toplevel, Radiobutton, StringVar, Frame, Label
            from tkinter import Listbox, Scrollbar
            from tkinter import ttk
            
            # Create dialog window
            dialog = Toplevel(self.root)
            dialog.title("Plot_3D Data Manager")
            dialog.geometry("500x600")
            dialog.resizable(False, False)
            
            # Center dialog
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Title
            Label(dialog, text="Plot_3D Data Manager", font=("Arial", 16, "bold")).pack(pady=10)
            Label(dialog, text="Choose your data source and action:", font=("Arial", 12)).pack(pady=5)
            
            # Data source selection
            source_var = StringVar(value="existing")
            
            source_frame = Frame(dialog)
            source_frame.pack(fill="x", padx=20, pady=10)
            
            Label(source_frame, text="Data Source:", font=("Arial", 12, "bold")).pack(anchor="w")
            
            Radiobutton(source_frame, text="New/Empty Template", variable=source_var, value="new").pack(anchor="w")
            Radiobutton(source_frame, text="Existing StampZ Database", variable=source_var, value="existing").pack(anchor="w")
            Radiobutton(source_frame, text="Load Existing File (.ods/.xlsx - load only)", variable=source_var, value="load_file").pack(anchor="w")
            Radiobutton(source_frame, text="Import External Data as New Worksheet", variable=source_var, value="import_new").pack(anchor="w")
            Radiobutton(source_frame, text="Import External CSV", variable=source_var, value="external").pack(anchor="w")
            
            # Sample set selection (for existing data)
            existing_frame = Frame(dialog)
            existing_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            Label(existing_frame, text="Available Sample Sets:", font=("Arial", 12, "bold")).pack(anchor="w")
            
            listbox_frame = Frame(existing_frame)
            listbox_frame.pack(fill="both", expand=True)
            
            sample_listbox = Listbox(listbox_frame, font=("Arial", 11))
            scrollbar = Scrollbar(listbox_frame, orient="vertical", command=sample_listbox.yview)
            sample_listbox.configure(yscrollcommand=scrollbar.set)
            
            sample_listbox.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Load available sample sets
            available_sets = self._get_available_sample_sets()
            for set_name in available_sets:
                sample_listbox.insert("end", set_name)
            
            if available_sets:
                sample_listbox.selection_set(0)  # Select first item
            
            # Action buttons
            action_frame = Frame(dialog)
            action_frame.pack(fill="x", padx=20, pady=20)
            
            def on_open_plot3d():
                self._handle_plot3d_action(dialog, "open_plot3d", source_var.get(), sample_listbox, available_sets)
            
            def on_cancel():
                dialog.destroy()
            
            # Use themed buttons for proper color handling
            open_button = ttk.Button(action_frame, text="Open in Plot_3D", command=on_open_plot3d, width=20)
            open_button.pack(side="left", padx=10)
            ttk.Button(action_frame, text="Cancel", command=on_cancel, width=10).pack(side="right", padx=5)
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open Plot_3D Data Manager:\\n\\n{str(e)}"
            )
    
    def _get_available_sample_sets(self):
        """Get list of available sample sets from color analysis database."""
        try:
            import sqlite3
            from utils.path_utils import get_color_analysis_dir
            
            analysis_dir = get_color_analysis_dir()
            db_files = [f for f in os.listdir(analysis_dir) if f.endswith('.db')]
            
            sample_sets = []
            for db_file in db_files:
                # Extract sample set name from filename (remove .db extension)
                set_name = os.path.splitext(db_file)[0]
                
                # Verify it has data
                db_path = os.path.join(analysis_dir, db_file)
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM measurements")
                    count = cursor.fetchone()[0]
                    conn.close()
                    
                    if count > 0:
                        sample_sets.append(f"{set_name} ({count} measurements)")
                except:
                    sample_sets.append(f"{set_name} (unknown count)")
            
            return sample_sets
            
        except Exception as e:
            logger.warning(f"Error getting sample sets: {e}")
            return []
    
    def _handle_plot3d_action(self, dialog, action, source_type, sample_listbox, available_sets):
        """Handle the selected Plot_3D action."""
        try:
            # Get selection BEFORE destroying dialog to avoid widget reference error
            selected_sample_set = None
            if source_type == "existing":
                selection = sample_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a sample set to use.")
                    return
                
                # Extract sample set name (remove the count info)
                selected_text = available_sets[selection[0]]
                selected_sample_set = selected_text.split(" (")[0]
            
            # NOW destroy the dialog after getting the data we need
            dialog.destroy()
            
            if source_type == "new":
                # Create new empty template and launch Plot_3D
                self._create_and_launch_new_template()
            elif source_type == "existing":
                # Use existing StampZ database - open real-time spreadsheet  
                self._open_realtime_spreadsheet(selected_sample_set)
            elif source_type == "load_file":
                # Load existing .ods/.xlsx file in Plot_3D
                self._load_existing_file_in_plot3d()
            elif source_type == "import_new":
                # Import external data as new worksheet
                self._import_external_as_new_worksheet()
            elif source_type == "external":
                # Import from external CSV and launch Plot_3D
                self._import_and_launch_csv()
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to perform Plot_3D action:\\n\\n{str(e)}"
            )
    
    def _import_external_as_new_worksheet(self):
        """Import external data as a completely new worksheet and database."""
        print("DEBUG: === STARTING IMPORT PROCESS ===")
        try:
            print("DEBUG: Step 1 - Importing required modules")
            from tkinter import filedialog, simpledialog
            from utils.external_data_importer import ExternalDataImporter
            print("DEBUG: Step 1 - Modules imported successfully")
            
            # Ask for file to import
            print("DEBUG: Step 2 - Opening file dialog")
            file_path = filedialog.askopenfilename(
                title="Import External Data as New Worksheet",
                filetypes=[
                    ('CSV Files', '*.csv'),
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('Excel Workbook', '*.xlsx'),
                    ('All files', '*.*')
                ]
            )
            print(f"DEBUG: Step 2 - File dialog returned: {file_path}")
            
            if not file_path:
                print("DEBUG: User cancelled file selection")
                return  # User cancelled
            
            # Ask user for new worksheet name
            print("DEBUG: Step 3 - Asking for worksheet name")
            new_worksheet_name = simpledialog.askstring(
                "New Worksheet Name",
                f"Enter name for the new worksheet:\n\n"
                f"This will create a separate worksheet and database.",
                initialvalue=f"ImportedData_{int(time.time())}"
            )
            print(f"DEBUG: Step 3 - Worksheet name: {new_worksheet_name}")
            
            # If user cancelled or no name provided, return
            if not new_worksheet_name or not new_worksheet_name.strip():
                print("DEBUG: User cancelled name entry or provided empty name")
                return
                
            new_worksheet_name = new_worksheet_name.strip()
            
            # Import the external data and save to database
            print("DEBUG: Step 4 - Starting real data import")
            try:
                importer = ExternalDataImporter()
                print("DEBUG: Step 4a - Created importer instance")
                
                result = self._import_external_data_helper(importer, file_path)
                print(f"DEBUG: Step 4b - Import completed, success: {result.success if result else 'None'}")
                
                if not result or not result.success:
                    print("DEBUG: Import failed or returned no result")
                    messagebox.showerror("Import Failed", "Failed to import external data. Check the file format.")
                    return
                
                # Save imported data directly to database
                print(f"DEBUG: Step 5 - Saving imported data to database: {new_worksheet_name}")
                saved_count = self._save_imported_data_to_database(new_worksheet_name, result)
                
                # Verify the database was created
                from utils.color_analysis_db import ColorAnalysisDB
                test_db = ColorAnalysisDB(new_worksheet_name)
                test_measurements = test_db.get_all_measurements()
                measurement_count = len(test_measurements) if test_measurements else 0
                print(f"DEBUG: Step 6 - Verification: database has {measurement_count} measurements")
                
                # Now create the worksheet window automatically
                print(f"DEBUG: Creating worksheet window for imported data: {new_worksheet_name}")
                try:
                    from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
                    print("DEBUG: Imported RealtimePlot3DSheet successfully")
                    
                    # Create worksheet WITHOUT initial data loading to avoid freezing
                    # User can manually refresh data once window is open
                    worksheet = RealtimePlot3DSheet(
                        parent=self.root,
                        sample_set_name=new_worksheet_name,
                        load_initial_data=False  # Prevent freezing during creation
                    )
                    print("DEBUG: Worksheet window created successfully")
                    
                    # Let worksheet window fully settle before showing dialog
                    print("DEBUG: Allowing worksheet window to settle...")
                    self.root.update()
                    self.root.update_idletasks()
                    
                    # Use a delayed messagebox to avoid modal conflicts on macOS
                    def show_success_message():
                        centroid_msg = f"\n• Imported {len(result.centroid_data)} K-means centroids" if result.centroid_data else ""
                        messagebox.showinfo(
                            "Import Successful",
                            f"✅ Successfully imported '{new_worksheet_name}'!\n\n"
                            f"• Imported {result.rows_imported} data rows{centroid_msg}\n"
                            f"• Saved {measurement_count} measurements to database\n"
                            f"• Worksheet window opened (empty to avoid freezing)\n\n"
                            f"📋 To load your data:\n"
                            f"Click the 'Refresh from StampZ' button in the worksheet\n\n"
                            f"This prevents UI freezing with large datasets!"
                        )
                    
                    # Show success message after a short delay to avoid modal conflicts
                    print("DEBUG: Scheduling success message...")
                    self.root.after(500, show_success_message)
                    
                except Exception as worksheet_error:
                    print(f"DEBUG: Failed to create worksheet window: {worksheet_error}")
                    import traceback
                    print(f"DEBUG: Worksheet creation error traceback: {traceback.format_exc()}")
                    
                    # Show fallback message if worksheet creation fails
                    messagebox.showinfo(
                        "Import Successful (Manual Open Required)",
                        f"✅ Data imported successfully as '{new_worksheet_name}'!\n\n"
                        f"• Imported {result.rows_imported} data rows\n"
                        f"• Saved {measurement_count} measurements to database\n\n"
                        f"📋 To view the data:\n"
                        f"1. Use Plot_3D Data Manager\n"
                        f"2. Select 'Existing StampZ Database'\n"
                        f"3. Choose '{new_worksheet_name}' from the list\n"
                        f"4. Click 'Open in Plot_3D'"
                    )
                
                print("DEBUG: === IMPORT PROCESS COMPLETED SUCCESSFULLY ===")
                logger.info(f"Successfully imported '{new_worksheet_name}' with {result.rows_imported} rows")
                return
                
            except Exception as import_error:
                print(f"DEBUG: Import process failed: {import_error}")
                messagebox.showerror("Import Failed", f"Import process failed: {import_error}")
                return
            
            # Create database directly without problematic UI (simpler approach)
            try:
                print(f"DEBUG: Starting direct database import for: {new_worksheet_name}")
                print(f"DEBUG: Import result has {len(result.data) if result.data else 0} data rows")
                print(f"DEBUG: Import result has {len(result.centroid_data) if result.centroid_data else 0} centroid rows")
                
                # Save imported data directly to database without creating UI first
                print(f"DEBUG: Saving imported data directly to database: {new_worksheet_name}")
                self._save_imported_data_to_database(new_worksheet_name, result)
                
                # Verify the database was created
                from utils.color_analysis_db import ColorAnalysisDB
                test_db = ColorAnalysisDB(new_worksheet_name)
                test_measurements = test_db.get_all_measurements()
                measurement_count = len(test_measurements) if test_measurements else 0
                print(f"DEBUG: Verification - new database has {measurement_count} measurements")
                
                # Now create the worksheet window to view the data
                print(f"DEBUG: Creating worksheet window to view the imported data")
                try:
                    # Import here to avoid circular import issues
                    from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
                    
                    # Create worksheet normally - it will load the data we just saved to database
                    new_worksheet = RealtimePlot3DSheet(
                        parent=self.root,
                        sample_set_name=new_worksheet_name,
                        load_initial_data=True  # Load the data we just saved
                    )
                    print(f"DEBUG: Worksheet window created and data loaded from database")
                    
                except Exception as ui_error:
                    print(f"DEBUG: UI creation failed, but database was saved: {ui_error}")
                    # Data is still saved, just show message without UI
                
                # Show success message
                centroid_msg = f"\n• Imported {len(result.centroid_data)} K-means centroids" if result.centroid_data else ""
                
                messagebox.showinfo(
                    "Import Successful",
                    f"✅ Successfully imported data as '{new_worksheet_name}'!\n\n"
                    f"• Imported {result.rows_imported} data rows{centroid_msg}\n"
                    f"• Data saved to database ({measurement_count} measurements)\n"
                    f"• Available in 'Existing StampZ Database' option\n\n"
                    f"You can now select it from the Plot_3D Data Manager."
                )
                
                logger.info(f"Successfully imported '{new_worksheet_name}' with {result.rows_imported} rows from Plot_3D Data Manager")
                
            except Exception as new_window_error:
                logger.error(f"Error creating new worksheet window: {new_window_error}")
                messagebox.showerror(
                    "New Worksheet Error", 
                    f"Failed to create new worksheet:\n{new_window_error}\n\n"
                    f"Please try again or check the terminal for details."
                )
                
        except Exception as e:
            logger.error(f"Error in import external as new: {e}")
            messagebox.showerror("Import Error", f"Failed to import external data: {e}")
    
    def _import_external_data_helper(self, importer, file_path):
        """Helper method to import external data using the importer.
        
        Args:
            importer: ExternalDataImporter instance
            file_path: Path to the file to import
            
        Returns:
            ImportResult object
        """
        try:
            print(f"DEBUG: Helper - Starting import of file: {file_path}")
            # Import the file
            result = importer.import_file(file_path)
            print(f"DEBUG: Helper - Import file completed, result type: {type(result)}")
            
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
    
    def _populate_worksheet_with_data(self, worksheet, import_result):
        """Populate a worksheet with imported data.
        
        Args:
            worksheet: RealtimePlot3DSheet instance to populate
            import_result: ImportResult with data to populate
        """
        try:
            print(f"DEBUG: Starting worksheet population")
            
            # Safety check - limit data size to prevent freezing
            imported_data_rows = len(import_result.data) if import_result.data else 0
            if imported_data_rows > 1000:
                print(f"WARNING: Large dataset ({imported_data_rows} rows), limiting to first 1000 rows")
                import_result.data = import_result.data[:1000]
                imported_data_rows = 1000
            
            print(f"DEBUG: Will populate {imported_data_rows} data rows")
            min_rows = 7 + imported_data_rows + 10  # 7 reserved rows + data + 10 buffer
            
            # Clear the new sheet carefully
            print(f"DEBUG: Clearing existing sheet data")
            try:
                current_rows = worksheet.sheet.get_total_rows()
                print(f"DEBUG: Current sheet has {current_rows} rows")
                if current_rows > 0:
                    worksheet.sheet.delete_rows(0, current_rows)
                    print(f"DEBUG: Deleted {current_rows} existing rows")
            except Exception as clear_error:
                print(f"DEBUG: Error clearing sheet: {clear_error}")
                # Continue anyway
            
            # Create structure safely
            print(f"DEBUG: Creating sheet structure with {min_rows} rows")
            try:
                PLOT3D_COLUMNS = ['Xnorm', 'Ynorm', 'Znorm', 'DataID', 'Cluster', '∆E', 'Marker', 'Color', 'Centroid_X', 'Centroid_Y', 'Centroid_Z', 'Sphere', 'Radius']
                empty_rows = [[''] * len(PLOT3D_COLUMNS)] * min_rows
                worksheet.sheet.insert_rows(rows=empty_rows, idx=0)
                print(f"DEBUG: Sheet structure created successfully")
            except Exception as structure_error:
                print(f"DEBUG: Error creating sheet structure: {structure_error}")
                raise
            
            # Insert centroid data first (rows 1-6) - only first 6
            print(f"DEBUG: Inserting centroid data")
            try:
                if import_result.centroid_data:
                    for cluster_id, centroid_row in import_result.centroid_data[:6]:  # Limit to first 6
                        if 0 <= cluster_id <= 5:  # Valid centroid area
                            centroid_row_idx = 1 + cluster_id  # Rows 1-6 for clusters 0-5
                            worksheet.sheet.set_row_data(centroid_row_idx, values=centroid_row)
                            print(f"DEBUG: Populated centroid for cluster {cluster_id}")
            except Exception as centroid_error:
                print(f"DEBUG: Error inserting centroid data: {centroid_error}")
                # Continue without centroids
            
            # Insert imported data starting at row 7 (data area)
            print(f"DEBUG: Inserting {imported_data_rows} data rows starting at row 7")
            try:
                if import_result.data and imported_data_rows > 0:
                    for i, row in enumerate(import_result.data[:imported_data_rows]):  # Safety limit
                        worksheet.sheet.set_row_data(7 + i, values=row)
                        # Progress indicator for large datasets
                        if i > 0 and i % 100 == 0:
                            print(f"DEBUG: Inserted {i} rows so far...")
                            # Update UI periodically to prevent freezing
                            worksheet.window.update_idletasks()
                    print(f"DEBUG: All {imported_data_rows} data rows inserted")
            except Exception as data_error:
                print(f"DEBUG: Error inserting data rows: {data_error}")
                raise
            
            # Skip formatting and validation for now to prevent freezing
            print(f"DEBUG: Skipping formatting to prevent freezing")
            # worksheet._apply_formatting()
            # worksheet._setup_validation()
            
            print(f"DEBUG: Worksheet population completed successfully")
            logger.info(f"Successfully populated new worksheet with {imported_data_rows} rows")
            
        except Exception as e:
            print(f"DEBUG: Error in worksheet population: {e}")
            logger.error(f"Error populating new worksheet: {e}")
            raise
    
    def _save_imported_data_to_database(self, sample_set_name, import_result):
        """Save imported data directly to database without UI.
        
        Args:
            sample_set_name: Name for the new database
            import_result: ImportResult with data to save
        """
        if self.database_manager:
            return self.database_manager.save_imported_data_to_database(sample_set_name, import_result)
        else:
            return self._legacy_save_imported_data_to_database(sample_set_name, import_result)
    
    def _create_new_plot3d_template(self):
        """Create a new empty Plot_3D template."""
        sample_set_name = simpledialog.askstring(
            "Sample Set Name",
            "Enter a name for the new sample set:",
            initialvalue="New_Analysis"
        )
        
        if sample_set_name:
            self.create_plot3d_worksheet_with_name(sample_set_name, populate=False)
    
    def _open_internal_viewer(self, sample_set_name):
        """Open real-time spreadsheet viewer for specific sample set."""
        if self.database_manager:
            return self.database_manager.open_internal_viewer(sample_set_name)
        else:
            return self._legacy_open_internal_viewer(sample_set_name)
    
    def _create_file_from_existing(self, sample_set_name):
        """Create file template from existing StampZ data."""
        self.create_plot3d_worksheet_with_name(sample_set_name, populate=True)
    
    def create_plot3d_worksheet_with_name(self, sample_set_name, populate=True):
        """Create Plot_3D worksheet with specified sample set name."""
        if self.database_manager:
            return self.database_manager.create_plot3d_worksheet_with_name(sample_set_name, populate)
        else:
            return self._legacy_create_plot3d_worksheet_with_name(sample_set_name, populate)
    
    def _execute_worksheet_creation(self, filepath, sample_set_name, populate):
        """Execute the actual worksheet creation logic (ODS format only)."""
        try:
            from utils.worksheet_manager import WorksheetManager
            
            # Create formatted ODS template
            success = self._create_clean_template(filepath, sample_set_name)
            
            if success and populate:
                # Populate ODS template with data
                self._populate_ods_template(filepath, sample_set_name)
            
            if success:
                data_info = "populated with existing data" if populate else "empty template ready for data entry"
                
                # Ask if user wants to launch Plot_3D with the created file
                launch_plot3d = messagebox.askyesno(
                    "Worksheet Created",
                    f"Plot_3D worksheet created successfully.\\n\\n"
                    f"File: {os.path.basename(filepath)}\\n"
                    f"Format: OpenDocument format (compatible with Excel)\\n"
                    f"Data: {data_info}\\n\\n"
                    f"Would you like to open this file in Plot_3D now?"
                )
                
                if launch_plot3d:
                    self._launch_plot3d_with_file(filepath)
            else:
                messagebox.showerror(
                    "Creation Failed",
                    f"Failed to create Plot_3D worksheet.\\n\\nPlease check file permissions and try again."
                )
                
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to create worksheet:\\n\\n{str(e)}"
            )
    
    def _launch_plot3d_with_file(self, file_path):
        """Launch Plot_3D with a specific data file."""
        try:
            from plot3d.Plot_3D import Plot3DApp
            
            # Launch Plot_3D with the specified file
            plot_app = Plot3DApp(parent=self.root, data_path=file_path)
            
            messagebox.showinfo(
                "Plot_3D Launched",
                f"Plot_3D opened with your worksheet:\\n\\n"
                f"{os.path.basename(file_path)}\\n\\n"
                f"You can now analyze your data in 3D space!"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch Plot_3D with file:\\n\\n{str(e)}\\n\\n"
                f"You can manually open Plot_3D and load the file:"
                f"\\n{os.path.basename(file_path)}"
            )
    
    def _create_and_launch_new_template(self):
        """Create new empty template and launch Plot_3D."""
        sample_set_name = simpledialog.askstring(
            "Sample Set Name",
            "Enter a name for the new sample set:",
            initialvalue="New_Analysis"
        )
        
        if sample_set_name:
            filepath = self._get_save_path(sample_set_name)
            if filepath:
                self._create_clean_template(filepath, sample_set_name)
                self._launch_plot3d_with_file(filepath)
    
    def _create_and_launch_from_database(self, sample_set_name):
        """Create template from database and launch Plot_3D."""
        if self.database_manager:
            return self.database_manager.create_and_launch_from_database(sample_set_name)
        else:
            return self._legacy_create_and_launch_from_database(sample_set_name)
    
    def _load_existing_file_in_plot3d(self):
        """Load existing Plot_3D file directly."""
        filepath = filedialog.askopenfilename(
            title="Open Existing Plot_3D File",
            filetypes=[
                ('OpenDocument Spreadsheet', '*.ods'),
                ('Excel Workbook', '*.xlsx'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
            ]
        )
        
        if filepath:
            self._launch_plot3d_with_file(filepath)
    
    def _import_and_launch_csv(self):
        """Import CSV and launch Plot_3D."""
        # Simplified CSV import that creates template and launches Plot_3D
        filepath = filedialog.askopenfilename(
            title="Import CSV for Plot_3D",
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        
        if filepath:
            # Create formatted template from CSV
            output_file = filepath.replace('.csv', '_Plot3D.ods')
            if self._convert_csv_to_plot3d(filepath, output_file):
                self._launch_plot3d_with_file(output_file)
    
    def _get_save_path(self, sample_set_name):
        """Get save path for new template (ODS format only)."""
        default_filename = f"{sample_set_name}_Plot3D_{datetime.now().strftime('%Y%m%d')}"
        
        return filedialog.asksaveasfilename(
            title="Save Plot_3D Template",
            defaultextension=".ods",
            filetypes=[
                ('OpenDocument Spreadsheet', '*.ods')
            ],
            initialfile=default_filename,
            initialdir=os.path.expanduser("~/Desktop")
        )
    
    def _create_clean_template(self, filepath, sample_set_name):
        """Create clean template using formatted Plot3D_Template.ods."""
        try:
            import shutil
            import sys
            
            # Path to the formatted template - handle both development and bundled environments
            if getattr(sys, 'frozen', False):
                # Running in a PyInstaller bundle
                if hasattr(sys, '_MEIPASS'):
                    # PyInstaller temp directory
                    base_path = sys._MEIPASS
                else:
                    # App bundle Contents/Resources
                    base_path = os.path.join(os.path.dirname(sys.executable), '..', 'Resources')
                template_path = os.path.join(base_path, 'data', 'templates', 'plot3d', 'Plot3D_Template.ods')
            else:
                # Development environment
                template_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'templates', 'plot3d', 'Plot3D_Template.ods')
            
            # Verify template exists
            if not os.path.exists(template_path):
                logger.warning(f"Formatted template not found at {template_path}, creating basic template")
                # Fallback to basic creation if template missing
                return self._create_basic_template(filepath, sample_set_name)
            
            # Copy the formatted template to the new location
            shutil.copy2(template_path, filepath)
            
            # Update the sample set name in the copied template
            self._update_template_sample_name(filepath, sample_set_name)
            
            logger.info(f"Created formatted ODS template from {template_path}: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating template from formatted file: {e}")
            logger.info("Falling back to basic template creation")
            # Fallback to basic creation if template copy fails
            return self._create_basic_template(filepath, sample_set_name)
    
    def _create_basic_template(self, filepath, sample_set_name):
        """Create basic template using pandas (fallback method)."""
        try:
            import pandas as pd
            from utils.worksheet_manager import WorksheetManager
            
            # Create clean DataFrame with just headers
            df = pd.DataFrame(columns=WorksheetManager.PLOT3D_COLUMNS)
            
            # Export to ODS format only
            df.to_excel(filepath, engine='odf', index=False)
            
            logger.info(f"Created basic ODS template: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating basic template: {e}")
            messagebox.showerror("Error", f"Failed to create template: {e}")
            return False
    
    def _update_template_sample_name(self, filepath, sample_set_name):
        """Update sample set name in the copied template."""
        try:
            # For now, just log that we should update the name
            # The template copying should work as-is, and the user can edit the name
            logger.info(f"Template copied successfully, sample set: {sample_set_name}")
            
        except Exception as e:
            logger.warning(f"Could not update template sample name: {e}")
    
    def _create_template_with_data(self, filepath, sample_set_name):
        """Create template populated with real StampZ data."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Get real measurements
            db = ColorAnalysisDB(sample_set_name)
            measurements = db.get_all_measurements()
            
            if not measurements:
                # No data - create clean template
                self._create_clean_template(filepath, sample_set_name)
                messagebox.showinfo(
                    "No Data",
                    f"No measurements found for '{sample_set_name}'.\\n\\n"
                    f"Created empty template instead."
                )
                return
            
            # Create DataFrame with real data - NORMALIZE the raw L*a*b* values from database
            data_rows = []
            for i, measurement in enumerate(measurements):
                # CRITICAL FIX: Normalize raw L*a*b* values from database for Plot_3D
                l_val = measurement.get('l_value', 0.0)
                a_val = measurement.get('a_value', 0.0)
                b_val = measurement.get('b_value', 0.0)
                
                # Apply proper normalization:
                # L*: 0-100 → 0-1
                # a*: -128 to +127 → 0-1 
                # b*: -128 to +127 → 0-1
                x_norm = max(0.0, min(1.0, (l_val if l_val is not None else 0.0) / 100.0))
                y_norm = max(0.0, min(1.0, ((a_val if a_val is not None else 0.0) + 128.0) / 255.0))
                z_norm = max(0.0, min(1.0, ((b_val if b_val is not None else 0.0) + 128.0) / 255.0))
                
                row = {
                    'Xnorm': round(x_norm, 4),  # Normalized L* value
                    'Ynorm': round(y_norm, 4),  # Normalized a* value 
                    'Znorm': round(z_norm, 4),  # Normalized b* value
                    'DataID': f"{sample_set_name}_Sample_{i+1:03d}",
                    'Cluster': '', '∆E': '', 'Marker': '.', 'Color': 'blue',
                    'Centroid_X': '', 'Centroid_Y': '', 'Centroid_Z': '',
                    'Sphere': '', 'Radius': ''
                }
                data_rows.append(row)
            
            # Save to file
            df = pd.DataFrame(data_rows)
            file_ext = os.path.splitext(filepath)[1].lower()
            if file_ext == '.xlsx':
                df.to_excel(filepath, index=False)
            else:
                df.to_excel(filepath, engine='odf', index=False)
            
            logger.info(f"Created template with {len(measurements)} measurements")
            
        except Exception as e:
            logger.error(f"Error creating template with data: {e}")
            messagebox.showerror("Error", f"Failed to create template: {e}")
    
    def _convert_csv_to_plot3d(self, csv_path, output_path):
        """Convert CSV to Plot_3D format."""
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['Xnorm', 'Ynorm', 'Znorm', 'DataID']
            if not all(col in df.columns for col in required_cols):
                messagebox.showerror(
                    "Invalid CSV",
                    f"CSV missing required columns: {required_cols}"
                )
                return False
            
            # Save as ODS
            df.to_excel(output_path, engine='odf', index=False)
            return True
            
        except Exception as e:
            logger.error(f"Error converting CSV: {e}")
            messagebox.showerror("Error", f"Failed to convert CSV: {e}")
            return False
    
    def _open_realtime_spreadsheet(self, sample_set_name):
        """Open the real-time Excel-like spreadsheet."""
        try:
            print(f"DEBUG: Attempting to open real-time spreadsheet for: {sample_set_name}")
            
            from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
            print("DEBUG: Successfully imported RealtimePlot3DSheet")
            
            # Create the real-time spreadsheet
            print(f"DEBUG: Creating RealtimePlot3DSheet instance...")
            sheet = RealtimePlot3DSheet(self.root, sample_set_name)
            print("DEBUG: RealtimePlot3DSheet created successfully")
            
            # Create custom dialog that appears on top
            self._show_spreadsheet_acknowledgment(sample_set_name, sheet.window)
            
        except ImportError as ie:
            logger.error(f"Import error for real-time spreadsheet: {ie}")
            messagebox.showerror(
                "Missing Component",
                f"Could not load real-time spreadsheet:\\n\\n{str(ie)}\\n\\n"
                f"tksheet library may not be properly installed."
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error opening real-time spreadsheet: {e}")
            print(f"DEBUG: Full error traceback:\\n{error_details}")
            messagebox.showerror(
                "Error",
                f"Failed to open real-time spreadsheet:\\n\\n{str(e)}\\n\\n"
                f"Check console for detailed error information."
            )
    
    def _show_spreadsheet_acknowledgment(self, sample_set_name, sheet_window):
        """Show acknowledgment dialog that appears on top of the spreadsheet."""
        from tkinter import Toplevel, Label, CENTER
        from tkinter import ttk
        
        # Create dialog as child of the spreadsheet window
        dialog = Toplevel(sheet_window)
        dialog.title("Real-time Spreadsheet Opened")
        dialog.geometry("500x350")
        dialog.resizable(False, False)
        
        # Make it modal and on top
        dialog.transient(sheet_window)
        dialog.grab_set()
        dialog.attributes('-topmost', True)
        
        # Center on the spreadsheet window
        sheet_window.update_idletasks()
        dialog.update_idletasks()
        
        # Get positions
        sheet_x = sheet_window.winfo_x()
        sheet_y = sheet_window.winfo_y()
        sheet_width = sheet_window.winfo_width()
        sheet_height = sheet_window.winfo_height()
        
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        
        # Calculate center position
        x = sheet_x + (sheet_width - dialog_width) // 2
        y = sheet_y + (sheet_height - dialog_height) // 2
        
        dialog.geometry(f"+{x}+{y}")
        
        # Content
        Label(dialog, text="Real-time Spreadsheet Opened", 
              font=("Arial", 14, "bold")).pack(pady=10)
        
        message = f"User-editable spreadsheet opened for '{sample_set_name}'.\n\n" + \
                  f"Features:\n" + \
                  f"• Pink cells: Protected areas (no manual entry)\n" + \
                  f"• Colored columns: G=Salmon, H=Yellow, L=Yellow\n" + \
                  f"• Auto-refresh: New StampZ data appears automatically\n" + \
                  f"• Direct Plot_3D integration (no external files needed!)\n\n" + \
                  f"This is your real-time workflow solution!"
        
        Label(dialog, text=message, font=("Arial", 11), justify=CENTER, 
              wraplength=450).pack(pady=10, padx=20)
        
        def close_dialog():
            dialog.grab_release()
            dialog.destroy()
        
        ttk.Button(dialog, text="OK", command=close_dialog, width=10).pack(pady=10)
        
        # Ensure dialog gets focus
        dialog.focus_force()
        dialog.lift()
    
    def _legacy_export_plot3d_flexible(self):
        """Legacy implementation of flexible Plot3D export."""
        if self.database_manager:
            return self.database_manager.export_plot3d_flexible()
        else:
            # Fallback implementation
            messagebox.showerror(
                "Feature Unavailable",
                "Plot3D export functionality is not available.\n\n"
                "DatabaseManager component is missing."
            )

    def export_color_data(self):
        """Export color analysis data to spreadsheet format."""
        try:
            from utils.ods_exporter import ODSExporter
            
            current_sample_set = None
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                current_sample_set = self.app.control_panel.sample_set_name.get().strip()

            exporter = ODSExporter(sample_set_name=current_sample_set)
            measurements = exporter.get_color_measurements()

            if not measurements:
                if current_sample_set:
                    messagebox.showinfo(
                        "No Data", 
                        f"No color analysis data found for sample set '{current_sample_set}'.\\n\\n"
                        "Please run some color analysis first using the coordinate sampling tool."
                    )
                else:
                    messagebox.showinfo(
                        "No Data", 
                        "No color analysis data found in the database.\\n\\n"
                        "Please run some color analysis first using the coordinate sampling tool."
                    )
                return

            if current_sample_set:
                default_filename = f"{current_sample_set}_{datetime.now().strftime('%Y%m%d')}.ods"
            else:
                default_filename = f"stampz_color_data_{datetime.now().strftime('%Y%m%d')}.ods"

            filepath = filedialog.asksaveasfilename(
                title="Export Color Data",
                defaultextension=".ods",
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('All files', '*.*')
                ],
                initialfile=default_filename
            )

            if filepath:
                success = exporter.export_and_open(filepath)
                if success:
                    if current_sample_set:
                        messagebox.showinfo(
                            "Export Successful",
                            f"Successfully exported {len(measurements)} color measurements from sample set '{current_sample_set}' to:\\n\\n"
                            f"{os.path.basename(filepath)}\\n\\n"
                            f"The spreadsheet has been opened in LibreOffice Calc for analysis."
                        )
                    else:
                        messagebox.showinfo(
                            "Export Successful",
                            f"Successfully exported {len(measurements)} color measurements to:\\n\\n"
                            f"{os.path.basename(filepath)}\\n\\n"
                            f"The spreadsheet has been opened in LibreOffice Calc for analysis."
                        )
                else:
                    messagebox.showerror(
                        "Export Failed",
                        "Failed to export color data or open spreadsheet. Please check that LibreOffice Calc is installed."
                    )

        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "The ODS export feature requires the 'odfpy' library.\\n\\n"
                "Please install it with: pip install odfpy==1.4.1"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"An error occurred during export:\\n\\n{str(e)}"
            )

    def open_color_library(self):
        """Open the Color Library Manager window."""
        try:
            from gui.color_library_manager import ColorLibraryManager
            library_manager = ColorLibraryManager(parent=self.root)
            library_manager.root.update()
        except ImportError as e:
            messagebox.showerror(
                "Missing Component",
                f"Color Library Manager not available:\\n\\n{str(e)}\\n\\n"
                "Please ensure all color library components are properly installed."
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open Color Library Manager:\\n\\n{str(e)}"
            )

    def compare_sample_to_library(self):
        """Compare analyzed samples to color library entries."""
        try:
            from gui.color_library_manager import ColorLibraryManager
            from utils.color_analyzer import ColorAnalyzer
            from utils.color_library import ColorLibrary

            if not hasattr(self.app.canvas, '_coord_markers') or not self.app.canvas._coord_markers:
                messagebox.showwarning(
                    "No Samples",
                    "Please analyze some color samples first using the Sample tool."
                )
                return

            if not self.app.current_file:
                messagebox.showwarning(
                    "No Image",
                    "Please open an image before comparing colors."
                )
                return

            analyzer = ColorAnalyzer()
            sample_data = []
            non_preview_markers = [m for m in self.app.canvas._coord_markers if not m.get('is_preview', False)]

            for marker in non_preview_markers:
                try:
                    image_x, image_y = marker['image_pos']
                    sample_type = marker.get('sample_type', 'rectangle')
                    sample_width = float(marker.get('sample_width', 20))
                    sample_height = float(marker.get('sample_height', 20))

                    measurement = {
                        'position': (image_x, image_y),
                        'type': sample_type,
                        'size': (sample_width, sample_height),
                        'anchor': marker.get('anchor', 'center')
                    }
                    sample_data.append(measurement)
                except Exception as e:
                    continue

            try:
                library_manager = ColorLibraryManager(parent=self.root)
                if not library_manager.library:
                    library_manager.library = ColorLibrary('basic_colors')
                library_manager._create_comparison_tab()
                library_manager.comparison_manager.set_analyzed_data(
                    image_path=self.app.current_file,
                    sample_data=sample_data
                )
                library_manager.notebook.select(1)
                library_manager.root.update()
                library_manager.root.lift()
                library_manager.root.focus_force()
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to initialize comparison window: {str(e)}"
                )

        except ImportError as e:
            messagebox.showerror(
                "Missing Component",
                f"Color Library Manager not available:\\n\\n{str(e)}\\n\\n"
                "Please ensure all color library components are properly installed."
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open Color Library Manager:\\n\\n{str(e)}"
            )

    def create_standard_libraries(self):
        """Create standard color libraries for philatelic analysis."""
        try:
            from utils.color_library_integration import create_standard_philatelic_libraries

            result = messagebox.askyesno(
                "Create Standard Libraries",
                "This will create standard color libraries for philatelic analysis:\\n\\n"
                "• Basic Colors (primary, secondary, neutral colors)\\n"
                "• Philatelic Colors (common stamp colors)\\n\\n"
                "If these libraries already exist, they will be updated.\\n\\n"
                "Do you want to continue?"
            )

            if result:
                progress_dialog = tk.Toplevel(self.root)
                progress_dialog.title("Creating Libraries")
                progress_dialog.geometry("300x100")
                progress_dialog.transient(self.root)
                progress_dialog.grab_set()

                progress_dialog.update_idletasks()
                x = (progress_dialog.winfo_screenwidth() // 2) - (progress_dialog.winfo_width() // 2)
                y = (progress_dialog.winfo_screenheight() // 2) - (progress_dialog.winfo_height() // 2)
                progress_dialog.geometry(f"+{x}+{y}")

                progress_label = ttk.Label(progress_dialog, text="Creating standard libraries...")
                progress_label.pack(expand=True)

                progress_dialog.update()

                created_libraries = create_standard_philatelic_libraries()

                progress_dialog.destroy()

                messagebox.showinfo(
                    "Libraries Created",
                    f"Successfully created standard libraries:\\n\\n"
                    f"• {created_libraries[0]}\\n"
                    f"• {created_libraries[1]}\\n\\n"
                    f"You can now access these through the Color Library Manager."
                )

        except ImportError as e:
            messagebox.showerror(
                "Missing Component",
                f"Color library system not available:\\n\\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Creation Error",
                f"Failed to create standard libraries:\\n\\n{str(e)}"
            )

    def export_with_library_matches(self, sample_set_name=None):
        """Export analysis data with color library matches."""
        try:
            from utils.color_library_integration import ColorLibraryIntegration

            if not sample_set_name:
                if (hasattr(self.app, 'control_panel') and 
                    hasattr(self.app.control_panel, 'sample_set_name') and 
                    self.app.control_panel.sample_set_name.get().strip()):
                    sample_set_name = self.app.control_panel.sample_set_name.get().strip()
                else:
                    messagebox.showwarning(
                        "No Sample Set",
                        "Please enter a sample set name in the control panel first."
                    )
                    return

            integration = ColorLibraryIntegration(['philatelic_colors', 'basic_colors'])

            default_filename = f"{sample_set_name}_with_library_matches_{datetime.now().strftime('%Y%m%d')}.ods"
            filepath = filedialog.asksaveasfilename(
                title="Export Analysis with Library Matches",
                defaultextension=".ods",
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('All files', '*.*')
                ],
                initialfile=default_filename
            )

            if filepath:
                workflow = integration.get_analysis_workflow_summary(sample_set_name, threshold=5.0)

                if workflow['status'] == 'analyzed':
                    messagebox.showinfo(
                        "Export Complete",
                        f"Would export analysis with library matches to:\\n\\n"
                        f"{os.path.basename(filepath)}\\n\\n"
                        f"This would include:\\n"
                        f"• {workflow['summary']['total_samples']} color samples\\n"
                        f"• Library matches with ΔE values\\n"
                        f"• Match quality ratings\\n"
                        f"• Complete analysis metadata\\n\\n"
                        f"Note: This feature requires ODSExporter integration."
                    )
                else:
                    messagebox.showwarning(
                        "No Data",
                        f"No analysis data found for sample set '{sample_set_name}'"
                    )

        except ImportError as e:
            messagebox.showerror(
                "Missing Component",
                f"Export functionality not available:\\n\\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export analysis:\\n\\n{str(e)}"
            )

    def open_spectral_analysis(self):
        """Open the spectral analysis window."""
        try:
            from utils.spectral_analyzer import SpectralAnalyzer
            
            if not self.app.current_file:
                messagebox.showwarning(
                    "No Image",
                    "Please open an image before performing spectral analysis."
                )
                return
            
            # Create and show spectral analysis window
            spectral_analyzer = SpectralAnalyzer(parent=self.root, image_path=self.app.current_file)
            
        except ImportError as e:
            messagebox.showerror(
                "Missing Component",
                f"Spectral analysis not available:\\n\\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open spectral analysis:\\n\\n{str(e)}"
            )

    def open_3d_analysis(self):
        """Open 3D color space analysis tool."""
        try:
            # Import Plot_3D module
            from plot3d.Plot_3D import Plot3DApp
            
            # Get current sample set name if available
            current_sample_set = None
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                current_sample_set = self.app.control_panel.sample_set_name.get().strip()
            
            # Check if we have data to analyze
            if current_sample_set and self.app.current_file:
                # Try to export data first, then launch Plot_3D
                try:
                    # Export data for Plot3D
                    self._export_data_for_plot3d(current_sample_set, measurements=None)
                    
                    # Launch Plot_3D (it will find the exported files)
                    messagebox.showinfo(
                        "3D Analysis",
                        f"Launching 3D color analysis with data from '{current_sample_set}'.\\n\\n"
                        "The 3D analysis window will open shortly."
                    )
                    
                    # Create Plot_3D app instance
                    plot_app = Plot3DApp(parent=self.root)
                    
                except Exception as e:
                    print(f"Error exporting data for Plot_3D: {e}")
                    # Still launch Plot_3D even if export fails
                    messagebox.showinfo(
                        "3D Analysis",
                        "Launching 3D color analysis tool.\\n\\n"
                        "You can load the exported data files manually."
                    )
                    plot_app = Plot3DApp(parent=self.root)
            else:
                # No current data - launch Plot_3D in standalone mode
                messagebox.showinfo(
                    "3D Analysis",
                    "Launching 3D color analysis tool.\\n\\n"
                    "You can load existing data files or import from spreadsheets."
                )
                
                # Create Plot_3D app instance without specific data
                plot_app = Plot3DApp(parent=self.root)
                
        except ImportError as e:
            messagebox.showerror(
                "Import Error",
                f"Could not load 3D analysis module:\\n\\n{str(e)}\\n\\n"
                "Please ensure all Plot_3D components are properly installed."
            )
        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch 3D analysis:\\n\\n{str(e)}"
            )

    def _export_data_for_plot3d(self, sample_set_name, measurements):
        """Export data specifically formatted for Plot3D analysis."""
        if self.database_manager:
            return self.database_manager.export_data_for_plot3d(sample_set_name, measurements)
        else:
            return self._legacy_export_data_for_plot3d(sample_set_name, measurements)

    def view_spreadsheet(self):
        """Open real-time spreadsheet view of color analysis data."""
        if self.database_manager:
            return self.database_manager.view_spreadsheet()
        else:
            return self._legacy_view_spreadsheet()

    def _show_realtime_data_selection_dialog(self):
        """Show dialog to select which spreadsheet data to view."""
        try:
            from tkinter import Toplevel, Listbox, Button, Frame, Label, Scrollbar
            from utils.color_analysis_db import ColorAnalysisDB
            from utils.path_utils import get_color_analysis_dir
            
            # Get available sample sets
            color_data_dir = get_color_analysis_dir()
            if not os.path.exists(color_data_dir):
                messagebox.showinfo(
                    "No Data",
                    "No color analysis data found.\\n\\n"
                    "Please run color analysis first using the Sample tool."
                )
                return
            
            available_sets = ColorAnalysisDB.get_all_sample_set_databases(color_data_dir)
            if not available_sets:
                messagebox.showinfo(
                    "No Data",
                    "No color analysis data found.\\n\\n"
                    "Please run color analysis first using the Sample tool."
                )
                return
                
            # Create a selection dialog
            dialog = Toplevel(self.root)
            dialog.title("Select Data to View")
            dialog.geometry("450x350")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Center dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Header
            from tkinter import Label, Listbox, Button, Frame, Scrollbar
            Label(dialog, text="Choose which data to view:", font=("Arial", 12, "bold")).pack(pady=10)
            
            # Sample sets listbox
            sets_frame = Frame(dialog)
            sets_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
            
            Label(sets_frame, text="Available Sample Sets:", font=("Arial", 10)).pack(anchor="w")
            
            listbox_frame = Frame(sets_frame)
            listbox_frame.pack(fill="both", expand=True, pady=5)
            
            sets_listbox = Listbox(listbox_frame, font=("Arial", 13, "bold"))
            sets_listbox.pack(side="left", fill="both", expand=True)
            
            scrollbar = Scrollbar(listbox_frame)
            scrollbar.pack(side="right", fill="y")
            sets_listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=sets_listbox.yview)
            
            # Populate listbox
            for sample_set in available_sets:
                sets_listbox.insert("end", sample_set)
                
            # Select first item by default
            if available_sets:
                sets_listbox.selection_set(0)
                
            selected_option = None
            selected_sample_set = None
            
            def on_view_selected():
                nonlocal selected_option, selected_sample_set
                selection = sets_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a sample set to view")
                    return
                
                selected_option = "specific"
                selected_sample_set = available_sets[selection[0]]
                dialog.quit()
                dialog.destroy()
            
            def on_cancel():
                nonlocal selected_option
                selected_option = None
                dialog.quit()
                dialog.destroy()
            
            # Buttons
            button_frame = Frame(dialog)
            button_frame.pack(pady=10)
            
            Button(button_frame, text="View Selected", command=on_view_selected, width=15).pack(side="left", padx=5)
            Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side="right", padx=5)
            
            # Keyboard bindings
            dialog.bind('<Return>', lambda e: on_view_selected())
            dialog.bind('<Escape>', lambda e: on_cancel())
            sets_listbox.bind("<Double-Button-1>", lambda e: on_view_selected())
            
            sets_listbox.focus_set()
            dialog.mainloop()
            
            # Process selection - open real-time spreadsheet
            if selected_option == "specific" and selected_sample_set:
                print(f"DEBUG: User selected sample set: {selected_sample_set}")
                
                # Handle both regular and averaged sample sets
                if selected_sample_set.endswith('_averages'):
                    base_name = selected_sample_set[:-9]  # Remove '_averages' suffix
                    sample_set_to_open = base_name
                else:
                    sample_set_to_open = selected_sample_set
                
                # Open real-time spreadsheet
                self._open_realtime_spreadsheet(sample_set_to_open)
            
        except Exception as e:
            messagebox.showerror(
                "Dialog Error",
                f"Failed to show data selection dialog:\\n\\n{str(e)}"
            )
    
    def open_black_ink_extractor(self):
        """Open the Black Ink Extractor for cancellation extraction."""
        # Use the new BlackInkManager for cleaner code organization
        try:
            from managers.black_ink_manager import BlackInkManager
            black_ink_manager = BlackInkManager(self.app)
            black_ink_manager.open_black_ink_extractor()
        except ImportError:
            # Fallback to legacy implementation
            self._legacy_black_ink_extractor()
            
    def _legacy_black_ink_extractor(self):
        """Legacy black ink extractor implementation (fallback)."""
        if not self.app.current_file:
            messagebox.showwarning(
                "No Image",
                "Please open a stamp image first.\\n\\n"
                "The Black Ink Extractor works best with colored stamps containing black cancellations."
            )
            return
            
        # Import the extraction functionality
        try:
            from black_ink_extractor import extract_black_ink, extract_colored_cancellation_rgb_only, safe_pil_fromarray
        except ImportError:
            messagebox.showerror(
                "Module Error",
                "Black Ink Extractor module not found.\\n\\n"
                "Please ensure 'black_ink_extractor.py' is in the StampZ directory."
            )
            return
            
        try:
            # Create extraction dialog (orphaned for multi-screen use)
            dialog = tk.Toplevel()
            dialog.title("Black Ink Extractor")
            dialog.geometry("650x700")  # Wider for better layout
            # Remove transient and grab_set to allow moving to different screens
            # dialog.transient(self.root)  # Commented out to orphan the dialog
            # dialog.grab_set()  # Commented out to allow interaction with other windows
            
            # Center dialog
            dialog.update_idletasks()
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = (screen_width // 2) - (dialog.winfo_reqwidth() // 2)
            y = (screen_height // 2) - (dialog.winfo_reqheight() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Header
            header_frame = ttk.Frame(dialog)
            header_frame.pack(fill="x", padx=20, pady=10)
            
            ttk.Label(
                header_frame,
                text="Black Ink Cancellation Extractor",
                font=("Arial", 14, "bold")
            ).pack()
            
            ttk.Label(
                header_frame,
                text=f"Current Image: {os.path.basename(self.app.current_file)}",
                font=("Arial", 10)
            ).pack(pady=5)
            
            # Description
            desc_frame = ttk.Frame(dialog)
            desc_frame.pack(fill="x", padx=20, pady=10)
            
            description = (
                "This tool extracts black ink cancellations from colored stamps.\\n\\n"
                "Perfect for isolating postmarks, cancellations, and overprints from \\n"
                "colored stamp backgrounds. The extracted black ink appears on a clean \\n"
                "white background for easy study and documentation.\\n\\n"
                "Works excellent with:\\n"
                "• Red cancellations on Penny Blacks and other dark stamps\\n"
                "• Black postmarks on colored stamps\\n"
                "• Overprints and surcharges\\n\\n"
                "Supports 48-bit TIFF files from VueScan and all image formats."
            )
            
            ttk.Label(
                desc_frame,
                text=description,
                font=("Arial", 9),
                justify="left"
            ).pack(anchor="w")
            
            # Focus message
            focus_frame = ttk.Frame(dialog)
            focus_frame.pack(fill="x", padx=20, pady=10)
            
            ttk.Label(
                focus_frame,
                text="Extracting Black Ink (cancellations, postmarks, overprints)",
                font=("Arial", 12, "bold"),
                foreground="#2E8B57"
            ).pack()
            
            # Settings frame
            settings_frame = ttk.LabelFrame(dialog, text="Extraction Settings", padding=10)
            settings_frame.pack(fill="x", padx=20, pady=10)
            
            # Black threshold setting
            ttk.Label(settings_frame, text="Black Threshold (0-255):").grid(row=0, column=0, sticky="w", pady=2)
            black_threshold = tk.IntVar(value=60)
            black_threshold_scale = ttk.Scale(
                settings_frame,
                from_=10, to=150,
                variable=black_threshold,
                orient="horizontal",
                length=200
            )
            black_threshold_scale.grid(row=0, column=1, padx=10, pady=2)
            black_threshold_label = ttk.Label(settings_frame, text="60")
            black_threshold_label.grid(row=0, column=2, pady=2)
            
            def update_black_threshold(event):
                black_threshold_label.config(text=str(int(black_threshold.get())))
            black_threshold_scale.bind("<Motion>", update_black_threshold)
            
            # Saturation threshold setting
            ttk.Label(settings_frame, text="Saturation Threshold (0-255):").grid(row=1, column=0, sticky="w", pady=2)
            saturation_threshold = tk.IntVar(value=30)
            saturation_threshold_scale = ttk.Scale(
                settings_frame,
                from_=10, to=100,
                variable=saturation_threshold,
                orient="horizontal",
                length=200
            )
            saturation_threshold_scale.grid(row=1, column=1, padx=10, pady=2)
            saturation_threshold_label = ttk.Label(settings_frame, text="30")
            saturation_threshold_label.grid(row=1, column=2, pady=2)
            
            def update_saturation_threshold(event):
                saturation_threshold_label.config(text=str(int(saturation_threshold.get())))
            saturation_threshold_scale.bind("<Motion>", update_saturation_threshold)
            
            # Red channel offset setting (for detecting black ink against red backgrounds)
            ttk.Label(settings_frame, text="Red Channel Offset:").grid(row=2, column=0, sticky="w", pady=2)
            red_offset = tk.IntVar(value=40)
            red_offset_scale = ttk.Scale(
                settings_frame,
                from_=10, to=100,
                variable=red_offset,
                orient="horizontal",
                length=200
            )
            red_offset_scale.grid(row=2, column=1, padx=10, pady=2)
            red_offset_label = ttk.Label(settings_frame, text="40")
            red_offset_label.grid(row=2, column=2, pady=2)
            
            def update_red_offset(event):
                red_offset_label.config(text=str(int(red_offset.get())))
            red_offset_scale.bind("<Motion>", update_red_offset)
            
            # Output settings
            output_frame = ttk.LabelFrame(dialog, text="Output Options", padding=10)
            output_frame.pack(fill="x", padx=20, pady=10)
            
            open_results = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                output_frame,
                text="Open results automatically",
                variable=open_results
            ).pack(anchor="w")
            
            save_to_stampz = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                output_frame,
                text="Load best result back into StampZ",
                variable=save_to_stampz
            ).pack(anchor="w")
            
            
            # Progress and status
            status_frame = ttk.Frame(dialog)
            status_frame.pack(fill="x", padx=20, pady=10)
            
            progress = ttk.Progressbar(status_frame, mode='indeterminate')
            progress.pack(fill="x", pady=5)
            
            status_label = ttk.Label(status_frame, text="Ready to extract...")
            status_label.pack()
            
            # Buttons
            button_frame = ttk.Frame(dialog)
            button_frame.pack(fill="x", padx=20, pady=20)
            
            def extract_cancellation():
                """Perform the black ink extraction."""
                try:
                    # Update UI
                    status_label.config(text="Loading image...")
                    progress.start()
                    dialog.update()
                    
                    # Load image using PIL with proper dtype handling
                    pil_image = Image.open(self.app.current_file)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # Convert to numpy array and ensure uint8 dtype
                    img_array = np.array(pil_image)
                    
                    # Debug info
                    print(f"DEBUG: Original array dtype: {img_array.dtype}, shape: {img_array.shape}")
                    print(f"DEBUG: Value range: {img_array.min()} - {img_array.max()}")
                    
                    # Ensure proper uint8 format
                    if img_array.dtype != np.uint8:
                        if img_array.dtype in [np.uint16, np.int16, np.int32, np.int64]:
                            # Scale down from higher bit depths
                            if img_array.max() > 255:
                                img_array = (img_array.astype(np.float64) / img_array.max() * 255).astype(np.uint8)
                            else:
                                img_array = img_array.astype(np.uint8)
                        else:
                            # For float types or others, clip and convert
                            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    
                    print(f"DEBUG: Final array dtype: {img_array.dtype}, range: {img_array.min()} - {img_array.max()}")
                    
                    # Extract black ink cancellations
                    status_label.config(text="Extracting black ink...")
                    dialog.update()
                    
                    results, mask, analysis = extract_black_ink(
                        img_array,
                        black_threshold=int(black_threshold.get()),
                        saturation_threshold=int(saturation_threshold.get()),
                        red_offset=int(red_offset.get())
                    )
                    
                    status_label.config(text="Saving results...")
                    dialog.update()
                    
                    # Create output directory
                    image_path = Path(self.app.current_file)
                    output_dir = image_path.parent / "black_ink_extraction"
                    output_dir.mkdir(exist_ok=True)
                    base_name = image_path.stem
                    
                    # Save results (streamlined - only essential files)
                    saved_files = []
                    
                    # Save black ink extraction results
                    saved_files = []
                    
                    # 1. Pure black cancellation (main result)
                    pure_result_path = output_dir / f"{base_name}_pure_black_cancellation.png"
                    safe_pil_fromarray(results['pure_black']).save(pure_result_path)
                    saved_files.append(pure_result_path)
                    best_result_path = pure_result_path
                    
                    # 2. Grayscale version (preserves ink density)
                    grayscale_path = output_dir / f"{base_name}_grayscale_cancellation.png"
                    safe_pil_fromarray(results['grayscale']).save(grayscale_path)
                    saved_files.append(grayscale_path)
                    
                    # 3. Create and save adaptive mask (the one you mentioned wanting)
                    # This shows the detection method that works well
                    try:
                        import cv2
                    except ImportError:
                        raise ImportError("OpenCV (cv2) is required for the Black Ink Extractor. Please install it with: pip install opencv-python")
                    
                    # Convert PIL image to proper format for cv2
                    img_for_cv2 = np.array(pil_image, dtype=np.uint8)
                    if img_for_cv2.shape[2] == 3:  # RGB
                        gray = cv2.cvtColor(img_for_cv2, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_for_cv2[:, :, 0]  # Take first channel if not RGB
                    
                    adaptive_mask = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
                    )
                    adaptive_path = output_dir / f"{base_name}_mask_adaptive.png"
                    safe_pil_fromarray(adaptive_mask).save(adaptive_path)
                    saved_files.append(adaptive_path)
                    
                    # 4. Save original for reference (smaller file)
                    original_path = output_dir / f"{base_name}_original.png"
                    # Save at reduced quality to save space
                    pil_image.save(original_path, optimize=True, quality=85)
                    saved_files.append(original_path)
                    
                    # Save analysis report
                    report_path = output_dir / f"{base_name}_analysis.txt"
                    with open(report_path, 'w') as f:
                        f.write(f"Black Ink Extraction Report\\n")
                        f.write(f"Image: {image_path.name}\\n")
                        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
                        f.write(f"Coverage: {analysis['coverage_percentage']:.1f}%\\n")
                        f.write(f"Black ink pixels: {analysis['cancellation_pixels']:,}\\n")
                        f.write(f"Total pixels: {analysis['total_pixels']:,}\\n\\n")
                        f.write(f"Settings used:\\n")
                        f.write(f"  Black threshold: {int(black_threshold.get())}\\n")
                        f.write(f"  Saturation threshold: {int(saturation_threshold.get())}\\n")
                        f.write(f"  Red offset: {int(red_offset.get())}\\n")
                    saved_files.append(report_path)
                    
                    progress.stop()
                    status_label.config(text="Extraction complete!")
                    
                    # Load best result back into StampZ if requested
                    if save_to_stampz.get() and best_result_path:
                        try:
                            from utils.image_processor import load_image
                            new_image, new_metadata = load_image(str(best_result_path))
                            self.app.canvas.load_image(new_image)
                            self.app.current_file = str(best_result_path)
                            self.app.current_image_metadata = new_metadata
                            self.app.root.title(f"StampZ - {best_result_path.name}")
                            self.app.control_panel.update_current_filename(str(best_result_path))
                        except Exception as load_error:
                            print(f"Warning: Could not load result back into StampZ: {load_error}")
                    
                    # Show success dialog (also orphaned)
                    success_dialog = tk.Toplevel()
                    success_dialog.title("Extraction Complete")
                    success_dialog.geometry("400x300")
                    # success_dialog.transient(dialog)  # Commented out to allow moving
                    # success_dialog.grab_set()  # Commented out for flexibility
                    
                    # Center success dialog
                    success_dialog.update_idletasks()
                    x = (screen_width // 2) - (success_dialog.winfo_reqwidth() // 2)
                    y = (screen_height // 2) - (success_dialog.winfo_reqheight() // 2)
                    success_dialog.geometry(f"+{x}+{y}")
                    
                    success_text = (
                        f"✅ Black ink extraction completed successfully!\\n\\n"
                        f"Coverage: {analysis['coverage_percentage']:.1f}% of image\\n"
                        f"Black ink pixels: {analysis['cancellation_pixels']:,}\\n\\n"
                        f"📁 Results saved to:\\n{output_dir.name}/\\n\\n"
                        f"📋 Generated files ({len(saved_files)} total):\\n"
                        f"• Pure black cancellation (loaded in StampZ)\\n"
                        f"• Grayscale version (ink density preserved)\\n"
                        f"• Adaptive detection mask\\n"
                        f"• Original image (for comparison)\\n"
                        f"• Analysis report with settings"
                    )
                    
                    ttk.Label(
                        success_dialog,
                        text=success_text,
                        font=("Arial", 10),
                        justify="left"
                    ).pack(padx=20, pady=20)
                    
                    success_button_frame = ttk.Frame(success_dialog)
                    success_button_frame.pack(pady=10)
                    
                    def open_results_folder():
                        import subprocess
                        subprocess.run(['open', str(output_dir)])
                    
                    ttk.Button(
                        success_button_frame,
                        text="Open Results Folder",
                        command=open_results_folder
                    ).pack(side="left", padx=5)
                    
                    ttk.Button(
                        success_button_frame,
                        text="Close",
                        command=lambda: [success_dialog.destroy(), dialog.destroy()]
                    ).pack(side="right", padx=5)
                    
                    # Open results folder automatically if requested
                    if open_results.get():
                        open_results_folder()
                    
                except Exception as e:
                    progress.stop()
                    status_label.config(text="Error occurred")
                    
                    # Get detailed error information
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"DETAILED ERROR: {error_details}")
                    
                    messagebox.showerror(
                        "Extraction Error",
                        f"Failed to extract ink:\\n\\n{str(e)}\\n\\nFull error details printed to terminal."
                    )
            
            ttk.Button(
                button_frame,
                text="Extract Black Ink",
                command=extract_cancellation,
                style="Accent.TButton"
            ).pack(side="left", padx=5)
            
            ttk.Button(
                button_frame,
                text="Cancel",
                command=dialog.destroy
            ).pack(side="right", padx=5)
            
        except Exception as e:
            messagebox.showerror(
                "Black Ink Extractor Error",
                f"Failed to open Black Ink Extractor:\\n\\n{str(e)}"
            )
    
    def open_precision_measurements(self):
        """Open the Precision Measurements tool for detailed stamp measurements."""
        if not self.app.current_file:
            messagebox.showwarning(
                "No Image",
                "Please open a stamp image first.\\n\\n"
                "The Precision Measurements tool requires an image to measure."
            )
            return
            
        try:
            # Import the precision measurement tool
            from gui.precision_measurement_tool import PrecisionMeasurementTool
            
            # Create and show the precision measurement tool
            tool = PrecisionMeasurementTool(
                parent=self.root,
                image_path=self.app.current_file,
                main_app=self.app
            )
            
        except ImportError:
            messagebox.showerror(
                "Module Error",
                "Precision Measurements module not found.\\n\\n"
                "Please ensure the precision measurement tool is properly installed."
            )
        except Exception as e:
            messagebox.showerror(
                "Precision Measurements Error",
                f"Failed to open Precision Measurements tool:\\n\\n{str(e)}"
            )
    
    # Legacy Database Methods (fallbacks when DatabaseManager not available)
    
    def _legacy_save_imported_data_to_database(self, sample_set_name, import_result):
        """Legacy implementation of save_imported_data_to_database."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            
            print(f"DEBUG: Creating database: {sample_set_name}")
            db = ColorAnalysisDB(sample_set_name)
            
            saved_count = 0
            
            # Save centroid data first
            if import_result.centroid_data:
                print(f"DEBUG: Saving {len(import_result.centroid_data)} centroids")
                for cluster_id, centroid_row in import_result.centroid_data:
                    try:
                        # Extract centroid info from the row
                        centroid_x = float(centroid_row[8]) if centroid_row[8] and str(centroid_row[8]).strip() else None
                        centroid_y = float(centroid_row[9]) if centroid_row[9] and str(centroid_row[9]).strip() else None
                        centroid_z = float(centroid_row[10]) if centroid_row[10] and str(centroid_row[10]).strip() else None
                        sphere_color = centroid_row[11] if centroid_row[11] and str(centroid_row[11]).strip() else None
                        sphere_radius = float(centroid_row[12]) if centroid_row[12] and str(centroid_row[12]).strip() else None
                        
                        if centroid_x is not None and centroid_y is not None and centroid_z is not None:
                            success = db.insert_or_update_centroid_data(
                                cluster_id=cluster_id,
                                centroid_x=centroid_x,
                                centroid_y=centroid_y,
                                centroid_z=centroid_z,
                                sphere_color=sphere_color,
                                sphere_radius=sphere_radius,
                                marker='.',
                                color='blue'
                            )
                            if success:
                                saved_count += 1
                                print(f"DEBUG: Saved centroid {cluster_id}: ({centroid_x}, {centroid_y}, {centroid_z})")
                    except Exception as centroid_error:
                        print(f"DEBUG: Error saving centroid {cluster_id}: {centroid_error}")
            
            # Save regular data
            if import_result.data:
                print(f"DEBUG: Saving {len(import_result.data)} data rows")
                for i, row in enumerate(import_result.data):
                    try:
                        # Extract data from row
                        x_norm = float(row[0]) if row[0] and str(row[0]).strip() else 0.0
                        y_norm = float(row[1]) if row[1] and str(row[1]).strip() else 0.0
                        z_norm = float(row[2]) if row[2] and str(row[2]).strip() else 0.0
                        data_id = str(row[3]) if row[3] else f"Data_{i+1:03d}"
                        cluster = int(row[4]) if row[4] and str(row[4]).strip() else None
                        delta_e = float(row[5]) if row[5] and str(row[5]).strip() else None
                        marker = str(row[6]) if row[6] else '.'
                        color = str(row[7]) if row[7] else 'blue'
                        
                        # Parse DataID for database format
                        if '_pt' in data_id:
                            parts = data_id.split('_pt')
                            image_name = parts[0]
                            coord_point = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
                        else:
                            image_name = data_id
                            coord_point = 1
                        
                        # Insert as new measurement
                        success = db.insert_new_measurement(
                            image_name=image_name,
                            coordinate_point=coord_point,
                            x_pos=0.0,  # Position not relevant for imported data
                            y_pos=0.0,
                            l_value=x_norm,  # Store normalized values as L*a*b*
                            a_value=y_norm,
                            b_value=z_norm,
                            rgb_r=0.0, rgb_g=0.0, rgb_b=0.0,
                            cluster_id=cluster,
                            delta_e=delta_e,
                            centroid_x=float(row[8]) if row[8] and str(row[8]).strip() else None,
                            centroid_y=float(row[9]) if row[9] and str(row[9]).strip() else None,
                            centroid_z=float(row[10]) if row[10] and str(row[10]).strip() else None,
                            sphere_color=str(row[11]) if row[11] and str(row[11]).strip() else None,
                            sphere_radius=float(row[12]) if row[12] and str(row[12]).strip() else None,
                            marker=marker,
                            color=color,
                            sample_type='imported_data',
                            notes=f'Imported from external file'
                        )
                        
                        if success:
                            saved_count += 1
                            if i % 50 == 0:  # Progress indicator
                                print(f"DEBUG: Saved {i+1} data rows so far...")
                    
                    except Exception as row_error:
                        print(f"DEBUG: Error saving data row {i}: {row_error}")
                        continue
            
            print(f"DEBUG: Database save completed - {saved_count} total records saved")
            return saved_count
            
        except Exception as e:
            print(f"DEBUG: Error saving to database: {e}")
            logger.error(f"Error saving imported data to database: {e}")
            raise
    
    def _legacy_open_internal_viewer(self, sample_set_name):
        """Legacy implementation of open_internal_viewer."""
        try:
            from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
            
            # Use the new real-time Excel-like spreadsheet
            spreadsheet = RealtimePlot3DSheet(
                parent=self.root,
                sample_set_name=sample_set_name
            )
            
            logger.info(f"Opened real-time spreadsheet for: {sample_set_name}")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open real-time viewer:\\n\\n{str(e)}"
            )
    
    def _legacy_view_spreadsheet(self):
        """Legacy implementation of view_spreadsheet."""
        try:
            # Get sample set name from control panel
            sample_set_name = self.app.control_panel.sample_set_name.get().strip()
            
            # Check if we have a current sample set with analysis data
            has_current_analysis = False
            if sample_set_name:
                # Check if there's analysis data for the current sample set
                from utils.color_analysis_db import ColorAnalysisDB
                try:
                    db = ColorAnalysisDB(sample_set_name)
                    measurements = db.get_all_measurements()
                    has_current_analysis = bool(measurements)
                except:
                    has_current_analysis = False
            
            if has_current_analysis:
                # Case 1: We have current analysis data, show it directly
                print(f"DEBUG: Opening real-time spreadsheet for current sample set: {sample_set_name}")
                self._legacy_open_realtime_spreadsheet(sample_set_name)
            else:
                # Case 2: No current analysis, show selection dialog
                print("DEBUG: No current analysis found, showing selection dialog")
                messagebox.showinfo(
                    "No Analysis Data",
                    "No current analysis data found. Please run color analysis first."
                )
            
        except Exception as e:
            messagebox.showerror(
                "View Error",
                f"Failed to open spreadsheet view:\\n\\n{str(e)}"
            )
    
    def _legacy_open_realtime_spreadsheet(self, sample_set_name):
        """Legacy implementation of open_realtime_spreadsheet."""
        try:
            from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
            
            # Create the real-time spreadsheet
            sheet = RealtimePlot3DSheet(self.root, sample_set_name)
            logger.info(f"Opened real-time spreadsheet for: {sample_set_name}")
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open real-time spreadsheet:\\n\\n{str(e)}"
            )
    
    def _legacy_create_plot3d_worksheet_with_name(self, sample_set_name, populate=True):
        """Legacy implementation of create_plot3d_worksheet_with_name."""
        messagebox.showinfo(
            "Feature Unavailable",
            "Plot3D worksheet creation is not available.\\n\\n"
            "DatabaseManager component is missing."
        )
    
    def _legacy_create_and_launch_from_database(self, sample_set_name):
        """Legacy implementation of create_and_launch_from_database."""
        messagebox.showinfo(
            "Feature Unavailable",
            "Database launch functionality is not available.\\n\\n"
            "DatabaseManager component is missing."
        )
    
    def _legacy_export_data_for_plot3d(self, sample_set_name, measurements):
        """Legacy implementation of export_data_for_plot3d."""
        messagebox.showinfo(
            "Feature Unavailable",
            "Plot3D export functionality is not available.\\n\\n"
            "DatabaseManager component is missing."
        )
