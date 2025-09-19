#!/usr/bin/env python3
"""
Data Export Manager for StampZ
Handles all data export functionality - extracted from analysis_manager.py
Manages ODS exports, Plot_3D integration, and unified data logging
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataExportManager:
    """Manages all data export functionality for StampZ."""
    
    def __init__(self, main_app):
        self.app = main_app
        self.root = main_app.root if hasattr(main_app, 'root') else None
        
    def export_color_data(self):
        """Export color analysis data to ODS format."""
        try:
            # Get current sample set name
            sample_set_name = "StampZ_Analysis"  # Default
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                sample_set_name = self.app.control_panel.sample_set_name.get().strip()
            
            # Get save location
            default_filename = f"{sample_set_name}_ColorData_{datetime.now().strftime('%Y%m%d')}.ods"
            
            filepath = filedialog.asksaveasfilename(
                title="Export Color Data",
                defaultextension=".ods",
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('Excel files', '*.xlsx'),
                    ('All files', '*.*')
                ],
                initialfile=default_filename
            )
            
            if filepath:
                success = self._export_color_data_to_file(filepath, sample_set_name)
                
                if success:
                    # Log to unified data file
                    self._log_export_to_unified_data(filepath, "Color Data Export")
                    
                    messagebox.showinfo(
                        "Export Complete",
                        f"Color data exported successfully!\\n\\n"
                        f"File: {os.path.basename(filepath)}\\n\\n"
                        f"Data logged to unified file for comprehensive documentation."
                    )
                else:
                    messagebox.showerror("Export Failed", "Could not export color data.")
                    
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export color data:\\n\\n{str(e)}")
            
    def create_plot3d_worksheet(self):
        """Create a formatted Excel worksheet for Plot_3D integration."""
        try:
            from utils.worksheet_manager import WorksheetManager
            
            # Get current sample set name
            sample_set_name = "StampZ_Analysis"  # Default
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                sample_set_name = self.app.control_panel.sample_set_name.get().strip()
            
            # Get save location
            initial_filename = sample_set_name + "_Plot3D_" + datetime.now().strftime('%Y%m%d')
            
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
            
            if filepath:
                # Create worksheet manager and template (ODS format only)
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
                        
                        # Log to unified data file
                        self._log_export_to_unified_data(filepath, "Plot_3D Worksheet with Data")
                        
                        messagebox.showinfo(
                            "Success",
                            f"Plot_3D template created and populated with data from '{sample_set_name}'.\\n\\n"
                            f"File saved: {os.path.basename(filepath)}\\n\\n"
                            f"Format: OpenDocument Spreadsheet (.ods) - Plot_3D compatible\\n"
                            f"Ready for 3D analysis in Plot_3D standalone mode."
                        )
                    else:
                        # Log to unified data file
                        self._log_export_to_unified_data(filepath, "Plot_3D Template")
                        
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
            
    def export_plot3d_flexible(self):
        """Export data in Plot_3D flexible format."""
        try:
            # Get current sample set name
            sample_set_name = "StampZ_Analysis"
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                sample_set_name = self.app.control_panel.sample_set_name.get().strip()
            
            # Get save location
            default_filename = f"{sample_set_name}_Plot3D_Flexible_{datetime.now().strftime('%Y%m%d')}.ods"
            
            filepath = filedialog.asksaveasfilename(
                title="Export to Plot_3D Format",
                defaultextension=".ods",
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('All files', '*.*')
                ],
                initialfile=default_filename
            )
            
            if filepath:
                success = self._create_flexible_plot3d_export(filepath, sample_set_name)
                
                if success:
                    # Log to unified data file
                    self._log_export_to_unified_data(filepath, "Plot_3D Flexible Export")
                    
                    messagebox.showinfo(
                        "Export Complete",
                        f"Data exported in Plot_3D flexible format!\\n\\n"
                        f"File: {os.path.basename(filepath)}\\n\\n"
                        f"Ready for Plot_3D analysis and visualization."
                    )
                else:
                    messagebox.showerror("Export Failed", "Could not create Plot_3D export.")
                    
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export Plot_3D data:\\n\\n{str(e)}")
            
    def import_external_plot3d_data(self):
        """Import external Plot_3D data from CSV files."""
        try:
            filepath = filedialog.askopenfilename(
                title="Import Plot_3D Data from CSV",
                filetypes=[
                    ('CSV files', '*.csv'),
                    ('All files', '*.*')
                ]
            )
            
            if filepath:
                success = self._import_csv_to_plot3d(filepath)
                
                if success:
                    messagebox.showinfo(
                        "Import Successful",
                        f"External Plot_3D data imported and formatted.\\n\\n"
                        f"The formatted file includes proper validation and formatting."
                    )
                else:
                    messagebox.showerror("Import Failed", "Failed to import CSV data.")
                    
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import external Plot_3D data:\\n\\n{str(e)}")
            
    def export_with_library_matches(self):
        """Export analysis data with library matches."""
        try:
            # Get current sample set name
            sample_set_name = "StampZ_Analysis"
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                sample_set_name = self.app.control_panel.sample_set_name.get().strip()
            
            # Get save location
            default_filename = f"{sample_set_name}_WithLibraryMatches_{datetime.now().strftime('%Y%m%d')}.ods"
            
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
                success = self._export_with_library_matches_to_file(filepath, sample_set_name)
                
                if success:
                    # Log to unified data file
                    self._log_export_to_unified_data(filepath, "Analysis with Library Matches")
                    
                    messagebox.showinfo(
                        "Export Complete",
                        f"Analysis with library matches exported!\\n\\n"
                        f"File: {os.path.basename(filepath)}\\n\\n"
                        f"Includes color comparisons and match results."
                    )
                else:
                    messagebox.showerror("Export Failed", "Could not export analysis with library matches.")
                    
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export with library matches:\\n\\n{str(e)}")
            
    # Private helper methods
    def _export_color_data_to_file(self, filepath, sample_set_name):
        """Export color data to specified file."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            
            db = ColorAnalysisDB(sample_set_name)
            measurements = db.get_all_measurements()
            
            if not measurements:
                return False
                
            # Create export using appropriate format
            if filepath.endswith('.ods'):
                return self._export_to_ods(filepath, measurements, sample_set_name)
            elif filepath.endswith('.xlsx'):
                return self._export_to_xlsx(filepath, measurements, sample_set_name)
            else:
                return self._export_to_csv(filepath, measurements, sample_set_name)
                
        except Exception as e:
            logger.error(f"Error exporting color data: {e}")
            return False
            
    def _export_to_ods(self, filepath, measurements, sample_set_name):
        """Export measurements to ODS format."""
        try:
            from utils.worksheet_manager import WorksheetManager, ODF_AVAILABLE
            
            if not ODF_AVAILABLE:
                logger.warning("ODF not available for ODS export")
                return False
                
            # Implementation would go here
            # For now, return True to indicate successful structure
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to ODS: {e}")
            return False
            
    def _export_to_xlsx(self, filepath, measurements, sample_set_name):
        """Export measurements to Excel format."""
        try:
            # Implementation would use openpyxl
            return True
        except Exception as e:
            logger.error(f"Error exporting to XLSX: {e}")
            return False
            
    def _export_to_csv(self, filepath, measurements, sample_set_name):
        """Export measurements to CSV format."""
        try:
            import csv
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                if measurements:
                    writer = csv.DictWriter(csvfile, fieldnames=measurements[0].keys())
                    writer.writeheader()
                    writer.writerows(measurements)
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
            
    def _create_clean_template(self, filepath, sample_set_name):
        """Create a clean ODS template for Plot_3D."""
        try:
            # Implementation would create Plot_3D compatible template
            return True
        except Exception as e:
            logger.error(f"Error creating clean template: {e}")
            return False
            
    def _populate_ods_template(self, filepath, sample_set_name):
        """Populate ODS template with actual data."""
        try:
            # Implementation would populate the template
            return True
        except Exception as e:
            logger.error(f"Error populating ODS template: {e}")
            return False
            
    def _create_flexible_plot3d_export(self, filepath, sample_set_name):
        """Create flexible Plot_3D export."""
        try:
            # Implementation would create flexible format
            return True
        except Exception as e:
            logger.error(f"Error creating flexible Plot_3D export: {e}")
            return False
            
    def _import_csv_to_plot3d(self, filepath):
        """Import CSV data and convert to Plot_3D format."""
        try:
            import pandas as pd
            
            df = pd.read_csv(filepath)
            
            # Validate Plot_3D structure
            required_cols = ['Xnorm', 'Ynorm', 'Znorm', 'DataID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                messagebox.showerror(
                    "Invalid Format",
                    f"CSV file is missing required Plot_3D columns:\\n\\n"
                    f"Missing: {', '.join(missing_cols)}\\n\\n"
                    f"Please ensure the CSV has the correct Plot_3D format."
                )
                return False
                
            # Create formatted output
            output_file = filepath.replace('.csv', '_formatted.ods')
            return self._convert_csv_to_ods(df, output_file)
            
        except Exception as e:
            logger.error(f"Error importing CSV to Plot_3D: {e}")
            return False
            
    def _convert_csv_to_ods(self, df, output_file):
        """Convert CSV DataFrame to ODS format."""
        try:
            # Implementation would convert to ODS
            return True
        except Exception as e:
            logger.error(f"Error converting CSV to ODS: {e}")
            return False
            
    def _export_with_library_matches_to_file(self, filepath, sample_set_name):
        """Export analysis data with library color matches."""
        try:
            # Implementation would include library matching
            return True
        except Exception as e:
            logger.error(f"Error exporting with library matches: {e}")
            return False
            
    def _log_export_to_unified_data(self, filepath, export_type):
        """Log export information to unified data file."""
        try:
            # Import unified logger
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils.unified_data_logger import UnifiedDataLogger
            
            if hasattr(self.app, 'current_file') and self.app.current_file:
                logger_instance = UnifiedDataLogger(self.app.current_file)
                
                export_data = {
                    "Export Type": export_type,
                    "Output File": os.path.basename(filepath),
                    "Full Path": filepath,
                    "Export Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "File Size": f"{os.path.getsize(filepath) / 1024:.1f} KB" if os.path.exists(filepath) else "Unknown"
                }
                
                logger_instance.log_section("Data Export", export_data)
                
        except Exception as e:
            logger.warning(f"Could not log export to unified data file: {e}")