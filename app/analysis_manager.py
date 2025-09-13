"""
Analysis Manager for StampZ Application

Handles all color analysis operations including spectral analysis, 
color library operations, and data export functionality.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import logging
import pandas as pd
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stampz_app import StampZApp

logger = logging.getLogger(__name__)


class AnalysisManager:
    """Manages color analysis operations for the StampZ application."""
    
    def __init__(self, app: 'StampZApp'):
        self.app = app
        self.root = app.root
        
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
                dialog = tk.Toplevel(self.root)
                dialog.title("Analysis Complete")
                
                dialog_width = 400
                dialog_height = 200
                
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                
                x = screen_width - dialog_width - 50
                y = (screen_height - dialog_height) // 2
                
                dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
                
                message = f"Successfully analyzed {len(measurements)} color samples from set '{actual_sample_set}'.\\n\\n"
                message += f"Color data has been saved to the database.\\n\\n"
                message += f"You can now view the spreadsheet or export the data."
                
                ttk.Label(dialog, text=message, wraplength=350, justify="left").pack(padx=20, pady=20)
                
                ttk.Button(dialog, text="OK", command=dialog.destroy).pack(pady=10)
                
                dialog.transient(self.root)
                dialog.grab_set()
                
                self.root.wait_window(dialog)
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
                
                # Create ODS template
                success = manager._create_simple_plot3d_template(filepath, sample_set_name)
                
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
            
            # Use WorksheetManager's new rigid ODS population method
            manager = WorksheetManager()
            success = manager._create_simple_plot3d_template(file_path, sample_set_name)
            
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
                row_data = {
                    'Xnorm': measurement.get('l_value', 0.0),
                    'Ynorm': measurement.get('a_value', 0.0),
                    'Znorm': measurement.get('b_value', 0.0),
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
            Radiobutton(source_frame, text="Load Existing File (.ods/.xlsx)", variable=source_var, value="load_file").pack(anchor="w")
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
            elif source_type == "external":
                # Import from external CSV and launch Plot_3D
                self._import_and_launch_csv()
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to perform Plot_3D action:\\n\\n{str(e)}"
            )
    
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
    
    def _create_file_from_existing(self, sample_set_name):
        """Create file template from existing StampZ data."""
        self.create_plot3d_worksheet_with_name(sample_set_name, populate=True)
    
    def create_plot3d_worksheet_with_name(self, sample_set_name, populate=True):
        """Create Plot_3D worksheet with specified sample set name."""
        # This is the refactored version of the original create_plot3d_worksheet method
        # but using the provided sample_set_name instead of guessing
        
        # Get save location
        default_filename = f"{sample_set_name}_Plot3D_{datetime.now().strftime('%Y%m%d')}"
        
        filepath = filedialog.asksaveasfilename(
            title="Create Plot_3D Worksheet",
            defaultextension=".ods",
            filetypes=[
                ('OpenDocument Spreadsheet', '*.ods'),
                ('Excel Workbook', '*.xlsx'),
                ('All files', '*.*')
            ],
            initialfile=default_filename,
            initialdir=os.path.expanduser("~/Desktop")
        )
        
        if filepath:
            # Rest of the creation logic...
            self._execute_worksheet_creation(filepath, sample_set_name, populate)
    
    def _execute_worksheet_creation(self, filepath, sample_set_name, populate):
        """Execute the actual worksheet creation logic."""
        try:
            from utils.worksheet_manager import WorksheetManager
            
            # Determine format from file extension
            file_ext = os.path.splitext(filepath)[1].lower()
            is_excel = file_ext == '.xlsx'
            
            # Create worksheet manager and template
            manager = WorksheetManager()
            
            if is_excel:
                # For Excel: Create formatted worksheet with validation
                success = manager.create_plot3d_worksheet(filepath, sample_set_name)
            else:
                # For ODS: Create simple template without advanced formatting
                success = manager._create_simple_plot3d_template(filepath, sample_set_name)
            
            if success and populate:
                if is_excel:
                    data_loaded = manager.load_stampz_data(sample_set_name)
                    if data_loaded:
                        manager.save_worksheet(filepath)
                else:
                    # For ODS, populate using simple method
                    self._populate_ods_template(filepath, sample_set_name)
            
            if success:
                format_info = "Excel format with validation" if is_excel else "OpenDocument format"
                data_info = "populated with existing data" if populate else "empty template ready for data entry"
                
                # Ask if user wants to launch Plot_3D with the created file
                launch_plot3d = messagebox.askyesno(
                    "Worksheet Created",
                    f"Plot_3D worksheet created successfully.\\n\\n"
                    f"File: {os.path.basename(filepath)}\\n"
                    f"Format: {format_info}\\n"
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
        filepath = self._get_save_path(sample_set_name)
        if filepath:
            self._create_template_with_data(filepath, sample_set_name)
            self._launch_plot3d_with_file(filepath)
    
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
        """Get save path for new template."""
        default_filename = f"{sample_set_name}_Plot3D_{datetime.now().strftime('%Y%m%d')}"
        
        return filedialog.asksaveasfilename(
            title="Save Plot_3D Template",
            defaultextension=".ods",
            filetypes=[
                ('OpenDocument Spreadsheet', '*.ods'),
                ('Excel Workbook', '*.xlsx')
            ],
            initialfile=default_filename,
            initialdir=os.path.expanduser("~/Desktop")
        )
    
    def _create_clean_template(self, filepath, sample_set_name):
        """Create clean template without placeholder data."""
        try:
            import pandas as pd
            
            # Create clean DataFrame with just headers
            from utils.worksheet_manager import WorksheetManager
            df = pd.DataFrame(columns=WorksheetManager.PLOT3D_COLUMNS)
            
            # Export to chosen format
            file_ext = os.path.splitext(filepath)[1].lower()
            if file_ext == '.xlsx':
                # Use openpyxl for Excel format
                df.to_excel(filepath, index=False)
            else:
                # Use ODS format
                df.to_excel(filepath, engine='odf', index=False)
            
            logger.info(f"Created clean template: {filepath}")
            
        except Exception as e:
            logger.error(f"Error creating clean template: {e}")
            messagebox.showerror("Error", f"Failed to create template: {e}")
    
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
            
            # Create DataFrame with real data
            data_rows = []
            for i, measurement in enumerate(measurements):
                row = {
                    'Xnorm': measurement.get('l_value', 0.0),
                    'Ynorm': measurement.get('a_value', 0.0),
                    'Znorm': measurement.get('b_value', 0.0),
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
    
    def export_plot3d_flexible(self):
        """Export current data to Plot_3D format with flexible format options."""
        try:
            # Get current sample set name
            sample_set_name = "StampZ_Analysis"  # Default
            if (hasattr(self.app, 'control_panel') and 
                hasattr(self.app.control_panel, 'sample_set_name') and 
                self.app.control_panel.sample_set_name.get().strip()):
                sample_set_name = self.app.control_panel.sample_set_name.get().strip()
            
            # Check if we have data
            try:
                from utils.color_analysis_db import ColorAnalysisDB
                db = ColorAnalysisDB(sample_set_name)
                measurements = db.get_all_measurements()
                
                if not measurements:
                    messagebox.showinfo(
                        "No Data",
                        f"No color analysis data found for sample set '{sample_set_name}'.\\n\\n"
                        "Please run color analysis first."
                    )
                    return
            except Exception as e:
                messagebox.showerror(
                    "Database Error",
                    f"Error accessing color analysis data:\\n\\n{str(e)}"
                )
                return
            
            # Get export format and location
            default_filename = f"{sample_set_name}_Plot3D_{datetime.now().strftime('%Y%m%d')}"
            
            filepath = filedialog.asksaveasfilename(
                title="Export Plot_3D Data",
                filetypes=[
                    ('Excel Workbook', '*.xlsx'),
                    ('OpenDocument Spreadsheet', '*.ods'), 
                    ('CSV files', '*.csv'),
                    ('All files', '*.*')
                ],
                initialfile=default_filename
            )
            
            if filepath:
                # Determine format from extension
                file_ext = os.path.splitext(filepath)[1].lower()
                format_map = {'.xlsx': 'xlsx', '.ods': 'ods', '.csv': 'csv'}
                export_format = format_map.get(file_ext, 'xlsx')
                
                from utils.worksheet_manager import WorksheetManager
                
                # Create worksheet with data
                manager = WorksheetManager()
                
                if export_format == 'xlsx':
                    # For Excel, create formatted worksheet
                    success = manager.create_plot3d_worksheet(filepath, sample_set_name)
                    if success:
                        manager.load_stampz_data(sample_set_name)
                        manager.save_worksheet(filepath)
                else:
                    # For ODS/CSV, create temporary Excel then export
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                    
                    success = manager.create_plot3d_worksheet(temp_path, sample_set_name)
                    if success:
                        manager.load_stampz_data(sample_set_name)
                        success = manager.export_to_format(filepath, export_format)
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                
                if success:
                    messagebox.showinfo(
                        "Export Successful",
                        f"Plot_3D data exported successfully.\\n\\n"
                        f"File: {os.path.basename(filepath)}\\n"
                        f"Format: {export_format.upper()}\\n"
                        f"Data: {len(measurements)} measurements from '{sample_set_name}'\\n\\n"
                        f"{'Formatted with validation (Excel only)' if export_format == 'xlsx' else 'Plain data format'}"
                    )
                else:
                    messagebox.showerror(
                        "Export Failed",
                        f"Failed to export Plot_3D data to {export_format.upper()} format."
                    )
                    
        except ImportError as e:
            if 'openpyxl' in str(e):
                messagebox.showerror(
                    "Missing Dependency",
                    "The Plot_3D export feature requires 'openpyxl'.\\n\\n"
                    "Please install it using: pip install openpyxl"
                )
            elif 'odfpy' in str(e):
                messagebox.showerror(
                    "Missing Dependency",
                    "ODS export requires 'odfpy'.\\n\\n"
                    "Please install it using: pip install odfpy"
                )
            else:
                messagebox.showerror(
                    "Import Error",
                    f"Missing required dependency:\\n\\n{str(e)}"
                )
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export Plot_3D data:\\n\\n{str(e)}"
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
        try:
            from utils.direct_plot3d_exporter import DirectPlot3DExporter
            
            exporter = DirectPlot3DExporter()
            created_files = exporter.export_to_plot3d(sample_set_name)
            
            if created_files:
                # Show success message with all created files
                files_list = "\\n".join([f"  - {os.path.basename(f)}" for f in created_files])
                messagebox.showinfo(
                    "Export Complete",
                    f"Successfully exported Plot_3D data for sample set '{sample_set_name}'.\\n\\n"
                    f"Created {len(created_files)} file(s):\\n{files_list}\\n\\n"
                    f"These files can be loaded in Plot_3D for 3D color space analysis."
                )
            else:
                messagebox.showerror(
                    "Export Failed",
                    f"Failed to export Plot_3D data for sample set '{sample_set_name}'.\\n\\n"
                    f"No files were created. Please check the sample set has valid data."
                )
                
        except ImportError as e:
            messagebox.showerror(
                "Missing Component",
                f"Plot3D export functionality not available:\\n\\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export for Plot3D:\\n\\n{str(e)}"
            )

    def view_spreadsheet(self):
        """Open real-time spreadsheet view of color analysis data.
        
        Logic:
        1. If there's a current sample set with analysis data -> show that specific analysis
        2. If there's no current analysis -> show dialog to choose which data to view
        """
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
                self._open_realtime_spreadsheet(sample_set_name)
            else:
                # Case 2: No current analysis, show selection dialog
                print("DEBUG: No current analysis found, showing selection dialog")
                self._show_realtime_data_selection_dialog()
            
        except Exception as e:
            messagebox.showerror(
                "View Error",
                f"Failed to open spreadsheet view:\\n\\n{str(e)}"
            )

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
