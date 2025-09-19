"""
Database Manager for StampZ Analysis Manager

This module handles all database-related operations including:
- Data import/export to databases
- Real-time spreadsheet viewing
- Plot3D integration and worksheet creation
- Database querying and management

Extracted from analysis_manager.py to improve code organization and maintainability.
"""

import os
import sys
import logging
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all database operations for StampZ analysis system.
    
    This class handles:
    - Saving imported data to databases
    - Opening real-time spreadsheet viewers
    - Database querying and management
    - Plot3D integration and worksheet creation
    """
    
    def __init__(self, app, root):
        """
        Initialize DatabaseManager.
        
        Args:
            app: Main StampZ application instance
            root: Tkinter root window
        """
        self.app = app
        self.root = root
        
    def save_imported_data_to_database(self, sample_set_name, import_result):
        """Save imported data directly to database without UI.
        
        Args:
            sample_set_name: Name for the new database
            import_result: ImportResult with data to save
            
        Returns:
            int: Number of records saved
        """
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
    
    def open_internal_viewer(self, sample_set_name):
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
    
    def open_realtime_spreadsheet(self, sample_set_name):
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
        
        message = f"User-editable spreadsheet opened for '{sample_set_name}'.\\n\\n" + \
                  f"Features:\\n" + \
                  f"• Pink cells: Protected areas (no manual entry)\\n" + \
                  f"• Colored columns: G=Salmon, H=Yellow, L=Yellow\\n" + \
                  f"• Auto-refresh: New StampZ data appears automatically\\n" + \
                  f"• Direct Plot_3D integration (no external files needed!)\\n\\n" + \
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
                self.open_realtime_spreadsheet(sample_set_name)
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
                self.open_realtime_spreadsheet(sample_set_to_open)
            
        except Exception as e:
            messagebox.showerror(
                "Dialog Error",
                f"Failed to show data selection dialog:\\n\\n{str(e)}"
            )
    
    # Plot3D Integration Methods
    
    def create_plot3d_worksheet_with_name(self, sample_set_name, populate=True):
        """Create Plot_3D worksheet with specified sample set name."""
        # Get save location
        default_filename = f"{sample_set_name}_Plot3D_{datetime.now().strftime('%Y%m%d')}"
        
        filepath = filedialog.asksaveasfilename(
            title="Create Plot_3D Worksheet",
            defaultextension=".ods",
            filetypes=[
                ('OpenDocument Spreadsheet', '*.ods')
            ],
            initialfile=default_filename,
            initialdir=os.path.expanduser("~/Desktop")
        )
        
        if filepath:
            # Execute the creation logic
            self._execute_worksheet_creation(filepath, sample_set_name, populate)
    
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
                    self.launch_plot3d_with_file(filepath)
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
    
    def launch_plot3d_with_file(self, file_path):
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
    
    def create_and_launch_new_template(self):
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
                self.launch_plot3d_with_file(filepath)
    
    def create_and_launch_from_database(self, sample_set_name):
        """Create template from database and launch Plot_3D."""
        filepath = self._get_save_path(sample_set_name)
        if filepath:
            self._create_template_with_data(filepath, sample_set_name)
            self.launch_plot3d_with_file(filepath)
    
    def load_existing_file_in_plot3d(self):
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
            self.launch_plot3d_with_file(filepath)
    
    def import_and_launch_csv(self):
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
                self.launch_plot3d_with_file(output_file)
    
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
                template_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'templates', 'plot3d', 'Plot3D_Template.ods')
            
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
    
    def _populate_ods_template(self, filepath, sample_set_name):
        """Populate ODS template with data from database."""
        # This method would populate the template with actual data
        # Implementation would depend on the specific template format
        try:
            logger.info(f"Populating template {filepath} with data from {sample_set_name}")
            # Add actual population logic here if needed
        except Exception as e:
            logger.error(f"Error populating template: {e}")
    
    def _create_template_with_data(self, filepath, sample_set_name):
        """Create template populated with real StampZ data."""
        try:
            from utils.color_analysis_db import ColorAnalysisDB
            import pandas as pd
            
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
            import pandas as pd
            
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
                success = False
                
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
    
    def export_data_for_plot3d(self, sample_set_name, measurements):
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
