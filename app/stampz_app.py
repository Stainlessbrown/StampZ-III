"""
Core StampZ Application Class

Main application window that coordinates all managers and UI components.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import logging

from gui.canvas import CropCanvas, ShapeType, ToolMode
from gui.controls_reorganized import ReorganizedControlPanel as ControlPanel
from utils.recent_files import RecentFilesManager
from utils.image_straightener import StraighteningTool
from utils.path_utils import ensure_data_directories
from utils.color_data_bridge import ColorDataBridge
from utils.advanced_color_plots import TernaryPlotter, QuaternaryPlotter
import webbrowser
import datetime

from .menu_manager import MenuManager
from .file_manager import FileManager
from .analysis_manager import AnalysisManager
from .settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class StampZApp:
    """Main application window for StampZ."""
    
    def __init__(self, root: tk.Tk):
        # Ensure data directories exist first
        ensure_data_directories()
        
        self.root = root
        self.root.title("StampZ-III")
        self._set_application_name()
        
        try:
            self.root.tk.call('wm', 'class', self.root, 'StampZ-III')
        except:
            pass
            
        # Initialize managers
        self.menu_manager = MenuManager(self)
        self.file_manager = FileManager(self)
        self.analysis_manager = AnalysisManager(self)
        self.settings_manager = SettingsManager(self)
        
        # Use the environment variable for recent files directory
        stampz_data_dir = os.getenv('STAMPZ_DATA_DIR')
        if stampz_data_dir:
            recent_dir = os.path.join(stampz_data_dir, 'recent')
            self.recent_files = RecentFilesManager(recent_dir=recent_dir)
        else:
            self.recent_files = RecentFilesManager()
            
        # Initialize window and UI
        self._setup_window()
        self.menu_manager.create_menu()
        self._create_widgets()
        self._bind_shortcuts()
        
        # Initialize state
        self.current_file = None
        self.current_image_metadata = None
        
        # Setup UI connections
        self._setup_ui_connections()
        self._apply_default_settings()
        
        # Check optional dependencies at startup
        self.settings_manager.check_dependencies()
        
        # Initialize advanced color components
        self._color_bridge = ColorDataBridge()
        self._ternary_plotter = TernaryPlotter()
        self._quaternary_plotter = QuaternaryPlotter()

    def _set_application_name(self):
        """Set the application name for the system."""
        try:
            self.root.tk.call('tk', 'appname', 'StampZ-III')
        except:
            pass

    def _setup_window(self):
        """Configure the main window size and position."""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Account for dock/taskbar - be very conservative for smaller monitors
        # Base sizing on 13" MacBooks (1440x900) and smaller screens first
        if screen_height <= 768:  # 13" laptops and smaller (1366x768)
            window_height = int(screen_height * 0.55)  # Very aggressive - max 422px on 768px screen
        elif screen_height <= 900:  # 13" MacBooks (1440x900)
            window_height = int(screen_height * 0.58)  # Aggressive - max 522px on 900px screen
        elif screen_height <= 1080:  # 21.5" iMacs and similar (1920x1080)
            window_height = int(screen_height * 0.62)  # Conservative - max 670px on 1080px screen
        else:  # Larger monitors
            window_height = int(screen_height * 0.68)  # More room for large displays
        
        # For width, we can be more generous since horizontal space is less constrained
        window_width = int(screen_width * 0.85)
        
        # Position window with some top margin to account for menu bars
        x_position = (screen_width - window_width) // 2
        y_position = max(50, (screen_height - window_height) // 2)  # At least 50px from top
        
        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
        # Set minimum size to ensure all UI elements can be visible, but smaller for small monitors
        self.root.minsize(900, 780)  # Reduced to fit on 1366x768 and smaller screens
        self.root.protocol("WM_DELETE_WINDOW", self.file_manager.quit_app)
        self.root.resizable(True, True)
        
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

    def _create_widgets(self):
        """Create the main UI widgets."""
        self.canvas = CropCanvas(self.main_container, bg='white', width=800, height=600)
        self.canvas.set_dimensions_callback(self._update_crop_dimensions)
        self.straightening_tool = StraighteningTool()
        self.canvas.main_app = self
        
        self.control_panel = ControlPanel(
            self.main_container,
            on_reset=self.reset_view,
            on_open=self.file_manager.open_image,
            on_save=self.file_manager.save_image,
            on_clear=self.file_manager.clear_image,
            on_quit=self.file_manager.quit_app,
            on_vertex_count_change=self._handle_vertex_count_change,
            on_fit_to_window=self.fit_to_window,
            on_transparency_change=self._handle_transparency_change,
            on_shape_type_change=self._handle_shape_type_change,
            on_tool_mode_change=self._handle_tool_mode_change,
        )
        
        self.control_panel.main_app = self
        self.control_panel.canvas = self.canvas
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

    def _setup_ui_connections(self):
        """Setup connections between UI components."""
        self.control_panel.on_ruler_toggle = self._handle_ruler_toggle
        self.control_panel.on_grid_toggle = self._handle_grid_toggle
        self.control_panel.tool_mode.set("view")
        self._handle_tool_mode_change("view")
        self._handle_shape_type_change(ShapeType.POLYGON)
        self.control_panel.on_line_color_change = self._handle_line_color_change
        self.canvas.set_coordinate_callback(self.control_panel.update_mouse_coordinates)

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind('<Control-o>', lambda e: self.file_manager.open_image())
        self.root.bind('<Control-s>', lambda e: self.file_manager.save_image())
        self.root.bind('<Control-q>', lambda e: self.file_manager.quit_app())
        self.root.bind('<Control-w>', lambda e: self.file_manager.clear_image())
        self.root.bind('<Control-r>', lambda e: self.reset_view())
        self.root.bind('<Escape>', lambda e: self.reset_view())
        self.root.bind('<F11>', lambda e: self.fit_to_window())
        self.root.bind('<Control-p>', lambda e: self.measure_perforations())
        self.root.bind('<plus>', lambda e: self._adjust_vertex_count(1))
        self.root.bind('<minus>', lambda e: self._adjust_vertex_count(-1))

    # Delegated methods to managers
    def open_image(self, filename=None):
        """Delegate to file manager."""
        return self.file_manager.open_image(filename)
        
    def save_image(self):
        """Delegate to file manager."""
        return self.file_manager.save_image()
        
    def clear_image(self):
        """Delegate to file manager."""
        return self.file_manager.clear_image()
        
    def quit_app(self):
        """Delegate to file manager."""
        return self.file_manager.quit_app()
        
    def export_color_data(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.export_color_data()
        
    def open_color_library(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.open_color_library()
        
    def compare_sample_to_library(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.compare_sample_to_library()
        
    def create_standard_libraries(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.create_standard_libraries()
        
    def export_with_library_matches(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.export_with_library_matches()
    
    def open_spectral_analysis(self):
        """Delegate to analysis manager."""
        try:
            return self.analysis_manager.open_spectral_analysis()
        except AttributeError:
            messagebox.showinfo("Feature Not Available", "Spectral analysis is not currently implemented.")
    
    def open_black_ink_extractor(self):
        """Delegate to analysis manager."""
        try:
            return self.analysis_manager.open_black_ink_extractor()
        except AttributeError:
            messagebox.showinfo("Feature Not Available", "Black ink extractor is not currently implemented.")
    
    def open_database_viewer(self):
        """Delegate to analysis manager."""
        try:
            return self.analysis_manager.open_database_viewer()
        except AttributeError:
            messagebox.showinfo("Feature Not Available", "Database viewer is not currently implemented.")
    
    def measure_perforations(self):
        """Delegate to analysis manager."""
        try:
            return self.analysis_manager.measure_perforations()
        except AttributeError:
            messagebox.showinfo("Feature Not Available", "Perforation measurement is not currently implemented.")
    
    def open_precision_measurements(self):
        """Delegate to analysis manager."""
        try:
            return self.analysis_manager.open_precision_measurements()
        except AttributeError:
            messagebox.showinfo("Feature Not Available", "Precision measurements are not currently implemented.")
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About StampZ-III",
            "StampZ-III\n\nPhilatelic Image Analysis Tool\n\nFeatures:\n• Color Analysis\n• RGB Ternary Plots\n• Plot_3D Integration\n• Perforation Measurement\n\nVersion: Development Build"
        )
    
    def open_preferences(self):
        """Open preferences dialog."""
        messagebox.showinfo("Preferences", "Preferences dialog not yet implemented.")
    
    def reset_view(self):
        """Reset the canvas view."""
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.reset_view()
    
    def fit_to_window(self):
        """Fit image to window size."""
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.fit_to_window()
        
    def open_advanced_color_visualization(self):
        """Open RGB ternary analysis dialog with optional Plot_3D integration."""
        try:
            # Gather available sample sets
            sample_sets = self._color_bridge.get_available_sample_sets()
            if not sample_sets:
                messagebox.showinfo(
                    "Advanced Color Visualization",
                    "No color analysis sample sets found.\n\nRun color analysis first to generate databases."
                )
                return
            
            # Create selection dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Ternary Color Analysis")
            dialog.geometry("420x420")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text="Select a sample set for RGB ternary analysis:", font=(None, 11, 'bold')).pack(pady=(12,6))
            
            listbox = tk.Listbox(dialog, height=8)
            for s in sample_sets:
                listbox.insert(tk.END, s)
            listbox.selection_set(0)
            listbox.pack(fill=tk.BOTH, expand=True, padx=12)
            
            options_frame = ttk.Frame(dialog)
            options_frame.pack(fill=tk.X, padx=12, pady=8)
            
            hull_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(options_frame, text="Show convex hull on ternary plot", variable=hull_var).pack(anchor='w')
            
            plot3d_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(options_frame, text="Also launch Plot_3D for L*a*b* analysis", variable=plot3d_var).pack(anchor='w')
            
            ttk.Separator(options_frame, orient='horizontal').pack(fill=tk.X, pady=(8,8))
            
            # Output format options
            output_frame = ttk.LabelFrame(options_frame, text="Display Options", padding=10)
            output_frame.pack(fill=tk.X, pady=(5,5))
            
            display_var = tk.StringVar(value="interactive")
            ttk.Radiobutton(output_frame, text="📊 Interactive Window (Plot_3D-style with K-means)", 
                           variable=display_var, value="interactive").pack(anchor='w')
            ttk.Radiobutton(output_frame, text="💾 Save Plot as File (PNG in output folder)", 
                           variable=display_var, value="file").pack(anchor='w')
            
            btns = ttk.Frame(dialog)
            btns.pack(fill=tk.X, padx=12, pady=12)
            
            def on_cancel():
                dialog.destroy()
            
            def on_run():
                selection = listbox.curselection()
                if not selection:
                    messagebox.showwarning("Selection Required", "Please select a sample set.")
                    return
                selected_set = sample_sets[selection[0]]
                display_mode = display_var.get()
                dialog.destroy()
                self._run_ternary_analysis(
                    selected_set, 
                    show_hull=hull_var.get(),
                    launch_plot3d=plot3d_var.get(),
                    display_mode=display_mode
                )
            
            ttk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)
            ttk.Button(btns, text="Run", command=on_run).pack(side=tk.RIGHT, padx=(0,8))
        except Exception as e:
            logger.exception("Failed to open advanced color visualization dialog")
            messagebox.showerror("Advanced Color Visualization", f"An error occurred:\n\n{e}")
    
    def _run_ternary_analysis(self, sample_set: str, show_hull: bool = True, launch_plot3d: bool = False, display_mode: str = "interactive"):
        """Run ternary analysis with optional Plot_3D integration."""
        try:
            # Load color points
            color_points = self._color_bridge.load_color_points_from_database(sample_set)
            if not color_points:
                messagebox.showwarning("Advanced Color Visualization", "No color points were loaded.")
                return
            
            if display_mode == "interactive":
                # Open interactive ternary window (Plot_3D style)
                try:
                    from gui.ternary_plot_window import TernaryPlotWindow
                    ternary_window = TernaryPlotWindow(
                        parent=self.root,
                        sample_set_name=sample_set,
                        color_points=color_points
                    )
                    # Set hull option
                    ternary_window.show_hull.set(show_hull)
                    ternary_window._refresh_plot()
                    
                    msg = f"Interactive ternary window opened for '{sample_set}'\n\n📊 {len(color_points)} colors loaded\n🔧 K-means clustering available\n📈 Interactive analysis tools"
                    if launch_plot3d:
                        try:
                            self._launch_plot3d_for_sample_set(sample_set)
                            msg += "\n🔬 Plot_3D also launched for L*a*b* analysis"
                        except Exception as e:
                            logger.warning(f"Plot_3D launch failed: {e}")
                    
                    messagebox.showinfo("Interactive Ternary Window", msg)
                    
                except Exception as e:
                    logger.exception("Failed to open interactive window")
                    messagebox.showerror("Window Error", f"Failed to open interactive window: {e}")
                    
            else:  # file mode
                # Create and save ternary plot file
                out_dir = os.path.join(os.getcwd(), 'output')
                os.makedirs(out_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                ternary_path = os.path.join(out_dir, f"ternary_plot_{sample_set}_{timestamp}.png")
                
                fig2d = self._ternary_plotter.create_ternary_plot(
                    color_points,
                    title=f"RGB Ternary Analysis - {sample_set} ({len(color_points)} colors)",
                    color_by="auto",
                    show_hull=show_hull
                )
                self._ternary_plotter.save_plot(fig2d, ternary_path)
                
                # Launch Plot_3D if requested
                plot3d_launched = False
                if launch_plot3d:
                    try:
                        self._launch_plot3d_for_sample_set(sample_set)
                        plot3d_launched = True
                    except Exception as e:
                        logger.warning(f"Plot_3D launch failed: {e}")
                
                # Show completion message
                msg = (
                    f"Ternary analysis completed for '{sample_set}'\\n\\n"
                    f"📊 RGB Ternary Plot: {os.path.basename(ternary_path)}\\n"
                    f"📈 Colors analyzed: {len(color_points)}\\n"
                )
                
                if plot3d_launched:
                    msg += f"🔬 Plot_3D launched for L*a*b* analysis\\n"
                
                if messagebox.askyesno("Ternary Analysis Complete", msg + "\\nOpen output folder now?"):
                    webbrowser.open(f"file://{out_dir}")
        except Exception as e:
            logger.exception("Failed during ternary analysis run")
            messagebox.showerror("Ternary Analysis", f"An error occurred during analysis:\n\n{e}")
    
    def _launch_plot3d_for_sample_set(self, sample_set: str):
        """Launch Plot_3D with the existing database for the sample set."""
        try:
            # Use existing Plot_3D integration to open the sample set
            from plot3d.standalone_plot3d import main as plot3d_main
            import threading
            
            # Get the database file path
            from utils.path_utils import get_color_analysis_dir
            analysis_dir = get_color_analysis_dir()
            db_path = os.path.join(analysis_dir, f"{sample_set}.db")
            
            if not os.path.exists(db_path):
                raise Exception(f"Database not found: {db_path}")
            
            # Launch Plot_3D in a separate thread
            def launch_plot3d():
                try:
                    # Let Plot_3D handle the database directly
                    plot3d_main()
                except Exception as e:
                    logger.warning(f"Plot_3D launch error: {e}")
            
            thread = threading.Thread(target=launch_plot3d, daemon=True)
            thread.start()
            
        except Exception as e:
            raise Exception(f"Failed to launch Plot_3D: {e}")
    
    def _export_to_plot3d(self, color_points, sample_set: str):
        """Delegate to analysis manager."""
        return self.analysis_manager.open_spectral_analysis()
        
    def open_3d_analysis(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.open_3d_analysis()
        
    def show_about(self):
        """Delegate to settings manager."""
        return self.settings_manager.show_about()
        
    def open_preferences(self):
        """Delegate to settings manager."""
        return self.settings_manager.open_preferences()
        
    def open_black_ink_extractor(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.open_black_ink_extractor()
        
    def open_precision_measurements(self):
        """Delegate to analysis manager."""
        return self.analysis_manager.open_precision_measurements()
        
    # Measurement Methods
    def measure_perforations(self):
        """Launch perforation measurement dialog."""
        if hasattr(self.analysis_manager, 'measurement_manager') and self.analysis_manager.measurement_manager:
            return self.analysis_manager.measurement_manager.measure_perforations()
        else:
            # Fallback - direct method call
            try:
                from gui.perforation_ui import PerforationMeasurementDialog
                import numpy as np
                import cv2
                
                # Get image data
                image_array = None
                image_filename = ""
                
                if hasattr(self, 'canvas') and self.canvas.original_image:
                    from PIL import Image
                    pil_image = self.canvas.original_image
                    image_array = np.array(pil_image)
                    if len(image_array.shape) == 3:
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
                    if hasattr(self, 'current_file') and self.current_file:
                        image_filename = self.current_file
                
                dialog = PerforationMeasurementDialog(
                    parent=self.root,
                    image_array=image_array,
                    image_filename=image_filename
                )
                
            except ImportError as e:
                from tkinter import messagebox
                messagebox.showerror(
                    "Feature Not Available",
                    f"Perforation measurement requires additional dependencies.\n"
                    f"Please install: pip install opencv-python\n\n"
                    f"Error: {str(e)}"
                )
            except Exception as e:
                from tkinter import messagebox
                messagebox.showerror(
                    "Error",
                    f"Failed to launch perforation measurement:\n{str(e)}"
                )
        
    def export_individual_to_unified_logger(self):
        """Export individual color measurements to unified data logger."""
        return self._export_to_unified_logger(export_type="individual")
        
    def export_averaged_to_unified_logger(self):
        """Export averaged color measurement to unified data logger."""
        return self._export_to_unified_logger(export_type="averaged")
        
    def export_both_to_unified_logger(self):
        """Export both individual and averaged measurements to unified data logger."""
        return self._export_to_unified_logger(export_type="both")
        
    def open_database_viewer(self):
        """Open the database viewer window."""
        try:
            from gui.database_viewer import DatabaseViewer
            DatabaseViewer(parent=self.root)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to open database viewer:\\n\\n{str(e)}")
            
    def view_spreadsheet(self):
        """Delegate to analysis manager for spreadsheet viewing."""
        return self.analysis_manager.view_spreadsheet()
        
    def _view_spreadsheet(self):
        """Legacy method name for compatibility with control panel."""
        return self.analysis_manager.view_spreadsheet()
        
    def _analyze_colors(self):
        """Legacy method name for compatibility with control panel."""
        return self.analysis_manager.analyze_colors()
        
    def _load_sample_set(self):
        """Load coordinate template from database."""
        from utils.coordinate_db import CoordinateDB
        from tkinter import Toplevel, Listbox, Button, Frame, Label, Scrollbar, messagebox
        
        if not self.canvas:
            messagebox.showerror("Error", "Cannot access canvas for loading coordinates")
            return
        
        db = CoordinateDB()
        all_sets = db.get_all_set_names()
        
        if not all_sets:
            messagebox.showinfo("No Sets", "No coordinate sets found in database")
            return
        
        dialog = Toplevel(self.root)
        dialog.title("Load Template")
        
        dialog_width = 400
        dialog_height = 300
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        x = screen_width - dialog_width - 50
        y = (screen_height - dialog_height) // 2
        
        dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
        dialog.resizable(False, False)
        
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.focus_force()
        
        Label(dialog, text="Select a coordinate set to load:", font=("Arial", 12)).pack(pady=10)
        
        listbox_frame = Frame(dialog)
        listbox_frame.pack(fill="both", expand=True, padx=20)
        
        listbox = Listbox(listbox_frame, font=("Arial", 14))
        listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        for set_name in all_sets:
            listbox.insert("end", set_name)
        
        selected_set = None
        
        def on_load(event=None):
            nonlocal selected_set
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a coordinate set to load")
                return
            
            selected_set = all_sets[selection[0]]
            dialog.quit()
            dialog.destroy()
        
        def on_cancel(event=None):
            dialog.quit()
            dialog.destroy()
        
        button_frame = Frame(dialog)
        button_frame.pack(pady=10)
        
        load_button = Button(button_frame, text="Load", command=on_load, width=10)
        load_button.pack(side="left", padx=5)
        cancel_button = Button(button_frame, text="Cancel", command=on_cancel, width=10)
        cancel_button.pack(side="left", padx=5)
        
        dialog.bind('<Return>', on_load)
        dialog.bind('<Escape>', on_cancel)
        
        listbox.bind("<Double-Button-1>", on_load)
        
        listbox.focus_set()
        if listbox.size() > 0:
            listbox.selection_set(0)
        
        dialog.mainloop()
        
        if not selected_set:
            return
        
        coordinates = db.load_coordinate_set(selected_set)
        
        if not coordinates:
            messagebox.showerror("Error", f"Failed to load coordinate set '{selected_set}'")
            return
        
        # Clear existing markers
        if hasattr(self.canvas, '_coord_markers'):
            for marker in self.canvas._coord_markers:
                self.canvas.delete(marker.get('tag', 'unknown_tag'))
            self.canvas._coord_markers.clear()
        else:
            self.canvas._coord_markers = []
        
        # Load the coordinates as visual markers
        from utils.coordinate_db import SampleAreaType
        for i, coord in enumerate(coordinates):
            canvas_x, canvas_y = self.canvas._image_to_screen_coords(coord.x, coord.y)
            
            sample_type = 'circle' if coord.sample_type == SampleAreaType.CIRCLE else 'rectangle'
            
            marker = {
                'index': i + 1,  # 1-based sample numbering
                'image_pos': (coord.x, coord.y),
                'canvas_pos': (canvas_x, canvas_y),
                'sample_type': sample_type,
                'sample_width': coord.sample_size[0],
                'sample_height': coord.sample_size[1],
                'anchor': coord.anchor_position,
                'is_preview': False
            }
            
            tag = f"coord_marker_{len(self.canvas._coord_markers)}"
            marker['tag'] = tag
            
            line_color = self.control_panel.get_line_color()
            if sample_type == 'circle':
                radius = coord.sample_size[0] / 2
                self.canvas.create_oval(
                    canvas_x - radius, canvas_y - radius,
                    canvas_x + radius, canvas_y + radius,
                    outline=line_color, width=2, tags=tag
                )
            else:
                half_w = coord.sample_size[0] / 2
                half_h = coord.sample_size[1] / 2
                self.canvas.create_rectangle(
                    canvas_x - half_w, canvas_y - half_h,
                    canvas_x + half_w, canvas_y + half_h,
                    outline=line_color, width=2, tags=tag
                )
            
            cross_size = 8
            self.canvas.create_line(
                canvas_x - cross_size, canvas_y,
                canvas_x + cross_size, canvas_y,
                fill=line_color, width=2, tags=tag
            )
            self.canvas.create_line(
                canvas_x, canvas_y - cross_size,
                canvas_x, canvas_y + cross_size,
                fill=line_color, width=2, tags=tag
            )
            
            self.canvas.create_text(
                canvas_x + 12, canvas_y - 12,
                text=str(i + 1),
                fill=line_color, font=("Arial", 10, "bold"),
                tags=tag
            )
            
            self.canvas._coord_markers.append(marker)
        
        self.control_panel.sample_set_name.set(selected_set)
        self.control_panel.update_sample_controls_from_coordinates(coordinates)
        
    def _save_sample_set(self):
        """Save the current coordinate sample set."""
        print("DEBUG: _save_sample_set() called in main.py")
        try:
            # Check if we have sample markers to save
            print(f"DEBUG: Checking for coord markers: hasattr={hasattr(self.canvas, '_coord_markers')}")
            if hasattr(self.canvas, '_coord_markers'):
                print(f"DEBUG: Number of coord markers: {len(self.canvas._coord_markers)}")
            
            if not hasattr(self.canvas, '_coord_markers') or not self.canvas._coord_markers:
                print("DEBUG: No sample markers found")
                from tkinter import messagebox
                messagebox.showwarning(
                    "No Samples",
                    "No sample points found. Please place some sample markers first."
                )
                return

            if not self.canvas.original_image:
                print("DEBUG: No original image found")
                from tkinter import messagebox
                messagebox.showwarning(
                    "No Image",
                    "Please open an image before saving samples."
                )
                return

            # Get sample set name
            sample_set_name = self.control_panel.sample_set_name.get().strip()
            print(f"DEBUG: Sample set name: '{sample_set_name}'")
            if not sample_set_name:
                print("DEBUG: No sample set name provided")
                from tkinter import messagebox
                messagebox.showwarning(
                    "No Name",
                    "Please enter a name for the sample set in the Template field."
                )
                return

            # Create coordinate points from markers
            from utils.coordinate_db import CoordinateDB, CoordinatePoint, SampleAreaType
            print("DEBUG: About to create coordinates from markers")
            coordinates = []
            for marker in self.canvas._coord_markers:
                if marker.get('is_preview', False):
                    continue

                # Get marker data
                image_x, image_y = marker['image_pos']
                sample_type = SampleAreaType.CIRCLE if marker['sample_type'] == 'circle' else SampleAreaType.RECTANGLE
                width = float(marker['sample_width'])
                height = float(marker['sample_height'])
                anchor = marker['anchor']

                # Create coordinate point
                coord = CoordinatePoint(
                    x=image_x,
                    y=image_y,
                    sample_type=sample_type,
                    sample_size=(width, height),
                    anchor_position=anchor
                )
                coordinates.append(coord)
                print(f"DEBUG: Added coordinate: x={image_x}, y={image_y}, type={sample_type}, size=({width}, {height})")

            print(f"DEBUG: Created {len(coordinates)} coordinates, about to save to database")
            # Save coordinates to database
            db = CoordinateDB()
            print("DEBUG: Created CoordinateDB instance")
            success, standardized_name = db.save_coordinate_set(
                name=sample_set_name,
                image_path=self.current_file,
                coordinates=coordinates
            )
            print(f"DEBUG: Database save result: success={success}, standardized_name={standardized_name}")

            if success:
                # Update the template name in control panel to the standardized name
                self.control_panel.sample_set_name.set(standardized_name)
                
                # Protect the newly saved template
                print(f"DEBUG: Protecting newly saved template '{standardized_name}'")
                if hasattr(self.control_panel, 'template_protection'):
                    self.control_panel.template_protection.protect_template(standardized_name, coordinates)
                
                from tkinter import messagebox
                messagebox.showinfo(
                    "Success",
                    f"Successfully saved {len(coordinates)} sample points to set '{standardized_name}'.\\n\\n"
                    f"Template is now protected from accidental modification."
                )
            else:
                from tkinter import messagebox
                messagebox.showerror(
                    "Save Error",
                    f"Failed to save sample set '{sample_set_name}'. Please try again."
                )

        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror(
                "Save Error",
                f"Failed to save sample set:\\n\\n{str(e)}"
            )

    # Core UI event handlers that need to stay in the main app
    def reset_view(self):
        """Reset the canvas view."""
        if self.canvas.original_image:
            self.canvas.reset_view()

    def fit_to_window(self):
        """Fit image to window."""
        if self.canvas and self.canvas.original_image:
            self.canvas.fit_to_window()

    def _handle_vertex_count_change(self, count: int):
        """Handle vertex count changes."""
        if self.canvas:
            self.canvas.set_max_vertices(count)
            self.control_panel.vertex_count.set(count)
            
            # Enable/disable Fine Square button based on vertex count
            if hasattr(self.control_panel, 'fine_square_btn'):
                if count == 4:
                    self.control_panel.fine_square_btn.config(state='normal')
                    print("DEBUG: Fine Square button enabled (4 vertices)")
                else:
                    self.control_panel.fine_square_btn.config(state='disabled')
                    print(f"DEBUG: Fine Square button disabled ({count} vertices, need 4)")

    def _adjust_vertex_count(self, delta: int):
        """Adjust vertex count by delta."""
        current = self.control_panel.vertex_count.get()
        new_count = current + delta
        if 3 <= new_count <= 8:
            self._handle_vertex_count_change(new_count)

    def _handle_transparency_change(self, value: int):
        """Handle transparency changes."""
        if self.canvas:
            self.canvas.set_mask_alpha(value)

    def _handle_tool_mode_change(self, mode: str):
        """Handle tool mode changes."""
        if self.canvas:
            if mode == "view":
                self.canvas.set_tool_mode(ToolMode.VIEW)
                self.canvas.configure(cursor='fleur')
            elif mode == "crop":
                self.canvas.set_tool_mode(ToolMode.CROP)
                self.canvas.configure(cursor='crosshair')
            elif mode == "coord":
                self.canvas.set_tool_mode(ToolMode.COORD)
                self.canvas.configure(cursor='crosshair')
            elif mode == "straighten":
                self.canvas.set_tool_mode(ToolMode.STRAIGHTENING)
                self.canvas.configure(cursor='crosshair')

    def _handle_ruler_toggle(self, show: bool):
        """Handle ruler visibility toggle."""
        if self.canvas:
            self.canvas.ruler_manager.toggle_visibility(show)
            self.canvas.update_display()

    def _handle_grid_toggle(self, show: bool):
        """Handle grid visibility toggle."""
        if self.canvas:
            self.canvas.ruler_manager.toggle_grid(show)
            self.canvas.update_display()

    def _handle_shape_type_change(self, shape_type: ShapeType):
        """Handle shape type changes."""
        if self.canvas:
            self.canvas.set_shape_type(shape_type)

    def _handle_line_color_change(self, color: str):
        """Handle line color changes."""
        if self.canvas:
            self.canvas.set_line_color(color)

    def _update_crop_dimensions(self, width: int, height: int):
        """Update crop dimensions in control panel."""
        if self.control_panel:
            self.control_panel.update_crop_dimensions(width, height)

    def _apply_default_settings(self):
        """Apply default application settings."""
        # This will be expanded with actual default settings
        pass
        
    def _apply_fine_square(self):
        """Apply fine square adjustment to the current crop vertices."""
        print("DEBUG: _apply_fine_square called")
        
        if not self.canvas or not self.canvas.original_image:
            print("DEBUG: No canvas or image available for fine square")
            from tkinter import messagebox
            messagebox.showwarning("No Image", "Please open an image before using the Fine Square tool.")
            return
        
        # Get current vertices
        vertices = self.canvas.get_vertices()
        if len(vertices) != 4:
            print(f"DEBUG: Wrong number of vertices: {len(vertices)} (need 4)")
            from tkinter import messagebox
            messagebox.showwarning(
                "Invalid Shape", 
                "Fine Square adjustment requires exactly 4 vertices. "
                f"Current shape has {len(vertices)} vertices."
            )
            return
        
        try:
            # Import the fine square adjustment function
            from utils.auto_square import fine_square_adjustment
            
            # Apply fine square adjustment with level alignment (horizontal/vertical sides)
            adjusted_vertices = fine_square_adjustment(vertices, method='preserve_center_level')
            
            # Update the canvas with the adjusted vertices
            self.canvas.set_vertices(adjusted_vertices)
            
            print(f"DEBUG: Fine square adjustment applied successfully")
            print(f"DEBUG: Original vertices: {[(v.x, v.y) for v in vertices]}")
            print(f"DEBUG: Adjusted vertices: {[(v.x, v.y) for v in adjusted_vertices]}")
            
        except Exception as e:
            print(f"ERROR: Fine square adjustment failed: {e}")
            from tkinter import messagebox
            messagebox.showerror(
                "Fine Square Error", 
                f"Failed to apply fine square adjustment:\n\n{str(e)}"
            )
    
    def _clear_straightening_points(self):
        """Clear all straightening reference points."""
        if not hasattr(self, 'straightening_tool'):
            print("DEBUG: No straightening tool available")
            return
            
        self.straightening_tool.clear_points()
        self.control_panel.update_straightening_status(0)
        if self.canvas:
            self.canvas.delete('straightening_point')
        print("DEBUG: Cleared all straightening points")
    
    def _apply_straightening(self):
        """Apply straightening/leveling to the current image."""
        if not self.canvas or not self.canvas.original_image:
            from tkinter import messagebox
            messagebox.showwarning("No Image", "Please open an image before straightening.")
            return

        if not hasattr(self, 'straightening_tool') or not self.straightening_tool.can_straighten():
            from tkinter import messagebox
            messagebox.showwarning("Insufficient Points", "Please place at least 2 reference points.")
            return

        try:
            straightened_image, angle = self.straightening_tool.straighten_image(
                self.canvas.original_image,
                background_color='white'
            )
            self.canvas.load_image(straightened_image)
            self.straightening_tool.clear_points()
            self.control_panel.update_straightening_status(0)
            if self.canvas:
                self.canvas.delete('straightening_point')
            
            print(f"DEBUG: Applied straightening with angle {angle:.2f} degrees")
            
            # Provide user feedback about the leveling operation
            from tkinter import messagebox
            if abs(angle) < 0.1:
                messagebox.showinfo(
                    "Leveling Complete", 
                    f"Image leveling applied successfully.\n\n"
                    f"Rotation angle: {angle:.2f}°\n\n"
                    f"The image was already nearly level."
                )
            else:
                messagebox.showinfo(
                    "Leveling Complete", 
                    f"Image leveling applied successfully.\n\n"
                    f"Rotation angle: {angle:.2f}°\n\n"
                    f"The image has been straightened."
                )
                
        except Exception as e:
            print(f"ERROR: Straightening failed: {e}")
            from tkinter import messagebox
            messagebox.showerror("Straightening Error", f"Failed to straighten image: {str(e)}")
        
    # Sample management methods
    def _clear_samples(self, skip_confirmation=False, reset_all=False):
        """Clear all sample markers and reset sample-related UI elements.
        
        Args:
            skip_confirmation (bool): If True, skips the confirmation dialog
            reset_all (bool): If True, resets all UI elements to default state
        """
        if not hasattr(self.canvas, '_coord_markers'):
            self.canvas._coord_markers = []
        
        # Ask for confirmation if there are sample markers and confirmation is not skipped
        if self.canvas._coord_markers and not skip_confirmation:
            from tkinter import messagebox
            result = messagebox.askyesno(
                "Clear All Samples",
                f"This will clear all {len(self.canvas._coord_markers)} sample markers.\n\n"
                "Are you sure you want to continue?"
            )
            if not result:
                return
        
        try:
            # Delete visual markers from canvas
            for marker in self.canvas._coord_markers:
                tag = marker.get('tag')
                if tag:
                    self.canvas.delete(tag)
            
            # Clear the markers list
            self.canvas._coord_markers.clear()
            
            # Reset control UI to defaults
            if hasattr(self.control_panel, 'sample_controls'):
                for control in self.control_panel.sample_controls:
                    control['shape'].set('circle')
                    control['width'].set('10')
                    control['height'].set('10')
                    control['anchor'].set('center')
            
            # Clear template and analysis names
            if hasattr(self.control_panel, 'sample_set_name'):
                self.control_panel.sample_set_name.set("")
            if hasattr(self.control_panel, 'analysis_name'):
                self.control_panel.analysis_name.set("")
            
            # Reset offset values and status
            if hasattr(self.control_panel, 'global_x_offset'):
                self.control_panel.global_x_offset.set(0)
            if hasattr(self.control_panel, 'global_y_offset'):
                self.control_panel.global_y_offset.set(0)
            if hasattr(self.control_panel, 'individual_x_offset'):
                self.control_panel.individual_x_offset.set(0)
            if hasattr(self.control_panel, 'individual_y_offset'):
                self.control_panel.individual_y_offset.set(0)
            if hasattr(self.control_panel, 'offset_status'):
                self.control_panel.offset_status.set("No offsets applied")
            
            # If resetting all, also reset additional UI elements
            if reset_all:
                if hasattr(self.control_panel, 'sample_mode'):
                    self.control_panel.sample_mode.set("template")
                    # This will trigger the UI update for template mode
                    if hasattr(self.control_panel, '_set_template_mode_ui'):
                        self.control_panel._set_template_mode_ui()
                
                # Reset any manual mode settings if they exist
                if hasattr(self.control_panel, 'manual_shape'):
                    self.control_panel.manual_shape.set('circle')
                if hasattr(self.control_panel, 'manual_width'):
                    self.control_panel.manual_width.set('10')
                if hasattr(self.control_panel, 'manual_height'):
                    self.control_panel.manual_height.set('10')
                if hasattr(self.control_panel, 'manual_anchor'):
                    self.control_panel.manual_anchor.set('center')
            
            # Force canvas update
            self.canvas.update_display()
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror(
                "Clear Error", 
                f"Failed to clear sample markers:\\n\\n{str(e)}"
            )
            
    def _export_to_unified_logger(self, export_type="individual"):
        """Export color measurements to unified data logger with sample set selection.
        
        Args:
            export_type: Either "individual" or "averaged"
        """
        try:
            # Check if we have a current image
            if not self.current_file:
                messagebox.showerror(
                    "No Image", 
                    "Please open an image first. The unified data logger creates files alongside the current image."
                )
                return
            
            # Get available sample sets
            available_sets = self._get_available_sample_sets_for_export()
            
            if not available_sets:
                messagebox.showinfo(
                    "No Data",
                    "No color analysis data found. Please perform color analysis first."
                )
                return
            
            # Show sample set selection dialog
            selected_set = self._show_sample_set_selection_dialog(
                available_sets, 
                export_type
            )
            
            if not selected_set:
                return  # User cancelled
            
            # Extract just the sample set name (remove count info)
            sample_set_name = selected_set.split(" (")[0]
            
            # Export to unified logger based on type
            if export_type == "individual":
                success = self._export_individual_measurements_to_logger(sample_set_name)
            elif export_type == "averaged":
                success = self._export_averaged_measurements_to_logger(sample_set_name)
            else:  # both
                success = self._export_both_measurements_to_logger(sample_set_name)
            
            if not success:
                messagebox.showerror(
                    "Export Failed",
                    f"Failed to export {export_type} measurements to unified data logger."
                )
                
        except Exception as e:
            messagebox.showerror(
                "Export Error",
                f"Failed to export to unified data logger:\\n\\n{str(e)}"
            )
            
    def _get_available_sample_sets_for_export(self):
        """Get list of sample sets with measurement data."""
        try:
            import sqlite3
            from utils.path_utils import get_color_analysis_dir
            
            analysis_dir = get_color_analysis_dir()
            if not os.path.exists(analysis_dir):
                return []
                
            db_files = [f for f in os.listdir(analysis_dir) if f.endswith('.db')]
            
            sample_sets = []
            for db_file in db_files:
                # Extract sample set name from filename (remove .db extension)
                set_name = os.path.splitext(db_file)[0]
                
                # Skip averaged databases for this listing
                if set_name.endswith('_averages'):
                    continue
                
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
                    # If we can't read it, skip it
                    pass
            
            return sorted(sample_sets)
            
        except Exception as e:
            print(f"Error getting sample sets for export: {e}")
            return []
            
    def _show_sample_set_selection_dialog(self, available_sets, export_type):
        """Show dialog to select sample set for export.
        
        Args:
            available_sets: List of available sample sets
            export_type: "individual" or "averaged"
            
        Returns:
            Selected sample set string or None if cancelled
        """
        try:
            from tkinter import Toplevel, Listbox, Scrollbar, Frame, Label, Button
            
            dialog = Toplevel(self.root)
            dialog.title(f"Export {export_type.title()} Measurements")
            
            dialog_width = 450
            dialog_height = 350
            
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            x = (screen_width - dialog_width) // 2
            y = (screen_height - dialog_height) // 2
            
            dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
            dialog.resizable(False, False)
            
            dialog.transient(self.root)
            dialog.grab_set()
            dialog.focus_force()
            
            # Header
            header_text = f"Export {export_type.title()} Measurements to Unified Data Logger"
            Label(dialog, text=header_text, font=("Arial", 12, "bold")).pack(pady=(15, 10))
            
            # Instructions
            instruction_text = (
                f"Select a sample set to export {export_type} measurements:\n"
                f"Data will be added to the unified log file for the current image."
            )
            Label(
                dialog, 
                text=instruction_text, 
                font=("Arial", 10),
                wraplength=400,
                justify="center"
            ).pack(pady=(0, 15))
            
            # Listbox frame
            listbox_frame = Frame(dialog)
            listbox_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))
            
            listbox = Listbox(listbox_frame, font=("Arial", 11))
            listbox.pack(side="left", fill="both", expand=True)
            
            scrollbar = Scrollbar(listbox_frame)
            scrollbar.pack(side="right", fill="y")
            listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=listbox.yview)
            
            # Populate listbox
            for set_name in available_sets:
                listbox.insert("end", set_name)
            
            if available_sets:
                listbox.selection_set(0)  # Select first item
            
            # Result variable
            selected_set = None
            
            def on_export():
                nonlocal selected_set
                selection = listbox.curselection()
                if not selection:
                    messagebox.showwarning(
                        "No Selection", 
                        "Please select a sample set to export."
                    )
                    return
                
                selected_set = available_sets[selection[0]]
                dialog.quit()
                dialog.destroy()
            
            def on_cancel():
                dialog.quit()
                dialog.destroy()
            
            # Button frame
            button_frame = Frame(dialog)
            button_frame.pack(fill="x", padx=20, pady=(0, 20))
            
            export_button = Button(
                button_frame, 
                text=f"Export {export_type.title()}", 
                command=on_export, 
                width=15,
                bg="#4CAF50",
                fg="white",
                font=("Arial", 10, "bold")
            )
            export_button.pack(side="left", padx=(0, 10))
            
            cancel_button = Button(
                button_frame, 
                text="Cancel", 
                command=on_cancel, 
                width=10
            )
            cancel_button.pack(side="right")
            
            # Bind events
            dialog.bind('<Return>', lambda e: on_export())
            dialog.bind('<Escape>', lambda e: on_cancel())
            listbox.bind("<Double-Button-1>", lambda e: on_export())
            
            # Focus on listbox
            listbox.focus_set()
            
            dialog.mainloop()
            
            return selected_set
            
        except Exception as e:
            print(f"Error showing sample set selection dialog: {e}")
            return None
            
    def _export_individual_measurements_to_logger(self, sample_set_name):
        """Export individual measurements to unified data logger.
        
        Args:
            sample_set_name: Name of the sample set to export
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from utils.unified_data_logger import UnifiedDataLogger
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Get measurements from database filtered by current image
            db = ColorAnalysisDB(sample_set_name)
            current_image_name = os.path.basename(self.current_file)
            
            # Debug: Check what's in the database vs what we're looking for
            all_measurements = db.get_all_measurements()
            if all_measurements:
                unique_image_names = set(m['image_name'] for m in all_measurements)
                print(f"DEBUG: Looking for image: '{current_image_name}'")
                print(f"DEBUG: Available images in database: {list(unique_image_names)}")
                
                # Try to find a match using the pattern extraction logic
                from utils.color_analyzer import ColorAnalyzer
                analyzer = ColorAnalyzer()
                sample_identifier = analyzer._extract_sample_identifier_from_filename(self.current_file)
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
                messagebox.showwarning(
                    "No Data",
                    f"No measurements found for sample set '{sample_set_name}'."
                )
                return False
            
            # Create data logger for current image
            logger = UnifiedDataLogger(self.current_file)
            
            # Log individual measurements
            log_file = logger.log_individual_color_measurements(
                db_measurements, sample_set_name, self.current_file
            )
            
            if log_file:
                messagebox.showinfo(
                    "Export Complete",
                    f"Individual color measurements exported to unified data logger.\n\n"
                    f"Sample Set: {sample_set_name}\n"
                    f"File: {log_file.name}\n"
                    f"Measurements: {len(db_measurements)} samples\n\n"
                    f"Data has been appended to your comprehensive analysis log."
                )
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error exporting individual measurements: {e}")
            return False
            
    def _export_averaged_measurements_to_logger(self, sample_set_name):
        """Export averaged measurement to unified data logger.
        
        Args:
            sample_set_name: Name of the sample set to export
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from utils.unified_data_logger import UnifiedDataLogger
            from utils.color_analyzer import ColorAnalyzer
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Calculate averaged measurement from individual measurements
            analyzer = ColorAnalyzer()
            
            # Get individual measurements from database filtered by current image
            db = ColorAnalysisDB(sample_set_name)
            current_image_name = os.path.basename(self.current_file)
            
            # Debug: Check what's in the database vs what we're looking for
            all_measurements = db.get_all_measurements()
            if all_measurements:
                unique_image_names = set(m['image_name'] for m in all_measurements)
                print(f"DEBUG: Looking for image: '{current_image_name}'")
                print(f"DEBUG: Available images in database: {list(unique_image_names)}")
                
                # Try to find a match using the pattern extraction logic
                from utils.color_analyzer import ColorAnalyzer
                analyzer = ColorAnalyzer()
                sample_identifier = analyzer._extract_sample_identifier_from_filename(self.current_file)
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
                messagebox.showwarning(
                    "No Data",
                    f"No measurements found for sample set '{sample_set_name}'."
                )
                return False
            
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
            logger = UnifiedDataLogger(self.current_file)
            
            # Log averaged measurement
            log_file = logger.log_averaged_color_measurement(
                averaged_data, sample_set_name, self.current_file, len(db_measurements)
            )
            
            if log_file:
                messagebox.showinfo(
                    "Export Complete",
                    f"Averaged color measurement exported to unified data logger.\n\n"
                    f"Sample Set: {sample_set_name}\n"
                    f"File: {log_file.name}\n"
                    f"Source: {len(db_measurements)} individual samples\n"
                    f"Quality: ΔE max = {averaging_result['max_delta_e']:.2f}\n\n"
                    f"High-quality averaged data has been added to your comprehensive analysis log."
                )
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error exporting averaged measurement: {e}")
            return False
            
    def _export_both_measurements_to_logger(self, sample_set_name):
        """Export both individual and averaged measurements to unified data logger.
        
        Args:
            sample_set_name: Name of the sample set to export
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from utils.unified_data_logger import UnifiedDataLogger
            from utils.color_analyzer import ColorAnalyzer
            from utils.color_analysis_db import ColorAnalysisDB
            
            # Get measurements from database filtered by current image
            db = ColorAnalysisDB(sample_set_name)
            current_image_name = os.path.basename(self.current_file)
            
            # Debug: Check what's in the database vs what we're looking for
            all_measurements = db.get_all_measurements()
            if all_measurements:
                unique_image_names = set(m['image_name'] for m in all_measurements)
                print(f"DEBUG: Looking for image: '{current_image_name}'")
                print(f"DEBUG: Available images in database: {list(unique_image_names)}")
                
                # Try to find a match using the pattern extraction logic
                from utils.color_analyzer import ColorAnalyzer
                analyzer = ColorAnalyzer()
                sample_identifier = analyzer._extract_sample_identifier_from_filename(self.current_file)
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
                messagebox.showwarning(
                    "No Data",
                    f"No measurements found for sample set '{sample_set_name}'."
                )
                return False
            
            # Create data logger for current image
            logger = UnifiedDataLogger(self.current_file)
            
            # First export individual measurements
            individual_log_file = logger.log_individual_color_measurements(
                db_measurements, sample_set_name, self.current_file
            )
            
            if not individual_log_file:
                messagebox.showerror("Export Failed", "Failed to export individual measurements to data logger.")
                return False
            
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
                averaged_data, sample_set_name, self.current_file, len(db_measurements)
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
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error exporting both measurements: {e}")
            return False
