"""
Core StampZ Application Class

Main application window that coordinates all managers and UI components.
"""

import os
import tkinter as tk
from tkinter import ttk
import logging

from gui.canvas import CropCanvas, ShapeType, ToolMode
from gui.controls_reorganized import ReorganizedControlPanel as ControlPanel
from utils.recent_files import RecentFilesManager
from utils.image_straightener import StraighteningTool
from utils.path_utils import ensure_data_directories

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
                f"Failed to clear sample markers:\n\n{str(e)}"
            )
