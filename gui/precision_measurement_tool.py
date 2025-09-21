#!/usr/bin/env python3
"""
StampZ Precision Measurement Tool
Advanced measurement tool with architectural-style dimension lines
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import math

# Import the measurement engine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precision_measurement_engine import MeasurementEngine, ArchitecturalMeasurement

class PrecisionMeasurementTool:
    """Advanced precision measurement tool with architectural dimension lines."""
    
    def __init__(self, parent=None, image_path=None, main_app=None):
        self.parent = parent
        self.main_app = main_app
        self.image_path = image_path
        
        # Create window
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Precision Measurements - StampZ")
        self.root.geometry("1400x900")
        
        # Initialize measurement engine
        self.measurement_engine = MeasurementEngine(image_path) if image_path else MeasurementEngine()
        
        # Measurement state
        self.measurements = []
        self.current_measurement_start = None
        self.measurement_mode = None
        self.click_count = 0
        self.selected_measurement = None
        self.selected_endpoint = None  # 'start' or 'end'
        self.dragging = False
        self.snap_tolerance = 5  # degrees for snap-to-horizontal/vertical
        
        # UI state
        self.precision_decimals = 2  # Default to 2 decimal places - adequate for stamps
        self.show_pixel_coords = False
        self.auto_label = True  # Skip label dialog for faster workflow
        self.data_logged = False  # Track if measurements have been logged to avoid duplicates
        
        self.setup_ui()
        
        if image_path:
            self.load_image_into_plot()
        
        # Bind keyboard shortcuts for nudging
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()  # Enable keyboard events
        
        # Center window
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (self.root.winfo_reqwidth() // 2)
        y = (screen_height // 2) - (self.root.winfo_reqheight() // 2)
        self.root.geometry(f"+{x}+{y}")
        
    def setup_ui(self):
        """Setup the measurement tool interface"""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right panel - Plot
        self.setup_plot_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel with measurement tools"""
        # Create overall control container
        control_container = ttk.Frame(parent, width=320)
        control_container.pack(side="left", fill="y", padx=(0, 10))
        control_container.pack_propagate(False)
        
        # Create scrollable upper section
        scrollable_container = ttk.Frame(control_container)
        scrollable_container.pack(fill="both", expand=True)
        
        # Canvas and scrollbar for scrolling
        canvas = tk.Canvas(scrollable_container, width=320)
        scrollbar = ttk.Scrollbar(scrollable_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        control_frame = self.scrollable_frame
        
        # Header
        header_frame = ttk.Frame(control_frame)
        header_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(header_frame, text="Precision Measurements", 
                 font=("Arial", 14, "bold")).pack()
        
        if self.image_path:
            ttk.Label(header_frame, text=f"Image: {os.path.basename(self.image_path)}", 
                     font=("Arial", 9)).pack()
        
        # Image loading
        self.setup_image_section(control_frame)
        
        # Calibration section
        self.setup_calibration_section(control_frame)
        
        # Measurement tools
        self.setup_measurement_tools(control_frame)
        
        # Measurements list
        self.setup_measurements_list(control_frame)
        
        # Export and actions (excluding Return button)
        self.setup_actions_section(control_frame)
        
        # Fixed navigation bar at bottom (outside scrollable area)
        self.setup_navigation_bar(control_container)
        
    def setup_image_section(self, parent):
        """Setup image loading section"""
        image_frame = ttk.LabelFrame(parent, text="Image", padding=5)
        image_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(image_frame, text="Load Image...", 
                  command=self.load_image_dialog).pack(fill="x", pady=2)
        
        if self.image_path:
            # Show image info
            if self.measurement_engine.image:
                size = self.measurement_engine.image.size
                ttk.Label(image_frame, text=f"Size: {size[0]}x{size[1]} px",
                         font=("Arial", 8)).pack()
        
    def setup_calibration_section(self, parent):
        """Setup calibration controls"""
        cal_frame = ttk.LabelFrame(parent, text="Calibration", padding=5)
        cal_frame.pack(fill="x", pady=(0, 10))
        
        # DPI info
        if self.measurement_engine.dpi:
            ttk.Label(cal_frame, text=f"DPI: {float(self.measurement_engine.dpi):.1f}",
                     font=("Arial", 9, "bold")).pack()
            ttk.Label(cal_frame, text=f"Source: {self.measurement_engine.calibration_source}",
                     font=("Arial", 8)).pack()
            
            if self.measurement_engine.precision_um:
                precision_color = "#008000" if self.measurement_engine.precision_um < 100 else "#FF6600"
                ttk.Label(cal_frame, text=f"Precision: ±{self.measurement_engine.precision_um:.1f}µm",
                         font=("Arial", 8), foreground=precision_color).pack()
                         
            # Warning for poor calibration
            if "Default" in self.measurement_engine.calibration_source or self.measurement_engine.dpi < 50:
                ttk.Label(cal_frame, text="⚠️ Manual calibration recommended",
                         font=("Arial", 8), foreground="#FF6600").pack()
        
        # Current image size (calculated from DPI)
        if self.measurement_engine.image:
            height, width = self.measurement_engine.image.size[1], self.measurement_engine.image.size[0]
            height_mm = height / self.measurement_engine.pixels_per_mm if self.measurement_engine.pixels_per_mm else 0
            width_mm = width / self.measurement_engine.pixels_per_mm if self.measurement_engine.pixels_per_mm else 0
            
            size_frame = ttk.Frame(cal_frame)
            size_frame.pack(fill="x", pady=2)
            ttk.Label(size_frame, text=f"Current: {width}×{height}px = {width_mm:.1f}×{height_mm:.1f}mm", 
                     font=("Arial", 8)).pack(anchor="w")
        
        # Direct DPI setting
        dpi_frame = ttk.Frame(cal_frame)
        dpi_frame.pack(fill="x", pady=2)
        
        ttk.Label(dpi_frame, text="Set DPI directly:", font=("Arial", 8, "bold")).pack(anchor="w")
        
        dpi_inputs = ttk.Frame(dpi_frame)
        dpi_inputs.pack(fill="x", pady=2)
        
        ttk.Label(dpi_inputs, text="DPI:").grid(row=0, column=0, sticky="w")
        self.dpi_var = tk.DoubleVar(value=self.measurement_engine.dpi if self.measurement_engine.dpi else 800)
        dpi_entry = ttk.Entry(dpi_inputs, textvariable=self.dpi_var, width=8)
        dpi_entry.grid(row=0, column=1, padx=2)
        
        ttk.Button(dpi_inputs, text="Set DPI", 
                  command=self.set_dpi_directly).grid(row=0, column=2, padx=2)
        
        ttk.Separator(cal_frame, orient='horizontal').pack(fill="x", pady=5)
        
        # Manual calibration
        manual_frame = ttk.Frame(cal_frame)
        manual_frame.pack(fill="x", pady=2)
        
        ttk.Label(manual_frame, text="Or calibrate: Measure → Enter real size → Calibrate", 
                 font=("Arial", 8, "bold")).pack(anchor="w")
        
        cal_inputs = ttk.Frame(manual_frame)
        cal_inputs.pack(fill="x", pady=2)
        
        ttk.Label(cal_inputs, text="Real size (mm):").grid(row=0, column=0, sticky="w")
        self.known_size_var = tk.DoubleVar(value=0.0)
        size_entry = ttk.Entry(cal_inputs, textvariable=self.known_size_var, width=8)
        size_entry.grid(row=0, column=1, padx=2)
        
        ttk.Button(cal_inputs, text="Calibrate", 
                  command=self.manual_calibrate).grid(row=0, column=2, padx=2)
        
        # Precision settings
        precision_frame = ttk.Frame(cal_frame)
        precision_frame.pack(fill="x", pady=5)
        
        ttk.Label(precision_frame, text="Decimal places:").pack()
        self.precision_var = tk.IntVar(value=2)  # Default 2 decimal places
        precision_spinbox = ttk.Spinbox(precision_frame, from_=0, to=5, 
                                       textvariable=self.precision_var, width=10,
                                       command=self.update_precision)
        precision_spinbox.pack()
        
    def setup_measurement_tools(self, parent):
        """Setup measurement tool buttons"""
        tools_frame = ttk.LabelFrame(parent, text="Measurement Tools", padding=5)
        tools_frame.pack(fill="x", pady=(0, 10))
        
        # Measurement mode buttons
        ttk.Button(tools_frame, text="🖱️ Select/Edit Mode", 
                  command=lambda: self.set_measurement_mode("select")).pack(fill="x", pady=2)
        
        ttk.Separator(tools_frame, orient='horizontal').pack(fill="x", pady=3)
        
        ttk.Button(tools_frame, text="📏 Distance Measurement", 
                  command=lambda: self.set_measurement_mode("distance")).pack(fill="x", pady=2)
        
        ttk.Button(tools_frame, text="↔️ Horizontal Measurement", 
                  command=lambda: self.set_measurement_mode("horizontal")).pack(fill="x", pady=2)
        
        ttk.Button(tools_frame, text="↕️ Vertical Measurement", 
                  command=lambda: self.set_measurement_mode("vertical")).pack(fill="x", pady=2)
        
        # Current mode indicator
        self.mode_label = ttk.Label(tools_frame, text="Mode: View", 
                                   font=("Arial", 8), foreground="blue")
        self.mode_label.pack(pady=5)
        
        # Instructions
        self.instruction_label = ttk.Label(tools_frame, 
                                          text="Select a measurement tool,\nthen click two points on the image.",
                                          font=("Arial", 8), justify="center")
        self.instruction_label.pack(pady=5)
        
        # Options
        options_frame = ttk.Frame(tools_frame)
        options_frame.pack(fill="x", pady=2)
        
        self.auto_label_var = tk.BooleanVar(value=self.auto_label)
        ttk.Checkbutton(options_frame, text="Auto-label measurements", 
                       variable=self.auto_label_var,
                       command=self.toggle_auto_label).pack(anchor="w")
        
        # Keyboard shortcuts info (compact)
        ttk.Label(tools_frame, text="Keys: ←↑↓→ nudge, Shift+arrows coarse", 
                 font=("Arial", 7), foreground="gray").pack(pady=2)
        
    def setup_measurements_list(self, parent):
        """Setup measurements list"""
        list_frame = ttk.LabelFrame(parent, text="Measurements", padding=5)
        list_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Measurements listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side="right", fill="y")
        
        self.measurements_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set, 
                                              font=("Monaco", 8))
        self.measurements_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.measurements_listbox.yview)
        
        # Add context menu for editing labels (multiple bindings for cross-platform)
        self.measurements_listbox.bind("<Button-2>", self.show_context_menu)  # Right-click on Mac
        self.measurements_listbox.bind("<Button-3>", self.show_context_menu)  # Right-click on PC
        self.measurements_listbox.bind("<Control-Button-1>", self.show_context_menu)  # Ctrl+click fallback
        
        # Also add double-click to edit
        self.measurements_listbox.bind("<Double-Button-1>", self.double_click_edit)
        
        # List control buttons
        list_buttons = ttk.Frame(list_frame)
        list_buttons.pack(fill="x", pady=5)
        
        ttk.Button(list_buttons, text="Delete Selected", 
                  command=self.delete_selected_measurement).pack(side="left", padx=2)
        ttk.Button(list_buttons, text="Clear All", 
                  command=self.clear_all_measurements).pack(side="right", padx=2)
        
    def setup_actions_section(self, parent):
        """Setup export and action buttons"""
        actions_frame = ttk.Frame(parent)
        actions_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(actions_frame, text="📊 Export Report", 
                  command=self.export_report).pack(fill="x", pady=2)
        
        ttk.Button(actions_frame, text="💾 Save Measurements", 
                  command=self.save_measurements).pack(fill="x", pady=2)
        
        ttk.Button(actions_frame, text="📁 Load Measurements", 
                  command=self.load_measurements).pack(fill="x", pady=2)
    
    def setup_navigation_bar(self, parent):
        """Setup fixed navigation bar at bottom of control panel"""
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill="x", side="bottom", padx=5, pady=10)
        
        # Add separator line
        separator = ttk.Separator(nav_frame, orient='horizontal')
        separator.pack(fill="x", pady=(0, 10))
        
        # Log to Unified Data button - always accessible
        log_button = ttk.Button(nav_frame, text="📝 Log to Unified Data", 
                               command=self.log_to_unified_data)
        log_button.pack(fill="x", pady=(0, 2))
        
        # Return to StampZ button - always visible
        return_button = ttk.Button(nav_frame, text="🔙 Return to StampZ", 
                                  command=self.close_window)
        return_button.pack(fill="x", pady=2)
        
        # Make return button more prominent with styling
        return_button.configure(style="Accent.TButton")
        
    def setup_plot_panel(self, parent):
        """Setup the matplotlib plot panel"""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(side="right", fill="both", expand=True)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.patch.set_facecolor('white')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        
        # Pack canvas
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Bind click events
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.canvas.mpl_connect('button_release_event', self.on_plot_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_plot_hover)
        
        # Initial plot setup
        self.ax.set_title("Click to Load Image or Use Measurement Tools", fontsize=12)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
    def load_image_dialog(self):
        """Load image via file dialog"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Load Image for Measurement",
            filetypes=filetypes
        )
        
        if filename:
            self.image_path = filename
            self.measurement_engine = MeasurementEngine(filename)
            self.load_image_into_plot()
            self.update_calibration_display()
            
    def load_image_into_plot(self):
        """Load the current image into the matplotlib plot"""
        if not self.measurement_engine.image:
            return
            
        self.ax.clear()
        
        # Display image
        image_array = np.array(self.measurement_engine.image)
        self.ax.imshow(image_array, origin='upper', aspect='equal')
        
        # Set title with image info
        if self.measurement_engine.dpi:
            # Convert DPI to float to handle IFDRational objects from TIFF metadata
            dpi_value = float(self.measurement_engine.dpi)
            title = f"{os.path.basename(self.image_path)} - {dpi_value:.0f} DPI"
        else:
            title = f"{os.path.basename(self.image_path)}"
            
        self.ax.set_title(title, fontsize=10)
        
        # Auto-fit the image with some padding
        height, width = image_array.shape[:2]
        padding = max(width, height) * 0.1  # 10% padding
        
        self.ax.set_xlim(-padding, width + padding)
        self.ax.set_ylim(height + padding, -padding)  # Inverted Y for image coordinates
        
        # Redraw existing measurements
        self.redraw_all_measurements()
        
        self.canvas.draw()
        
    def update_calibration_display(self):
        """Update calibration info in the UI"""
        # Rebuild the calibration section with updated info
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                # Find the main frame and rebuild calibration section
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame) and child.winfo_children():
                        # This is a bit hacky, but we'll just refresh the whole UI
                        pass
                        
        # For now, just update measurements list and redraw
        self.update_measurements_list()
        if self.measurement_engine.image:
            self.load_image_into_plot()
        
    def set_measurement_mode(self, mode):
        """Set the current measurement mode"""
        self.measurement_mode = mode
        self.current_measurement_start = None
        self.click_count = 0
        self.selected_measurement = None
        self.selected_endpoint = None
        
        # Clear any preview lines
        self.clear_preview_lines()
        
        self.mode_label.config(text=f"Mode: {mode.title()}")
        
        if mode == "select":
            measurement_count = len(self.measurements)
            if measurement_count > 0:
                self.instruction_label.config(text=f"Select mode: {measurement_count} measurements\nClick endpoints to drag/adjust\nUse arrow keys to nudge")
            else:
                self.instruction_label.config(text="Select mode: No measurements yet\nCreate measurements first")
        else:
            snap_info = "" if mode == "distance" else f"\n({mode.upper()} constraint active)"
            self.instruction_label.config(text=f"Click two points to create\n{mode} measurement{snap_info}")
        
    def on_plot_click(self, event):
        """Handle plot click events for measurement creation and selection"""
        if not self.measurement_mode or not event.inaxes:
            return
            
        x, y = event.xdata, event.ydata
        
        if self.measurement_mode == "select":
            # Selection mode - find nearest endpoint
            self.select_nearest_endpoint(x, y)
            return  # Don't proceed to measurement creation
            
        elif self.measurement_mode in ["distance", "horizontal", "vertical"]:
            # Measurement creation mode
            if self.click_count == 0:
                # First click - start point
                self.current_measurement_start = (x, y)
                self.click_count = 1
                self.instruction_label.config(text=f"Click second point\nfor {self.measurement_mode} measurement")
                
                # Draw a temporary marker for the first click
                self.draw_first_click_marker(x, y)
                
            elif self.click_count == 1:
                # Second click - end point
                end_point = (x, y)
                
                # Apply snap-to-horizontal/vertical for horizontal and vertical modes
                if self.measurement_mode in ["horizontal", "vertical"]:
                    end_point = self.apply_snap_constraint(self.current_measurement_start, end_point)
                
                # Ensure we have two different points (minimum distance check)
                dx = end_point[0] - self.current_measurement_start[0]
                dy = end_point[1] - self.current_measurement_start[1]
                distance = (dx*dx + dy*dy)**0.5
                
                if distance < 5:  # Minimum 5 pixel distance
                    self.instruction_label.config(text=f"Click farther away for {self.measurement_mode}\nmeasurement (min 5 pixels)")
                    return
                
                # Create measurement with optional label
                default_label = f"{self.measurement_mode.title()} {len(self.measurements) + 1}"
                
                if self.auto_label:
                    user_label = default_label
                else:
                    from tkinter import simpledialog
                    user_label = simpledialog.askstring(
                        "Measurement Label",
                        f"Enter a label for this {self.measurement_mode} measurement:",
                        initialvalue=default_label
                    )
                    
                    if user_label is None:  # User cancelled
                        user_label = default_label
                    
                measurement = ArchitecturalMeasurement(
                    start_point=self.current_measurement_start,
                    end_point=end_point,
                    measurement_type=self.measurement_mode,
                    label=user_label
                )
                
                self.measurements.append(measurement)
                self.add_measurement_to_list(measurement)
                self.draw_measurement(measurement)
                
                # Reset logged flag since we have new data
                self.data_logged = False
                
                # Clear preview lines
                self.clear_preview_lines()
                
                # Reset for next measurement
                self.click_count = 0
                self.current_measurement_start = None
                self.instruction_label.config(text=f"Click two points to create\n{self.measurement_mode} measurement")
            
    def on_plot_hover(self, event):
        """Handle mouse hover for coordinate display and dragging"""
        if not event.inaxes:
            return
            
        x, y = event.xdata, event.ydata
        
        # Handle dragging
        if self.dragging and self.selected_measurement and self.selected_endpoint:
            self.drag_endpoint(x, y)
            
        # Show preview line for horizontal/vertical measurements
        elif self.current_measurement_start and self.click_count == 1:
            if self.measurement_mode in ["horizontal", "vertical"]:
                self.draw_preview_line(self.current_measurement_start, (x, y))
            
        # Show coordinates if enabled
        if self.show_pixel_coords:
            # Could show coordinates in status bar if implemented
            pass
            
    def on_plot_release(self, event):
        """Handle mouse button release"""
        if self.dragging:
            self.dragging = False
            self.instruction_label.config(text="Click endpoints to select and drag\nthem for fine adjustment")
            
    def select_nearest_endpoint(self, x, y):
        """Find and select the nearest endpoint"""
        min_distance = float('inf')
        selected_measurement = None
        selected_endpoint = None
        
        # Search through all measurements
        for measurement in self.measurements:
            # Check start point
            dx = x - measurement.start_point[0]
            dy = y - measurement.start_point[1]
            distance = (dx*dx + dy*dy)**0.5
            
            if distance < min_distance:
                min_distance = distance
                selected_measurement = measurement
                selected_endpoint = 'start'
                
            # Check end point
            dx = x - measurement.end_point[0]
            dy = y - measurement.end_point[1]
            distance = (dx*dx + dy*dy)**0.5
            
            if distance < min_distance:
                min_distance = distance
                selected_measurement = measurement
                selected_endpoint = 'end'
        
        # Only select if within reasonable distance (20 pixels)
        if min_distance < 20:
            self.selected_measurement = selected_measurement
            self.selected_endpoint = selected_endpoint
            self.dragging = True
            
            point = (selected_measurement.start_point if selected_endpoint == 'start' 
                    else selected_measurement.end_point)
            self.instruction_label.config(text=f"Dragging {selected_endpoint} point\nof {selected_measurement.label}")
            
            # Highlight selected measurement
            self.highlight_selected_measurement()
        else:
            self.selected_measurement = None
            self.selected_endpoint = None
            
    def drag_endpoint(self, x, y):
        """Drag the selected endpoint to new position"""
        if not self.selected_measurement or not self.selected_endpoint:
            return
            
        # Update the endpoint position
        if self.selected_endpoint == 'start':
            self.selected_measurement.start_point = (x, y)
        else:
            self.selected_measurement.end_point = (x, y)
            
        # Update measurements list
        self.update_measurements_list()
        
        # Redraw
        if self.measurement_engine.image:
            self.load_image_into_plot()
            
    def highlight_selected_measurement(self):
        """Highlight the selected measurement"""
        if self.selected_measurement:
            # Change color temporarily to indicate selection
            original_color = self.selected_measurement.color
            self.selected_measurement.color = 'orange'  # Highlight color
            
            # Redraw
            if self.measurement_engine.image:
                self.load_image_into_plot()
                
            # Restore original color after a brief moment
            self.root.after(200, lambda: self.restore_measurement_color(original_color))
            
    def restore_measurement_color(self, original_color):
        """Restore original measurement color"""
        if self.selected_measurement:
            self.selected_measurement.color = original_color
            if self.measurement_engine.image:
                self.load_image_into_plot()
                
    def update_measurements_list(self):
        """Update the measurements list display"""
        self.measurements_listbox.delete(0, tk.END)
        for measurement in self.measurements:
            self.add_measurement_to_list(measurement)
            
    def on_key_press(self, event):
        """Handle keyboard shortcuts for nudging"""
        if not self.selected_measurement or not self.selected_endpoint:
            return
            
        # Determine nudge amount
        nudge_amount = 5 if event.state & 0x1 else 1  # Shift key for coarse nudge
        
        # Get current position
        if self.selected_endpoint == 'start':
            x, y = self.selected_measurement.start_point
        else:
            x, y = self.selected_measurement.end_point
            
        # Apply nudge based on key
        if event.keysym == 'Left':
            x -= nudge_amount
        elif event.keysym == 'Right':
            x += nudge_amount
        elif event.keysym == 'Up':
            y -= nudge_amount
        elif event.keysym == 'Down':
            y += nudge_amount
        else:
            return  # Not an arrow key
            
        # Update position
        if self.selected_endpoint == 'start':
            self.selected_measurement.start_point = (x, y)
        else:
            self.selected_measurement.end_point = (x, y)
            
        # Update display
        self.update_measurements_list()
        if self.measurement_engine.image:
            self.load_image_into_plot()
            
    def draw_first_click_marker(self, x, y):
        """Draw a temporary marker for the first click"""
        # Add a temporary cross marker
        marker_size = 8
        self.ax.plot([x-marker_size, x+marker_size], [y, y], 'r-', linewidth=2, alpha=0.7)
        self.ax.plot([x, x], [y-marker_size, y+marker_size], 'r-', linewidth=2, alpha=0.7)
        self.ax.plot(x, y, 'ro', markersize=4, alpha=0.7)
        self.canvas.draw()
        
    def apply_snap_constraint(self, start_point, end_point):
        """Apply snap-to-horizontal/vertical constraint"""
        x1, y1 = start_point
        x2, y2 = end_point
        
        if self.measurement_mode == "horizontal":
            # Force horizontal line
            return (x2, y1)
        elif self.measurement_mode == "vertical":
            # Force vertical line
            return (x1, y2)
        
        return end_point
        
    def draw_preview_line(self, start_point, current_point):
        """Draw a preview line showing snap constraint"""
        if not hasattr(self, '_preview_lines'):
            self._preview_lines = []
            
        # Remove old preview lines
        for line in self._preview_lines:
            if line in self.ax.lines:
                line.remove()
        self._preview_lines.clear()
        
        # Apply constraint to current point
        constrained_point = self.apply_snap_constraint(start_point, current_point)
        
        x1, y1 = start_point
        x2, y2 = constrained_point
        
        # Draw preview line
        preview_color = 'green' if self.measurement_mode == 'horizontal' else 'blue'
        line = self.ax.plot([x1, x2], [y1, y2], '--', 
                           color=preview_color, linewidth=2, alpha=0.6)[0]
        self._preview_lines.append(line)
        
        # No text indicators on image - keep it clean
        
        self.canvas.draw()
        
    def clear_preview_lines(self):
        """Clear all preview lines"""
        if hasattr(self, '_preview_lines'):
            for item in self._preview_lines:
                try:
                    if hasattr(item, 'remove'):
                        item.remove()
                except:
                    pass
            self._preview_lines.clear()
            
    def draw_measurement(self, measurement):
        """Draw a single measurement on the plot"""
        if not self.measurement_engine.pixels_per_mm:
            return
            
        # Calculate distance
        distance_mm = measurement.calculate_distance_mm(self.measurement_engine.pixels_per_mm)
        if distance_mm is None:
            return
            
        # Get dimension line geometry
        geometry = measurement.get_dimension_line_geometry(offset=30 + len(self.measurements) * 15)
        
        x1, y1 = measurement.start_point
        x2, y2 = measurement.end_point
        
        # Define line width for consistency
        line_width = 1.5
        
        # Draw extension lines
        for line in geometry["extension_lines"]:
            start, end = line
            self.ax.plot([start[0], end[0]], [start[1], end[1]], 
                        color=measurement.color, linewidth=line_width*0.7, alpha=0.7)
            
        # Draw dimension line with arrows
        dim_start, dim_end = geometry["dimension_line"]
        arrow = FancyArrowPatch(dim_start, dim_end,
                               arrowstyle='<->', mutation_scale=12,
                               color=measurement.color, linewidth=line_width)
        self.ax.add_patch(arrow)
        
        # No measurement text on image - keep it clean and professional
                    
        # Draw smaller endpoint markers that match line width
        marker_size = line_width + 0.5  # Slightly larger than line for visibility
        self.ax.plot(x1, y1, 'o', color=measurement.color, markersize=marker_size, 
                    markeredgewidth=0.5, markeredgecolor='white')
        self.ax.plot(x2, y2, 'o', color=measurement.color, markersize=marker_size,
                    markeredgewidth=0.5, markeredgecolor='white')
        
        self.canvas.draw()
        
    def redraw_all_measurements(self):
        """Redraw all measurements"""
        for measurement in self.measurements:
            self.draw_measurement(measurement)
            
    def add_measurement_to_list(self, measurement):
        """Add measurement to the list display"""
        if not self.measurement_engine.pixels_per_mm:
            return
            
        distance_mm = measurement.calculate_distance_mm(self.measurement_engine.pixels_per_mm)
        precision = self.precision_var.get()
        
        # Format: "Label: 12.345mm (type)"
        list_text = f"{measurement.label}: {distance_mm:.{precision}f}mm ({measurement.measurement_type})"
        self.measurements_listbox.insert(tk.END, list_text)
        
    def update_precision(self):
        """Update precision and redraw measurements"""
        self.precision_decimals = self.precision_var.get()
        
        # Update measurements list
        self.measurements_listbox.delete(0, tk.END)
        for measurement in self.measurements:
            self.add_measurement_to_list(measurement)
            
        # Redraw plot
        if self.measurement_engine.image:
            self.load_image_into_plot()
            
    def close_window(self):
        """Return to main StampZ application"""
        # Log to unified data first if measurements exist and not already logged
        if len(self.measurements) > 0 and not self.data_logged:
            result = messagebox.askyesnocancel("Log Measurements?", 
                                             f"You have {len(self.measurements)} measurements.\n\n"
                                             "Log them to unified data file before returning?")
            if result is True:  # Yes, log
                self.log_to_unified_data()
            elif result is None:  # Cancel
                return
        
        # Hide this window and show main StampZ
        self.root.withdraw()
        if self.main_app and hasattr(self.main_app, 'root'):
            self.main_app.root.deiconify()
            self.main_app.root.lift()
            self.main_app.root.focus_force()
            print("Returned to main StampZ application")
        else:
            print("Warning: Could not return to main StampZ - closing measurement tool")
            self.root.destroy()
        
    def manual_calibrate(self):
        """Perform manual calibration"""
        if len(self.measurements) == 0:
            messagebox.showwarning("No Reference", 
                                 "Please create a measurement first to use as calibration reference.\n\n"
                                 "Steps:\n"
                                 "1. Select a measurement tool (Distance/Horizontal/Vertical)\n"
                                 "2. Click two points on a feature with known size\n"
                                 "3. Enter the real-world size below\n"
                                 "4. Click Calibrate")
            return
            
        # Use the first measurement as reference
        reference = self.measurements[0]
        pixel_distance = reference.calculate_distance_pixels()
        known_mm = self.known_size_var.get()
        
        if known_mm <= 0:
            messagebox.showerror("Invalid Size", "Please enter a valid size greater than 0 mm")
            return
            
        # Update calibration
        old_dpi = self.measurement_engine.dpi
        capabilities = self.measurement_engine.calibrate_manually(pixel_distance, known_mm)
        
        # Update UI
        self.update_precision()
        
        messagebox.showinfo("Calibration Complete", 
                          f"Calibration updated!\n\n"
                          f"Reference measurement: {known_mm}mm\n"
                          f"Pixel distance: {pixel_distance:.1f} pixels\n\n"
                          f"Old DPI: {float(old_dpi):.1f}\n"
                          f"New DPI: {float(self.measurement_engine.dpi):.1f}\n"
                          f"Precision: ±{self.measurement_engine.precision_um:.1f}µm")
                          
    def toggle_auto_label(self):
        """Toggle auto-labeling of measurements"""
        self.auto_label = self.auto_label_var.get()
        
    def set_dpi_directly(self):
        """Set DPI directly without needing a reference measurement"""
        new_dpi = self.dpi_var.get()
        
        if new_dpi <= 0:
            messagebox.showerror("Invalid DPI", "Please enter a DPI value greater than 0")
            return
            
        # Update the measurement engine
        old_dpi = self.measurement_engine.dpi
        self.measurement_engine.dpi = new_dpi
        self.measurement_engine.pixels_per_mm = new_dpi / 25.4
        self.measurement_engine.precision_um = 1000 / self.measurement_engine.pixels_per_mm / 2
        self.measurement_engine.calibration_source = "User set directly"
        
        # Update UI
        self.update_precision()
        
        messagebox.showinfo("DPI Set", 
                          f"DPI updated!\n\n"
                          f"Old DPI: {float(old_dpi):.1f}\n"
                          f"New DPI: {new_dpi:.1f}\n"
                          f"Precision: ±{self.measurement_engine.precision_um:.1f}µm\n\n"
                          f"All measurements will now use {new_dpi} DPI")
                          
    def show_context_menu(self, event):
        """Show context menu for measurement editing"""
        selection = self.measurements_listbox.curselection()
        if not selection:
            return
            
        # Create context menu
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="Edit Label", command=lambda: self.edit_measurement_label(selection[0]))
        context_menu.add_command(label="Delete", command=lambda: self.delete_selected_measurement())
        
        # Show menu at cursor position
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
            
    def edit_measurement_label(self, index):
        """Edit the label of a measurement"""
        if index >= len(self.measurements):
            return
            
        measurement = self.measurements[index]
        from tkinter import simpledialog
        
        new_label = simpledialog.askstring(
            "Edit Measurement Label",
            f"Enter new label for this {measurement.measurement_type} measurement:",
            initialvalue=measurement.label
        )
        
        if new_label and new_label.strip():
            measurement.label = new_label.strip()
            self.update_measurements_list()
            
            # Reset logged flag since measurement was modified
            self.data_logged = False
            
            # Redraw if image is loaded
            if self.measurement_engine.image:
                self.load_image_into_plot()
                
    def double_click_edit(self, event):
        """Handle double-click to edit measurement label"""
        selection = self.measurements_listbox.curselection()
        if selection:
            self.edit_measurement_label(selection[0])
                
    def log_to_unified_data(self):
        """Log measurements to unified StampZ data file"""
        if not self.measurements:
            messagebox.showwarning("No Measurements", "No measurements to log.")
            return
            
        if not self.image_path:
            messagebox.showwarning("No Image Path", "Cannot log without an image path.")
            return
            
        try:
            # Import the unified logger
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils.unified_data_logger import UnifiedDataLogger
            
            # Create logger and log measurements
            logger = UnifiedDataLogger(self.image_path)
            
            # Prepare measurement data
            measurement_lines = []
            for i, measurement in enumerate(self.measurements, 1):
                distance_mm = measurement.calculate_distance_mm(self.measurement_engine.pixels_per_mm)
                precision = self.precision_var.get()
                measurement_lines.append(
                    f"  {i}. {measurement.label}: {distance_mm:.{precision}f}mm ({measurement.measurement_type})"
                )
            
            data = {
                "DPI": f"{float(self.measurement_engine.dpi):.1f}",
                "Calibration Source": self.measurement_engine.calibration_source,
                "Precision": f"±{self.measurement_engine.precision_um:.1f}µm",
                "Number of Measurements": len(self.measurements),
                "Measurements": "\n" + "\n".join(measurement_lines)
            }
            
            data_file = logger.log_section("Precision Measurements", data)
            
            if data_file:
                self.data_logged = True  # Mark as logged
                messagebox.showinfo("Data Logged", 
                                  f"Measurements logged to unified data file:\n\n"
                                  f"{data_file}\n\n"
                                  f"This file consolidates all StampZ analysis data for this image.")
            else:
                messagebox.showerror("Log Error", "Failed to log measurements to unified data file.")
                
        except ImportError:
            messagebox.showerror("Module Error", "Unified data logger not found.")
        except Exception as e:
            messagebox.showerror("Log Error", f"Failed to log measurements:\n{e}")
        
    def delete_selected_measurement(self):
        """Delete selected measurement"""
        selection = self.measurements_listbox.curselection()
        if selection:
            index = selection[0]
            del self.measurements[index]
            self.measurements_listbox.delete(index)
            
            # Reset logged flag since measurements changed
            self.data_logged = False
            
            if self.measurement_engine.image:
                self.load_image_into_plot()
                
    def clear_all_measurements(self):
        """Clear all measurements"""
        self.measurements.clear()
        self.measurements_listbox.delete(0, tk.END)
        
        # Reset logged flag since measurements cleared
        self.data_logged = False
        
        if self.measurement_engine.image:
            self.load_image_into_plot()
            
    def export_report(self):
        """Export measurement report"""
        if not self.measurements:
            messagebox.showwarning("No Measurements", "No measurements to export.")
            return
            
        report = self.generate_measurement_report()
        
        filename = filedialog.asksaveasfilename(
            title="Export Measurement Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(report)
                messagebox.showinfo("Export Complete", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save report:\\n{e}")
                
    def generate_measurement_report(self):
        """Generate measurement report text"""
        report = "STAMPZ PRECISION MEASUREMENT REPORT\\n"
        report += "=" * 45 + "\\n\\n"
        
        report += f"Image: {os.path.basename(self.image_path) if self.image_path else 'Unknown'}\\n"
        report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        
        if self.measurement_engine.dpi:
            report += f"Calibration: {float(self.measurement_engine.dpi):.1f} DPI\\n"
            report += f"Source: {self.measurement_engine.calibration_source}\\n"
            if self.measurement_engine.precision_um:
                report += f"Precision: ±{self.measurement_engine.precision_um:.1f}µm\\n"
        report += "\\n"
        
        report += "MEASUREMENTS:\\n"
        report += "-" * 15 + "\\n"
        
        for i, measurement in enumerate(self.measurements, 1):
            distance_mm = measurement.calculate_distance_mm(self.measurement_engine.pixels_per_mm)
            precision = self.precision_var.get()
            
            report += f"{i}. {measurement.measurement_type.title()}: {distance_mm:.{precision}f}mm\\n"
            report += f"   From: ({measurement.start_point[0]:.0f}, {measurement.start_point[1]:.0f})\\n"
            report += f"   To: ({measurement.end_point[0]:.0f}, {measurement.end_point[1]:.0f})\\n"
            report += f"   Created: {measurement.created.strftime('%H:%M:%S')}\\n\\n"
            
        return report
        
    def save_measurements(self):
        """Save measurements to JSON file"""
        if not self.measurements:
            messagebox.showwarning("No Measurements", "No measurements to save.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Measurements",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                data = {
                    "image_path": self.image_path,
                    "dpi": self.measurement_engine.dpi,
                    "calibration_source": self.measurement_engine.calibration_source,
                    "created": datetime.now().isoformat(),
                    "measurements": []
                }
                
                for measurement in self.measurements:
                    data["measurements"].append({
                        "id": measurement.id,
                        "start_point": measurement.start_point,
                        "end_point": measurement.end_point,
                        "measurement_type": measurement.measurement_type,
                        "label": measurement.label,
                        "color": measurement.color,
                        "created": measurement.created.isoformat()
                    })
                    
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                messagebox.showinfo("Save Complete", f"Measurements saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save measurements:\\n{e}")
                
    def load_measurements(self):
        """Load measurements from JSON file"""
        filename = filedialog.askopenfilename(
            title="Load Measurements",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    
                # Clear existing measurements
                self.clear_all_measurements()
                
                # Load measurements
                for m_data in data["measurements"]:
                    measurement = ArchitecturalMeasurement(
                        start_point=tuple(m_data["start_point"]),
                        end_point=tuple(m_data["end_point"]),
                        measurement_type=m_data["measurement_type"],
                        label=m_data["label"],
                        color=m_data["color"]
                    )
                    measurement.id = m_data["id"]
                    measurement.created = datetime.fromisoformat(m_data["created"])
                    
                    self.measurements.append(measurement)
                    self.add_measurement_to_list(measurement)
                    
                # Redraw
                if self.measurement_engine.image:
                    self.load_image_into_plot()
                    
                messagebox.showinfo("Load Complete", 
                                  f"Loaded {len(self.measurements)} measurements from {filename}")
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load measurements:\\n{e}")


def demo_precision_tool():
    """Demo the precision measurement tool"""
    root = tk.Tk()
    root.withdraw()
    
    tool = PrecisionMeasurementTool(parent=root)
    tool.root.mainloop()


if __name__ == "__main__":
    demo_precision_tool()