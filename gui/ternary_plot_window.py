#!/usr/bin/env python3
"""
Ternary Plot Window - Plot_3D style interface for RGB ternary analysis

Features:
- Plot_3D-style windowing and toolbar
- K-means clustering on ternary coordinates
- Realtime datasheet integration
- Convex hull visualization
- Interactive cluster analysis

This provides a unified visualization experience similar to Plot_3D
but focused on RGB ternary relationships.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os

# StampZ imports
from utils.color_data_bridge import ColorDataBridge
from utils.advanced_color_plots import TernaryPlotter, ColorPoint
from gui.ternary_clustering import TernaryClusterManager
from gui.ternary_datasheet import TernaryDatasheetManager
from gui.ternary_export import TernaryExportManager

# Optional ML imports for backward compatibility
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class TernaryPlotWindow:
    """Plot_3D-style window for ternary plot visualization with K-means clustering."""
    
    # Dictionary of marker sizes for different marker types (same as Plot_3D)
    MARKER_SIZES = {
        '.': 25,  # Dot
        'o': 25,  # Circle
        '*': 60,  # Star
        '^': 40,  # Triangle up
        '<': 40,  # Triangle left
        '>': 40,  # Triangle right
        'v': 40,  # Triangle down
        's': 40,  # Square
        'D': 40,  # Diamond
        '+': 50,  # Plus (increased for better visibility)
        'x': 40,  # Cross (increased for better visibility)
    }
    
    def __init__(self, parent=None, sample_set_name="StampZ_Analysis", color_points=None, datasheet_ref=None, app_ref=None, db_format=None):
        """Initialize ternary plot window.
        
        Args:
            parent: Parent window (for integration)
            sample_set_name: Name of the sample set
            color_points: Optional pre-loaded color points
            datasheet_ref: Optional reference to associated realtime datasheet
            app_ref: Optional reference to main application for integration features
            db_format: Optional detected database format ('ACTUAL_LAB', 'NORMALIZED', 'MIXED', 'UNKNOWN')
        """
        self.parent = parent
        self.sample_set_name = sample_set_name
        self.color_points = color_points or []
        self.db_format = db_format or 'UNKNOWN'
        
        # Initialize clustering manager
        self.cluster_manager = TernaryClusterManager()
        
        # Initialize datasheet manager
        self.datasheet_manager = TernaryDatasheetManager(ternary_window_ref=self)
        
        # Initialize export manager
        self.export_manager = TernaryExportManager(ternary_window_ref=self)
        
        # Keep legacy references for backward compatibility
        self.clusters = {}
        self.cluster_colors = self.cluster_manager.cluster_colors
        
        # Use command callbacks instead of trace callbacks for better macOS compatibility
        
        # Track external file for integration
        self.external_file_path = None
        
        # Reference to associated realtime datasheet for bidirectional data flow
        self.datasheet_ref = datasheet_ref
        
        # Store app reference for integration features (spectral analyzer, etc.)
        self.app_ref = app_ref
        
        # Initialize bridge and plotter
        try:
            logger.info("Initializing ColorDataBridge...")
            self.bridge = ColorDataBridge()
            
            logger.info("Initializing TernaryPlotter...")
            self.ternary_plotter = TernaryPlotter()
            
            logger.info("Creating window...")
            self._create_window()
            
            logger.info("Setting up UI...")
            self._setup_ui()
            
            # Update format indicator if we have format info
            if self.db_format != 'UNKNOWN':
                logger.info(f"Setting format indicator to: {self.db_format}")
                self._update_format_indicator(self.db_format)
            
            logger.info("Loading initial data...")
            self._load_initial_data()
            
            logger.info("Creating initial plot...")
            self._create_initial_plot()
            
            # Ensure window is visible on macOS
            import platform
            if platform.system() == 'Darwin':
                self.window.update()
                self.window.deiconify()  # Ensure it's not minimized
                self.window.lift()
            
            # If this is an orphaned window (no parent), just ensure it's visible
            if not self.parent:
                logger.info("Orphaned window detected - ensuring visibility")
            
            logger.info("TernaryPlotWindow initialization complete")
            
        except Exception as init_error:
            logger.exception(f"TernaryPlotWindow initialization failed: {init_error}")
            # Try to show error to user if possible
            try:
                import tkinter.messagebox as mb
                mb.showerror("Ternary Window Error", 
                           f"Failed to initialize ternary window:\n\n{init_error}\n\n"
                           f"Please check the console for detailed error information.")
            except:
                pass  # If even the messagebox fails, just log
            raise
    
    def _create_window(self):
        """Create the main window with Plot_3D styling."""
        if self.parent:
            self.window = tk.Toplevel(self.parent)
        else:
            self.window = tk.Tk()
        
        self.window.title("RGB Ternary Analysis")
        self.window.geometry("1200x800")
        
        # macOS-specific window handling
        import platform
        if platform.system() == 'Darwin':  # macOS
            # Ensure window is visible and comes to front
            self.window.lift()
            self.window.attributes('-topmost', True)
            self.window.after(100, lambda: self.window.attributes('-topmost', False))
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Window configuration
        self.window.resizable(True, True)
        self.window.minsize(800, 600)
        
        # NOTE: Don't set transient on macOS to avoid multi-monitor disappearing issues
        # The window can still be launched from integrated mode but remains independently movable
        import platform
        if self.parent and platform.system() != 'Darwin':
            self.window.transient(self.parent)
        # On macOS, leave window orphaned even in integrated mode for better window management
        
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Initialize tkinter variables after window creation
        self.show_hull = tk.BooleanVar(value=False)  # Start unchecked
        self.show_clusters = tk.BooleanVar(value=False) 
        self.show_spheres = tk.BooleanVar(value=False)  # Initialize spheres variable
        self.n_clusters = tk.IntVar(value=3)
        
        # macOS: Force window to appear
        if platform.system() == 'Darwin':
            self.window.focus_force()
    
    def _start_orphaned_mainloop(self):
        """Start the mainloop for orphaned windows."""
        try:
            logger.info("Starting orphaned window mainloop...")
            self.window.mainloop()
        except Exception as e:
            logger.exception(f"Error in orphaned mainloop: {e}")
    
    def _setup_ui(self):
        """Setup the user interface with modern Plot_3D-style layout."""
        # Main container with grid layout
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure grid weights for responsive layout
        main_frame.grid_columnconfigure(0, weight=2)  # Plot area gets ~40% width
        main_frame.grid_columnconfigure(1, weight=3)  # Control panel gets ~60% width
        main_frame.grid_rowconfigure(0, weight=1)     # Content area expands
        main_frame.grid_rowconfigure(1, weight=0)     # Status bar fixed height
        
        # === LEFT SIDE: Plot Area ===
        plot_frame = ttk.Frame(main_frame, relief='solid', borderwidth=1)
        plot_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=(0, 5))
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(1, weight=1)  # Canvas expands
        
        # Plot title and info
        self.title_frame = ttk.Frame(plot_frame)
        self.title_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=(5, 2))
        
        self.plot_title_label = ttk.Label(self.title_frame, text=f"RGB Ternary Analysis: {self.sample_set_name}", 
                 font=('Arial', 12, 'bold'))
        self.plot_title_label.pack(side=tk.LEFT)
        
        # Format indicator in title area
        self.format_frame = ttk.Frame(self.title_frame)
        self.format_frame.pack(side=tk.RIGHT)
        ttk.Label(self.format_frame, text="Format:", font=('Arial', 9)).pack(side=tk.LEFT, padx=(0, 2))
        self.format_indicator = ttk.Label(self.format_frame, text="UNKNOWN", font=('Arial', 9, 'bold'), 
                                        foreground='gray', background='lightgray', relief='sunken', width=12)
        self.format_indicator.pack(side=tk.LEFT)
        
        # Canvas container with navigation toolbar
        canvas_container = ttk.Frame(plot_frame)
        canvas_container.grid(row=1, column=0, sticky='nsew', padx=5, pady=(0, 5))
        canvas_container.grid_columnconfigure(0, weight=1)
        canvas_container.grid_rowconfigure(1, weight=1)  # Canvas expands
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_container)
        
        # Navigation toolbar
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, canvas_container)
        self.nav_toolbar.grid(row=0, column=0, sticky='ew', pady=(0, 2))
        self.nav_toolbar.update()
        
        # Canvas
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        
        # Add click handler for point selection
        self.selected_points = set()
        self.point_labels = []
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        
        # === RIGHT SIDE: Control Panel ===
        self._create_control_panel(main_frame)
        
        # === BOTTOM: Status Bar ===
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, font=('Arial', 9))
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(5, 0))
    
    def _create_control_panel(self, parent):
        """Create modern, right-side control panel with organized sections."""
        # Control panel frame with scroll support and minimum width
        control_outer_frame = ttk.Frame(parent, relief='groove', borderwidth=2)
        control_outer_frame.grid(row=0, column=1, sticky='nsew', ipadx=15, ipady=8)
        control_outer_frame.grid_columnconfigure(0, weight=1, minsize=450)  # Minimum 450px width
        control_outer_frame.grid_rowconfigure(0, weight=1)
        
        # Create a canvas for scrolling
        control_canvas = tk.Canvas(control_outer_frame)
        control_canvas.grid(row=0, column=0, sticky='nsew')
        
        # Scrollbar for canvas
        scrollbar = ttk.Scrollbar(control_outer_frame, orient="vertical", command=control_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        control_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create inner frame for controls
        self.control_frame = ttk.Frame(control_canvas)
        control_window = control_canvas.create_window((0, 0), window=self.control_frame, anchor='nw')
        
        # Configure inner frame
        self.control_frame.grid_columnconfigure(0, weight=1)
        
        # Update scroll region when size changes
        def update_scrollregion(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
            # Also update canvas width to match frame width
            control_canvas.itemconfig(control_window, width=event.width)
        
        self.control_frame.bind('<Configure>', update_scrollregion)
        
        # Control Panel Title
        title_lbl = ttk.Label(self.control_frame, text="Controls", 
                            font=('Arial', 12, 'bold'), anchor='center')
        title_lbl.grid(row=0, column=0, pady=(10, 5), sticky='ew')
        
        # Add sections
        row_idx = 1
        
        # SECTION 1: DATA CONTROLS
        row_idx = self._create_data_controls_section(row_idx)
        
        # SECTION 2: VISUALIZATION OPTIONS
        row_idx = self._create_visualization_section(row_idx)
        
        # SECTION 3: K-MEANS CLUSTERING
        row_idx = self._create_clustering_section(row_idx)
        
        # SECTION 4: EXPORT OPTIONS
        row_idx = self._create_export_section(row_idx)
        
        # SECTION 5: APPLICATION CONTROL
        row_idx = self._create_app_control_section(row_idx)
        
        # Add empty space at the bottom
        spacer = ttk.Frame(self.control_frame)
        spacer.grid(row=row_idx, column=0, sticky='ew', pady=10)
    
    def _create_section_header(self, title, row):
        """Create a section header with title and separator."""
        # Section Frame
        section_frame = ttk.Frame(self.control_frame, relief='flat')
        section_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=(15, 0))
        section_frame.grid_columnconfigure(0, weight=1)
        
        # Section Title
        header = ttk.Label(section_frame, text=title, font=('Arial', 10, 'bold'), foreground='#2a6eb2')
        header.grid(row=0, column=0, sticky='w')
        
        # Separator
        sep = ttk.Separator(self.control_frame, orient='horizontal')
        sep.grid(row=row+1, column=0, sticky='ew', padx=10, pady=(2, 5))
        
        return section_frame, row+2
    
    def _create_data_controls_section(self, start_row):
        """Create data controls section."""
        section_frame, row_idx = self._create_section_header("Data Controls", start_row)
        
        # Container for database info
        info_frame = ttk.Frame(self.control_frame)
        info_frame.grid(row=row_idx, column=0, sticky='ew', padx=20)
        info_frame.grid_columnconfigure(0, weight=1)
        row_idx += 1
        
        # Sample set info
        sample_frame = ttk.Frame(info_frame)
        sample_frame.grid(row=0, column=0, sticky='ew', pady=2)
        ttk.Label(sample_frame, text="Sample Set:", width=10).pack(side=tk.LEFT)
        ttk.Label(sample_frame, text=self.sample_set_name, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Load/Reload buttons in grid layout
        buttons_frame = ttk.Frame(self.control_frame)
        buttons_frame.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=5)
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        row_idx += 1
        
        ttk.Button(buttons_frame, text="Load DB", command=self._load_data_dialog).grid(row=0, column=0, sticky='ew', padx=2, pady=4, ipadx=25)
        ttk.Button(buttons_frame, text="Load File", command=self._load_external_file).grid(row=0, column=1, sticky='ew', padx=2, pady=4, ipadx=25)
        ttk.Button(buttons_frame, text="↻ Reload", command=self._reload_current_database).grid(row=1, column=0, sticky='ew', padx=2, pady=4, ipadx=25)
        ttk.Button(buttons_frame, text="Refresh", command=self._refresh_plot_only).grid(row=1, column=1, sticky='ew', padx=2, pady=4, ipadx=25)
        
        # Datasheet sync button
        self.refresh_from_datasheet_btn = ttk.Button(self.control_frame, text="↻ Sync from Datasheet", 
                                                  command=self._refresh_from_datasheet)
        self.refresh_from_datasheet_btn.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=4, ipadx=15)
        self.refresh_from_datasheet_btn.config(state='disabled')  # Initially disabled
        row_idx += 1
        
        # Integration buttons
        integration_frame = ttk.Frame(self.control_frame)
        integration_frame.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=5)
        integration_frame.grid_columnconfigure(0, weight=1)
        integration_frame.grid_columnconfigure(1, weight=1)
        row_idx += 1
        
        ttk.Button(integration_frame, text="📊 Open Datasheet", 
                  command=self._open_plot3d).grid(row=0, column=0, columnspan=2, sticky='ew', padx=4, pady=4, ipadx=15)
        ttk.Button(integration_frame, text="🔬 Spectral Analysis", 
                  command=self._open_spectral_analyzer).grid(row=1, column=0, columnspan=2, sticky='ew', padx=4, pady=4, ipadx=15)
        
        return row_idx
    
    def _create_visualization_section(self, start_row):
        """Create visualization options section."""
        section_frame, row_idx = self._create_section_header("Visualization Options", start_row)
        
        # Hull display
        hull_frame = ttk.Frame(self.control_frame)
        hull_frame.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=3)
        row_idx += 1
        
        # Convex Hull checkbox with improved styling - using manual toggle for macOS compatibility
        hull_cb = ttk.Checkbutton(hull_frame, text="Show Convex Hull", 
                                 variable=self.show_hull, 
                                 command=self._manual_hull_toggle)
        hull_cb.pack(fill=tk.X, padx=5)
        
        # Spheres checkbox
        spheres_frame = ttk.Frame(self.control_frame)
        spheres_frame.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=3)
        row_idx += 1
        
        spheres_cb = ttk.Checkbutton(spheres_frame, text="Use Sphere Visualization", 
                                    variable=self.show_spheres, 
                                    command=self._manual_spheres_toggle)
        spheres_cb.pack(fill=tk.X, padx=5)
        
        return row_idx
    
    def _create_clustering_section(self, start_row):
        """Create K-means clustering section."""
        section_frame, row_idx = self._create_section_header("K-means Clustering", start_row)
        
        # Clusters control
        cluster_frame = ttk.Frame(self.control_frame)
        cluster_frame.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=3)
        cluster_frame.grid_columnconfigure(1, weight=1)
        row_idx += 1
        
        # K-means activation checkbox - using manual toggle for macOS compatibility
        clusters_cb = ttk.Checkbutton(cluster_frame, text="Enable K-means Clusters", 
                                     variable=self.show_clusters, 
                                     command=self._manual_clusters_toggle)
        clusters_cb.grid(row=0, column=0, columnspan=2, sticky='w', padx=5)
        
        # Cluster count control
        ttk.Label(cluster_frame, text="Number of Clusters:").grid(row=1, column=0, sticky='w', padx=5, pady=(5, 0))
        
        # Spinbox with modern styling
        spinbox_frame = ttk.Frame(cluster_frame)
        spinbox_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=(2, 5))
        
        cluster_spin = ttk.Spinbox(spinbox_frame, from_=2, to=8, width=3, textvariable=self.n_clusters,
                                   command=self._update_clusters)
        cluster_spin.pack(side=tk.LEFT)
        self.cluster_spin = cluster_spin
        cluster_spin.set(self.n_clusters.get())
        
        # Add event bindings
        cluster_spin.bind('<KeyRelease>', lambda e: self._on_spinbox_change())
        cluster_spin.bind('<Button-1>', lambda e: self.window.after(100, self._on_spinbox_change))
        cluster_spin.bind('<ButtonRelease-1>', lambda e: self.window.after(100, self._on_spinbox_change))
        
        # NEW: Manual Save Clusters button
        save_clusters_btn = ttk.Button(cluster_frame, text="💾 Save Clusters to Database", 
                                      command=self._save_clusters_to_database)
        save_clusters_btn.grid(row=3, column=0, columnspan=2, sticky='ew', padx=8, pady=(8, 0), ipadx=15)
        
        return row_idx
    
    def _create_export_section(self, start_row):
        """Create export options section."""
        section_frame, row_idx = self._create_section_header("Export Options", start_row)
        
        # Export buttons
        export_frame = ttk.Frame(self.control_frame)
        export_frame.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=5)
        export_frame.grid_columnconfigure(0, weight=1)
        row_idx += 1
        
        ttk.Button(export_frame, text="💾 Save As Database", 
                  command=self._save_as_database).grid(row=0, column=0, sticky='ew', pady=4, ipadx=15)
        ttk.Button(export_frame, text="📸 Save Plot as PNG", 
                  command=self._save_plot).grid(row=1, column=0, sticky='ew', pady=4, ipadx=15)
        ttk.Button(export_frame, text="📤 Export Data", 
                  command=self._export_data).grid(row=2, column=0, sticky='ew', pady=4, ipadx=15)
        
        return row_idx
    
    def _create_app_control_section(self, start_row):
        """Create application control section with exit buttons."""
        section_frame, row_idx = self._create_section_header("Application Control", start_row)
        
        # Exit buttons - standardized for both internal and standalone modes
        exit_frame = ttk.Frame(self.control_frame)
        exit_frame.grid(row=row_idx, column=0, sticky='ew', padx=20, pady=5)
        exit_frame.grid_columnconfigure(0, weight=1)
        row_idx += 1
        
        # Both modes show the same buttons
        ttk.Button(exit_frame, text="← Return to Launcher", 
                  command=self._exit_ternary).grid(row=0, column=0, sticky='ew', pady=4, ipadx=15)
        
        ttk.Button(exit_frame, text="🚪 Exit Application", 
                  command=self._exit_application).grid(row=1, column=0, sticky='ew', pady=4, ipadx=15)
        
        return row_idx
    
    def _load_initial_data(self):
        """Load initial data if color_points not provided."""
        if not self.color_points:
            # Don't auto-load if launched with placeholder name indicating user selection required
            if self.sample_set_name == "No Database Selected":
                logger.info("DEBUG: Ternary window launched without database - user will select manually")
                self._update_status("Ready - Use 'Load Database' to select a database")
                self._update_format_indicator('UNKNOWN')
                return
                
            try:
                self.color_points, db_format = self.bridge.load_color_points_from_database(self.sample_set_name)
                logger.info(f"DEBUG: Loaded {len(self.color_points)} color points from {self.sample_set_name} (Format: {db_format})")
                
                # Update format indicator
                self._update_format_indicator(db_format)
                
                # Debug: Print first few points to verify loading
                for i, point in enumerate(self.color_points[:5]):
                    logger.info(f"DEBUG Point {i}: ID={point.id}, RGB={point.rgb}, Lab={point.lab}")
                
                self._update_status(f"Loaded {len(self.color_points)} color points ({db_format} format)")
            except Exception as e:
                logger.exception(f"Failed to load initial data: {e}")
                self._update_status(f"No data loaded: {e}")
                self._update_format_indicator('UNKNOWN')
    
    def _create_initial_plot(self):
        """Create the initial ternary plot."""
        self._refresh_plot()
    
    def _refresh_plot(self):
        """Refresh the ternary plot with current settings.
        
        Legacy method - maintains original behavior for backward compatibility.
        """
        # This is now just a redirection to the appropriate method based on context
        if self.datasheet_manager.is_datasheet_linked():
            self._refresh_from_datasheet()
        else:
            self._refresh_plot_only()
    
    def _refresh_plot_only(self):
        """Refresh only the plot without syncing from datasheet."""
        # This refreshes the plot using current color_points without datasheet sync
        if not self.color_points:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No data available\nUse "Load Database" to load sample set', 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        try:
            # Clear and create new plot
            self.ax.clear()
            
            # Create base ternary plot
            self._draw_ternary_framework()
            self._plot_data_points()
            
            # Plot clusters if enabled and available
            print(f"DEBUG: In _refresh_plot_only - show_clusters: {self.show_clusters.get()}, has_sklearn: {self.cluster_manager.has_sklearn()}")
            if self.show_clusters.get() and self.cluster_manager.has_sklearn():
                # Ensure clusters are computed if checkbox is on but no clusters exist
                if not hasattr(self, 'clusters') or not self.clusters:
                    print("DEBUG: Clusters enabled but not computed, computing now...")
                    logger.info("Clusters enabled but not computed, computing now...")
                    self._compute_clusters()
                
                # Plot clusters if they exist
                if hasattr(self, 'clusters') and self.clusters:
                    print(f"DEBUG: Plotting {len(self.clusters)} clusters")
                    self._plot_clusters()
                else:
                    print("DEBUG: No clusters to plot after computation attempt")
            
            # Update plot and force axis limits (prevent matplotlib auto-adjustment)
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            
            # Force axis limits to prevent compression when clusters/legends are added/removed
            self.ax.set_xlim(-0.15, 1.15)
            self.ax.set_ylim(-0.15, 1.05)
            
            self.canvas.draw()
            
            self._update_status(f"Plot refreshed: {len(self.color_points)} points")
            
        except Exception as e:
            logger.exception("Error refreshing plot")
            self._update_status(f"Plot error: {e}")
    
    def _refresh_from_datasheet(self):
        """Refresh the plot by syncing data from linked datasheet using datasheet manager."""
        if not self.datasheet_manager.is_datasheet_linked():
            self._update_status("No datasheet linked for sync")
            return
        
        try:
            logger.info("🔄 REFRESH DEBUG: Starting datasheet sync from ternary plot window")
            print(f"\n=== REFRESH FROM DATASHEET DEBUG ===")
            print(f"BEFORE sync: self.color_points length = {len(self.color_points) if self.color_points else 0}")
            print(f"=====================================\n")
            self._update_status("Syncing from datasheet...")
            
            # Use datasheet manager to sync data
            updated_color_points = self.datasheet_manager.sync_from_datasheet()
            
            if updated_color_points:
                logger.info(f"🔄 REFRESH DEBUG: Received {len(updated_color_points)} updated points from datasheet")
                
                # Debug: Show first few updated points with their metadata
                for i, point in enumerate(updated_color_points[:3]):
                    if hasattr(point, 'metadata'):
                        logger.info(f"🔄 REFRESH DEBUG: Point {i} ({point.id}): marker={point.metadata.get('marker', 'N/A')}, color={point.metadata.get('marker_color', 'N/A')}")
                
                # Update the color points
                self.color_points = updated_color_points
                
                # Force a complete plot refresh to show updated markers/colors
                self._refresh_plot_only()
                
                # Update status with success message
                points_with_prefs = sum(1 for p in self.color_points 
                                      if hasattr(p, 'metadata') and 
                                         (p.metadata.get('marker', '.') != '.' or 
                                          p.metadata.get('marker_color', 'blue') != 'blue'))
                
                self._update_status(f"Synced {len(self.color_points)} points ({points_with_prefs} with custom preferences)")
                logger.info("🔄 REFRESH DEBUG: Plot refresh completed with synced data")
                
                # Flash the window to indicate sync completed
                if hasattr(self, 'window'):
                    self.window.bell()  # Audio feedback
                
            else:
                logger.warning("🔄 REFRESH DEBUG: No data synced from datasheet")
                self._update_status("No data synced from datasheet")
                
        except Exception as e:
            logger.exception(f"🔄 REFRESH DEBUG: Failed to sync from datasheet: {e}")
            self._update_status(f"Datasheet sync error: {e}")
            # Fall back to regular refresh
            self._refresh_plot_only()
    
    def _draw_ternary_framework(self):
        """Draw ternary triangle framework."""
        # Triangle coordinates (same as TernaryPlotter)
        coords = {
            'red': (0.5, 0.866025404),    # Top vertex
            'green': (0.0, 0.0),          # Bottom left
            'blue': (1.0, 0.0)            # Bottom right
        }
        
        # Draw triangle outline
        triangle_x = [coords['green'][0], coords['blue'][0], coords['red'][0], coords['green'][0]]
        triangle_y = [coords['green'][1], coords['blue'][1], coords['red'][1], coords['green'][1]]
        self.ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
        
        # Enhanced vertex labels
        self.ax.text(coords['red'][0], coords['red'][1] + 0.08, 'RED\n100%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))
        self.ax.text(coords['green'][0] - 0.08, coords['green'][1], 'GREEN\n100%', 
                    ha='right', va='center', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
        self.ax.text(coords['blue'][0] + 0.08, coords['blue'][1], 'BLUE\n100%', 
                    ha='left', va='center', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.2))
        
        # Draw grid lines
        for pct in [0.2, 0.4, 0.6, 0.8]:
            self._draw_percentage_lines(pct, coords)
        
        # Add percentage scales along each side
        self._add_side_legends(coords)
        
        # Set limits
        self.ax.set_xlim(-0.15, 1.15)
        self.ax.set_ylim(-0.15, 1.05)
    
    def _draw_percentage_lines(self, percentage, coords):
        """Draw percentage grid lines."""
        # Lines parallel to each side
        # Red percentage lines (parallel to green-blue edge)
        p1 = (coords['green'][0] + percentage * (coords['red'][0] - coords['green'][0]),
              coords['green'][1] + percentage * (coords['red'][1] - coords['green'][1]))
        p2 = (coords['blue'][0] + percentage * (coords['red'][0] - coords['blue'][0]),
              coords['blue'][1] + percentage * (coords['red'][1] - coords['blue'][1]))
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.3, linewidth=0.5)
        
        # Green percentage lines
        p3 = (coords['red'][0] + percentage * (coords['green'][0] - coords['red'][0]),
              coords['red'][1] + percentage * (coords['green'][1] - coords['red'][1]))
        p4 = (coords['blue'][0] + percentage * (coords['green'][0] - coords['blue'][0]),
              coords['blue'][1] + percentage * (coords['green'][1] - coords['blue'][1]))
        self.ax.plot([p3[0], p4[0]], [p3[1], p4[1]], 'k--', alpha=0.3, linewidth=0.5)
        
        # Blue percentage lines
        p5 = (coords['red'][0] + percentage * (coords['blue'][0] - coords['red'][0]),
              coords['red'][1] + percentage * (coords['blue'][1] - coords['red'][1]))
        p6 = (coords['green'][0] + percentage * (coords['blue'][0] - coords['green'][0]),
              coords['green'][1] + percentage * (coords['blue'][1] - coords['green'][1]))
        self.ax.plot([p5[0], p6[0]], [p5[1], p6[1]], 'k--', alpha=0.3, linewidth=0.5)
    
    def _add_side_legends(self, coords):
        """Add percentage scales along each side of the triangle."""
        # Red axis (left side, from green to red vertex)
        for pct in [0.2, 0.4, 0.6, 0.8]:
            # Position along green-red edge
            x = coords['green'][0] + pct * (coords['red'][0] - coords['green'][0])
            y = coords['green'][1] + pct * (coords['red'][1] - coords['green'][1])
            
            # Red percentage label (left side)
            self.ax.text(x - 0.05, y, f'{int(pct*100)}%', 
                        ha='right', va='center', fontsize=8, 
                        color='red', fontweight='bold')
        
        # Green axis (bottom side, from blue to green vertex) 
        for pct in [0.2, 0.4, 0.6, 0.8]:
            # Position along blue-green edge
            x = coords['blue'][0] + pct * (coords['green'][0] - coords['blue'][0])
            y = coords['blue'][1] + pct * (coords['green'][1] - coords['blue'][1])
            
            # Green percentage label (bottom)
            self.ax.text(x, y - 0.05, f'{int(pct*100)}%', 
                        ha='center', va='top', fontsize=8, 
                        color='green', fontweight='bold')
        
        # Blue axis (right side, from red to blue vertex)
        for pct in [0.2, 0.4, 0.6, 0.8]:
            # Position along red-blue edge
            x = coords['red'][0] + pct * (coords['blue'][0] - coords['red'][0])
            y = coords['red'][1] + pct * (coords['blue'][1] - coords['red'][1])
            
            # Blue percentage label (right side)
            self.ax.text(x + 0.05, y, f'{int(pct*100)}%', 
                        ha='left', va='center', fontsize=8, 
                        color='blue', fontweight='bold')
    
    def _plot_data_points(self):
        """Plot the color data points with proper marker types and highlighting."""
        if not self.color_points:
            logger.warning("DEBUG: No color_points available for plotting")
            return
        
        logger.info(f"DEBUG: Starting to plot {len(self.color_points)} color points")
        
        # Color map for named colors from datasheet
        color_map = {
            'red': '#FF0000', 'blue': '#0000FF', 'green': '#00FF00', 'yellow': '#FFFF00',
            'purple': '#800080', 'orange': '#FFA500', 'pink': '#FFC0CB', 'brown': '#A52A2A',
            'black': '#000000', 'white': '#FFFFFF', 'gray': '#808080', 'grey': '#808080',
            'cyan': '#00FFFF', 'magenta': '#FF00FF'
        }
        
        # Convert Plot_3D marker symbols to matplotlib symbols
        marker_map = {'.': 'o', 'o': 'o', 's': 's', '^': '^', 'v': 'v', 
                     'd': 'D', 'x': 'x', '+': '+', '*': '*'}
        
        # Group points by marker type for efficient plotting
        marker_groups = {}
        
        plotted_count = 0
        for i, point in enumerate(self.color_points):
            logger.debug(f"DEBUG: Processing point {i}: ID={point.id}, RGB={point.rgb}")
            
            # Ensure ternary coordinates are computed
            if not hasattr(point, 'ternary_coords') or point.ternary_coords is None:
                point.ternary_coords = self.ternary_plotter.rgb_to_ternary(point.rgb)
            
            logger.debug(f"DEBUG: Point {i} ternary coords: {point.ternary_coords}")
            
            # Get marker style from metadata (use ternary-specific keys)
            marker_style = 'o'  # default
            if hasattr(point, 'metadata') and 'ternary_marker' in point.metadata and point.metadata['ternary_marker']:
                marker_style = marker_map.get(point.metadata['ternary_marker'], 'o')
            
            # Get color from metadata or RGB (use ternary-specific keys)
            if (hasattr(point, 'metadata') and 'ternary_marker_color' in point.metadata and 
                point.metadata['ternary_marker_color'] and point.metadata['ternary_marker_color'].lower() in color_map):
                point_color = color_map[point.metadata['ternary_marker_color'].lower()]
            else:
                # Use RGB color of the point
                point_color = tuple(c/255.0 for c in point.rgb)
            
            logger.debug(f"DEBUG: Point {i} - marker:{marker_style}, color:{point_color}")
            plotted_count += 1
            
            # Get marker-specific base size from dictionary (like Plot_3D)
            base_size = self.MARKER_SIZES.get(marker_style, 25)
            
            # Determine size and edge color based on selection
            if i in getattr(self, 'selected_points', set()):
                size = base_size * 6  # Much larger size for selected (6x multiplier)
                edgecolor = '#FFD700'  # Gold color for better visibility
                linewidth = 4  # Thicker border
            else:
                size = base_size  # Use marker-specific size
                edgecolor = 'black'
                linewidth = 0.5
            
            # Group by marker type
            if marker_style not in marker_groups:
                marker_groups[marker_style] = {
                    'x': [], 'y': [], 'colors': [], 'sizes': [], 
                    'edgecolors': [], 'linewidths': [], 'indices': []
                }
            
            marker_groups[marker_style]['x'].append(point.ternary_coords[0])
            marker_groups[marker_style]['y'].append(point.ternary_coords[1])
            marker_groups[marker_style]['colors'].append(point_color)
            marker_groups[marker_style]['sizes'].append(size)
            marker_groups[marker_style]['edgecolors'].append(edgecolor)
            marker_groups[marker_style]['linewidths'].append(linewidth)
            marker_groups[marker_style]['indices'].append(i)
        
        logger.info(f"DEBUG: Processed {plotted_count} points, created {len(marker_groups)} marker groups")
        for marker_style, group_data in marker_groups.items():
            logger.info(f"DEBUG: Marker group '{marker_style}' has {len(group_data['x'])} points")
        
        # Plot each marker type separately
        self.plotted_points = {}  # Store for click detection
        
        # Plot spheres if requested (as background elements)
        if self.show_spheres.get():
            self._plot_spheres()
        
        total_plotted_points = 0
        for marker_style, group_data in marker_groups.items():
            logger.info(f"DEBUG: Plotting {len(group_data['x'])} points for marker '{marker_style}'")
            scatter = self.ax.scatter(
                group_data['x'], group_data['y'], 
                c=group_data['colors'], 
                s=group_data['sizes'], 
                marker=marker_style,
                alpha=0.8, 
                edgecolors=group_data['edgecolors'], 
                linewidths=group_data['linewidths'], 
                zorder=3,
                picker=True  # Enable picking for click detection
            )
            
            total_plotted_points += len(group_data['x'])
            logger.info(f"DEBUG: Scatter plot created for marker '{marker_style}' with {len(group_data['x'])} points")
            
            # Store mapping for click detection
            for i, point_idx in enumerate(group_data['indices']):
                self.plotted_points[point_idx] = {
                    'x': group_data['x'][i],
                    'y': group_data['y'][i], 
                    'scatter': scatter
                }
        
        logger.info(f"DEBUG: TOTAL PLOTTED POINTS: {total_plotted_points} out of {len(self.color_points)} available")
        
        # Check for overlapping coordinates and apply display jitter if needed (without modifying actual data)
        coordinates = [(point.ternary_coords[0], point.ternary_coords[1]) for point in self.color_points]
        unique_coordinates = set(coordinates)
        if len(unique_coordinates) != len(coordinates):
            logger.warning(f"DEBUG: Found overlapping coordinates! {len(coordinates)} points but only {len(unique_coordinates)} unique positions")
            # Find duplicates
            from collections import Counter
            coord_counts = Counter(coordinates)
            duplicates = {coord: count for coord, count in coord_counts.items() if count > 1}
            logger.warning(f"DEBUG: Duplicate coordinates: {duplicates}")
            
            # Apply display jitter to marker groups WITHOUT modifying actual point data
            logger.info("DEBUG: Applying display jitter to overlapping points for visibility...")
            import random
            random.seed(42)  # Consistent jitter across runs
            
            # Apply jitter to the plotting coordinates in marker_groups, not the actual data
            coord_seen = {}
            jittered_count = 0
            
            for marker_style, group_data in marker_groups.items():
                for i in range(len(group_data['x'])):
                    coord_key = (round(group_data['x'][i], 6), round(group_data['y'][i], 6))
                    if coord_key in coord_seen:
                        # Add small random offset for display only
                        jitter_x = random.uniform(-0.015, 0.015)
                        jitter_y = random.uniform(-0.015, 0.015)
                        group_data['x'][i] = max(0.0, min(1.0, group_data['x'][i] + jitter_x))
                        group_data['y'][i] = max(0.0, min(1.0, group_data['y'][i] + jitter_y))
                        coord_seen[coord_key] += 1
                        jittered_count += 1
                        if jittered_count <= 5:  # Log first 5 for debugging
                            point_idx = group_data['indices'][i]
                            logger.debug(f"DEBUG: Display jitter applied to point {self.color_points[point_idx].id}: {coord_key} -> ({group_data['x'][i]:.4f}, {group_data['y'][i]:.4f})")
                    else:
                        coord_seen[coord_key] = 1
                        
            logger.info(f"DEBUG: Applied display jitter to {jittered_count} overlapping points (original data unchanged)")
        else:
            logger.info(f"DEBUG: All {len(coordinates)} points have unique coordinates")
        
        # Add point labels for small datasets
        if len(self.color_points) <= 15:
            for i, point in enumerate(self.color_points):
                label = point.id
                if hasattr(point, 'metadata') and 'marker' in point.metadata and point.metadata['marker']:
                    label += f" ({point.metadata['marker']})"
                if hasattr(point, 'metadata') and 'marker_color' in point.metadata and point.metadata['marker_color']:
                    label += f" [{point.metadata['marker_color']}]"
                self.ax.annotate(label, (point.ternary_coords[0], point.ternary_coords[1]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.7)
        
        # Add convex hull if requested
        if self.show_hull.get() and len(self.color_points) >= 3:
            print(f"DEBUG: Attempting to draw convex hull for {len(self.color_points)} points")
            x_coords = [p.ternary_coords[0] for p in self.color_points]
            y_coords = [p.ternary_coords[1] for p in self.color_points]
            self._add_convex_hull(x_coords, y_coords)
        else:
            print(f"DEBUG: Convex hull not drawn - show_hull: {self.show_hull.get()}, points: {len(self.color_points) if self.color_points else 0}")
    
    def _add_convex_hull(self, x_coords, y_coords):
        """Add convex hull to the plot."""
        try:
            print(f"DEBUG: _add_convex_hull called with {len(x_coords)} coordinates")
            from scipy.spatial import ConvexHull
            points = np.column_stack((x_coords, y_coords))
            print(f"DEBUG: Created points array with shape: {points.shape}")
            hull = ConvexHull(points)
            print(f"DEBUG: ConvexHull computed with {len(hull.vertices)} vertices")
            
            # Plot hull with high visibility
            hull_lines_added = 0
            for simplex in hull.simplices:
                # Make hull lines much more visible
                self.ax.plot(points[simplex, 0], points[simplex, 1], 
                           'red', alpha=0.8, linewidth=3, linestyle='--', zorder=10)
                hull_lines_added += 1
            print(f"DEBUG: Added {hull_lines_added} hull edge lines")
            
            # Fill hull area with more visible color
            hull_points = points[hull.vertices]
            self.ax.fill(hull_points[:, 0], hull_points[:, 1], 
                        alpha=0.2, color='yellow', zorder=1)
            print(f"DEBUG: Added hull fill area")
            
            # Force canvas refresh to make sure changes are visible
            if hasattr(self, 'canvas'):
                self.canvas.draw_idle()
                print(f"DEBUG: Forced canvas refresh")
            
        except ImportError as e:
            print(f"DEBUG: scipy not available for convex hull: {e}")
        except Exception as e:
            print(f"DEBUG: Convex hull failed with error: {e}")
            logger.warning(f"Convex hull failed: {e}")
    
    def _plot_spheres(self):
        """Plot sphere-style cluster visualization using cluster manager."""
        if not self.clusters:
            return
        
        # Use cluster manager to create sphere patches
        try:
            patches = self.cluster_manager.create_sphere_patches()
            
            for circle, color, label in patches:
                self.ax.add_patch(circle)
            
            # Create centroid markers
            markers = self.cluster_manager.create_centroid_markers()
            
            for marker in markers:
                self.ax.scatter(marker['x'], marker['y'], 
                              c=[marker['color']], s=marker['size'], 
                              marker=marker['marker'], 
                              edgecolors='black', linewidths=1, 
                              zorder=2, alpha=0.7, 
                              label=marker['label'])
        
        except Exception as e:
            logger.warning(f"Failed to plot spheres using cluster manager: {e}")
            # Fallback to legacy method
            self._plot_spheres_legacy()
        
        # Legend will be added by the main _plot_clusters method to avoid duplicates
    
    def _plot_spheres_legacy(self):
        """Legacy sphere plotting method as fallback."""
        if not self.clusters:
            return
        
        # Plot cluster spheres as background elements
        for i, (label, points) in enumerate(self.clusters.items()):
            color = self.cluster_colors[i % len(self.cluster_colors)]
            
            # Extract coordinates for this cluster
            x_coords = [p.ternary_coords[0] for p in points]
            y_coords = [p.ternary_coords[1] for p in points]
            
            # Calculate cluster center and spread
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            # Set sphere radius to represent ΔE (0.02 for normalized coordinates)
            radius = 0.02  # Fixed radius representing ΔE like in Plot_3D
            
            # Draw sphere as a circle
            circle = plt.Circle((center_x, center_y), radius, 
                              color=color, alpha=0.15, zorder=1)
            self.ax.add_patch(circle)
            
            # Add subtle centroid marker (smaller and less intrusive)
            self.ax.scatter(center_x, center_y, 
                          c=[color], s=60, marker='o', 
                          edgecolors='black', linewidths=1, 
                          zorder=2, alpha=0.7, 
                          label=f'Cluster {label} ({len(points)} pts)')
    
    def _toggle_clusters(self):
        """Toggle K-means cluster display."""
        print(f"DEBUG: _toggle_clusters called, checkbox state: {self.show_clusters.get()}")
        print(f"DEBUG: Available color points: {len(self.color_points) if self.color_points else 0}")
        
        if self.show_clusters.get():
            print("DEBUG: K-means checkbox is ON, checking sklearn...")
            if not self.cluster_manager.has_sklearn():
                print("DEBUG: sklearn not available!")
                messagebox.showwarning("K-means Not Available", 
                                     "Scikit-learn is required for K-means clustering.\n\n"
                                     "Install with: pip install scikit-learn")
                self.show_clusters.set(False)
                return
            print("DEBUG: sklearn available, calling _compute_clusters...")
            self._compute_clusters()
        else:
            print("DEBUG: K-means checkbox is OFF, clearing clusters")
            # Clear clusters when toggled off
            self.clusters = {}
            self.cluster_manager.clear_clusters()
        print("DEBUG: Calling _refresh_plot...")
        self._refresh_plot()
    
    def _save_original_coordinates(self):
        """Save original ternary coordinates before clustering transformations."""
        if not self.color_points:
            return
        
        print("DEBUG: Saving original coordinates for restoration")
        self.original_coordinates = {}
        
        for i, point in enumerate(self.color_points):
            if hasattr(point, 'ternary_coords') and point.ternary_coords is not None:
                # Store a deep copy of the original coordinates
                self.original_coordinates[i] = tuple(point.ternary_coords)
        
        print(f"DEBUG: Saved {len(self.original_coordinates)} original coordinate sets")
    
    def _restore_original_coordinates(self):
        """Restore original ternary coordinates after disabling clustering."""
        if not hasattr(self, 'original_coordinates') or not self.original_coordinates:
            print("DEBUG: No original coordinates saved, regenerating from RGB")
            # If we don't have saved coordinates, regenerate from RGB
            self._regenerate_ternary_coordinates()
            return
        
        print("DEBUG: Restoring original coordinates")
        restored_count = 0
        
        for i, point in enumerate(self.color_points):
            if i in self.original_coordinates:
                point.ternary_coords = self.original_coordinates[i]
                restored_count += 1
        
        print(f"DEBUG: Restored {restored_count} coordinate sets")
        
        # Clear the saved coordinates
        self.original_coordinates = {}
    
    def _regenerate_ternary_coordinates(self):
        """Regenerate ternary coordinates from RGB values."""
        if not self.color_points:
            return
        
        print("DEBUG: Regenerating ternary coordinates from RGB values")
        regenerated_count = 0
        
        for point in self.color_points:
            if hasattr(point, 'rgb') and point.rgb:
                # Regenerate fresh coordinates from RGB
                point.ternary_coords = self.ternary_plotter.rgb_to_ternary(point.rgb)
                regenerated_count += 1
        
        print(f"DEBUG: Regenerated {regenerated_count} ternary coordinates")
    
    def _compute_clusters(self):
        """Compute K-means clusters using the cluster manager."""
        print(f"DEBUG: _compute_clusters called with {len(self.color_points) if self.color_points else 0} color points")
        
        if not self.color_points:
            print("DEBUG: No color points available for clustering!")
            self._update_status("No data available for K-means clustering")
            return
            
        if not self.cluster_manager.has_sklearn():
            print("DEBUG: sklearn not available in _compute_clusters!")
            self._update_status("Scikit-learn not available for clustering")
            return
        
        try:
            # Use cluster manager to compute clusters
            n_clusters = min(self.n_clusters.get(), len(self.color_points) // 2)
            print(f"DEBUG: Requested {self.n_clusters.get()} clusters, using {n_clusters}")
            
            if n_clusters < 2:
                print("DEBUG: Not enough clusters requested or data points")
                self.clusters = {}
                self.cluster_manager.clear_clusters()
                self._update_status(f"Need at least 2 clusters and {n_clusters*2} data points for K-means")
                return
            
            print(f"DEBUG: Calling cluster_manager.compute_clusters with {len(self.color_points)} points, {n_clusters} clusters")
            
            # Compute clusters using the manager
            clusters = self.cluster_manager.compute_clusters(self.color_points, n_clusters)
            
            print(f"DEBUG: cluster_manager returned: {clusters}")
            print(f"DEBUG: Number of clusters found: {len(clusters) if clusters else 0}")
            
            if clusters:
                # Update legacy reference for backward compatibility
                self.clusters = clusters
                print(f"DEBUG: Updated self.clusters with {len(self.clusters)} clusters")
                
                # Save cluster assignments to database
                for cluster_id, cluster_points in clusters.items():
                    for point in cluster_points:
                        self._save_cluster_assignment(point.id, int(cluster_id))
                
                status_msg = f"K-means: {len(clusters)} clusters computed and saved"
                print(f"DEBUG: Setting status: {status_msg}")
                self._update_status(status_msg)
                
                # Refresh datasheet if it's open to show cluster assignments
                if self.datasheet_manager.is_datasheet_linked():
                    self.datasheet_manager.refresh_datasheet_cluster_data(
                        self.color_points, clusters, self.cluster_manager)
            else:
                print("DEBUG: No clusters returned by cluster manager")
                self.clusters = {}
                self._update_status("Clustering failed")
            
        except Exception as e:
            logger.exception("K-means clustering failed")
            self._update_status(f"Clustering error: {e}")
            self.clusters = {}
            self.cluster_manager.clear_clusters()
    
    def _plot_clusters(self):
        """Plot K-means cluster results using sphere or convex hull visualization."""
        print(f"DEBUG: _plot_clusters called with {len(self.clusters) if self.clusters else 0} clusters")
        if not self.clusters:
            print("DEBUG: No clusters to plot")
            return
        
        print(f"DEBUG: Sphere mode: {self.show_spheres.get()}")
        # Choose visualization mode based on sphere toggle
        if self.show_spheres.get():
            print("DEBUG: Using sphere visualization")
            self._plot_spheres()
            return
        
        print("DEBUG: Using convex hull visualization")
        
        # Traditional convex hull visualization
        for i, (label, points) in enumerate(self.clusters.items()):
            color = self.cluster_colors[i % len(self.cluster_colors)]
            
            # Extract coordinates for this cluster
            x_coords = [p.ternary_coords[0] for p in points]
            y_coords = [p.ternary_coords[1] for p in points]
            
            # Plot cluster boundary (convex hull)
            if len(points) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    cluster_points = np.column_stack((x_coords, y_coords))
                    hull = ConvexHull(cluster_points)
                    hull_pts = cluster_points[hull.vertices]
                    
                    # Draw cluster boundary
                    self.ax.plot(np.append(hull_pts[:, 0], hull_pts[0, 0]), 
                               np.append(hull_pts[:, 1], hull_pts[0, 1]), 
                               color=color, linewidth=2, alpha=0.7, linestyle='--')
                    
                    # Fill cluster area lightly
                    self.ax.fill(hull_pts[:, 0], hull_pts[:, 1], 
                               color=color, alpha=0.1)
                
                except ImportError:
                    pass  # Skip if scipy not available
            
            # Plot improved centroid (much more visible)
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            print(f"DEBUG: Plotting centroid for cluster {label} at ({centroid_x:.4f}, {centroid_y:.4f}) with {len(points)} points")
            
            # Make centroids VERY visible
            self.ax.scatter(centroid_x, centroid_y, c='red', s=300, marker='X', 
                          edgecolors='black', linewidths=3, zorder=20, alpha=1.0)
            
            # Add text label next to centroid
            self.ax.text(centroid_x + 0.02, centroid_y + 0.02, f'C{label}', 
                        fontsize=12, fontweight='bold', color='red', zorder=21)
        
        # No legend needed - clusters identified by color in datasheet
        print(f"DEBUG: Cluster visualization complete - {len(self.clusters)} clusters shown with centroids")
    
    def _toggle_spheres(self):
        """Toggle sphere visualization mode."""
        print(f"DEBUG: _toggle_spheres called, spheres: {self.show_spheres.get()}, clusters enabled: {self.show_clusters.get()}")
        if self.show_clusters.get():
            print("DEBUG: Refreshing plot after sphere toggle")
            self._refresh_plot()
        else:
            print("DEBUG: Clusters not enabled, sphere toggle has no effect")
    
    def _update_clusters(self):
        """Update clusters when spinbox changes."""
        n_clusters_value = self.n_clusters.get()
        print(f"DEBUG: _update_clusters called, new value: {n_clusters_value}, clusters enabled: {self.show_clusters.get()}")
        if self.show_clusters.get():
            print(f"DEBUG: Recomputing clusters with {n_clusters_value} clusters")
            self._compute_clusters()
            self._refresh_plot()
        else:
            print("DEBUG: Clusters not enabled, spinbox change has no effect")
    
    def _on_spinbox_change(self):
        """Handle spinbox value changes with debouncing to avoid rapid-fire updates."""
        # Cancel any pending update
        if hasattr(self, '_spinbox_update_pending'):
            self.window.after_cancel(self._spinbox_update_pending)
        
        # Schedule update after brief delay to debounce rapid changes
        self._spinbox_update_pending = self.window.after(200, self._update_clusters)
    
    def _on_hull_toggle(self):
        """Handle convex hull checkbox toggle with debugging."""
        # Manually toggle the state since macOS checkboxes aren't auto-toggling
        current_state = self.show_hull.get()
        new_state = not current_state
        self.show_hull.set(new_state)
        print(f"DEBUG: Convex Hull checkbox clicked! Changed from {current_state} to {new_state}")
        self._refresh_plot()
    
    def _on_clusters_toggle(self):
        """Handle K-means clusters checkbox toggle with debugging."""
        # Manually toggle the state since macOS checkboxes aren't auto-toggling
        current_state = self.show_clusters.get()
        new_state = not current_state
        self.show_clusters.set(new_state)
        print(f"DEBUG: K-means Clusters checkbox clicked! Changed from {current_state} to {new_state}")
        self._toggle_clusters()
    
    def _on_spheres_toggle(self):
        """Handle spheres checkbox toggle with debugging."""
        # Manually toggle the state since macOS checkboxes aren't auto-toggling
        current_state = self.show_spheres.get()
        new_state = not current_state
        self.show_spheres.set(new_state)
        print(f"DEBUG: Spheres checkbox clicked! Changed from {current_state} to {new_state}")
        self._toggle_spheres()
    
    def _save_clusters_to_database(self):
        """Manually save cluster assignments to database - like Plot_3D approach."""
        try:
            if not hasattr(self, 'clusters') or not self.clusters:
                messagebox.showwarning("No Clusters", "No clusters found to save.\n\nPlease enable K-means clustering first.")
                return
                
            # Count total assignments
            total_assignments = sum(len(cluster_points) for cluster_points in self.clusters.values())
            
            # Confirm with user
            if not messagebox.askyesno("Save Clusters", 
                                     f"Save {len(self.clusters)} clusters with {total_assignments} point assignments to database?\n\n"
                                     f"This will update the database with cluster assignments and centroid data."):
                return
                
            # Save cluster assignments
            saved_count = 0
            for cluster_id, cluster_points in self.clusters.items():
                for point in cluster_points:
                    self._save_cluster_assignment(point.id, int(cluster_id))
                    saved_count += 1
            
            # Also save using cluster manager
            if hasattr(self, 'cluster_manager'):
                try:
                    self.cluster_manager.save_cluster_assignments_to_db(
                        database_manager=None,  # Will create its own connection
                        sample_set_name=self.sample_set_name
                    )
                except Exception as cluster_save_error:
                    logger.warning(f"Cluster manager save failed: {cluster_save_error}")
            
            # Update status and show success
            status_msg = f"Saved {len(self.clusters)} clusters ({saved_count} assignments) to database"
            self._update_status(status_msg)
            
            messagebox.showinfo("Clusters Saved", 
                              f"Successfully saved cluster data:\n\n"
                              f"• {len(self.clusters)} clusters\n"
                              f"• {saved_count} point assignments\n"
                              f"• Centroid coordinates\n\n"
                              f"Data has been written to the database.")
            
            logger.info(f"Manual cluster save completed: {len(self.clusters)} clusters, {saved_count} assignments")
            
        except Exception as e:
            logger.exception(f"Failed to save clusters: {e}")
            messagebox.showerror("Save Failed", f"Failed to save clusters to database:\n\n{e}")
    
    def _manual_hull_toggle(self):
        """Manually handle convex hull checkbox click with delay."""
        # Use after_idle to ensure checkbox state has been updated by tkinter
        self.window.after_idle(self._delayed_hull_toggle)
    
    def _delayed_hull_toggle(self):
        """Execute hull toggle after checkbox state update."""
        # Trust tkinter's checkbox state (no manual flipping needed)
        current_state = self.show_hull.get()
        print(f"🔧 MANUAL: Hull checkbox is now {current_state}")
        print(f"🔧 MANUAL: Calling _refresh_plot() for hull toggle...")
        self._refresh_plot()
        print(f"🔧 MANUAL: Hull toggle refresh completed")
    
    def _manual_clusters_toggle(self):
        """Manually handle clusters checkbox click with delay."""
        # Use after_idle to ensure checkbox state has been updated by tkinter
        self.window.after_idle(self._delayed_clusters_toggle)
    
    def _delayed_clusters_toggle(self):
        """Execute clusters toggle after checkbox state update."""
        # Trust tkinter's checkbox state (no manual flipping needed)
        current_state = self.show_clusters.get()
        print(f"🔧 MANUAL: Clusters checkbox is now {current_state}")
        print(f"🔧 MANUAL: Calling _toggle_clusters() for clusters toggle...")
        self._toggle_clusters()
        print(f"🔧 MANUAL: Clusters toggle completed")
    
    def _manual_spheres_toggle(self):
        """Manually handle spheres checkbox click with delay."""
        # Use after_idle to ensure checkbox state has been updated by tkinter
        self.window.after_idle(self._delayed_spheres_toggle)
    
    def _delayed_spheres_toggle(self):
        """Execute spheres toggle after checkbox state update."""
        # Trust tkinter's checkbox state (no manual flipping needed)
        current_state = self.show_spheres.get()
        print(f"🔧 MANUAL: Spheres checkbox is now {current_state}")
        self._toggle_spheres()
    
    def _rgb_to_ternary_coords(self, rgb):
        """Convert RGB values to ternary coordinates."""
        return self.ternary_plotter.rgb_to_ternary(rgb)
    
    def _show_datasheet_sync_button(self):
        """Enable the datasheet sync button when a datasheet is linked."""
        if hasattr(self, 'refresh_from_datasheet_btn'):
            self.refresh_from_datasheet_btn.config(state='normal')
            self._update_status("Datasheet linked - sync button enabled")
    
    def _hide_datasheet_sync_button(self):
        """Disable the datasheet sync button when no datasheet is linked."""
        if hasattr(self, 'refresh_from_datasheet_btn'):
            self.refresh_from_datasheet_btn.config(state='disabled')
            self._update_status("Datasheet disconnected - sync button disabled")
    
    def _load_data_dialog(self):
        """Open dialog to load different sample set."""
        try:
            sample_sets = self.bridge.get_available_sample_sets()
            if not sample_sets:
                messagebox.showinfo("No Data", "No sample sets found in database.")
                return
            
            # Simple selection dialog
            dialog = tk.Toplevel(self.window)
            dialog.title("Load Sample Set")
            dialog.geometry("350x400")
            dialog.transient(self.window)
            dialog.grab_set()
            
            ttk.Label(dialog, text="Select sample set:", font=('Arial', 11, 'bold')).pack(pady=10)
            
            listbox = tk.Listbox(dialog, height=15)
            for sample_set in sample_sets:
                listbox.insert(tk.END, sample_set)
            listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            
            def on_load():
                selection = listbox.curselection()
                if selection:
                    selected_set = sample_sets[selection[0]]
                    self.sample_set_name = selected_set
                    self.color_points, db_format = self.bridge.load_color_points_from_database(selected_set)
                    
                    # Update format indicator
                    self._update_format_indicator(db_format)
                    
                    # Clear any existing clusters since this is a new database
                    self.clusters = {}
                    self.cluster_manager.clear_clusters()
                    
                    # Reset cluster UI state
                    self.show_clusters.set(False)
                    self.show_spheres.set(False)
                    self.n_clusters.set(3)
                    
                    # Update plot title (window title stays simple)
                    self._update_plot_title(selected_set)
                    
                    self._refresh_plot()
                    self._update_status(f"Loaded {selected_set} ({db_format} format) - {len(self.color_points)} points")
                    print(f"DEBUG: Database loaded: {selected_set}, {len(self.color_points)} points, format: {db_format}")
                    dialog.destroy()
            
            ttk.Button(dialog, text="Load", command=on_load).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
    def _reload_current_database(self):
        """Reload the current database to see persisted changes."""
        try:
            if not self.sample_set_name or self.sample_set_name.startswith("External:"):
                messagebox.showinfo("No Database", "No database sample set loaded to reload.")
                return
            
            # Reload from database
            original_sample = self.sample_set_name
            self.color_points, db_format = self.bridge.load_color_points_from_database(original_sample)
            
            # Update format indicator
            self._update_format_indicator(db_format)
            
            # Refresh plot
            self._refresh_plot()
            
            # Count points with preferences for feedback
            points_with_prefs = sum(1 for p in self.color_points 
                                  if hasattr(p, 'metadata') and 
                                     (p.metadata.get('marker', '.') != '.' or 
                                      p.metadata.get('marker_color', 'blue') != 'blue'))
            
            self._update_status(f"Reloaded {original_sample}: {len(self.color_points)} points ({points_with_prefs} with saved preferences)")
            
        except Exception as e:
            logger.exception("Failed to reload database")
            messagebox.showerror("Reload Error", f"Failed to reload database: {e}")
    
    def _load_external_file(self):
        """Load color data from external .ods/.xlsx file."""
        try:
            # File selection dialog
            filetypes = [
                ("Spreadsheet files", "*.ods *.xlsx"),
                ("OpenDocument Spreadsheet", "*.ods"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="Load External Color Data",
                filetypes=filetypes
            )
            
            if not filename:
                return
            
            # Determine file type and load appropriately
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.ods':
                # Load ODS file - check if it's Plot_3D format or standard format
                color_points = self._load_ods_file(filename)
                
            elif file_ext == '.xlsx':
                # Load Excel file - convert to color points
                color_points = self._load_xlsx_file(filename)
                
            else:
                messagebox.showerror("Unsupported Format", 
                                   f"File format {file_ext} is not supported.\n\n"
                                   "Supported formats: .ods, .xlsx")
                return
            
            if color_points:
                self.color_points = color_points
                self.external_file_path = filename  # Track for integration
                self.sample_set_name = f"External: {os.path.basename(filename)}"
                self.window.title(f"RGB Ternary Analysis - {self.sample_set_name}")
                self._refresh_plot()
                self._update_status(f"Loaded {len(color_points)} points from external file")
            else:
                messagebox.showwarning("No Data", "No color data found in the selected file.")
                
        except Exception as e:
            logger.exception("Failed to load external file")
            messagebox.showerror("Load Error", f"Failed to load external file: {e}")
    
    def _load_xlsx_file(self, filename: str):
        """Load color data from Excel (.xlsx) file."""
        try:
            import pandas as pd
            
            # Read Excel file
            df = pd.read_excel(filename)
            
            # Look for color data columns (flexible column name matching)
            color_points = []
            
            # Expected column patterns
            rgb_patterns = {
                'r': ['R', 'r', 'Red', 'red', 'rgb_r'],
                'g': ['G', 'g', 'Green', 'green', 'rgb_g'], 
                'b': ['B', 'b', 'Blue', 'blue', 'rgb_b']
            }
            
            lab_patterns = {
                'l': ['L*', 'L_star', 'L', 'l_value', 'Lightness'],
                'a': ['a*', 'a_star', 'a', 'a_value'],
                'b': ['b*', 'b_star', 'b', 'b_value']
            }
            
            # Find actual column names
            rgb_cols = {}
            lab_cols = {}
            
            for color, patterns in rgb_patterns.items():
                for pattern in patterns:
                    if pattern in df.columns:
                        rgb_cols[color] = pattern
                        break
            
            for color, patterns in lab_patterns.items():
                for pattern in patterns:
                    if pattern in df.columns:
                        lab_cols[color] = pattern
                        break
            
            # Check if we have sufficient data
            has_rgb = len(rgb_cols) == 3
            has_lab = len(lab_cols) == 3
            
            if not (has_rgb or has_lab):
                raise ValueError("No recognizable RGB or L*a*b* columns found.\n\n"
                               "Expected columns: R,G,B or L*,a*,b* (or similar variants)")
            
            # Process each row
            for i, row in df.iterrows():
                try:
                    # Get ID (try multiple column names)
                    point_id = None
                    for id_col in ['DataID', 'ID', 'Name', 'Sample', 'Point']:
                        if id_col in df.columns:
                            point_id = str(row[id_col])
                            break
                    if not point_id:
                        point_id = f"Point_{i+1}"
                    
                    # Get RGB values
                    if has_rgb:
                        rgb = (
                            float(row[rgb_cols['r']]),
                            float(row[rgb_cols['g']]), 
                            float(row[rgb_cols['b']])
                        )
                    else:
                        rgb = (128.0, 128.0, 128.0)  # Default gray
                    
                    # Get L*a*b* values
                    if has_lab:
                        lab = (
                            float(row[lab_cols['l']]),
                            float(row[lab_cols['a']]),
                            float(row[lab_cols['b']])
                        )
                    else:
                        # Convert RGB to approximate L*a*b* if needed
                        try:
                            from utils.color_conversions import rgb_to_lab
                            lab = rgb_to_lab(rgb)
                        except ImportError:
                            # Simple approximation if color conversions not available
                            # This is rough but sufficient for ternary analysis
                            r, g, b = [c/255.0 for c in rgb]
                            lab = (
                                50.0 + (r + g + b) * 16.67,  # Rough L* approximation
                                (r - g) * 50.0,              # Rough a* approximation  
                                (b - (r + g)/2) * 50.0       # Rough b* approximation
                            )
                    
                    # Calculate ternary coordinates
                    ternary_coords = self.ternary_plotter.rgb_to_ternary(rgb)
                    
                    # Create metadata
                    metadata = {
                        'source': 'external_xlsx',
                        'file_path': filename,
                        'row_index': i
                    }
                    
                    # Add any other columns as metadata
                    for col in df.columns:
                        if col not in list(rgb_cols.values()) + list(lab_cols.values()) + ['DataID', 'ID', 'Name', 'Sample', 'Point']:
                            try:
                                metadata[col] = row[col]
                            except:
                                pass  # Skip problematic columns
                    
                    # Create ColorPoint
                    color_point = ColorPoint(
                        id=point_id,
                        rgb=rgb,
                        lab=lab,
                        ternary_coords=ternary_coords,
                        metadata=metadata
                    )
                    
                    color_points.append(color_point)
                    
                except Exception as row_error:
                    logger.warning(f"Failed to process row {i}: {row_error}")
                    continue
            
            return color_points
            
        except Exception as e:
            raise Exception(f"Failed to load Excel file: {e}")
    
    def _load_ods_file(self, filename: str):
        """Load color data from ODS file, supporting both standard and Plot_3D normalized formats."""
        try:
            # First try the existing bridge (for standard format)
            try:
                color_points = self.bridge.load_color_points_from_ods(filename)
                if color_points:
                    return color_points
            except Exception as bridge_error:
                logger.info(f"Bridge loading failed, trying Plot_3D format: {bridge_error}")
            
            # If bridge fails, try Plot_3D normalized format
            import pandas as pd
            
            # Read ODS file
            df = pd.read_excel(filename, engine='odf')
            
            print(f"DEBUG: ODS columns found: {list(df.columns)}")
            
            color_points = []
            
            # Plot_3D normalized format patterns
            plot3d_patterns = {
                'x': ['Xnorm', 'X', 'x'],
                'y': ['Ynorm', 'Y', 'y'],
                'z': ['Znorm', 'Z', 'z']
            }
            
            # Standard RGB patterns (0-255 or 0-1)
            rgb_patterns = {
                'r': ['R', 'r', 'Red', 'red', 'rgb_r'],
                'g': ['G', 'g', 'Green', 'green', 'rgb_g'], 
                'b': ['B', 'b', 'Blue', 'blue', 'rgb_b']
            }
            
            # Standard L*a*b* patterns
            lab_patterns = {
                'l': ['L*', 'L_star', 'L', 'l_value', 'Lightness'],
                'a': ['a*', 'a_star', 'a', 'a_value'],
                'b': ['b*', 'b_star', 'b', 'b_value']
            }
            
            # Find actual column names
            plot3d_cols = {}
            rgb_cols = {}
            lab_cols = {}
            
            for coord, patterns in plot3d_patterns.items():
                for pattern in patterns:
                    if pattern in df.columns:
                        plot3d_cols[coord] = pattern
                        break
            
            for color, patterns in rgb_patterns.items():
                for pattern in patterns:
                    if pattern in df.columns:
                        rgb_cols[color] = pattern
                        break
            
            for color, patterns in lab_patterns.items():
                for pattern in patterns:
                    if pattern in df.columns:
                        lab_cols[color] = pattern
                        break
            
            # Determine data format
            has_plot3d = len(plot3d_cols) == 3  # Xnorm, Ynorm, Znorm
            has_rgb = len(rgb_cols) == 3
            has_lab = len(lab_cols) == 3
            
            print(f"DEBUG: has_plot3d={has_plot3d}, has_rgb={has_rgb}, has_lab={has_lab}")
            print(f"DEBUG: plot3d_cols={plot3d_cols}")
            print(f"DEBUG: rgb_cols={rgb_cols}")
            print(f"DEBUG: lab_cols={lab_cols}")
            
            if not (has_plot3d or has_rgb or has_lab):
                available_cols = ', '.join(df.columns[:10])  # Show first 10 columns
                raise ValueError(f"No recognizable color columns found.\n\n"
                               f"Available columns: {available_cols}...\n\n"
                               f"Expected formats:\n"
                               f"• Plot_3D: Xnorm, Ynorm, Znorm (0-1 normalized)\n"
                               f"• RGB: R, G, B (0-255 or 0-1)\n"
                               f"• L*a*b*: L*, a*, b* (standard ranges)")
            
            # Process each row - SKIP ROWS 1-7 for Plot_3D format (header + centroids)
            start_row = 7 if has_plot3d else 0  # Skip first 7 rows for Plot_3D format
            print(f"DEBUG: Processing rows starting from row {start_row+1} (has_plot3d: {has_plot3d})")
            
            for i, row in df.iterrows():
                # Skip reserved rows for Plot_3D format
                if has_plot3d and i < start_row:
                    print(f"DEBUG: Skipping reserved row {i+1} (Plot_3D format)")
                    continue
                    
                try:
                    # Get ID (try multiple column names)
                    point_id = None
                    for id_col in ['DataID', 'ID', 'Name', 'Sample', 'Point', 'Label']:
                        if id_col in df.columns and pd.notna(row[id_col]) and str(row[id_col]).strip():
                            point_id = str(row[id_col]).strip()
                            break
                    if not point_id:
                        point_id = f"Point_{i+1}"
                    
                    print(f"DEBUG: Processing row {i+1} with ID '{point_id}'")
                    
                    # Skip rows with empty or invalid coordinate data
                    if has_plot3d:
                        # Check for valid Plot_3D coordinates
                        try:
                            x_test = float(row[plot3d_cols['x']])
                            y_test = float(row[plot3d_cols['y']])
                            z_test = float(row[plot3d_cols['z']])
                            if not all(0 <= val <= 1 for val in [x_test, y_test, z_test]):
                                print(f"DEBUG: Skipping row {i+1} - invalid Plot_3D coordinates: {x_test}, {y_test}, {z_test}")
                                continue
                        except (ValueError, TypeError, KeyError):
                            print(f"DEBUG: Skipping row {i+1} - missing or invalid Plot_3D coordinates")
                            continue
                    
                    # Handle Plot_3D normalized format
                    if has_plot3d:
                        # Plot_3D uses normalized L*a*b* coordinates (0-1 range)
                        x_norm = float(row[plot3d_cols['x']])  # L* normalized
                        y_norm = float(row[plot3d_cols['y']])  # a* normalized  
                        z_norm = float(row[plot3d_cols['z']])  # b* normalized
                        
                        # Convert back to standard ranges
                        # L*: 0-1 → 0-100
                        # a*, b*: 0-1 → -128 to +127 (assuming 0.5 = 0)
                        lab = (
                            x_norm * 100.0,                    # L*: 0-100
                            (y_norm - 0.5) * 255.0,           # a*: -127.5 to +127.5  
                            (z_norm - 0.5) * 255.0            # b*: -127.5 to +127.5
                        )
                        
                        # For ternary analysis, we need RGB. Try to convert from Lab
                        # This is approximate but sufficient for ternary analysis
                        try:
                            from utils.color_conversions import lab_to_rgb
                            rgb = lab_to_rgb(lab)
                        except ImportError:
                            # Simple approximation Lab → RGB 
                            l, a, b = lab
                            # Very rough approximation
                            r = max(0, min(255, l * 2.55 + a * 1.5))
                            g = max(0, min(255, l * 2.55 - a * 0.5))
                            b = max(0, min(255, l * 2.55 - b * 1.5))
                            rgb = (r, g, b)
                    
                    # Handle standard RGB format
                    elif has_rgb:
                        rgb_values = (
                            float(row[rgb_cols['r']]),
                            float(row[rgb_cols['g']]), 
                            float(row[rgb_cols['b']])
                        )
                        
                        # Check if normalized (0-1) or standard (0-255)
                        max_val = max(rgb_values)
                        if max_val <= 1.0:
                            # Normalized RGB, convert to 0-255
                            rgb = tuple(v * 255.0 for v in rgb_values)
                        else:
                            # Already in 0-255 range
                            rgb = rgb_values
                        
                        # Get or approximate L*a*b*
                        if has_lab:
                            lab = (
                                float(row[lab_cols['l']]),
                                float(row[lab_cols['a']]),
                                float(row[lab_cols['b']])
                            )
                        else:
                            # Approximate Lab from RGB
                            try:
                                from utils.color_conversions import rgb_to_lab
                                lab = rgb_to_lab(rgb)
                            except ImportError:
                                # Simple approximation
                                r, g, b = [c/255.0 for c in rgb]
                                lab = (
                                    50.0 + (r + g + b) * 16.67,
                                    (r - g) * 50.0,
                                    (b - (r + g)/2) * 50.0
                                )
                    
                    # Handle L*a*b* only format
                    elif has_lab:
                        lab = (
                            float(row[lab_cols['l']]),
                            float(row[lab_cols['a']]),
                            float(row[lab_cols['b']])
                        )
                        
                        # Approximate RGB from Lab
                        try:
                            from utils.color_conversions import lab_to_rgb
                            rgb = lab_to_rgb(lab)
                        except ImportError:
                            # Simple approximation
                            l, a, b = lab
                            r = max(0, min(255, l * 2.55 + a * 1.5))
                            g = max(0, min(255, l * 2.55 - a * 0.5))
                            b_val = max(0, min(255, l * 2.55 - b * 1.5))
                            rgb = (r, g, b_val)
                    
                    else:
                        raise ValueError("No valid color data format detected")
                    
                    # Calculate ternary coordinates
                    ternary_coords = self.ternary_plotter.rgb_to_ternary(rgb)
                    
                    # Create metadata
                    metadata = {
                        'source': 'external_ods',
                        'file_path': filename,
                        'row_index': i,
                        'format': 'plot3d' if has_plot3d else 'standard'
                    }
                    
                    # Add any other columns as metadata
                    exclude_cols = (list(plot3d_cols.values()) + list(rgb_cols.values()) + 
                                  list(lab_cols.values()) + ['DataID', 'ID', 'Name', 'Sample', 'Point', 'Label'])
                    
                    for col in df.columns:
                        if col not in exclude_cols:
                            try:
                                metadata[col] = row[col]
                            except:
                                pass  # Skip problematic columns
                    
                    # Create ColorPoint
                    color_point = ColorPoint(
                        id=point_id,
                        rgb=rgb,
                        lab=lab,
                        ternary_coords=ternary_coords,
                        metadata=metadata
                    )
                    
                    color_points.append(color_point)
                    
                except Exception as row_error:
                    logger.warning(f"Failed to process row {i}: {row_error}")
                    continue
            
            print(f"DEBUG: Successfully processed {len(color_points)} points")
            return color_points
            
        except Exception as e:
            raise Exception(f"Failed to load ODS file: {e}")
    
    def _save_plot(self):
        """Save current plot to file using export manager."""
        self.export_manager.save_plot_image(self.fig, self.sample_set_name)
    
    def _save_as_database(self):
        """Save current color data as a new Ternary database using export manager."""
        self.export_manager.save_as_database(self.color_points, self.sample_set_name)
    
    def _open_plot3d(self):
        """Open realtime datasheet with current data using datasheet manager."""
        try:
            # DEBUG: Check current state
            print(f"\n=== IMMEDIATE DEBUG CHECK ===")
            print(f"self.color_points type: {type(self.color_points)}")
            print(f"self.color_points length: {len(self.color_points) if self.color_points else 0}")
            print(f"self.color_points is None: {self.color_points is None}")
            print(f"self.color_points is empty list: {self.color_points == []}")
            if self.color_points and len(self.color_points) > 0:
                print(f"First color point ID: {self.color_points[0].id}")
            print(f"================================\n")
            
            logger.info(f"📊 DATASHEET DEBUG: Opening datasheet with {len(self.color_points) if self.color_points else 0} color points")
            logger.info(f"📊 DATASHEET DEBUG: Sample set name: {self.sample_set_name}")
            logger.info(f"📊 DATASHEET DEBUG: Has clusters: {hasattr(self, 'clusters') and bool(self.clusters)}")
            
            # Verify we have color points to share
            if not self.color_points:
                messagebox.showwarning("No Data", 
                                     "No color data available to open in datasheet.\n\n"
                                     "Please load a database or external file first using the \"Load Database\" button.")
                return
            
            # Ensure clusters are computed if K-means is enabled
            if self.show_clusters.get() and self.cluster_manager.has_sklearn():
                if not hasattr(self, 'clusters') or not self.clusters:
                    logger.info("K-means enabled but no clusters found, computing...")
                    self._compute_clusters()
                else:
                    logger.info(f"K-means enabled, found {len(self.clusters)} existing clusters")
            
            # Use datasheet manager to open datasheet
            logger.info(f"📊 DATASHEET DEBUG: Calling datasheet manager with {len(self.color_points)} points")
            datasheet = self.datasheet_manager.open_realtime_datasheet(
                color_points=self.color_points,
                clusters=self.clusters if self.clusters else None,
                sample_set_name=self.sample_set_name,
                external_file_path=getattr(self, 'external_file_path', None)
            )
            
            if datasheet:
                # Store legacy reference for backward compatibility
                self.datasheet_ref = datasheet
                
                # Show the "Sync from Datasheet" button now that datasheet is linked
                self._show_datasheet_sync_button()
                
                self._update_status(f"Realtime datasheet opened via manager")
            else:
                self._update_status("Failed to open datasheet")
            
        except Exception as e:
            error_msg = f"Failed to open realtime datasheet: {e}"
            logger.exception(error_msg)
            messagebox.showerror("Datasheet Error", error_msg)
    
    def _show_datasheet_ready_dialog(self, point_count, cluster_info):
        """Show the datasheet ready dialog with proper focus."""
        try:
            # Ensure the dialog appears in front
            if hasattr(self, 'window'):
                self.window.lift()
                self.window.focus_force()
            
            messagebox.showinfo(
                "Datasheet Ready",
                f"Realtime datasheet opened with {point_count} color points.\n\n"
                "• Data converted to Plot_3D normalized format (0-1 range)\n"
                "• Use 'Launch Plot_3D' button in datasheet for 3D visualization\n"
                "• Edit data in datasheet and refresh ternary plot to see changes"
                f"{cluster_info}"
            )
        except Exception as e:
            logger.exception(f"Error showing dialog: {e}")
    
    def _open_datasheet(self):
        """Open realtime datasheet with current data."""
        try:
            if self.external_file_path:
                # Open realtime datasheet with external file data
                from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
                
                # Create datasheet with the same external data
                datasheet = RealtimePlot3DSheet(
                    parent=self.window,
                    sample_set_name=f"External: {os.path.basename(self.external_file_path)}",
                    load_initial_data=False  # We'll load our data manually
                )
                
                # Convert color points to datasheet format
                self._populate_datasheet(datasheet)
                
                self._update_status(f"Datasheet opened with {os.path.basename(self.external_file_path)}")
                
            elif self.sample_set_name and not self.sample_set_name.startswith("External:"):
                # Open datasheet for database sample set
                from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
                
                datasheet = RealtimePlot3DSheet(
                    parent=self.window,
                    sample_set_name=self.sample_set_name,
                    load_initial_data=True
                )
                
                self._update_status(f"Datasheet opened for {self.sample_set_name}")
                
            else:
                messagebox.showinfo("No Data", "No sample set or external file loaded to open in datasheet.")
                
        except Exception as e:
            logger.exception("Failed to open datasheet")
            messagebox.showerror("Datasheet Error", f"Failed to open datasheet: {e}")
    
    def _populate_datasheet_with_current_data(self, datasheet):
        """Populate realtime datasheet with current ternary data in proper Plot_3D format."""
        try:
            if not self.color_points:
                logger.warning("No color points to populate datasheet")
                return
                
            # First, calculate cluster centroids if clusters exist
            cluster_centroids = {}
            if hasattr(self, 'clusters') and self.clusters:
                logger.info(f"DEBUG: Found {len(self.clusters)} clusters for centroid calculation")
                import matplotlib.pyplot as plt
                import numpy as np
                colors = plt.cm.Set3(np.linspace(0, 1, len(self.clusters)))
                
                for cluster_idx, (cluster_id, cluster_points) in enumerate(self.clusters.items()):
                    # Calculate cluster centroid in L*a*b* space
                    cluster_l = [cp.lab[0] for cp in cluster_points]
                    cluster_a = [cp.lab[1] for cp in cluster_points]
                    cluster_b = [cp.lab[2] for cp in cluster_points]
                    
                    # Convert centroid to normalized 0-1 range
                    centroid_l_norm = round(sum(cluster_l) / len(cluster_l) / 100.0, 6)
                    centroid_a_norm = round((sum(cluster_a) / len(cluster_a) + 127.5) / 255.0, 6)
                    centroid_b_norm = round((sum(cluster_b) / len(cluster_b) + 127.5) / 255.0, 6)
                    
                    # Get cluster sphere color from matplotlib colormap
                    cluster_color = colors[cluster_idx]
                    sphere_color = '#{:02x}{:02x}{:02x}'.format(
                        int(cluster_color[0] * 255),
                        int(cluster_color[1] * 255), 
                        int(cluster_color[2] * 255)
                    )
                    
                    cluster_centroids[cluster_id] = {
                        'centroid_x': centroid_l_norm,
                        'centroid_y': centroid_a_norm, 
                        'centroid_z': centroid_b_norm,
                        'sphere_color': sphere_color,
                        'sphere_radius': 0.02,
                        'point_count': len(cluster_points)
                    }
                    
                    # Debug centroid calculation
                    logger.info(f"DEBUG: Cluster {cluster_id} centroid - L:{centroid_l_norm:.4f}, a:{centroid_a_norm:.4f}, b:{centroid_b_norm:.4f} ({len(cluster_points)} points)")
            
            # Convert ColorPoint objects to Plot_3D normalized format (data rows only)
            plot3d_data_rows = []
            
            for i, point in enumerate(self.color_points):
                # Convert L*a*b* to normalized 0-1 range (Plot_3D format)
                l_norm = max(0.0, min(1.0, point.lab[0] / 100.0))  # L*: 0-100 → 0-1
                a_norm = max(0.0, min(1.0, (point.lab[1] + 127.5) / 255.0))  # a*: -127.5 to +127.5 → 0-1  
                b_norm = max(0.0, min(1.0, (point.lab[2] + 127.5) / 255.0))  # b*: -127.5 to +127.5 → 0-1
                
                # Store original RGB in metadata for accurate reconstruction
                if not hasattr(point, 'metadata'):
                    point.metadata = {}
                point.metadata['original_rgb'] = point.rgb
                point.metadata['original_ternary_coords'] = point.ternary_coords
                
                # Find cluster assignment (but no centroid data in individual rows)
                cluster_assignment = ''
                if hasattr(self, 'clusters') and self.clusters:
                    for cluster_id, cluster_points in self.clusters.items():
                        if point in cluster_points:
                            cluster_assignment = str(cluster_id)
                            break
                
                # Create row data in Plot_3D format (NO centroid/sphere data in individual rows)
                row_data = [
                    round(l_norm, 6),      # Xnorm (L*)
                    round(a_norm, 6),      # Ynorm (a*) 
                    round(b_norm, 6),      # Znorm (b*)
                    point.id,              # DataID
                    cluster_assignment,    # Cluster (just assignment, no centroid data)
                    '',                    # ∆E (empty - to be calculated)
                    '.',                   # Marker (default)
                    'blue',                # Color (default)
                    '',                    # Centroid_X (empty - centroids go in restricted area)
                    '',                    # Centroid_Y (empty - centroids go in restricted area)
                    '',                    # Centroid_Z (empty - centroids go in restricted area)
                    '',                    # Sphere (empty - centroids go in restricted area)
                    ''                     # Radius (empty - centroids go in restricted area)
                ]
                
                plot3d_data_rows.append(row_data)
            
            # Setup proper Plot_3D sheet structure
            if hasattr(datasheet, 'sheet'):
                # Calculate total rows needed: 7 reserved + data + buffer
                total_rows_needed = 7 + len(plot3d_data_rows) + 10  # 7 reserved + data + 10 buffer
                
                # Clear existing sheet and create proper structure
                current_rows = datasheet.sheet.get_total_rows()
                if current_rows > 0:
                    datasheet.sheet.delete_rows(0, current_rows)
                
                # Create empty rows for proper structure
                empty_rows = [[''] * len(datasheet.PLOT3D_COLUMNS)] * total_rows_needed
                datasheet.sheet.insert_rows(rows=empty_rows, idx=0)
                
                # Set headers in row 1 (index 0)
                datasheet.sheet.set_row_data(0, values=datasheet.PLOT3D_COLUMNS)
                
                # Populate rows 2-7 (indices 1-6) with SEQUENTIAL CLUSTER DATA in the protected pink area
                logger.info(f"DEBUG: Populating {len(cluster_centroids)} centroids in rows 2-7")
                
                # First clear rows 2-7 completely to avoid leftover data
                for clear_idx in range(1, 7):  # Clear rows 2-7 (indices 1-6)
                    empty_row = [''] * len(datasheet.PLOT3D_COLUMNS)
                    datasheet.sheet.set_row_data(clear_idx, values=empty_row)
                
                # Sort cluster IDs for consistent placement
                sorted_cluster_items = sorted(cluster_centroids.items(), key=lambda x: int(x[0]))
                logger.info(f"DEBUG: Processing clusters in order: {[cid for cid, _ in sorted_cluster_items]}")
                
                centroid_row_index = 1  # Start at row 2 (index 1)
                for cluster_id, centroid_data in sorted_cluster_items:
                    if centroid_row_index >= 7:  # Don't exceed reserved area
                        logger.warning(f"DEBUG: Too many clusters, skipping cluster {cluster_id}")
                        break
                    
                    # Create centroid row with cluster data only
                    centroid_row = [
                        centroid_data['centroid_x'],     # Xnorm (cluster centroid L*)
                        centroid_data['centroid_y'],     # Ynorm (cluster centroid a*)
                        centroid_data['centroid_z'],     # Znorm (cluster centroid b*)
                        f"Cluster_{cluster_id}",         # DataID (cluster identifier)
                        str(cluster_id),                 # Cluster (cluster ID)
                        '',                              # ΔE (empty)
                        'x',                             # Marker (centroid marker)
                        centroid_data['sphere_color'],   # Color (cluster color)
                        centroid_data['centroid_x'],     # Centroid_X (self-reference)
                        centroid_data['centroid_y'],     # Centroid_Y (self-reference)
                        centroid_data['centroid_z'],     # Centroid_Z (self-reference)
                        centroid_data['sphere_color'],   # Sphere (cluster sphere color)
                        centroid_data['sphere_radius']   # Radius (sphere radius)
                    ]
                    
                    logger.info(f"DEBUG: Placing Cluster {cluster_id} in row {centroid_row_index + 1} (index {centroid_row_index})")
                    datasheet.sheet.set_row_data(centroid_row_index, values=centroid_row)
                    centroid_row_index += 1
                
                logger.info(f"Populated {len(cluster_centroids)} cluster centroids in rows 2-{centroid_row_index}")
                
                # Insert individual data points starting at row 8 (index 7) as per Plot_3D format
                for i, row_data in enumerate(plot3d_data_rows):
                    row_index = 7 + i  # Start at row 7 (display row 8)
                    datasheet.sheet.set_row_data(row_index, values=row_data)
                
                logger.info(f"Populated {len(plot3d_data_rows)} data rows starting at row 8")
                
                # Apply proper formatting and validation
                datasheet._apply_formatting()
                datasheet._setup_validation()
                
                # Force refresh the datasheet display
                if hasattr(datasheet.sheet, 'refresh'):
                    datasheet.sheet.refresh()
                elif hasattr(datasheet.sheet, 'update'):
                    datasheet.sheet.update()
                
                logger.info(f"Populated datasheet with {len(plot3d_data_rows)} data rows in proper Plot_3D format")
                logger.info(f"Headers in row 1, protected area in rows 2-7, data starts at row 8")
                
            else:
                logger.warning("Datasheet sheet widget not found")
                
        except Exception as e:
            logger.exception(f"Failed to populate datasheet: {e}")
            raise Exception(f"Datasheet population failed: {e}")
            
    def _populate_datasheet(self, datasheet):
        """Legacy method - calls the new method for backward compatibility."""
        return self._populate_datasheet_with_current_data(datasheet)
        
    def _bring_ternary_to_front(self):
        """Bring the ternary window to front and focus it."""
        try:
            if hasattr(self, 'window') and self.window:
                self.window.lift()
                self.window.focus_force()
                self.window.attributes('-topmost', True)
                self.window.after(100, lambda: self.window.attributes('-topmost', False))
        except Exception as e:
            logger.warning(f"Could not bring ternary window to front: {e}")
    
    def _save_ternary_preferences_to_db(self, point_id: str, marker: str = None, color: str = None):
        """Save ternary-specific marker/color preferences to database."""
        try:
            # Parse the point ID to get image_name and coordinate_point
            if '_pt' in point_id:
                image_name, pt_part = point_id.rsplit('_pt', 1)
                coordinate_point = int(pt_part)
            else:
                # Single point entries
                image_name = point_id
                coordinate_point = 1
            
            # Use the database's ternary preference update method
            from utils.color_analysis_db import ColorAnalysisDB
            db = ColorAnalysisDB(self.sample_set_name)
            
            success = db.update_ternary_preferences(
                image_name=image_name,
                coordinate_point=coordinate_point,
                marker=marker,
                color=color
            )
            
            if success:
                logger.debug(f"Saved ternary preferences for {point_id}: marker={marker}, color={color}")
            else:
                logger.warning(f"Failed to save ternary preferences for {point_id}")
                
            return success
            
        except Exception as e:
            logger.exception(f"Error saving ternary preferences for {point_id}: {e}")
            return False
    
    def _handle_datasheet_changes(self, row_data):
        """Handle changes from the datasheet and save to ternary-specific columns."""
        try:
            if len(row_data) < 8:  # Need at least DataID, Marker, Color columns
                return
                
            data_id = row_data[3]  # DataID column
            marker = row_data[6] if len(row_data) > 6 else '.'  # Marker column
            color = row_data[7] if len(row_data) > 7 else 'blue'  # Color column
            
            # Check if this is a cluster centroid row (in restricted area)
            if data_id.startswith('Cluster_'):
                cluster_id = data_id.replace('Cluster_', '')
                self._handle_cluster_centroid_changes(cluster_id, row_data)
            else:
                # Regular data point - save individual preferences
                success = self._save_ternary_preferences_to_db(
                    point_id=data_id,
                    marker=marker,
                    color=color
                )
                
                if success:
                    logger.debug(f"Saved ternary preferences from datasheet: {data_id} -> marker={marker}, color={color}")
                else:
                    logger.warning(f"Failed to save ternary preferences from datasheet for {data_id}")
                
        except Exception as e:
            logger.exception(f"Error handling datasheet changes: {e}")
    
    def _handle_cluster_centroid_changes(self, cluster_id: str, centroid_row_data):
        """Handle changes to cluster centroid and propagate to all cluster members."""
        try:
            if not (hasattr(self, 'clusters') and self.clusters):
                return
            
            cluster_id_int = int(cluster_id)
            if cluster_id_int not in self.clusters:
                return
                
            # Extract centroid changes
            centroid_color = centroid_row_data[7] if len(centroid_row_data) > 7 else None  # Color column
            sphere_color = centroid_row_data[11] if len(centroid_row_data) > 11 else None  # Sphere column
            sphere_radius = centroid_row_data[12] if len(centroid_row_data) > 12 else None  # Radius column
            
            logger.info(f"Cluster {cluster_id} centroid changed: color={centroid_color}, sphere={sphere_color}, radius={sphere_radius}")
            
            # Apply changes to ALL members of this cluster
            cluster_points = self.clusters[cluster_id_int]
            updated_count = 0
            
            for point in cluster_points:
                try:
                    # Parse point ID to get image_name and coordinate_point
                    if '_pt' in point.id:
                        image_name, pt_part = point.id.rsplit('_pt', 1)
                        coordinate_point = int(pt_part)
                    else:
                        image_name = point.id
                        coordinate_point = 1
                    
                    # Update database with cluster-wide changes
                    success = self._save_ternary_preferences_to_db(
                        point_id=point.id,
                        marker=None,  # Don't change individual markers
                        color=centroid_color if centroid_color else None
                    )
                    
                    if success:
                        updated_count += 1
                        # Update point metadata for immediate refresh
                        if not hasattr(point, 'metadata'):
                            point.metadata = {}
                        if centroid_color:
                            point.metadata['marker_color'] = centroid_color
                    
                except Exception as point_error:
                    logger.warning(f"Failed to update cluster member {point.id}: {point_error}")
            
            logger.info(f"Updated {updated_count} points in cluster {cluster_id} with centroid changes")
            
            # Refresh the ternary plot to show changes
            self._refresh_plot_only()
            
        except Exception as e:
            logger.exception(f"Error handling cluster centroid changes for cluster {cluster_id}: {e}")
    
    def _refresh_datasheet_cluster_data(self):
        """Refresh cluster data in the open datasheet."""
        try:
            if not (hasattr(self, 'datasheet_ref') and self.datasheet_ref and hasattr(self.datasheet_ref, 'sheet')):
                return
                
            if not (hasattr(self, 'clusters') and self.clusters):
                return
                
            # Update cluster column for each data row
            total_rows = self.datasheet_ref.sheet.get_total_rows()
            
            for row_idx in range(7, total_rows):  # Start at row 8 (index 7) where data begins
                try:
                    row_data = self.datasheet_ref.sheet.get_row_data(row_idx)
                    if not row_data or len(row_data) < 5:  # Need at least DataID column
                        continue
                        
                    data_id = row_data[3]  # DataID is column 4 (index 3)
                    if not data_id:
                        continue
                        
                    # Find matching color point and its cluster assignment
                    matching_point = None
                    for point in self.color_points:
                        if point.id == data_id:
                            matching_point = point
                            break
                            
                    if matching_point:
                        # Find cluster assignment and centroid data for this point
                        cluster_assignment = ''
                        centroid_x = centroid_y = centroid_z = ''
                        sphere_color = sphere_radius = ''
                        
                        # Use cluster manager to get cluster assignment
                        cluster_assignment = self.cluster_manager.get_cluster_assignment(matching_point)
                        
                        if cluster_assignment:
                            # Get centroid data from cluster manager
                            try:
                                centroids = self.cluster_manager.calculate_centroids()
                                cluster_id_int = int(cluster_assignment)
                                
                                if cluster_id_int in centroids:
                                    centroid_data = centroids[cluster_id_int]
                                    centroid_x = centroid_data['centroid_x']
                                    centroid_y = centroid_data['centroid_y']
                                    centroid_z = centroid_data['centroid_z']
                                    sphere_color = centroid_data['sphere_color']
                                    sphere_radius = str(centroid_data['sphere_radius'])
                                    
                            except Exception as centroid_error:
                                logger.warning(f"Failed to get centroids from cluster manager: {centroid_error}")
                                # Fallback to legacy calculation
                                for cluster_id, cluster_points in self.clusters.items():
                                    if matching_point in cluster_points:
                                        cluster_assignment = str(cluster_id)
                                        
                                        # Calculate cluster centroid (legacy)
                                        cluster_l = [cp.lab[0] for cp in cluster_points]
                                        cluster_a = [cp.lab[1] for cp in cluster_points]
                                        cluster_b = [cp.lab[2] for cp in cluster_points]
                                        
                                        centroid_x = round(sum(cluster_l) / len(cluster_l) / 100.0, 6)
                                        centroid_y = round((sum(cluster_a) / len(cluster_a) + 127.5) / 255.0, 6)
                                        centroid_z = round((sum(cluster_b) / len(cluster_b) + 127.5) / 255.0, 6)
                                        
                                        sphere_color = '#808080'  # Gray fallback
                                        sphere_radius = '0.02'
                                        break
                                
                        # Update the cluster and centroid data if assignment found
                        if cluster_assignment:
                            self.datasheet_ref.sheet.set_cell_data(row_idx, 4, cluster_assignment)  # Cluster
                            if centroid_x:
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 8, centroid_x)   # Centroid_X
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 9, centroid_y)   # Centroid_Y
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 10, centroid_z)  # Centroid_Z
                            if sphere_color:
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 11, sphere_color)  # Sphere color
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 12, sphere_radius) # Sphere radius
                            
                except Exception as row_error:
                    logger.warning(f"Failed to update cluster data for row {row_idx}: {row_error}")
                    continue
                    
            # Refresh the datasheet display
            if hasattr(self.datasheet_ref.sheet, 'refresh'):
                self.datasheet_ref.sheet.refresh()
                
            logger.info(f"Updated cluster data in datasheet for {len(self.clusters)} clusters")
            
        except Exception as e:
            logger.exception(f"Failed to refresh datasheet cluster data: {e}")
    
    def _save_cluster_assignment(self, point_id: str, cluster_id: int):
        """Save cluster assignment to database for K-means persistence."""
        try:
            # Parse the point ID to get image_name and coordinate_point
            if '_pt' in point_id:
                image_name, pt_part = point_id.rsplit('_pt', 1)
                coordinate_point = int(pt_part)
            else:
                # Single point entries
                image_name = point_id
                coordinate_point = 1
            
            # Use the database's update method to save cluster assignment
            from utils.color_analysis_db import ColorAnalysisDB
            db = ColorAnalysisDB(self.sample_set_name)
            
            success = db.update_plot3d_extended_values(
                image_name=image_name,
                coordinate_point=coordinate_point,
                cluster_id=cluster_id
            )
            
            if success:
                logger.debug(f"Saved cluster assignment: {point_id} -> cluster {cluster_id}")
            else:
                logger.warning(f"Failed to save cluster assignment for {point_id}")
                
        except Exception as e:
            logger.exception(f"Error saving cluster assignment for {point_id}: {e}")
    
    def _sync_from_datasheet(self):
        """Sync data from linked datasheet back to ternary plot."""
        try:
            if not self.datasheet_ref or not hasattr(self.datasheet_ref, 'sheet'):
                return
                
            # Get current data from datasheet (skip header row)
            sheet_data = self.datasheet_ref.sheet.get_sheet_data()
            
            # Skip header row (index 0) and protected rows (indices 1-6), start from index 7
            if len(sheet_data) > 7:
                data_rows = sheet_data[7:]  # Data starts at row 8 (index 7)
            else:
                data_rows = []
            
            if not data_rows:
                logger.warning("No data rows in linked datasheet")
                return
                
            # Convert datasheet data back to ColorPoint objects
            updated_color_points = []
            
            for row in data_rows:
                try:
                    # Parse Plot_3D format: [Xnorm, Ynorm, Znorm, DataID, Cluster, ΔE, Marker, Color, ...]
                    if len(row) < 4 or not row[3]:  # Skip incomplete rows or rows without DataID
                        continue
                        
                    x_norm = float(row[0]) if row[0] != '' else 0.0  # L* normalized
                    y_norm = float(row[1]) if row[1] != '' else 0.0  # a* normalized 
                    z_norm = float(row[2]) if row[2] != '' else 0.0  # b* normalized
                    data_id = str(row[3]) if row[3] != '' else f"Point_{len(updated_color_points)}"
                    
                    # Extract marker information from datasheet if available
                    marker_style = str(row[6]) if len(row) > 6 and row[6] != '' else '.'
                    marker_color = str(row[7]) if len(row) > 7 and row[7] != '' else 'blue'
                    
                    # CRITICAL: Save ternary preferences to database for persistence
                    # This maintains separation from Plot_3D preferences
                    self._save_ternary_preferences_to_db(
                        point_id=data_id, 
                        marker=marker_style, 
                        color=marker_color
                    )
                    
                    # Convert normalized values back to L*a*b*
                    l_star = x_norm * 100.0  # 0-1 → 0-100
                    a_star = (y_norm * 255.0) - 127.5  # 0-1 → -127.5 to +127.5
                    b_star = (z_norm * 255.0) - 127.5  # 0-1 → -127.5 to +127.5
                    
                    # Try to use original RGB from existing points first to maintain accuracy
                    rgb = None
                    original_ternary_coords = None
                    
                    if hasattr(self, 'color_points') and len(updated_color_points) < len(self.color_points):
                        # Use original point RGB to preserve ternary accuracy
                        original_point = self.color_points[len(updated_color_points)]
                        
                        # Check if we have preserved RGB in metadata
                        if (hasattr(original_point, 'metadata') and 
                            'original_rgb' in original_point.metadata):
                            rgb = original_point.metadata['original_rgb']
                            logger.debug(f"Using preserved RGB from metadata: {rgb}")
                        else:
                            rgb = original_point.rgb
                            logger.debug(f"Using original RGB: {rgb}")
                        
                        # Also preserve original ternary coordinates if available
                        if (hasattr(original_point, 'metadata') and 
                            'original_ternary_coords' in original_point.metadata):
                            original_ternary_coords = original_point.metadata['original_ternary_coords']
                    
                    if rgb is None:
                        # Convert L*a*b* to RGB for ternary coordinates
                        try:
                            from utils.color_conversions import lab_to_rgb
                            rgb = lab_to_rgb((l_star, a_star, b_star))
                        except ImportError:
                            # More accurate approximation preserving color balance
                            # Convert normalized L*a*b* back to RGB more carefully
                            l_norm = l_star / 100.0  # L*: 0-100 -> 0-1
                            a_norm = (a_star + 127.5) / 255.0  # a*: -127.5 to +127.5 -> 0-1
                            b_norm = (b_star + 127.5) / 255.0  # b*: -127.5 to +127.5 -> 0-1
                            
                            # Better RGB approximation maintaining ratios
                            r = max(0, min(255, l_norm * 255 + (a_norm - 0.5) * 100))
                            g = max(0, min(255, l_norm * 255 - (a_norm - 0.5) * 50 + (b_norm - 0.5) * 20))
                            b = max(0, min(255, l_norm * 255 - (b_norm - 0.5) * 100))
                            rgb = (r, g, b)
                    
                    # Calculate ternary coordinates - use preserved if available
                    if original_ternary_coords is not None:
                        ternary_coords = original_ternary_coords
                        logger.debug(f"Using preserved ternary coords: {ternary_coords}")
                    else:
                        ternary_coords = self.ternary_plotter.rgb_to_ternary(rgb)
                    
                    # Create ColorPoint object with preserved metadata
                    metadata = {
                        'source': 'datasheet_sync', 
                        'original_rgb': rgb,
                        'marker': marker_style,
                        'marker_color': marker_color
                    }
                    if original_ternary_coords is not None:
                        metadata['original_ternary_coords'] = original_ternary_coords
                    
                    # Preserve any additional metadata from original point
                    if (hasattr(self, 'color_points') and len(updated_color_points) < len(self.color_points) and
                        hasattr(self.color_points[len(updated_color_points)], 'metadata')):
                        original_metadata = self.color_points[len(updated_color_points)].metadata
                        for key, value in original_metadata.items():
                            if key not in metadata:  # Don't overwrite our sync metadata
                                metadata[key] = value
                    
                    color_point = ColorPoint(
                        id=data_id,
                        rgb=rgb,
                        lab=(l_star, a_star, b_star),
                        ternary_coords=ternary_coords,
                        metadata=metadata
                    )
                    
                    updated_color_points.append(color_point)
                    
                except (ValueError, IndexError, TypeError) as row_error:
                    logger.warning(f"Skipping invalid row in datasheet: {row_error}")
                    continue
            
            # Update color points if we got valid data
            if updated_color_points:
                self.color_points = updated_color_points
                logger.info(f"Synced {len(updated_color_points)} points from datasheet")
            else:
                logger.warning("No valid data found in datasheet for sync")
                
        except Exception as e:
            logger.exception(f"Failed to sync from datasheet: {e}")
            raise
    
    
    def _update_status(self, message: str):
        """Update status bar with message."""
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text=message)
    
    def _update_format_indicator(self, db_format: str):
        """Update the database format indicator with appropriate styling.
        
        Args:
            db_format: 'NORMALIZED', 'ACTUAL_LAB', 'MIXED', or 'UNKNOWN'
        """
        if not hasattr(self, 'format_indicator'):
            return
            
        # Format-specific styling
        format_styles = {
            'ACTUAL_LAB': {
                'text': '✅ ACTUAL LAB',
                'foreground': 'darkgreen',
                'background': 'lightgreen'
            },
            'NORMALIZED': {
                'text': '🔧 NORMALIZED', 
                'foreground': 'darkorange',
                'background': 'lightyellow'
            },
            'MIXED': {
                'text': '⚠️ MIXED',
                'foreground': 'darkred',
                'background': 'lightpink'
            },
            'UNKNOWN': {
                'text': '❓ UNKNOWN',
                'foreground': 'gray',
                'background': 'lightgray'
            }
        }
        
        style = format_styles.get(db_format, format_styles['UNKNOWN'])
        
        self.format_indicator.config(
            text=style['text'],
            foreground=style['foreground'],
            background=style['background']
        )
        
        # Add tooltip for more information
        tooltip_text = {
            'ACTUAL_LAB': 'Database contains proper L*a*b* values (L*: 0-100, a*b*: ±127.5)',
            'NORMALIZED': 'Database contains normalized values (0-1 range) - auto-correcting',
            'MIXED': 'Database contains mixed formats - using heuristic detection',
            'UNKNOWN': 'Could not determine database format'
        }
        
        # Simple tooltip implementation
        tooltip = tooltip_text.get(db_format, 'Format detection status')
        self.format_indicator.config(text=f"{style['text']}")  # Keep it simple for now
    
    def _update_plot_title(self, sample_set_name):
        """Update the plot title label to show current sample set."""
        if hasattr(self, 'plot_title_label'):
            self.plot_title_label.config(text=f"RGB Ternary Analysis: {sample_set_name}")
    
    def _on_plot_click(self, event):
        """Handle clicks on the plot for point identification and selection."""
        if event.inaxes != self.ax or not self.color_points:
            return
        
        try:
            # Find the closest point to the click
            click_x, click_y = event.xdata, event.ydata
            if click_x is None or click_y is None:
                return
            
            closest_point_idx = None
            min_distance = float('inf')
            
            for i, point in enumerate(self.color_points):
                if not hasattr(point, 'ternary_coords') or point.ternary_coords is None:
                    continue
                
                point_x, point_y = point.ternary_coords[0], point.ternary_coords[1]
                distance = ((click_x - point_x) ** 2 + (click_y - point_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_point_idx = i
            
            # Only select if click is reasonably close to a point (within 0.05 units)
            if closest_point_idx is not None and min_distance < 0.05:
                # Clear previous labels
                for label in self.point_labels:
                    try:
                        label.remove()
                    except:
                        pass
                self.point_labels.clear()
                
                # Toggle selection
                if closest_point_idx in self.selected_points:
                    self.selected_points.remove(closest_point_idx)
                else:
                    # Clear other selections for single-select behavior
                    self.selected_points.clear()
                    self.selected_points.add(closest_point_idx)
                
                # Show point information if selected
                if closest_point_idx in self.selected_points:
                    point = self.color_points[closest_point_idx]
                    self._show_point_info(point, closest_point_idx)
                
                # Refresh plot to show selection highlighting
                self._refresh_plot_only()
            
        except Exception as e:
            logger.exception(f"Error handling plot click: {e}")
    
    def _show_point_info(self, point, point_index):
        """Show information about the selected point."""
        try:
            # Create info text
            info_lines = [f"🔍 Point: {point.id}"]
            
            # Add RGB and Lab values
            info_lines.append(f"RGB: ({point.rgb[0]:.1f}, {point.rgb[1]:.1f}, {point.rgb[2]:.1f})")
            info_lines.append(f"L*a*b*: ({point.lab[0]:.1f}, {point.lab[1]:.1f}, {point.lab[2]:.1f})")
            
            # Add ternary coordinates
            if hasattr(point, 'ternary_coords'):
                x, y = point.ternary_coords[0], point.ternary_coords[1]
                info_lines.append(f"Ternary: ({x:.3f}, {y:.3f})")
            
            # Add metadata if available
            if hasattr(point, 'metadata') and point.metadata:
                if 'image_name' in point.metadata:
                    info_lines.append(f"Image: {point.metadata['image_name']}")
                if 'coordinate_point' in point.metadata:
                    info_lines.append(f"Point #{point.metadata['coordinate_point']}")
            
            # Check if point is in a cluster
            cluster_info = ""
            if hasattr(self, 'clusters') and self.clusters:
                for cluster_id, cluster_points in self.clusters.items():
                    if point in cluster_points:
                        cluster_info = f" (Cluster {cluster_id})"
                        break
            
            # Display info in status bar
            status_text = f"Selected: {point.id}{cluster_info} | " + " | ".join(info_lines[1:])
            self._update_status(status_text)
            
            # Add floating label near the point
            point_x, point_y = point.ternary_coords[0], point.ternary_coords[1]
            
            # Main label with DataID
            main_label = self.ax.annotate(
                f"{point.id}{cluster_info}",
                (point_x, point_y),
                xytext=(10, 15),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black'),
                fontsize=10,
                fontweight='bold',
                zorder=10
            )
            
            # Secondary label with color info
            color_label = self.ax.annotate(
                f"RGB({point.rgb[0]:.0f},{point.rgb[1]:.0f},{point.rgb[2]:.0f})",
                (point_x, point_y),
                xytext=(10, -5),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7),
                fontsize=8,
                zorder=10
            )
            
            self.point_labels.extend([main_label, color_label])
            self.canvas.draw()
            
            logger.info(f"Selected point: {point.id} at index {point_index}")
            
        except Exception as e:
            logger.exception(f"Error showing point info: {e}")
            self.window.update_idletasks()
        else:
            logger.info(message)
    
    def _open_spectral_analyzer(self):
        """Open spectral analyzer window."""
        try:
            # Check if we have an app reference with current file
            if hasattr(self, 'app_ref') and hasattr(self.app_ref, 'current_file') and self.app_ref.current_file:
                # Use app's analysis manager to open spectral analysis
                if hasattr(self.app_ref, 'analysis_manager'):
                    self.app_ref.analysis_manager.open_spectral_analysis()
                else:
                    # Fallback: create spectral analyzer directly
                    self._create_spectral_analyzer()
            else:
                # No app context - try to create spectral analyzer directly
                self._create_spectral_analyzer()
                
        except Exception as e:
            logger.exception("Failed to open spectral analyzer")
            messagebox.showerror("Spectral Analysis Error", f"Failed to open spectral analyzer: {e}")
    
    def _create_spectral_analyzer(self):
        """Create spectral analyzer window directly."""
        try:
            from utils.spectral_analyzer import SpectralAnalyzer
            
            # Create analyzer instance (no parent parameter needed)
            spectral_analyzer = SpectralAnalyzer()
            
            # If we have color points, analyze them
            if self.color_points:
                # Convert ColorPoint objects to ColorMeasurement objects for analysis
                from utils.color_analyzer import ColorMeasurement
                from datetime import datetime
                measurements = []
                for i, point in enumerate(self.color_points):
                    measurement = ColorMeasurement(
                        coordinate_id=i,
                        coordinate_point=1,  # Default to 1 for single point
                        position=(point.metadata.get('x_position', 0), point.metadata.get('y_position', 0)),
                        rgb=point.rgb,
                        lab=point.lab,
                        sample_area={'id': point.id, 'metadata': point.metadata},
                        measurement_date=datetime.now().isoformat(),
                        notes=f"Color point: {point.id}"
                    )
                    measurements.append(measurement)
                
                # Perform spectral analysis
                spectral_data = spectral_analyzer.analyze_spectral_response(measurements)
                
                # Plot the results
                spectral_analyzer.plot_spectral_response(spectral_data, interactive=True)
                
                self._update_status(f"Spectral analysis complete: {len(measurements)} samples analyzed")
            else:
                self._update_status("No color data available for spectral analysis")
            
        except ImportError:
            messagebox.showwarning(
                "Component Missing", 
                "Spectral analyzer module not available.\n\n"
                "This feature may require additional components to be installed."
            )
        except Exception as e:
            raise Exception(f"Could not create spectral analyzer: {e}")
    
    def _export_data(self):
        """Export current color data using export manager with Plot_3D format compatibility."""
        # Ensure clusters are computed if K-means is enabled
        if self.show_clusters.get() and self.cluster_manager.has_sklearn() and (not hasattr(self, 'clusters') or not self.clusters):
            logger.info("Computing clusters for export...")
            self._compute_clusters()
        
        # Use export manager to handle the export
        self.export_manager.export_data(
            color_points=self.color_points,
            clusters=self.clusters if self.clusters else None,
            cluster_manager=self.cluster_manager,
            sample_set_name=self.sample_set_name
        )
    
    def _exit_ternary(self):
        """Exit ternary plot window and return to launcher."""
        try:
            # Ask for confirmation
            if messagebox.askyesno("Close Ternary Window", 
                                 "Close the Ternary Plot window?\n\n"
                                 "You will return to the StampZ-III launcher where you can\n"
                                 "choose a different analysis mode or exit completely.\n\n"
                                 "Any unsaved cluster data will be lost unless you've used the \"Save Clusters\" button."):
                
                # Clean up current window (don't auto-return to launcher)
                self._on_close(return_to_launcher=False)
                
                # Return to the main launcher
                self._return_to_launcher()
                
                logger.info("User requested ternary window closure, returning to launcher")
        except Exception as e:
            logger.exception(f"Error during ternary window exit: {e}")
            # Force close if there's an error but still try to return to launcher
            try:
                self._on_close(return_to_launcher=False)
                self._return_to_launcher()
            except:
                # If everything fails, just exit
                import sys
                sys.exit(0)
    
    def _return_to_launcher(self):
        """Return to the main StampZ-III launcher with crash protection."""
        try:
            logger.info("Attempting to return to launcher...")
            
            # Add delay to ensure window cleanup is complete
            import time
            time.sleep(0.2)  # Increased delay for better cleanup
            
            # Handle launcher return based on context
            if self.parent:
                # Internal mode - close this window and let parent continue
                logger.info("Internal mode - closing ternary window, returning to main app")
                try:
                    if hasattr(self, 'window') and self.window:
                        self.window.destroy()
                    # Don't exit the entire application, just return to parent
                    return
                except Exception as cleanup_error:
                    logger.warning(f"Error during internal mode cleanup: {cleanup_error}")
                    return
            else:
                # Standalone mode - return to launcher
                logger.info("Standalone mode - showing launcher selector")
                try:
                    # Import and show the launcher with protection
                    from launch_selector import LaunchSelector
                    
                    # Create launcher with error handling
                    selector = LaunchSelector()
                    selected_mode = selector.show()
                    
                    # Handle launcher cleanup more safely
                    try:
                        if hasattr(selector, 'root') and selector.root:
                            if hasattr(selector.root, 'winfo_exists'):
                                try:
                                    if selector.root.winfo_exists():
                                        selector.root.quit()
                                        selector.root.destroy()
                                except tk.TclError:
                                    pass  # Window already destroyed
                    except Exception as cleanup_error:
                        logger.warning(f"Selector cleanup warning (safe to ignore): {cleanup_error}")
                    
                    # Handle the selected mode with better error protection
                    self._handle_launcher_selection(selected_mode)
                    
                except ImportError as ie:
                    logger.error(f"Could not import launcher: {ie}")
                    import sys
                    sys.exit(0)
                except Exception as launcher_error:
                    logger.exception(f"Error with launcher: {launcher_error}")
                    import sys
                    sys.exit(0)
                    
        except Exception as e:
            logger.exception(f"Critical error in _return_to_launcher: {e}")
            # If anything fails, exit gracefully without crashing
            try:
                import sys
                logger.info("Performing safe exit due to launcher failure")
                sys.exit(0)  # Use 0 instead of 1 to avoid error appearance
            except:
                # Last resort - silent exit
                import os
                os._exit(0)
    
    def _handle_launcher_selection(self, selected_mode):
        """Handle launcher selection with protection against crashes."""
        try:
            if selected_mode == "full":
                logger.info("Launching full StampZ application")
                from app import StampZApp
                import tkinter as tk
                
                root = tk.Tk()
                app = StampZApp(root)
                root.mainloop()
                
            elif selected_mode == "ternary":
                logger.info("Relaunching ternary viewer")
                from main import launch_ternary_viewer
                launch_ternary_viewer()
                
            elif selected_mode == "plot3d":
                logger.info("Launching Plot_3D mode")
                from plot3d.standalone_plot3d import main as plot3d_main
                plot3d_main()
                
            else:
                # User cancelled or chose to exit
                logger.info("User cancelled launcher or chose to exit")
                import sys
                sys.exit(0)
                
        except Exception as launch_error:
            logger.exception(f"Error launching selected mode '{selected_mode}': {launch_error}")
            # Exit cleanly instead of with error code
            import sys
            sys.exit(0)
    
    def _exit_application(self):
        """Exit entire application - works for both internal and standalone modes."""
        try:
            # Determine appropriate message based on context
            if self.parent:
                message = ("Exit the StampZ-III application?\n\n"
                          "This will close all windows including the main application.\n"
                          "Make sure you've saved any important work.")
            else:
                message = ("Exit the entire StampZ-III application?\n\n"
                          "This will close all windows and stop all background processes.\n"
                          "Make sure you've saved any important work.")
            
            # Double confirmation for full application exit
            if messagebox.askyesno("Exit Application", message):
                
                # Clean up this window first
                try:
                    if self.datasheet_manager.is_datasheet_linked():
                        self.datasheet_manager.close_datasheet()
                        logger.info("Cleaned up datasheet references before application exit")
                except Exception as cleanup_error:
                    logger.warning(f"Error during cleanup before exit: {cleanup_error}")
                
                # Handle cleanup based on context
                if self.parent:
                    # Internal mode: Clean up properly and close parent application
                    try:
                        # Close any child windows
                        for widget in self.window.winfo_children():
                            if isinstance(widget, tk.Toplevel):
                                widget.destroy()
                        
                        # Close this window
                        self.window.destroy()
                        
                        # Close parent application if it exists
                        if hasattr(self.parent, 'quit'):
                            self.parent.quit()
                        if hasattr(self.parent, 'destroy'):
                            self.parent.destroy()
                            
                        logger.info("User requested full application exit from internal ternary window")
                    except Exception as parent_cleanup_error:
                        logger.warning(f"Error during parent cleanup: {parent_cleanup_error}")
                else:
                    # Standalone mode: Direct exit
                    try:
                        # Close any child windows
                        for widget in self.window.winfo_children():
                            if isinstance(widget, tk.Toplevel):
                                widget.destroy()
                    except Exception as window_cleanup_error:
                        logger.warning(f"Error closing child windows: {window_cleanup_error}")
                    
                    # Quit the window
                    self.window.quit()  # Stop the mainloop
                    self.window.destroy()  # Destroy the window
                    
                    logger.info("User requested full application exit from standalone ternary window")
                
                # Final exit for both modes
                try:
                    import sys
                    sys.exit(0)
                except:
                    pass
        
        except Exception as e:
            logger.exception(f"Error during application exit: {e}")
            # Force exit if there's an error
            try:
                import sys
                sys.exit(1)
            except:
                pass
    
    def _on_close(self, return_to_launcher=True):
        """Handle window close with improved crash protection.
        
        Args:
            return_to_launcher: If True, return to launcher after closing. If False, just destroy window.
        """
        try:
            # Prevent multiple close attempts
            if hasattr(self, '_closing'):
                return
            self._closing = True
            
            logger.info(f"Window close initiated, return_to_launcher={return_to_launcher}")
            
            # Clean up datasheet references
            try:
                if hasattr(self, 'datasheet_manager') and self.datasheet_manager.is_datasheet_linked():
                    self.datasheet_manager.close_datasheet()
                    if hasattr(self, '_hide_datasheet_sync_button'):
                        self._hide_datasheet_sync_button()
                    logger.info("Cleaned up datasheet references on window close")
            except Exception as e:
                logger.warning(f"Error during datasheet cleanup: {e}")
            
            # Destroy the window safely with better error handling
            try:
                if hasattr(self, 'window') and self.window:
                    try:
                        # Check if window still exists before destroying
                        if hasattr(self.window, 'winfo_exists') and self.window.winfo_exists():
                            self.window.quit()  # Stop mainloop first
                            self.window.destroy()  # Then destroy window
                            logger.info("Window destroyed successfully")
                    except tk.TclError as tcl_error:
                        logger.info(f"Window already destroyed: {tcl_error}")
                    except Exception as destroy_error:
                        logger.warning(f"Window destruction error: {destroy_error}")
            except Exception as window_error:
                logger.warning(f"Window handling error: {window_error}")
            
            # Return to launcher if requested (and not already in a return-to-launcher flow)
            if return_to_launcher and not hasattr(self, '_returning_to_launcher'):
                self._returning_to_launcher = True  # Prevent recursion
                try:
                    logger.info("Window closed via X button, returning to launcher")
                    # Delay to ensure window cleanup is complete
                    import time
                    time.sleep(0.1)  # Slightly longer delay
                    self._return_to_launcher()
                except Exception as e:
                    logger.exception(f"Failed to return to launcher from _on_close: {e}")
                    # If launcher fails, exit gracefully
                    try:
                        import sys
                        sys.exit(0)
                    except:
                        import os
                        os._exit(0)
        
        except Exception as close_error:
            logger.exception(f"Critical error in _on_close: {close_error}")
            # Emergency exit if all else fails
            try:
                import sys
                sys.exit(0)
            except:
                import os
                os._exit(0)


def demo_ternary_window():
    """Demo function for testing."""
    print("🎨 Launching Ternary Plot Window Demo...")
    
    # Create demo window
    root = tk.Tk()
    root.withdraw()  # Hide root
    
    # Load some sample data
    bridge = ColorDataBridge()
    sample_sets = bridge.get_available_sample_sets()
    
    if sample_sets:
        points, db_format = bridge.load_color_points_from_database(sample_sets[0])
        window = TernaryPlotWindow(parent=root, sample_set_name=sample_sets[0], 
                                 color_points=points, db_format=db_format)
        root.mainloop()
    else:
        print("❌ No sample data available for demo")


if __name__ == "__main__":
    demo_ternary_window()