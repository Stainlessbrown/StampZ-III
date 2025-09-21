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
from tkinter import ttk, messagebox, filedialog
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

# Optional ML imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class TernaryPlotWindow:
    """Plot_3D-style window for ternary plot visualization with K-means clustering."""
    
    def __init__(self, parent=None, sample_set_name="StampZ_Analysis", color_points=None, datasheet_ref=None):
        """Initialize ternary plot window.
        
        Args:
            parent: Parent window (for integration)
            sample_set_name: Name of the sample set
            color_points: Optional pre-loaded color points
            datasheet_ref: Optional reference to associated realtime datasheet
        """
        self.parent = parent
        self.sample_set_name = sample_set_name
        self.color_points = color_points or []
        self.clusters = {}
        self.cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        self.show_hull = tk.BooleanVar(value=True)
        self.show_clusters = tk.BooleanVar(value=False)
        self.n_clusters = tk.IntVar(value=3)
        
        # Track external file for integration
        self.external_file_path = None
        
        # Reference to associated realtime datasheet for bidirectional data flow
        self.datasheet_ref = datasheet_ref
        
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
            
            logger.info("Loading initial data...")
            self._load_initial_data()
            
            logger.info("Creating initial plot...")
            self._create_initial_plot()
            
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
        
        self.window.title(f"RGB Ternary Analysis - {self.sample_set_name}")
        self.window.geometry("1200x800")
        
        # Center window
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (self.window.winfo_width() // 2)
        y = (self.window.winfo_screenheight() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")
        
        # Window configuration
        self.window.resizable(True, True)
        self.window.minsize(800, 600)
        
        if self.parent:
            self.window.transient(self.parent)
        
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_ui(self):
        """Setup the user interface with Plot_3D-style layout."""
        # Main container
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top toolbar
        self._create_toolbar(main_frame)
        
        # Plot area
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Create matplotlib figure with interactive capabilities
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        
        # Enhanced navigation toolbar with zoom/pan for dense data points (pack FIRST)
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.nav_toolbar.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        
        # Add toolbar info
        nav_info = ttk.Label(plot_frame, text="💡 Use zoom/pan toolbar above when points overlap", 
                            font=('Arial', 9, 'italic'), foreground='gray')
        nav_info.pack(side=tk.TOP, pady=(0, 2))
        
        # Pack canvas AFTER toolbar so toolbar appears above
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add click handler for point selection
        self.selected_points = set()
        self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(2, 0))
    
    def _create_toolbar(self, parent):
        """Create toolbar similar to Plot_3D."""
        self.toolbar = ttk.Frame(parent)
        self.toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Left side - Data controls
        left_frame = ttk.Frame(self.toolbar)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(left_frame, text="Sample Set:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(left_frame, text=self.sample_set_name, font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Button(left_frame, text="Load Database", command=self._load_data_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(left_frame, text="Load External File", command=self._load_external_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(left_frame, text="Refresh Plot", command=self._refresh_plot_only).pack(side=tk.LEFT, padx=(0, 5))
        
        # Datasheet sync button (initially hidden)
        self.refresh_from_datasheet_btn = ttk.Button(left_frame, text="↻ Sync from Datasheet", command=self._refresh_from_datasheet)
        # Don't pack initially - will be shown when datasheet is opened
        
        # Middle - Visualization options
        middle_frame = ttk.Frame(self.toolbar)
        middle_frame.pack(side=tk.LEFT)
        
        ttk.Checkbutton(middle_frame, text="Convex Hull", variable=self.show_hull, 
                       command=self._refresh_plot).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Checkbutton(middle_frame, text="K-means Clusters", variable=self.show_clusters,
                       command=self._toggle_clusters).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(middle_frame, text="Clusters:").pack(side=tk.LEFT, padx=(0, 2))
        cluster_spin = ttk.Spinbox(middle_frame, from_=2, to=8, width=3, textvariable=self.n_clusters,
                                  command=self._update_clusters)
        cluster_spin.pack(side=tk.LEFT, padx=(0, 15))
        
        # Right side - Export controls
        right_frame = ttk.Frame(self.toolbar)
        right_frame.pack(side=tk.RIGHT)
        
        ttk.Button(right_frame, text="Save Plot", command=self._save_plot).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(right_frame, text="Open Datasheet", command=self._open_plot3d).pack(side=tk.LEFT, padx=(0, 5))
    
    def _load_initial_data(self):
        """Load initial data if color_points not provided."""
        if not self.color_points:
            try:
                self.color_points = self.bridge.load_color_points_from_database(self.sample_set_name)
                self._update_status(f"Loaded {len(self.color_points)} color points")
            except Exception as e:
                self._update_status(f"No data loaded: {e}")
    
    def _create_initial_plot(self):
        """Create the initial ternary plot."""
        self._refresh_plot()
    
    def _refresh_plot(self):
        """Refresh the ternary plot with current settings.
        
        Legacy method - maintains original behavior for backward compatibility.
        """
        # This is now just a redirection to the appropriate method based on context
        if self.datasheet_ref and hasattr(self.datasheet_ref, 'sheet'):
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
            
            if self.show_clusters.get() and HAS_SKLEARN:
                self._plot_clusters()
            
            # Update plot
            self.ax.set_aspect('equal')
            self.ax.axis('off')
            self.canvas.draw()
            
            self._update_status(f"Plot refreshed: {len(self.color_points)} points")
            
        except Exception as e:
            logger.exception("Error refreshing plot")
            self._update_status(f"Plot error: {e}")
    
    def _refresh_from_datasheet(self):
        """Refresh the plot by syncing data from linked datasheet."""
        if not self.datasheet_ref or not hasattr(self.datasheet_ref, 'sheet'):
            self._update_status("No datasheet linked for sync")
            return
        
        try:
            self._sync_from_datasheet()
            # Now refresh the plot with the synced data
            self._refresh_plot_only()
            self._update_status(f"Plot updated from datasheet: {len(self.color_points)} points")
        except Exception as e:
            logger.exception("Failed to sync from datasheet")
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
        """Plot the color data points."""
        if not self.color_points:
            return
        
        # Extract coordinates and colors, ensure ternary coordinates are computed
        x_coords = []
        y_coords = []
        colors = []
        markers = []
        
        # Color map for named colors from datasheet
        color_map = {
            'red': 'red', 'blue': 'blue', 'green': 'green', 'yellow': 'yellow',
            'purple': 'purple', 'orange': 'orange', 'pink': 'pink', 'brown': 'brown',
            'black': 'black', 'white': 'white', 'gray': 'gray', 'grey': 'gray',
            'cyan': 'cyan', 'magenta': 'magenta'
        }
        
        for point in self.color_points:
            # Ensure ternary coordinates are computed
            if not hasattr(point, 'ternary_coords') or point.ternary_coords is None:
                point.ternary_coords = self.ternary_plotter.rgb_to_ternary(point.rgb)
            
            x_coords.append(point.ternary_coords[0])
            y_coords.append(point.ternary_coords[1])
            
            # Check if point has datasheet marker color, otherwise use RGB color
            if (hasattr(point, 'metadata') and 'marker_color' in point.metadata and 
                point.metadata['marker_color'] and point.metadata['marker_color'].lower() in color_map):
                colors.append(color_map[point.metadata['marker_color'].lower()])
            else:
                # Use RGB color of the point
                colors.append(tuple(c/255.0 for c in point.rgb))
            
            # Get marker style from metadata
            if hasattr(point, 'metadata') and 'marker' in point.metadata:
                marker_style = point.metadata['marker']
                # Convert Plot_3D marker symbols to matplotlib symbols
                marker_map = {'.': 'o', 'o': 'o', 's': 's', '^': '^', 'v': 'v', 
                            'd': 'D', 'x': 'x', '+': '+', '*': '*'}
                markers.append(marker_map.get(marker_style, 'o'))
            else:
                markers.append('o')  # Default circle
        
        # Create size and edge color arrays for highlighting selected points
        sizes = []
        edgecolors = []
        linewidths = []
        
        for i in range(len(self.color_points)):
            if i in getattr(self, 'selected_points', set()):
                sizes.append(100)  # Larger size for selected
                edgecolors.append('yellow')
                linewidths.append(3)
            else:
                sizes.append(60)  # Normal size
                edgecolors.append('black')
                linewidths.append(0.5)
        
        # Plot points with highlighting and different markers
        # For simplicity, we'll plot all points as circles but with proper colors
        # Future enhancement: plot each marker type separately
        scatter = self.ax.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.8, 
                                edgecolors=edgecolors, linewidths=linewidths, zorder=3)
        
        # If we have marker metadata, show it in point labels (for small datasets)
        if len(self.color_points) <= 15:
            for i, point in enumerate(self.color_points):
                label = point.id
                if hasattr(point, 'metadata') and 'marker' in point.metadata:
                    label += f" ({point.metadata['marker']})"
                if hasattr(point, 'metadata') and 'marker_color' in point.metadata:
                    label += f" [{point.metadata['marker_color']}]"
                self.ax.annotate(label, (x_coords[i], y_coords[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.7)
        
        # Add convex hull if requested
        if self.show_hull.get() and len(self.color_points) >= 3:
            self._add_convex_hull(x_coords, y_coords)
    
    def _add_convex_hull(self, x_coords, y_coords):
        """Add convex hull to the plot."""
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack((x_coords, y_coords))
            hull = ConvexHull(points)
            
            # Plot hull
            for simplex in hull.simplices:
                self.ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.4, linewidth=1)
            
            # Fill hull area
            hull_points = points[hull.vertices]
            self.ax.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.1, color='gray')
            
        except ImportError:
            pass  # Skip if scipy not available
        except Exception as e:
            logger.warning(f"Convex hull failed: {e}")
    
    def _toggle_clusters(self):
        """Toggle K-means cluster display."""
        if self.show_clusters.get():
            if not HAS_SKLEARN:
                messagebox.showwarning("K-means Not Available", 
                                     "Scikit-learn is required for K-means clustering.\n\n"
                                     "Install with: pip install scikit-learn")
                self.show_clusters.set(False)
                return
            self._compute_clusters()
        self._refresh_plot()
    
    def _compute_clusters(self):
        """Compute K-means clusters on ternary coordinates."""
        if not self.color_points or not HAS_SKLEARN:
            return
        
        try:
            # Extract ternary coordinates
            ternary_coords = np.array([point.ternary_coords for point in self.color_points])
            
            # Apply K-means
            n_clusters = min(self.n_clusters.get(), len(self.color_points) // 2)
            if n_clusters < 2:
                self.clusters = {}
                return
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(ternary_coords)
            
            # Group points by cluster
            self.clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in self.clusters:
                    self.clusters[label] = []
                self.clusters[label].append(self.color_points[i])
            
            self._update_status(f"K-means: {len(self.clusters)} clusters computed")
            
        except Exception as e:
            logger.exception("K-means clustering failed")
            self._update_status(f"Clustering error: {e}")
            self.clusters = {}
    
    def _plot_clusters(self):
        """Plot K-means cluster results."""
        if not self.clusters:
            return
        
        # Plot cluster boundaries and centroids
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
            
            # Plot centroid
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            self.ax.scatter(centroid_x, centroid_y, c=color, s=150, marker='X', 
                          edgecolors='black', linewidths=2, zorder=4,
                          label=f'Cluster {label} ({len(points)} pts)')
        
        # Add legend
        self.ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
    
    def _update_clusters(self):
        """Update clusters when spinbox changes."""
        if self.show_clusters.get():
            self._compute_clusters()
            self._refresh_plot()
    
    def _rgb_to_ternary_coords(self, rgb):
        """Convert RGB values to ternary coordinates."""
        return self.ternary_plotter.rgb_to_ternary(rgb)
    
    def _show_datasheet_sync_button(self):
        """Show the datasheet sync button when a datasheet is linked."""
        if hasattr(self, 'refresh_from_datasheet_btn'):
            self.refresh_from_datasheet_btn.pack(side=tk.LEFT, padx=(5, 15))
    
    def _hide_datasheet_sync_button(self):
        """Hide the datasheet sync button when no datasheet is linked."""
        if hasattr(self, 'refresh_from_datasheet_btn'):
            self.refresh_from_datasheet_btn.pack_forget()
    
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
                    self.color_points = self.bridge.load_color_points_from_database(selected_set)
                    self.window.title(f"RGB Ternary Analysis - {selected_set}")
                    self._refresh_plot()
                    dialog.destroy()
            
            ttk.Button(dialog, text="Load", command=on_load).pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
    
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
            
            # Process each row
            for i, row in df.iterrows():
                try:
                    # Get ID (try multiple column names)
                    point_id = None
                    for id_col in ['DataID', 'ID', 'Name', 'Sample', 'Point', 'Label']:
                        if id_col in df.columns:
                            point_id = str(row[id_col])
                            break
                    if not point_id:
                        point_id = f"Point_{i+1}"
                    
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
        """Save current plot to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"ternary_plot_{self.sample_set_name}_{timestamp}.png"
            
            filename = filedialog.asksaveasfilename(
                title="Save Ternary Plot",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=default_name
            )
            
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self._update_status(f"Plot saved: {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plot: {e}")
    
    def _open_plot3d(self):
        """Open realtime datasheet with current data (Plot_3D workflow)."""
        try:
            # Open realtime datasheet just like Plot_3D does
            from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
            
            # Create datasheet name based on current data
            if self.external_file_path:
                datasheet_name = f"Ternary: {os.path.basename(self.external_file_path, '.ods').replace('.xlsx', '')}"
            else:
                datasheet_name = f"Ternary: {self.sample_set_name}"
            
            # Create realtime datasheet (don't load initial data, we'll populate it)
            datasheet = RealtimePlot3DSheet(
                parent=self.window,
                sample_set_name=datasheet_name,
                load_initial_data=False
            )
            
            # Populate datasheet with current ternary data converted to Plot_3D format
            self._populate_datasheet_with_current_data(datasheet)
            
            # Store reference for bidirectional updates
            self.datasheet_ref = datasheet
            
            # Show the "Sync from Datasheet" button now that datasheet is linked
            self._show_datasheet_sync_button()
            
            # Add back-reference to datasheet for navigation
            datasheet.ternary_window_ref = self
            
            # Add "Return to Ternary" button to datasheet toolbar
            if hasattr(datasheet, 'plot3d_btn') and hasattr(datasheet.plot3d_btn, 'pack_info'):
                # Add button after the Plot_3D button
                return_btn = ttk.Button(
                    datasheet.plot3d_btn.master, 
                    text="Return to Ternary", 
                    command=self._bring_ternary_to_front
                )
                return_btn.pack(side=tk.LEFT, padx=5)
            
            self._update_status(f"Realtime datasheet opened: {datasheet_name}")
            
            messagebox.showinfo(
                "Datasheet Ready",
                f"Realtime datasheet opened with {len(self.color_points)} color points.\n\n"
                "• Data converted to Plot_3D normalized format (0-1 range)\n"
                "• Use 'Launch Plot_3D' button in datasheet for 3D visualization\n"
                "• Edit data in datasheet and refresh ternary plot to see changes"
            )
                    
        except Exception as e:
            error_msg = f"Failed to open realtime datasheet: {e}"
            logger.exception(error_msg)
            messagebox.showerror("Datasheet Error", error_msg)
    
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
                
            # Convert ColorPoint objects to Plot_3D normalized format
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
                
                # Create row data in Plot_3D format
                row_data = [
                    round(l_norm, 6),      # Xnorm (L*)
                    round(a_norm, 6),      # Ynorm (a*) 
                    round(b_norm, 6),      # Znorm (b*)
                    point.id,              # DataID
                    '',                    # Cluster (empty for user to assign)
                    '',                    # ∆E (empty - to be calculated)
                    '.',                   # Marker (default)
                    'blue',                # Color (default)
                    '',                    # Centroid_X (empty)
                    '',                    # Centroid_Y (empty)
                    '',                    # Centroid_Z (empty)
                    '',                    # Sphere (empty)
                    ''                     # Radius (empty)
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
                
                # Leave rows 2-7 (indices 1-6) empty for centroid data - this is the protected pink area
                
                # Insert data starting at row 8 (index 7) as per Plot_3D format
                for i, row_data in enumerate(plot3d_data_rows):
                    row_index = 7 + i  # Start at row 7 (display row 8)
                    datasheet.sheet.set_row_data(row_index, values=row_data)
                
                # Apply proper formatting and validation
                datasheet._apply_formatting()
                datasheet._setup_validation()
                
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
    
    def _on_plot_click(self, event):
        """Handle clicks on the ternary plot for point selection."""
        if event.inaxes != self.ax or not self.color_points:
            return
            
        # Find closest point to click
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
            
        min_distance = float('inf')
        closest_point_idx = None
        
        for i, point in enumerate(self.color_points):
            x, y = point.ternary_coords
            distance = ((x - click_x) ** 2 + (y - click_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_point_idx = i
        
        # Only select if click is reasonably close (within 0.05 units)
        if min_distance < 0.05 and closest_point_idx is not None:
            if closest_point_idx in self.selected_points:
                self.selected_points.remove(closest_point_idx)
            else:
                self.selected_points.add(closest_point_idx)
            
            # Show point info
            point = self.color_points[closest_point_idx]
            self._update_status(f"Selected: {point.id} - RGB({point.rgb[0]:.0f},{point.rgb[1]:.0f},{point.rgb[2]:.0f})")
            
            # Refresh plot to show selection
            self._refresh_plot()
    
    def _update_status(self, message: str):
        """Update status bar."""
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text=message)
            self.window.update_idletasks()
        else:
            logger.info(message)
    
    def _on_close(self):
        """Handle window close."""
        self.window.destroy()


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
        points = bridge.load_color_points_from_database(sample_sets[0])
        window = TernaryPlotWindow(parent=root, sample_set_name=sample_sets[0], color_points=points)
        root.mainloop()
    else:
        print("❌ No sample data available for demo")


if __name__ == "__main__":
    demo_ternary_window()