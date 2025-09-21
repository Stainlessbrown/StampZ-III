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
    
    def __init__(self, parent=None, sample_set_name="StampZ_Analysis", color_points=None):
        """Initialize ternary plot window.
        
        Args:
            parent: Parent window (for integration)
            sample_set_name: Name of the sample set
            color_points: Optional pre-loaded color points
        """
        self.parent = parent
        self.sample_set_name = sample_set_name
        self.color_points = color_points or []
        self.clusters = {}
        self.cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        self.show_hull = tk.BooleanVar(value=True)
        self.show_clusters = tk.BooleanVar(value=False)
        self.n_clusters = tk.IntVar(value=3)
        
        # Initialize bridge and plotter
        self.bridge = ColorDataBridge()
        self.ternary_plotter = TernaryPlotter()
        
        self._create_window()
        self._setup_ui()
        self._load_initial_data()
        self._create_initial_plot()
    
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
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        nav_toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        nav_toolbar.update()
        
        # Status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=(2, 0))
    
    def _create_toolbar(self, parent):
        """Create toolbar similar to Plot_3D."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # Left side - Data controls
        left_frame = ttk.Frame(toolbar)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(left_frame, text="Sample Set:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(left_frame, text=self.sample_set_name, font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Button(left_frame, text="Load Database", command=self._load_data_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(left_frame, text="Load External File", command=self._load_external_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(left_frame, text="Refresh", command=self._refresh_plot).pack(side=tk.LEFT, padx=(0, 15))
        
        # Middle - Visualization options
        middle_frame = ttk.Frame(toolbar)
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
        right_frame = ttk.Frame(toolbar)
        right_frame.pack(side=tk.RIGHT)
        
        ttk.Button(right_frame, text="Save Plot", command=self._save_plot).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(right_frame, text="Open Plot_3D", command=self._open_plot3d).pack(side=tk.LEFT, padx=(0, 5))
    
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
        """Refresh the ternary plot with current settings."""
        if not self.color_points:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'No data available\nUse "Load Data" to load sample set', 
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
            
            self._update_status(f"Plot updated: {len(self.color_points)} points")
            
        except Exception as e:
            logger.exception("Error refreshing plot")
            self._update_status(f"Plot error: {e}")
    
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
        
        # Extract coordinates and colors
        x_coords = [point.ternary_coords[0] for point in self.color_points]
        y_coords = [point.ternary_coords[1] for point in self.color_points]
        colors = [tuple(c/255.0 for c in point.rgb) for point in self.color_points]
        
        # Plot points
        scatter = self.ax.scatter(x_coords, y_coords, c=colors, s=60, alpha=0.8, 
                                edgecolors='black', linewidths=0.5, zorder=3)
        
        # Add convex hull if requested
        if self.show_hull.get() and len(self.color_points) >= 3:
            self._add_convex_hull(x_coords, y_coords)
        
        # Add labels for small datasets
        if len(self.color_points) <= 15:
            for i, point in enumerate(self.color_points):
                self.ax.annotate(point.id, (x_coords[i], y_coords[i]), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, alpha=0.7)
    
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
                # Load ODS file using existing bridge functionality
                color_points = self.bridge.load_color_points_from_ods(filename)
                
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
        """Open Plot_3D with current sample set."""
        try:
            from plot3d.standalone_plot3d import main as plot3d_main
            import threading
            
            def launch_plot3d():
                try:
                    plot3d_main()
                except Exception as e:
                    logger.warning(f"Plot_3D launch failed: {e}")
            
            thread = threading.Thread(target=launch_plot3d, daemon=True)
            thread.start()
            
            self._update_status("Plot_3D launched for L*a*b* analysis")
            
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch Plot_3D: {e}")
    
    def _update_status(self, message: str):
        """Update status bar."""
        self.status_bar.config(text=message)
        self.window.update_idletasks()
    
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