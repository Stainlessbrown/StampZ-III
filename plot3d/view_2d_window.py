"""
Separate 2D View Window for precise point selection

This module creates a dedicated window for 2D views that allows users to:
1. Keep the original 3D plot visible
2. Select points in true 2D space with high precision
3. See real-time highlighting in both windows
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np


class View2DWindow:
    """Separate window for 2D plot views with point selection"""
    
    def __init__(self, parent_highlight_manager, data_df, use_rgb=False):
        """Initialize the 2D view window
        
        Args:
            parent_highlight_manager: Reference to the main highlight manager
            data_df: DataFrame with the plot data
            use_rgb: Whether to use RGB or L*a*b* coordinate system
        """
        self.parent_highlight_manager = parent_highlight_manager
        self.data_df = data_df
        self.use_rgb = use_rgb
        
        self.window = None
        self.canvas = None
        self.ax = None
        self.current_view = None
        
        # Track highlight elements for cleanup
        self.highlight_elements = []
        
    def create_window(self, view_type='xz'):
        """Create and show the 2D view window
        
        Args:
            view_type: 'xy' (L*a*), 'xz' (L*b*), or 'yz' (a*b*)
        """
        try:
            # Close existing window if open
            if self.window:
                self.close_window()
            
            # Create new independent window (orphan)
            self.window = tk.Tk()
            self.window.title("StampZ-III: 2D Point Selection View")
            self.window.geometry("900x700")
            
            # Make it completely independent (orphan window)
            self.window.withdraw()  # Hide initially
            self.window.deiconify()  # Show it again (makes it independent)
            
            # Configure as standalone window
            self.window.attributes('-topmost', False)
            self.window.focus_set()
            
            # Set window icon if available (optional)
            try:
                # You can set an icon here if you have one
                pass
            except:
                pass
            
            # Handle window close event
            self.window.protocol("WM_DELETE_WINDOW", self.close_window)
            
            # Create main frame
            main_frame = ttk.Frame(self.window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create control frame at top
            control_frame = ttk.Frame(main_frame)
            control_frame.pack(fill=tk.X, pady=(0, 10))
            
            # View selection buttons
            self.view_buttons = {}
            labels = {
                'xy': 'L*a*' if not self.use_rgb else 'R/G',
                'xz': 'L*b*' if not self.use_rgb else 'R/B',
                'yz': 'a*b*' if not self.use_rgb else 'G/B'
            }
            
            for i, (view, label) in enumerate(labels.items()):
                btn = ttk.Button(
                    control_frame,
                    text=f"{label} View",
                    command=lambda v=view: self.switch_view(v)
                )
                btn.pack(side=tk.LEFT, padx=5)
                self.view_buttons[view] = btn
            
            # Status separator
            ttk.Separator(control_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=10)
            
            # Info label
            self.info_label = ttk.Label(
                control_frame,
                text="🎯 Independent 2D View - Click points for precise selection",
                font=('Arial', 10, 'bold'),
                foreground='darkblue'
            )
            self.info_label.pack(side=tk.LEFT, padx=15)
            
            # Connection status
            self.connection_label = ttk.Label(
                control_frame,
                text="🔗 Connected to 3D Plot",
                font=('Arial', 9),
                foreground='darkgreen'
            )
            self.connection_label.pack(side=tk.LEFT, padx=10)
            
            # Window controls
            window_controls = ttk.Frame(control_frame)
            window_controls.pack(side=tk.RIGHT, padx=5)
            
            # Always on top toggle
            self.always_on_top_var = tk.BooleanVar(value=False)
            always_on_top_cb = ttk.Checkbutton(
                window_controls,
                text="Stay on Top",
                variable=self.always_on_top_var,
                command=self._toggle_always_on_top
            )
            always_on_top_cb.pack(side=tk.LEFT, padx=5)
            
            # Close button
            close_btn = ttk.Button(
                window_controls,
                text="Close Window",
                command=self.close_window
            )
            close_btn.pack(side=tk.LEFT, padx=5)
            
            # Create matplotlib figure and canvas
            self.fig = plt.figure(figsize=(10, 8), facecolor='white')
            self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Set up event handling
            self.canvas.mpl_connect('pick_event', self._on_pick)
            
            # Create initial plot
            self.switch_view(view_type)
            
            print(f"DEBUG: Created 2D view window with {view_type} view")
            
        except Exception as e:
            print(f"Error creating 2D view window: {e}")
            import traceback
            traceback.print_exc()
    
    def switch_view(self, view_type):
        """Switch to a different 2D view
        
        Args:
            view_type: 'xy', 'xz', or 'yz'
        """
        try:
            self.current_view = view_type
            
            # Clear existing plot
            self.fig.clear()
            self.ax = self.fig.add_subplot(111)
            
            # Clear highlight tracking
            self.highlight_elements.clear()
            
            # Filter valid data points
            valid_mask = (self.data_df['Xnorm'].notna() & 
                         self.data_df['Ynorm'].notna() & 
                         self.data_df['Znorm'].notna())
            valid_data = self.data_df[valid_mask]
            
            print(f"DEBUG: Plotting {len(valid_data)} points in 2D window {view_type} view")
            
            # Get coordinates and labels based on view type
            if view_type == 'xy':  # L*a* or R/G view
                x_data = valid_data['Xnorm'].values
                y_data = valid_data['Ynorm'].values
                x_label = 'L*' if not self.use_rgb else 'R'
                y_label = 'a*' if not self.use_rgb else 'G'
            elif view_type == 'xz':  # L*b* or R/B view
                x_data = valid_data['Xnorm'].values
                y_data = valid_data['Znorm'].values
                x_label = 'L*' if not self.use_rgb else 'R'
                y_label = 'b*' if not self.use_rgb else 'B'
            elif view_type == 'yz':  # a*b* or G/B view
                x_data = valid_data['Ynorm'].values
                y_data = valid_data['Znorm'].values
                x_label = 'a*' if not self.use_rgb else 'G'
                y_label = 'b*' if not self.use_rgb else 'B'
            else:
                return
            
            # Create scatter plot with individual pickable points
            for i, (idx, row) in enumerate(valid_data.iterrows()):
                marker = row['Marker'] if pd.notna(row['Marker']) else 'o'
                color = row['Color'] if pd.notna(row['Color']) else 'blue'
                
                scatter = self.ax.scatter(
                    x_data[i], y_data[i],
                    c=color, marker=marker, s=60,
                    picker=True, pickradius=8,
                    alpha=0.8, edgecolors='black', linewidths=0.5
                )
                # Store the DataFrame index for point identification
                scatter._df_index = idx
                scatter._data_id = row.get('DataID', f'Point_{idx}')
            
            # Style the plot
            self.ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
            self.ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
            
            view_names = {
                'xy': 'L*a*' if not self.use_rgb else 'R/G',
                'xz': 'L*b*' if not self.use_rgb else 'R/B',
                'yz': 'a*b*' if not self.use_rgb else 'G/B'
            }
            
            self.ax.set_title(f"{view_names[view_type]} View - Click Points for Precise Selection", 
                             fontsize=16, fontweight='bold', pad=20)
            
            # Set equal aspect ratio and styling
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add distinctive background color for the orphan window
            self.ax.set_facecolor('#f0fff0')  # Very light green background (honeydew)
            
            # Update view button states
            for view, btn in self.view_buttons.items():
                if view == view_type:
                    btn.configure(style='Accent.TButton')
                else:
                    btn.configure(style='TButton')
            
            # Draw the plot
            self.fig.tight_layout()
            self.canvas.draw()
            
            print(f"DEBUG: Successfully switched to {view_type} view in 2D window")
            
        except Exception as e:
            print(f"Error switching to {view_type} view: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_pick(self, event):
        """Handle point picking in the 2D window"""
        try:
            print(f"DEBUG: Point picked in 2D window!")
            
            # Get the DataFrame index from the picked artist
            df_idx = None
            data_id = "Unknown"
            
            if hasattr(event.artist, '_df_index'):
                df_idx = event.artist._df_index
                data_id = getattr(event.artist, '_data_id', f'Point_{df_idx}')
                print(f"DEBUG: Picked point {data_id} (DataFrame index: {df_idx})")
            
            if df_idx is not None:
                # Clear previous highlights in this window
                self.clear_highlights()
                
                # Highlight in this 2D window
                self._highlight_point_2d(df_idx)
                
                # Tell the parent highlight manager to highlight in 3D
                if self.parent_highlight_manager:
                    print(f"DEBUG: Notifying parent to highlight point {data_id}")
                    self.parent_highlight_manager._select_point_by_index(df_idx)
                
                # Update info
                self.info_label.config(
                    text=f"✅ Selected: {data_id}",
                    foreground='darkgreen'
                )
                
                # Update connection status
                self.connection_label.config(
                    text="🟢 Point sent to 3D Plot",
                    foreground='green'
                )
                
                # Reset connection status after a delay
                self.window.after(2000, lambda: self.connection_label.config(
                    text="🔗 Connected to 3D Plot",
                    foreground='darkgreen'
                ))
            
        except Exception as e:
            print(f"Error in 2D window pick handler: {e}")
            import traceback
            traceback.print_exc()
    
    def _highlight_point_2d(self, df_idx):
        """Add highlight to a point in the 2D window"""
        try:
            point = self.data_df.loc[df_idx]
            data_id = point.get('DataID', f'Point_{df_idx}')
            
            # Get coordinates based on current view
            if self.current_view == 'xy':
                x, y = point['Xnorm'], point['Ynorm']
            elif self.current_view == 'xz':
                x, y = point['Xnorm'], point['Znorm']
            elif self.current_view == 'yz':
                x, y = point['Ynorm'], point['Znorm']
            else:
                return
            
            print(f"DEBUG: Adding 2D highlight for {data_id} at ({x:.4f}, {y:.4f})")
            
            # Create highlight circle
            highlight_circle = plt.Circle(
                (x, y), 0.025,  # Larger radius for visibility
                fill=False, edgecolor='red', linewidth=3,
                zorder=1000
            )
            self.ax.add_patch(highlight_circle)
            self.highlight_elements.append(highlight_circle)
            
            # Add label with smart positioning
            self._add_smart_label(x, y, data_id)
            
            # Redraw
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error highlighting point in 2D window: {e}")
    
    def _add_smart_label(self, x, y, data_id):
        """Add a label with smart positioning to avoid plot edges"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Smart positioning logic
        margin = 0.08
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        
        # Default position (top-right)
        offset_x, offset_y = 0.04, 0.04
        ha, va = 'left', 'bottom'
        
        # Adjust if too close to right edge
        if x + offset_x * x_range > xlim[1] - margin * x_range:
            offset_x = -0.04
            ha = 'right'
        
        # Adjust if too close to top edge
        if y + offset_y * y_range > ylim[1] - margin * y_range:
            offset_y = -0.04
            va = 'top'
        
        text_x = x + offset_x * x_range
        text_y = y + offset_y * y_range
        
        # Create label
        highlight_text = self.ax.text(
            text_x, text_y, data_id,
            color='red', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.9, edgecolor='red'),
            zorder=1001, ha=ha, va=va
        )
        self.highlight_elements.append(highlight_text)
    
    def clear_highlights(self):
        """Clear all highlights from the 2D window"""
        try:
            for element in self.highlight_elements:
                if hasattr(element, 'remove'):
                    element.remove()
            self.highlight_elements.clear()
            
            if self.canvas:
                self.canvas.draw_idle()
                
        except Exception as e:
            print(f"Error clearing 2D highlights: {e}")
    
    def close_window(self):
        """Close the 2D view window"""
        try:
            print("DEBUG: Closing 2D view window")
            
            # Clear highlights first
            self.clear_highlights()
            
            # Close matplotlib figure
            if self.fig:
                plt.close(self.fig)
                self.fig = None
            
            # Destroy window
            if self.window:
                self.window.destroy()
                self.window = None
            
            # Clean up references
            self.canvas = None
            self.ax = None
            self.current_view = None
            
            print("DEBUG: 2D view window closed successfully")
            
        except Exception as e:
            print(f"Error closing 2D window: {e}")
    
    def _toggle_always_on_top(self):
        """Toggle the always on top behavior"""
        try:
            if self.always_on_top_var.get():
                self.window.attributes('-topmost', True)
                print("DEBUG: 2D window set to always on top")
            else:
                self.window.attributes('-topmost', False)
                print("DEBUG: 2D window normal stacking")
        except Exception as e:
            print(f"Error toggling always on top: {e}")
    
    def update_coordinate_system(self, use_rgb):
        """Update the coordinate system labels when RGB/L*a*b* changes"""
        self.use_rgb = use_rgb
        if self.current_view:
            self.switch_view(self.current_view)  # Refresh the current view
    
    def is_open(self):
        """Check if the 2D window is currently open"""
        return self.window is not None and self.window.winfo_exists()
