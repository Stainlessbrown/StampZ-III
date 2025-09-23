import pandas as pd
import tkinter as tk
from tkinter import ttk
from matplotlib.collections import PathCollection
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.pyplot as plt
import numpy as np

class HighlightManager:
    def __init__(self, master, frame, ax, canvas, data_df, use_rgb=False, rotation_controls=None):
        self.master = master
        self.frame = frame
        self.ax = ax
        self.canvas = canvas
        self.data_df = data_df
        self.use_rgb = use_rgb
        self.rotation_controls = rotation_controls  # Reference to rotation controls for 2D views
        
        # Initialize highlight lists for multiple highlights
        self.highlight_scatters = []
        self.highlight_texts = []
        self.highlight_lines = []
        
        # Data tracking variables for debugging
        self.data_id_to_index = {}       # Map DataID to current DataFrame index
        self.scatter_point_order = []    # List of DataIDs in order they appear in scatter plot
        self.data_id_to_scatter_idx = {} # Map DataID to scatter plot point index
        self.original_data_order = []    # Original DataID order before any sorting
        self.pre_sort_indices = {}       # Map DataID to pre-sort DataFrame index
        self.last_highlighted_index = None
        self.last_highlighted_data_id = None
        self.dataframe_version = 0       # Track dataframe updates
        self.sorting_applied = False     # Track if sorting has been applied
        self.current_scatter = None      # Reference to current scatter plot object
        
        # Click-to-highlight variables
        self.selected_point_index = None
        self.selected_data_id = None
        self.click_tolerance = 0.1       # Distance tolerance for 3D click detection
        
        # Track mouse operations to distinguish clicks from drags
        self.mouse_press_pos = None
        self.drag_threshold = 5  # pixels - minimum movement to consider it a drag
        
        # Point selection mode toggle
        self.point_selection_mode = tk.BooleanVar(value=False)
        self.original_toolbar_mode = None
        
        # 2D view selection mode
        self.view_2d_mode = tk.BooleanVar(value=False)
        self.current_2d_view = None  # 'xy', 'xz', or 'yz'
        self.original_view_state = None  # Store original 3D view for restoration
        
        # Separate 2D window system
        self.view_2d_window = None
        
        # Store the original row indices to handle blank rows
        self.update_row_mapping()
        
        # Create controls
        self.create_controls()
        
        # Set up click event handlers
        self._setup_click_detection()
        
    def create_controls(self):
        """Create highlight controls with 2D view point selection"""
        try:
            # Main frame with border
            controls_frame = tk.Frame(self.frame, relief='sunken', bd=2)
            controls_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
            
            # Configure grid to expand
            controls_frame.grid_columnconfigure(0, weight=1)
            
            # Title label with correct font
            title_label = tk.Label(
                controls_frame,
                text="Point Identification",
                font=('Arial', 9, 'bold')
            )
            title_label.grid(row=0, column=0, pady=(5,2))

            # Point Selection Mode toggle
            self.mode_toggle = ttk.Checkbutton(
                controls_frame,
                text="🎯 Enable Point Selection",
                variable=self.point_selection_mode,
                command=self._toggle_selection_mode
            )
            self.mode_toggle.grid(row=1, column=0, sticky='w', padx=5, pady=5)
            
            # 2D View Mode toggle (initially disabled)
            self.view_2d_toggle = ttk.Checkbutton(
                controls_frame,
                text="🕸️ Open 2D Window for Selection",
                variable=self.view_2d_mode,
                command=self._toggle_2d_window_mode,
                state='disabled'
            )
            self.view_2d_toggle.grid(row=2, column=0, sticky='w', padx=5, pady=2)
            
            # Info about 2D window (initially hidden)
            self.window_info_label = ttk.Label(
                controls_frame,
                text="🕸️ Separate 2D window will open for precise point selection",
                font=('Arial', 9),
                foreground='darkblue'
            )
            self.window_info_label.grid(row=3, column=0, sticky='w', padx=5, pady=2)
            self.window_info_label.grid_remove()  # Initially hidden
            
            # Info display
            self.info_label = tk.Label(
                controls_frame,
                text="Enable Point Selection, optionally open 2D window for precise clicking",
                font=('Arial', 9),
                foreground='gray',
                justify='left',
                wraplength=340
            )
            self.info_label.grid(row=4, column=0, sticky='ew', padx=5, pady=5)
            
        except Exception as e:
            print(f"Error creating controls: {str(e)}")
            import traceback
            traceback.print_exc()
    def _setup_click_detection(self):
        """Set up comprehensive click detection for both 3D and 2D modes"""
        try:
            if self.canvas:
                print("DEBUG: Setting up click detection...")
                
                # Connect pick events (for matplotlib's native picker system)
                self.pick_handler_id = self.canvas.mpl_connect('pick_event', self._on_pick)
                
                # Connect button press events (for direct click handling)
                self.click_handler_id = self.canvas.mpl_connect('button_press_event', self._on_button_press)
                
                # Enable picking on all existing artists
                self.enable_picking_on_all_artists()
                
                print("DEBUG: Click detection set up successfully")
                print(f"DEBUG: Pick handler ID: {self.pick_handler_id}")
                print(f"DEBUG: Click handler ID: {self.click_handler_id}")
            else:
                print("WARNING: No canvas available for click detection")
        except Exception as e:
            print(f"Error setting up click detection: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_button_press(self, event):
        """Handle direct button press events as fallback for 2D views"""
        if not self.point_selection_mode.get() or event.inaxes != self.ax:
            return
        
        # Only handle left mouse button
        if event.button != 1:
            return
            
        print(f"DEBUG: Button press detected at ({event.xdata}, {event.ydata})")
        print(f"DEBUG: Current 2D view: {self.current_2d_view}")
        print(f"DEBUG: In 2D mode: {self.view_2d_mode.get()}")
        
        try:
            # If we're in 2D view mode, try direct coordinate-based selection
            if self.view_2d_mode.get() and self.current_2d_view:
                closest_idx = self._find_closest_point_2d(event)
                if closest_idx is not None:
                    print(f"DEBUG: Found closest point via direct click: {closest_idx}")
                    self._select_point_by_index(closest_idx)
                else:
                    print("DEBUG: No point found within click tolerance")
            else:
                print("DEBUG: Not in 2D mode, relying on picker system")
                
        except Exception as e:
            print(f"Error in button press handler: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_closest_point_2d(self, event):
        """Find closest point in 2D view using matplotlib's 3D-to-2D projection"""
        if not self.current_2d_view or event.xdata is None or event.ydata is None:
            return None
        
        try:
            click_x, click_y = event.xdata, event.ydata
            print(f"DEBUG: Looking for point near ({click_x:.4f}, {click_y:.4f})")
            
            # Filter out NaN values from the data
            valid_mask = (self.data_df['Xnorm'].notna() & 
                         self.data_df['Ynorm'].notna() & 
                         self.data_df['Znorm'].notna())
            
            if not valid_mask.any():
                print("DEBUG: No valid coordinate data found")
                return None
            
            valid_data = self.data_df[valid_mask]
            print(f"DEBUG: Found {len(valid_data)} valid data points to check")
            
            # Get 3D coordinates for all valid points
            points_3d = np.column_stack([
                valid_data['Xnorm'].values,
                valid_data['Ynorm'].values, 
                valid_data['Znorm'].values
            ])
            
            print(f"DEBUG: First few 3D points: {points_3d[:3]}")
            
            # Project 3D points to 2D screen coordinates using matplotlib's transform
            try:
                # Use the axes' 3D projection to convert to 2D screen coords
                points_2d = self.ax.transData.transform(points_3d)
                
                # Get click position in screen coordinates
                click_screen = self.ax.transData.transform([(click_x, click_y, 0)])[0][:2]
                
                print(f"DEBUG: Click screen coords: {click_screen}")
                print(f"DEBUG: First few projected points: {points_2d[:3, :2]}")
                
                # Calculate distances in screen space (pixels)
                distances = np.sqrt(np.sum((points_2d[:, :2] - click_screen) ** 2, axis=1))
                
                # Find closest point within pixel tolerance
                pixel_tolerance = 20  # pixels
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                print(f"DEBUG: Closest point distance: {min_distance:.2f} pixels (tolerance: {pixel_tolerance})")
                
                if min_distance <= pixel_tolerance:
                    # Get the original DataFrame index
                    df_idx = valid_data.index[min_idx]
                    data_id = valid_data.iloc[min_idx].get('DataID', f'Point_{df_idx}')
                    print(f"DEBUG: Found point {data_id} at distance {min_distance:.2f} pixels")
                    return df_idx
                
            except Exception as proj_error:
                print(f"DEBUG: 3D projection failed: {proj_error}")
                
                # Fallback: Use data coordinates directly based on current view
                print("DEBUG: Falling back to direct coordinate comparison")
                
                if self.current_2d_view == 'xy':  # L*a* or R/G view (top view)
                    data_x = valid_data['Xnorm'].values  # L* or R
                    data_y = valid_data['Ynorm'].values  # a* or G
                elif self.current_2d_view == 'xz':  # L*b* or R/B view (front view)
                    data_x = valid_data['Xnorm'].values  # L* or R
                    data_y = valid_data['Znorm'].values  # b* or B
                elif self.current_2d_view == 'yz':  # a*b* or G/B view (side view)
                    data_x = valid_data['Ynorm'].values  # a* or G
                    data_y = valid_data['Znorm'].values  # b* or B
                else:
                    return None
                
                print(f"DEBUG: First few data coords: X={data_x[:3]}, Y={data_y[:3]}")
                
                # Calculate 2D distances in data space
                distances = np.sqrt((data_x - click_x) ** 2 + (data_y - click_y) ** 2)
                
                # Find closest point within tolerance
                tolerance = 0.05  # Larger tolerance for data coordinates
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                print(f"DEBUG: Fallback - Closest point distance: {min_distance:.4f} (tolerance: {tolerance})")
                
                if min_distance <= tolerance:
                    df_idx = valid_data.index[min_idx]
                    data_id = valid_data.iloc[min_idx].get('DataID', f'Point_{df_idx}')
                    print(f"DEBUG: Fallback - Found point {data_id} at distance {min_distance:.4f}")
                    return df_idx
            
            return None
            
        except Exception as e:
            print(f"Error finding closest 2D point: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_point_by_index(self, df_idx):
        """Select and highlight a point by its DataFrame index"""
        try:
            print(f"DEBUG: Selecting point at DataFrame index: {df_idx}")
            
            # Clear existing highlights
            self._clear_highlight(keep_info=False)
            
            # Highlight the point based on current mode
            if self.view_2d_mode.get() and self.current_2d_view and hasattr(self, 'ax_2d'):
                self._highlight_point_2d(df_idx)
            else:
                self._highlight_point_by_index(df_idx)
            
            # Display detailed point information
            point = self.data_df.loc[df_idx]
            self._display_point_info(point, df_idx)
            
        except Exception as e:
            print(f"Error selecting point: {e}")
            import traceback
            traceback.print_exc()
    
    def _highlight_point_2d(self, df_idx):
        """Highlight a point in the 2D view"""
        try:
            point = self.data_df.loc[df_idx]
            data_id = point.get('DataID', f'Point_{df_idx}')
            
            # Get coordinates based on current view
            if self.current_2d_view == 'xy':
                x, y = point['Xnorm'], point['Ynorm']
            elif self.current_2d_view == 'xz':
                x, y = point['Xnorm'], point['Znorm']
            elif self.current_2d_view == 'yz':
                x, y = point['Ynorm'], point['Znorm']
            else:
                return
            
            print(f"DEBUG: Highlighting 2D point {data_id} at ({x:.4f}, {y:.4f})")
            
            # Create highlight circle around the point
            highlight_circle = plt.Circle(
                (x, y), 0.02,  # radius in data coordinates
                fill=False, edgecolor='red', linewidth=3,
                zorder=1000
            )
            self.ax_2d.add_patch(highlight_circle)
            
            # Add DataID label with smart positioning to avoid edges
            xlim = self.ax_2d.get_xlim()
            ylim = self.ax_2d.get_ylim()
            
            # Smart label positioning to keep it within plot bounds
            margin = 0.05  # 5% margin from edges
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # Default offset
            text_offset_x, text_offset_y = 0.03, 0.03
            
            # Adjust if too close to right edge
            if x + text_offset_x > xlim[1] - margin * x_range:
                text_offset_x = -0.05  # Place to the left
            
            # Adjust if too close to top edge
            if y + text_offset_y > ylim[1] - margin * y_range:
                text_offset_y = -0.05  # Place below
            
            text_x, text_y = x + text_offset_x, y + text_offset_y
            
            highlight_text = self.ax_2d.text(
                text_x, text_y, f'{data_id}',
                color='red', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.9),
                zorder=1001,
                ha='center', va='center'  # Center alignment for better positioning
            )
            
            # Store highlight elements for later cleanup
            self.highlight_scatters.append(highlight_circle)
            self.highlight_texts.append(highlight_text)
            
            # Redraw the canvas
            self.canvas.draw_idle()
            
            print(f"DEBUG: Successfully highlighted 2D point {data_id}")
            
        except Exception as e:
            print(f"Error highlighting 2D point: {e}")
            import traceback
            traceback.print_exc()
    
    def _force_enable_picking(self):
        """Force enable picking on all artists - more aggressive approach"""
        try:
            if not self.ax:
                print("DEBUG: No axes available for picking setup")
                return False
                
            print("DEBUG: Force enabling picking on all artists...")
            artists_found = 0
            
            # Enable picking on all collections (scatter plots)
            for i, collection in enumerate(self.ax.collections):
                if hasattr(collection, 'set_picker'):
                    collection.set_picker(True)  # Enable picker
                    collection.set_pickradius(10)  # Set larger pick radius
                    artists_found += 1
                    print(f"DEBUG: Enabled picking on collection {i} with radius 10")
            
            # Enable picking on all lines (plot markers)
            for i, line in enumerate(self.ax.lines):
                if hasattr(line, 'set_picker'):
                    line.set_picker(True)
                    line.set_pickradius(10)
                    artists_found += 1
                    print(f"DEBUG: Enabled picking on line {i} with radius 10")
            
            # Force canvas to update
            self.canvas.draw_idle()
            
            print(f"DEBUG: Force enabled picking on {artists_found} artists")
            return artists_found > 0
            
        except Exception as e:
            print(f"Error force enabling picking: {e}")
            return False
    
    
    def _toggle_selection_mode(self):
        """Toggle point selection mode"""
        try:
            if self.point_selection_mode.get():
                print("DEBUG: Point Selection Mode ENABLED")
                self.view_2d_toggle.config(state='normal')
                self.info_label.config(
                    text="🎯 Point selection active! Use 3D or enable 2D views for easier clicking.",
                    foreground='darkgreen'
                )
            else:
                print("DEBUG: Point Selection Mode DISABLED")
                # Disable and hide 2D window options
                self.view_2d_mode.set(False)
                self._toggle_2d_window_mode()
                self.view_2d_toggle.config(state='disabled')
                self.info_label.config(
                    text="Enable Point Selection, optionally use 2D views for easier clicking",
                    foreground='gray'
                )
                # Clear any current highlight
                self._clear_highlight(keep_info=False)
                
        except Exception as e:
            print(f"Error toggling selection mode: {e}")
    
    def _toggle_2d_window_mode(self):
        """Toggle 2D window mode for point selection"""
        try:
            if self.view_2d_mode.get():
                print("DEBUG: 2D Window Mode ENABLED")
                
                # Import and create 2D window
                from .view_2d_window import View2DWindow
                
                if not self.view_2d_window or not hasattr(self.view_2d_window, 'window') or not self.view_2d_window.is_open():
                    self.view_2d_window = View2DWindow(
                        parent_highlight_manager=self,
                        data_df=self.data_df,
                        use_rgb=self.use_rgb
                    )
                    
                    # Create the orphan window with default L*b* view (most commonly used)
                    self.view_2d_window.create_window('xz')
                    
                    print("DEBUG: Created new orphan 2D window")
                else:
                    # Window already exists, bring it to front
                    self.view_2d_window.window.lift()
                    self.view_2d_window.window.focus_set()
                    print("DEBUG: Brought existing 2D window to front")
                
                # Show info
                self.window_info_label.grid()
                
                self.info_label.config(
                    text="🕸️ Independent 2D window opened - click points there for precise selection",
                    foreground='darkblue'
                )
            else:
                print("DEBUG: 2D Window Mode DISABLED")
                
                # Close the 2D window if it exists
                if self.view_2d_window and hasattr(self.view_2d_window, 'window'):
                    self.view_2d_window.close_window()
                    self.view_2d_window = None
                    print("DEBUG: Closed 2D window")
                
                # Hide info
                self.window_info_label.grid_remove()
                
                if self.point_selection_mode.get():
                    self.info_label.config(
                        text="🎯 Point selection active in 3D mode",
                        foreground='darkgreen'
                    )
                else:
                    self.info_label.config(
                        text="Enable Point Selection, optionally open 2D window for precise clicking",
                        foreground='gray'
                    )
                    
        except Exception as e:
            print(f"Error toggling 2D window mode: {e}")
            import traceback
            traceback.print_exc()
    
    def _set_2d_view(self, view_type):
        """Create a true 2D subplot for precise point selection"""
        try:
            if not self.view_2d_mode.get() or not self.point_selection_mode.get():
                return
            
            print(f"DEBUG: Creating true 2D view: {view_type}")
            
            # Store original 3D plot state
            if not hasattr(self, 'original_3d_state'):
                self.original_3d_state = {
                    'figure': self.canvas.figure,
                    'ax': self.ax
                }
                print(f"DEBUG: Stored original 3D plot state")
            
            # Clear the current figure
            self.canvas.figure.clear()
            
            # Create a true 2D subplot
            self.ax_2d = self.canvas.figure.add_subplot(111)
            
            # Filter valid data points
            valid_mask = (self.data_df['Xnorm'].notna() & 
                         self.data_df['Ynorm'].notna() & 
                         self.data_df['Znorm'].notna())
            valid_data = self.data_df[valid_mask]
            
            print(f"DEBUG: Plotting {len(valid_data)} points in 2D view")
            
            # Get coordinates based on view type
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
            
            # Create the 2D scatter plot with individual points
            colors = valid_data['Color'].fillna('blue')
            markers = valid_data['Marker'].fillna('o')
            
            # Plot each point individually to maintain picker compatibility
            for i, (idx, row) in enumerate(valid_data.iterrows()):
                marker = row['Marker'] if pd.notna(row['Marker']) else 'o'
                color = row['Color'] if pd.notna(row['Color']) else 'blue'
                
                scatter = self.ax_2d.scatter(
                    x_data[i], y_data[i],
                    c=color, marker=marker, s=50,
                    picker=True, pickradius=10,
                    label=row.get('DataID', f'Point_{idx}')
                )
                # Store the DataFrame index for later retrieval
                scatter._df_index = idx
            
            # Set labels and title
            self.ax_2d.set_xlabel(x_label, fontsize=12)
            self.ax_2d.set_ylabel(y_label, fontsize=12)
            
            view_names = {
                'xy': 'L*a*' if not self.use_rgb else 'R/G',
                'xz': 'L*b*' if not self.use_rgb else 'R/B',
                'yz': 'a*b*' if not self.use_rgb else 'G/B'
            }
            
            self.ax_2d.set_title(f"{view_names[view_type]} View - Click points to select", fontsize=14, pad=20)
            
            # Set equal aspect ratio and grid
            self.ax_2d.set_aspect('equal', adjustable='box')
            self.ax_2d.grid(True, alpha=0.3)
            
            # Update current view and axis references
            self.current_2d_view = view_type
            self.ax = self.ax_2d  # Update axis reference for click handling
            
            # Redraw the canvas
            self.canvas.draw()
            
            # Update info display
            self.info_label.config(
                text=f"📐 True {view_names[view_type]} 2D view - click points for precise selection",
                foreground='darkblue'
            )
            
            print(f"DEBUG: Successfully created true 2D {view_type} view")
            
        except Exception as e:
            print(f"Error setting 2D view {view_type}: {e}")
            self.info_label.config(
                text=f"❌ Error setting {view_type} view - check console",
                foreground='red'
            )
    
    def _return_to_3d(self):
        """Restore the original 3D plot completely"""
        try:
            if hasattr(self, 'original_3d_state'):
                print(f"DEBUG: Restoring original 3D plot")
                
                # Clear the current 2D plot
                self.canvas.figure.clear()
                
                # We need to recreate the 3D plot by triggering a refresh from the parent
                # This is the cleanest way to restore the full 3D visualization
                
                # Try to find the Plot_3D instance through various references
                plot_3d_instance = None
                
                # Check if master has refresh_plot
                if hasattr(self, 'master') and hasattr(self.master, 'refresh_plot'):
                    plot_3d_instance = self.master
                # Check if we can find it through the canvas
                elif hasattr(self.canvas, 'master') and hasattr(self.canvas.master, 'master'):
                    parent = self.canvas.master.master
                    if hasattr(parent, 'refresh_plot'):
                        plot_3d_instance = parent
                    
                if plot_3d_instance:
                    print("DEBUG: Triggering full 3D plot refresh")
                    plot_3d_instance.refresh_plot()
                else:
                    # Fallback: create a basic 3D plot structure
                    print("DEBUG: Creating fallback 3D axes")
                    self.ax = self.canvas.figure.add_subplot(111, projection='3d')
                    self.canvas.draw()
                
                # Clean up 2D state
                if hasattr(self, 'ax_2d'):
                    delattr(self, 'ax_2d')
                if hasattr(self, 'original_3d_state'):
                    delattr(self, 'original_3d_state')
                
                print("DEBUG: Successfully restored 3D view")
            
            # Reset view state
            self.current_2d_view = None
            
            # Update info display
            if self.point_selection_mode.get():
                self.info_label.config(
                    text="🎯 Returned to 3D view - point selection still active",
                    foreground='darkgreen'
                )
            else:
                self.info_label.config(
                    text="Returned to 3D view",
                    foreground='gray'
                )
            
        except Exception as e:
            print(f"Error returning to 3D view: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.config(
                text="❌ Error returning to 3D - check console",
                foreground='red'
            )
    
    def _on_pick(self, event):
        """Handle pick events from matplotlib - works with both 3D and true 2D views"""
        if not self.point_selection_mode.get():
            return
            
        try:
            # Determine if we're in true 2D view mode
            in_2d_mode = self.view_2d_mode.get() and self.current_2d_view is not None
            
            print(f"DEBUG: Pick event detected! 2D mode: {in_2d_mode}")
            print(f"DEBUG: Artist type: {type(event.artist)}")
            print(f"DEBUG: Indices: {event.ind}")
            
            df_idx = None
            
            if in_2d_mode and hasattr(self, 'ax_2d'):
                # In true 2D mode, the artist should have the DataFrame index stored
                print(f"DEBUG: Using true 2D view picking in {self.current_2d_view} view")
                
                if hasattr(event.artist, '_df_index'):
                    df_idx = event.artist._df_index
                    print(f"DEBUG: Found stored DataFrame index: {df_idx}")
                else:
                    print("DEBUG: No stored index on artist, falling back to position mapping")
                    # Fallback: try to use the picked index to map to valid data
                    indices = event.ind
                    if len(indices) > 0:
                        picked_idx = indices[0]
                        # Get valid data (same filter as used in _set_2d_view)
                        valid_mask = (self.data_df['Xnorm'].notna() & 
                                     self.data_df['Ynorm'].notna() & 
                                     self.data_df['Znorm'].notna())
                        valid_data = self.data_df[valid_mask]
                        
                        if 0 <= picked_idx < len(valid_data):
                            df_idx = valid_data.index[picked_idx]
                            print(f"DEBUG: Mapped picked index {picked_idx} to DataFrame index: {df_idx}")
                    
            else:
                # In 3D mode, use the original mapping logic
                print(f"DEBUG: Using 3D view picking")
                indices = event.ind
                if len(indices) > 0:
                    picked_idx = indices[0]
                    print(f"DEBUG: Picked index: {picked_idx}")
                    
                    if hasattr(self, 'data_df') and 0 <= picked_idx < len(self.data_df):
                        df_idx = self.data_df.index[picked_idx]
                        print(f"DEBUG: Mapped to DataFrame index: {df_idx}")
            
            # If we found a valid DataFrame index, select the point
            if df_idx is not None:
                self._select_point_by_index(df_idx)
                
                # Get point data for display
                point = self.data_df.loc[df_idx]
                data_id = point.get('DataID', f'Point_{df_idx}')
                
                # Enhanced info display based on mode
                if in_2d_mode:
                    view_name = {
                        'xy': 'L*a*' if not self.use_rgb else 'R/G',
                        'xz': 'L*b*' if not self.use_rgb else 'R/B',
                        'yz': 'a*b*' if not self.use_rgb else 'G/B'
                    }.get(self.current_2d_view, '2D')
                    
                    self.info_label.config(
                        text=f"✅ Selected in {view_name} 2D view: {data_id}",
                        foreground='darkblue'
                    )
                else:
                    self.info_label.config(
                        text=f"✅ Selected in 3D: {data_id}",
                        foreground='darkgreen'
                    )
                
            else:
                print(f"DEBUG: Could not identify clicked point")
                error_msg = "⚠️ Point picked but could not identify"
                if in_2d_mode:
                    error_msg += f" in {self.current_2d_view} 2D view"
                error_msg += " - try another point"
                
                self.info_label.config(
                    text=error_msg,
                    foreground='orange'
                )
                    
        except Exception as e:
            print(f"Error in pick handler: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = "❌ Error picking point"
            if self.view_2d_mode.get() and self.current_2d_view:
                error_msg += f" in {self.current_2d_view} view"
            error_msg += " - see console"
            
            self.info_label.config(
                text=error_msg,
                foreground='red'
            )
    
    def _display_point_info(self, point, df_idx):
        """Display detailed information about the selected point"""
        try:
            data_id = point.get('DataID', f'Point_{df_idx}')
            x, y, z = point['Xnorm'], point['Ynorm'], point['Znorm']
            
            # Build detailed info text
            info_parts = [f"Point: {data_id}"]
            
            # Add coordinate information based on current view
            if self.current_2d_view == 'xy':  # L*a* or R/G view
                coord_labels = ('L*', 'a*') if not self.use_rgb else ('R', 'G')
                info_parts.append(f"{coord_labels[0]}: {x:.4f}, {coord_labels[1]}: {y:.4f}")
            elif self.current_2d_view == 'xz':  # L*b* or R/B view
                coord_labels = ('L*', 'b*') if not self.use_rgb else ('R', 'B')
                info_parts.append(f"{coord_labels[0]}: {x:.4f}, {coord_labels[1]}: {z:.4f}")
            elif self.current_2d_view == 'yz':  # a*b* or G/B view
                coord_labels = ('a*', 'b*') if not self.use_rgb else ('G', 'B')
                info_parts.append(f"{coord_labels[0]}: {y:.4f}, {coord_labels[1]}: {z:.4f}")
            else:
                # 3D mode - show all coordinates
                if self.use_rgb:
                    info_parts.append(f"R: {x:.4f}, G: {y:.4f}, B: {z:.4f}")
                else:
                    info_parts.append(f"L*: {x:.4f}, a*: {y:.4f}, b*: {z:.4f}")
            
            # Add cluster information if available
            if 'Cluster' in point and pd.notna(point['Cluster']):
                info_parts.append(f"Cluster: {point['Cluster']}")
            
            # Add ΔE information if available
            if '∆E' in point and pd.notna(point['∆E']):
                info_parts.append(f"ΔE: {point['∆E']:.2f}")
            
            # Join with newlines for display (but keep it concise for the label)
            info_text = " | ".join(info_parts)
            
            print(f"DEBUG: Point info - {info_text}")
            
        except Exception as e:
            print(f"Error displaying point info: {e}")
    
    def _on_hover(self, event):
        """Handle mouse hover - clear highlight when moving to empty space"""
        if not self.point_selection_mode.get():
            return
            
        # Simple logic: if mouse moves and we have a highlight, clear it after a delay
        # This gives natural "hover to clear" behavior
        if self.selected_point_index is not None:
            # You could add a small delay here if desired
            # For now, let's keep highlights until explicitly cleared or new point selected
            pass
    
    def enable_picking_on_all_artists(self):
        """Enable picking on all scatter plots and line plots in the current axes"""
        try:
            if not self.ax or not hasattr(self, 'data_df'):
                print("DEBUG: No axes or data available for picking setup")
                return False
                
            # Find all scatter and line artists
            artists_found = 0
            
            # Check all collections (scatter plots)
            for collection in self.ax.collections:
                if hasattr(collection, 'set_picker'):
                    collection.set_picker(5)  # 5 pixel tolerance
                    artists_found += 1
                    print(f"DEBUG: Enabled picking on collection (scatter plot)")
            
            # Check all lines (plot markers)
            for line in self.ax.lines:
                if hasattr(line, 'set_picker'):
                    line.set_picker(5)  # 5 pixel tolerance
                    artists_found += 1
                    print(f"DEBUG: Enabled picking on line (plot markers)")
            
            print(f"DEBUG: Enabled picking on {artists_found} artists")
            return artists_found > 0
            
        except Exception as e:
            print(f"Error enabling picking on artists: {e}")
            return False
            
    # Complex 3D detection methods removed - now using matplotlib's built-in picker system
    
    def _highlight_point_by_index(self, idx):
        """Highlight a specific point by its DataFrame index"""
        try:
            # Store selection info
            self.selected_point_index = idx
            
            # Get point data
            point = self.data_df.iloc[idx]
            x, y, z = point['Xnorm'], point['Ynorm'], point['Znorm']
            data_id = point.get('DataID', f'Point_{idx}')
            self.selected_data_id = data_id
            
            print(f"DEBUG: Highlighting point at index {idx}, DataID: {data_id}")
            print(f"DEBUG: Point coordinates: ({x:.4f}, {y:.4f}, {z:.4f})")
            
            # Create highlight scatter
            highlight_scatter = self.ax.scatter(
                [x], [y], [z],
                facecolors='none',
                edgecolors='red',  # Changed to red for better visibility
                s=120,  # Slightly larger than original
                marker='o',
                linewidth=3,  # Thicker border
                zorder=1000
            )
            self.highlight_scatters.append(highlight_scatter)
            
            # Add DataID label with improved positioning
            text_offset = 0.02  # Smaller offset for better positioning
            text_x = x + text_offset
            text_y = y + text_offset
            text_z = z + text_offset
            
            highlight_text = self.ax.text(
                text_x, text_y, text_z,
                f'{data_id}',
                color='red',
                fontsize=10,
                fontweight='bold',
                zorder=1001
            )
            self.highlight_texts.append(highlight_text)
            
            # Add connecting line
            from mpl_toolkits.mplot3d.art3d import Line3D
            highlight_line = Line3D(
                [x, text_x],
                [y, text_y],
                [z, text_z],
                linestyle=':',
                color='red',
                linewidth=2,
                zorder=999
            )
            self.ax.add_line(highlight_line)
            self.highlight_lines.append(highlight_line)
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error highlighting point by index: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_info_display(self, idx):
        """Update the info display with selected point information"""
        try:
            if not hasattr(self, 'info_label'):
                return
                
            point = self.data_df.iloc[idx]
            data_id = point.get('DataID', f'Point_{idx}')
            x, y, z = point['Xnorm'], point['Ynorm'], point['Znorm']
            
            # Create info text with point details
            info_lines = []
            info_lines.append(f"Selected: {data_id}")
            info_lines.append(f"Coordinates: ({x:.4f}, {y:.4f}, {z:.4f})")
            
            # Add cluster info if available
            if 'Cluster' in point and pd.notna(point['Cluster']):
                info_lines.append(f"Cluster: {point['Cluster']}")
            
            # Add ΔE info if available
            if '∆E' in point and pd.notna(point['∆E']):
                info_lines.append(f"ΔE: {point['∆E']:.2f}")
            
            # Update the info label
            info_text = "\n".join(info_lines)
            self.info_label.config(text=info_text, foreground='darkblue')
            
            print(f"DEBUG: Updated info display for {data_id}")
            
        except Exception as e:
            print(f"Error updating info display: {e}")
            
    def update_row_mapping(self):
        """Create a mapping between spreadsheet row numbers and DataFrame indices
        that accounts for blank rows in the original data"""
        try:
            # Get the dataframe length for validation
            df_length = len(self.data_df)
            
            # Create both mappings:
            # 1. From spreadsheet row to DataFrame index
            # 2. From DataFrame index to spreadsheet row
            self.row_mapping = {}
            self.index_to_row = {}
            
            # CRITICAL FIX: Check for _original_sheet_row column from internal worksheet
            # This provides exact mapping between display rows and DataFrame indices
            if '_original_sheet_row' in self.data_df.columns:
                print("DEBUG: Using '_original_sheet_row' column for precise row mapping (REALTIME WORKSHEET)")
                for idx, row in self.data_df.iterrows():
                    orig_sheet_row = int(row['_original_sheet_row'])
                    # Display row = sheet row + 1 (sheet rows are 0-based, display rows are 1-based)
                    display_row = orig_sheet_row + 1
                    self.row_mapping[display_row] = idx
                    self.index_to_row[idx] = display_row
                    
                    # DEBUG: Show the mapping for first few rows
                    if idx < 5:
                        print(f"DEBUG: Display row {display_row} → DataFrame index {idx} (sheet row {orig_sheet_row})")
                        
            elif 'original_row' in self.data_df.columns:
                print("DEBUG: Using 'original_row' column for precise row mapping (FILE-BASED)")
                for idx, row in self.data_df.iterrows():
                    orig_row = int(row['original_row'])
                    # Adjust for header row (+1) and zero-indexing (+1) = +2
                    spreadsheet_row = orig_row + 2
                    self.row_mapping[spreadsheet_row] = idx
                    self.index_to_row[idx] = spreadsheet_row
            else:
                # Fall back to default sequential mapping (LEGACY)
                print("DEBUG: Using default sequential row mapping (LEGACY - may be incorrect)")
                # Row 2 in spreadsheet = index 0 in DataFrame (accounting for header row)
                self.row_mapping = {i+2: i for i in range(df_length)}
                self.index_to_row = {i: i+2 for i in range(df_length)}
            
            # Log mapping information for debugging
            print(f"DEBUG: Created row mapping for {df_length} data points")
            if len(self.row_mapping) <= 20:  # Only print full mapping for small datasets
                print(f"DEBUG: Row mapping: {self.row_mapping}")
            else:
                print(f"DEBUG: First 5 row mappings: {dict(list(self.row_mapping.items())[:5])}")
                print(f"DEBUG: Last 5 row mappings: {dict(list(self.row_mapping.items())[-5:])}")
                
            # Build DataID to index mapping for verification
            self.data_id_to_index = {}
            for idx, row in self.data_df.iterrows():
                if 'DataID' in row:
                    data_id = row['DataID']
                    self.data_id_to_index[data_id] = idx
            
            print(f"DEBUG: Created DataID mapping for {len(self.data_id_to_index)} unique DataIDs")
            
            # Initialize scatter point tracking if not already set
            # Note: This could be incomplete if update_row_mapping is called before 
            # scatter plot is created, but update_references will fix this
            if not self.scatter_point_order and 'DataID' in self.data_df.columns:
                # Initially, scatter points will be in the same order as DataFrame
                self.scatter_point_order = self.data_df['DataID'].tolist()
                self.data_id_to_scatter_idx = {data_id: idx for idx, data_id in enumerate(self.scatter_point_order)}
                
                # Also store the original data order if not already set
                if not self.original_data_order:
                    self.original_data_order = self.scatter_point_order.copy()
                    self.pre_sort_indices = self.data_id_to_index.copy()
                    print(f"DEBUG: Stored original data order with {len(self.original_data_order)} points")
                
                print(f"DEBUG: Initialized scatter point order tracking with {len(self.scatter_point_order)} points")
        except Exception as e:
            print(f"Error creating row mapping: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def find_df_index(self, user_row):
        """
        Find the DataFrame index corresponding to a spreadsheet row number.
        Handles potential gaps in data due to blank rows and the +1 offset issue.
        """
        try:
            print(f"DEBUG: Finding DataFrame index for spreadsheet row {user_row}")
            
            # Check if this exact row exists in our mapping
            if user_row in self.row_mapping:
                df_index = self.row_mapping[user_row]
                print(f"DEBUG: Direct mapping found: spreadsheet row {user_row} → DataFrame index {df_index}")
                return df_index
            
            # If we don't have an exact match, we need to find the appropriate index
            valid_rows = sorted(self.row_mapping.keys())
            
            if not valid_rows:
                # No valid rows in mapping (should not happen)
                print("WARNING: No valid rows in mapping, using basic calculation")
                return max(0, user_row - 2)
                
            # Handle edge cases: before first valid row or after last valid row
            if user_row < min(valid_rows):
                print(f"WARNING: Row {user_row} is before the first valid row {min(valid_rows)}")
                return self.row_mapping[min(valid_rows)]  # Return first mapped index
                
            elif user_row > max(valid_rows):
                print(f"WARNING: Row {user_row} is after the last valid row {max(valid_rows)}")
                return self.row_mapping[max(valid_rows)]  # Return last mapped index
                
            # Find the closest valid row before the requested row
            prev_row = None
            next_row = None
            
            for row in valid_rows:
                if row < user_row:
                    prev_row = row
                elif row > user_row:
                    next_row = row
                    break
                    
            # Log what we found
            if prev_row is not None:
                print(f"DEBUG: Previous valid row: {prev_row} → index {self.row_mapping[prev_row]}")
            if next_row is not None:
                print(f"DEBUG: Next valid row: {next_row} → index {self.row_mapping[next_row]}")
                
            # Calculate offset from blank rows
            if prev_row is not None and next_row is not None:
                # We have valid rows both before and after
                # Check if there's a gap that suggests blank rows
                row_gap = next_row - prev_row
                index_gap = self.row_mapping[next_row] - self.row_mapping[prev_row]
                
                if row_gap > index_gap + 1:
                    # There are blank rows in between
                    print(f"DEBUG: Detected blank rows: row gap = {row_gap}, index gap = {index_gap}")
                    
                # Calculate how far we are from the previous valid row
                offset = user_row - prev_row
                
                # If we're asking for a row between prev and next, and there's a gap,
                # calculate the appropriate index
                if offset <= (row_gap - index_gap):
                    # We're in the blank row region, snap to previous index
                    df_index = self.row_mapping[prev_row]
                    print(f"DEBUG: In blank row region, snapping to previous index {df_index}")
                else:
                    # Past the blank rows, calculate adjusted index
                    df_index = self.row_mapping[prev_row] + (offset - (row_gap - index_gap))
                    print(f"DEBUG: Past blank rows, calculated adjusted index {df_index}")
                    
                return df_index
                
            elif next_row is not None:
                # We only have rows after, use the next valid row's index
                print(f"DEBUG: Only have rows after, using next valid row's index")
                return self.row_mapping[next_row]
                
            elif prev_row is not None:
                # We only have rows before, calculate based on offset from last valid row
                offset = user_row - prev_row
                # Basic check to ensure we don't go out of bounds
                df_index = min(self.row_mapping[prev_row] + offset, len(self.data_df) - 1)
                print(f"DEBUG: Only have rows before, calculated index {df_index} based on offset {offset}")
                return df_index
                
            # Final fallback - use basic calculation
            df_index = max(0, min(user_row - 2, len(self.data_df) - 1))
            print(f"DEBUG: Using fallback calculation, spreadsheet row {user_row} → DataFrame index {df_index}")
            return df_index
            
        except Exception as e:
            print(f"Error finding DataFrame index: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to simple conversion with bounds checking
            return max(0, min(user_row - 2, len(self.data_df) - 1))

    # Old _highlight_data method removed - replaced with click-to-highlight functionality
            
    def _debug_row_mapping(self):
        """Debug function to test and verify row mapping"""
        try:
            print("\nDEBUG: Testing row mapping functionality")
            print(f"Current DataFrame version: {self.dataframe_version}")
            # Test a few sample rows
            test_rows = [2, 5, 10]
            
            for row in test_rows:
                try:
                    df_index = self.find_df_index(row)
                    print(f"Test: Spreadsheet row {row} → DataFrame index {df_index}")
                    
                    # Verify that we can get the row's data
                    if 0 <= df_index < len(self.data_df):
                        data_id = self.data_df.iloc[df_index]['DataID']
                        print(f"  Data at index {df_index}: DataID = {data_id}")
                    else:
                        print(f"  Index {df_index} is out of range")
                except Exception as e:
                    print(f"  Error testing row {row}: {str(e)}")
            
            # Add a verification test for DataID consistency
            print("\nDEBUG: Testing DataID consistency")
            
            # Sample a few DataIDs to verify 
            if not self.data_df.empty and 'DataID' in self.data_df.columns:
                sample_size = min(5, len(self.data_df))
                sample_indices = sorted([i for i in range(0, len(self.data_df), max(1, len(self.data_df) // sample_size))])[:sample_size]
                
                for idx in sample_indices:
                    data_id = self.data_df.iloc[idx]['DataID']
                    expected_idx = self.data_id_to_index.get(data_id)
                    scatter_idx = self.data_id_to_scatter_idx.get(data_id)
                    
                    if expected_idx != idx:
                        print(f"  CONSISTENCY ERROR: DataID {data_id} is at index {idx} but mapped to index {expected_idx}")
                    else:
                        print(f"  Verified: DataID {data_id} correctly mapped to index {idx}")
                    
                    if scatter_idx is not None:
                        print(f"  Scatter point index for DataID {data_id}: {scatter_idx}")
                    else:
                        print(f"  WARNING: No scatter point index for DataID {data_id}")
                        
            print("DEBUG: Row mapping test complete")
        except Exception as e:
            print(f"Error in debug row mapping: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def _highlight_point(self, idx):
        """Highlight the selected point"""
        try:
            print(f"DEBUG: Highlighting point at index {idx}")
            # Don't clear existing highlights to allow multiple highlights
            
            # Get normalized point coordinates and DataID from the DataFrame
            # This must be done first before any other operations
            point = self.data_df.iloc[idx]
            x, y, z = point['Xnorm'], point['Ynorm'], point['Znorm']
            data_id = point['DataID']
            print(f"DEBUG: Found point at index {idx} with coordinates ({x}, {y}, {z}) with DataID: {data_id}")
            
            # Verify the DataID is valid and update if needed
            if data_id not in self.data_id_to_index:
                print(f"WARNING: DataID {data_id} is not in the data_id_to_index mapping!")
                # Add it now to ensure it's tracked
                self.data_id_to_index[data_id] = idx
                print(f"Added DataID {data_id} to index mapping at position {idx}")
            
            # If this DataID isn't in scatter_point_order, add it now
            if data_id not in self.data_id_to_scatter_idx and self.scatter_point_order:
                print(f"WARNING: DataID {data_id} is not in the scatter_point_order mapping!")
                if not self.sorting_applied:
                    # Only add to scatter point order if sorting hasn't been applied
                    scatter_idx = len(self.scatter_point_order)
                    self.scatter_point_order.append(data_id)
                    self.data_id_to_scatter_idx[data_id] = scatter_idx
                    print(f"Added DataID {data_id} to scatter mapping at position {scatter_idx}")
            
            # If sorting has been applied, we need to handle the mismatch between 
            # DataFrame indices and scatter plot indices
            scatter_idx = None
            if self.sorting_applied and data_id in self.data_id_to_scatter_idx:
                scatter_idx = self.data_id_to_scatter_idx[data_id]
                print(f"DEBUG: Using scatter plot index {scatter_idx} for DataID {data_id} (DataFrame index {idx})")
                
                # Log comparison - this helps track if mapping is correct
                if scatter_idx < len(self.scatter_point_order):
                    expected_data_id = self.scatter_point_order[scatter_idx]
                    if expected_data_id != data_id:
                        print(f"ERROR: Scatter point mapping mismatch! DataID {data_id} maps to scatter_idx {scatter_idx}, but that position contains DataID {expected_data_id}")
                        # CRITICAL: Don't update the mapping! 
                        # If we have a mismatch, we need to find the correct scatter_idx
                        for test_idx, test_data_id in enumerate(self.scatter_point_order):
                            if test_data_id == data_id:
                                print(f"FIXING: Found correct scatter_idx {test_idx} for DataID {data_id}")
                                scatter_idx = test_idx
                                self.data_id_to_scatter_idx[data_id] = test_idx
                                break
                    else:
                        print(f"INFO: Verified scatter point mapping is correct for DataID {data_id}")
                
                # CRITICAL INSIGHT: When sorting is applied, the scatter plot points remain in their
                # original positions, but the DataFrame rows are reordered. The coordinates (x,y,z)
                # from the DataFrame are correct and should be used for highlighting.
                print("INFO: Using DataFrame coordinates for highlighting during sorting")
                
                # When sorting has been applied, we need to make sure we're highlighting the correct point
                # in the scatter plot, which may be different from the DataFrame index
                # The coordinates in the DataFrame are correct, but we should verify against original positions
                print(f"DEBUG: Original point coordinates from DataFrame: ({x}, {y}, {z})")
                
                # Check if we have access to the original (pre-sort) indices
                if data_id in self.pre_sort_indices:
                    original_idx = self.pre_sort_indices[data_id]
                    print(f"DEBUG: Original index for DataID {data_id} was {original_idx}")
            
            # If sorting has been applied, we always trust the DataID-based mapping
            # since the scatter plot points remain in their original order
            if self.sorting_applied:
                # No need to verify or correct index - the coordinates from the DataFrame
                # are correct for highlighting even when sorted
                print(f"SORTING: Using DataFrame coordinates for highlighting (no correction needed)")
            else:
                # For non-sorted data, verify DataID matches our index mapping
                expected_idx = self.data_id_to_index.get(data_id)
                if expected_idx is not None and expected_idx != idx:
                    print(f"WARNING: DataID mismatch detected! DataID {data_id} should be at index {expected_idx}, not {idx}")
                    print(f"STATE CHECK: DataFrame version: {self.dataframe_version}, Sorting applied: {self.sorting_applied}")
                    
                    # Check if we should correct the index
                    verified_idx = expected_idx
                    verified_data_id = self.data_df.iloc[verified_idx]['DataID']
                    
                    if verified_data_id == data_id:
                        print(f"CORRECTION: Using verified index {verified_idx} instead of {idx}")
                        idx = verified_idx
                        # Re-fetch point data with corrected index
                        point = self.data_df.iloc[idx]
                        x, y, z = point['Xnorm'], point['Ynorm'], point['Znorm']
                        data_id = point['DataID']
                        print(f"CORRECTION: Updated point at coordinates ({x}, {y}, {z}) with DataID: {data_id}")
                    else:
                        print(f"VALIDATION ERROR: Verification failed. DataID at index {verified_idx} is {verified_data_id}, not {data_id}")
                        # Dump a portion of the DataFrame for debugging
                        print("Current DataFrame state (first few rows):")
                        print(self.data_df.head().to_string())
                        if len(self.data_df) > 10:
                            print("Current DataFrame state (rows around the issue):")
                            start_idx = max(0, min(idx, expected_idx) - 2)
                            end_idx = min(len(self.data_df), max(idx, expected_idx) + 3)
                            print(self.data_df.iloc[start_idx:end_idx].to_string())
            
            # Store for tracking
            self.last_highlighted_index = idx
            self.last_highlighted_data_id = data_id
            
            # Create scatter plot for highlight
            # Define the coordinates for the highlight point
            highlight_coords = [x, y, z]
            
            # Log the exact coordinates we're using for the highlight
            print(f"DEBUG: Creating highlight scatter at exact coordinates: ({x}, {y}, {z})")
            
            # When sorting has been applied, the coordinates in the DataFrame are correct,
            # but the scatter plot points are still in their original order
            # We're highlighting using coordinates, which is the correct approach
            # The issue occurs when trying to match scatter plot points by index
            
            highlight_scatter = self.ax.scatter(
                [x], [y], [z],
                facecolors='none',
                edgecolors='black',
                s=100,  # Size parameter - only one 's' parameter allowed
                marker='o',
                linewidth=2,
                zorder=1000
            )
            self.highlight_scatters.append(highlight_scatter)
            
            # Store the fact that we highlighted this specific point
            print(f"DEBUG: Highlighted DataID {data_id} (DataFrame index: {idx}, Scatter index: {scatter_idx})")
            # Add DataID label
            # Calculate offset coordinates for the text
            text_x = x + 0.05
            text_y = y + 0.05
            text_z = z
            
            # Use text3D for proper 3D annotation
            print(f"DEBUG: Creating text3D at ({text_x}, {text_y}, {text_z})")
            highlight_text = self.ax.text(
                text_x, text_y, text_z,
                f'{data_id}',
                color='red',
                zorder=1000
            )
            self.highlight_texts.append(highlight_text)
            
            # Create a dotted line connecting the point and the text
            print(f"DEBUG: Creating dotted line from ({x}, {y}, {z}) to ({text_x}, {text_y}, {text_z})")
            highlight_line = Line3D(
                [x, text_x],
                [y, text_y],
                [z, text_z],
                linestyle=':',
                color='black',
                zorder=999
            )
            self.ax.add_line(highlight_line)
            self.highlight_lines.append(highlight_line)
            
            # Print selected point info
            print(f"Selected row index: {idx}")
            print("Data at this index:")
            print(self.data_df.iloc[idx])
            
            # Refresh the canvas
            self.canvas.draw()
            print("DEBUG: Canvas refreshed")
            
        except Exception as e:
            print(f"Error highlighting point: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def _clear_highlight(self, keep_info=True):
        """Clear all current highlights"""
        try:
            print("DEBUG: Clearing all highlights")
            
            # Remove all highlight scatters
            for scatter in self.highlight_scatters:
                try:
                    if scatter in self.ax.collections:
                        scatter.remove()
                    print("DEBUG: Removed highlight scatter")
                except Exception as e:
                    print(f"DEBUG: Could not remove scatter: {e}")
            self.highlight_scatters = []
                
            # Remove all highlight texts
            for text in self.highlight_texts:
                try:
                    if text in self.ax.texts:
                        text.remove()
                    print("DEBUG: Removed highlight text")
                except Exception as e:
                    print(f"DEBUG: Could not remove text: {e}")
            self.highlight_texts = []

            # Remove all highlight lines
            for line in self.highlight_lines:
                try:
                    if line in self.ax.lines:
                        line.remove()
                    print("DEBUG: Removed highlight line")
                except Exception as e:
                    print(f"DEBUG: Could not remove line: {e}")
            self.highlight_lines = []
                
            # Clear selection state
            self.selected_point_index = None
            self.selected_data_id = None
            
            # Reset info display unless keep_info is True
            if not keep_info and hasattr(self, 'info_label'):
                self.info_label.config(text="No point selected", foreground='gray')
                print("DEBUG: Reset info display")
            
            # Refresh the canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Warning: Non-critical error in highlight clearing: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def update_coordinate_labels(self, use_rgb):
        """Update 2D view button labels when coordinate system changes"""
        try:
            self.use_rgb = use_rgb
            
            # Update 2D view button labels
            if hasattr(self, 'view_buttons'):
                labels = {
                    'xy': 'L*a*' if not use_rgb else 'R/G',
                    'xz': 'L*b*' if not use_rgb else 'R/B',
                    'yz': 'a*b*' if not use_rgb else 'G/B'
                }
                
                for view, btn in self.view_buttons.items():
                    if view in labels:
                        btn.config(text=f"{labels[view]} View")
                
                print(f"DEBUG: Updated 2D view button labels for {'RGB' if use_rgb else 'L*a*b*'} mode")
            
        except Exception as e:
            print(f"Error updating coordinate labels: {e}")
    
    def update_references(self, ax, data_df, use_rgb=False):
        """Update references to axis, data, and use_rgb flag"""
        try:
            # Update coordinate labels if RGB mode changed
            if hasattr(self, 'use_rgb') and self.use_rgb != use_rgb:
                self.update_coordinate_labels(use_rgb)
            
            # Increment DataFrame version counter
            self.dataframe_version += 1
            print(f"DEBUG: Updating references (DataFrame version: {self.dataframe_version})")
            
            # Check if DataFrame order has changed by comparing DataID order
            # This is an indicator of sorting being applied
            if 'DataID' in self.data_df.columns and 'DataID' in data_df.columns:
                old_data_ids = self.data_df['DataID'].tolist() if not self.data_df.empty else []
                new_data_ids = data_df['DataID'].tolist() if not data_df.empty else []
                
                # Before updating anything, store the pre-sort state if this is the first data load
                if not self.original_data_order and len(old_data_ids) > 0:
                    self.original_data_order = old_data_ids.copy()
                    self.pre_sort_indices = {data_id: idx for idx, data_id in enumerate(old_data_ids)}
                    
                    # Initialize scatter point order to match original data order if not already set
                    if not self.scatter_point_order:
                        self.scatter_point_order = self.original_data_order.copy()
                        self.data_id_to_scatter_idx = {data_id: idx for idx, data_id in enumerate(self.scatter_point_order)}
                    
                    print("DEBUG: Stored original data order before first sort")
                
                # Check if we have the same DataIDs but in a different order (sorted)
                if (len(old_data_ids) == len(new_data_ids) and 
                    set(old_data_ids) == set(new_data_ids) and 
                    old_data_ids != new_data_ids):
                    self.sorting_applied = True
                    print("DEBUG: DataFrame sorting detected - maintaining scatter plot to DataFrame index mapping")
                    
                    # Log the sort change details
                    print("DEBUG: Sorting details:")
                    sample_size = min(5, len(old_data_ids))
                    for i in range(sample_size):
                        old_id = old_data_ids[i]
                        if old_id in new_data_ids:
                            new_idx = new_data_ids.index(old_id)
                            print(f"  DataID {old_id} moved from index {i} to {new_idx}")
                else:
                    # Reset sorting flag if data content has changed (not just order)
                    old_sorting = self.sorting_applied
                    self.sorting_applied = False
                    if old_sorting:
                        print("DEBUG: New data detected - resetting sorting flag")
            
            # First, verify if we had a previously highlighted point
            if self.last_highlighted_data_id is not None and len(self.data_df) > 0:
                try:
                    # Check if the DataID exists in the new DataFrame
                    matches = data_df[data_df['DataID'] == self.last_highlighted_data_id]
                    if not matches.empty:
                        new_idx = matches.index[0]
                        old_idx = self.last_highlighted_index
                        if new_idx != old_idx:
                            print(f"WARNING: After data update, DataID {self.last_highlighted_data_id} moved from index {old_idx} to {new_idx}")
                except Exception as e:
                    print(f"DEBUG: Error checking previous highlight: {str(e)}")
            
            # Before updating dataframe, preserve the original scatter point order if needed
            if 'DataID' in self.data_df.columns:
                if not self.scatter_point_order:
                    print("DEBUG: Preserving original scatter point order before reference update")
                    self.scatter_point_order = self.data_df['DataID'].tolist()
                    self.data_id_to_scatter_idx = {data_id: idx for idx, data_id in enumerate(self.scatter_point_order)}
                elif self.sorting_applied:
                    # If sorting was applied, we need to keep the scatter point order the same,
                    # since the actual scatter plot points don't change order, only the DataFrame indices
                    print("DEBUG: Maintaining existing scatter point order during sort")
                    
                    # CRITICAL: We should NEVER update scatter_point_order after its initial creation
                    # The scatter plot points maintain their original positions even when the DataFrame is sorted
                    
                    # CRITICAL: We should always use the original_data_order for scatter point mapping
                    # This ensures we maintain the original plot order after sorting
                    if self.original_data_order:
                        # Verify scatter_point_order matches original_data_order
                        if self.scatter_point_order != self.original_data_order:
                            print("DEBUG: Resetting scatter_point_order to match original_data_order")
                            self.scatter_point_order = self.original_data_order.copy()
                            self.data_id_to_scatter_idx = {data_id: idx for idx, data_id in enumerate(self.scatter_point_order)}
                    elif not self.scatter_point_order:
                        print("DEBUG: Setting scatter_point_order from current data")
                        self.scatter_point_order = self.data_df['DataID'].tolist()
                        self.data_id_to_scatter_idx = {data_id: idx for idx, data_id in enumerate(self.scatter_point_order)}
            # Store the current scatter plot object reference if available
            scatter_plots = [c for c in self.ax.collections if isinstance(c, PathCollection)]
            if scatter_plots:
                self.current_scatter = scatter_plots[0]
                print(f"DEBUG: Captured reference to current scatter plot")
            
            # Update references
            self.ax = ax
            self.data_df = data_df
            self.use_rgb = use_rgb
            
            # Check if the scatter plot has been recreated
            new_scatter_plots = [c for c in self.ax.collections if isinstance(c, PathCollection)]
            if new_scatter_plots and self.current_scatter != new_scatter_plots[0]:
                print("DEBUG: Detected new scatter plot object - this is crucial to understand sorting behavior")
                self.current_scatter = new_scatter_plots[0]
            
            # Verify DataID uniqueness
            if 'DataID' in self.data_df.columns:
                data_ids = self.data_df['DataID'].tolist()
                unique_ids = set(data_ids)
                if len(data_ids) != len(unique_ids):
                    print(f"WARNING: DataFrame contains {len(data_ids) - len(unique_ids)} duplicate DataIDs!")
                    # Find and report duplicates
                    from collections import Counter
                    duplicates = [item for item, count in Counter(data_ids).items() if count > 1]
                    print(f"Duplicate DataIDs: {duplicates[:10]}")
                    for dup_id in duplicates[:3]:  # Show details for first few duplicates
                        dup_rows = self.data_df[self.data_df['DataID'] == dup_id]
                        print(f"Duplicate rows for DataID {dup_id}:")
                        print(dup_rows.to_string())
            
            # Reset tracking variables
            self.last_highlighted_index = None
            self.last_highlighted_data_id = None
            
            # Update row mapping for the new data
            self.update_row_mapping()
            
            # Clear existing highlight
            self._clear_highlight()
            
            # Update data_id_to_scatter_idx if we're in sorted mode
            if self.sorting_applied and 'DataID' in self.data_df.columns:
                # After sorting, update the DataFrame index mapping while maintaining scatter point order
                new_data_id_to_index = {}
                for idx, row in self.data_df.iterrows():
                    data_id = row['DataID']
                    new_data_id_to_index[data_id] = idx
                
                # Store the updated mapping to use for highlighting
                self.data_id_to_index = new_data_id_to_index
                print(f"DEBUG: Updated DataID to DataFrame index mapping after sorting")
                
                # Critical step: After sorting we need to verify that our scatter point mapping still works
                # Compare DataFrame order with scatter plot order
                df_data_ids = self.data_df['DataID'].tolist()
                
                # CRITICAL INFORMATION FOR DEBUGGING:
                print("\nIMPORTANT SORTING DIAGNOSTIC:")
                print("When DataFrame is sorted, scatter plot points remain in their ORIGINAL order")
                print("This means the order of DataIDs in self.scatter_point_order should be preserved")
                print("We need to ensure our DataID-to-ScatterIdx mapping remains consistent with the original order")
                
                if set(df_data_ids) != set(self.scatter_point_order):
                    print("WARNING: DataFrame DataIDs don't match scatter plot DataIDs after sorting!")
                    print(f"  DataFrame has {len(df_data_ids)} DataIDs, scatter plot has {len(self.scatter_point_order)}")
                    print(f"  First few DataFrame DataIDs: {df_data_ids[:5]}")
                    print(f"  First few scatter plot DataIDs: {self.scatter_point_order[:5]}")
                else:
                    print("DEBUG: Verified all DataIDs present in both DataFrame and scatter plot mapping")
                    
                    # Key behavior check - did the DataIDs change order in self.scatter_point_order?
                    # They should NOT change order since the scatter plot points remain in original order
                    if self.original_data_order and self.scatter_point_order != self.original_data_order:
                        print("ERROR: Scatter point order changed from original! This should not happen.")
                        print(f"Original first 5: {self.original_data_order[:5]}")
                        print(f"Current first 5: {self.scatter_point_order[:5]}")
                        
                        # Reset to original order
                        print("Resetting scatter_point_order to match original order")
                        self.scatter_point_order = self.original_data_order.copy()
                        self.data_id_to_scatter_idx = {data_id: idx for idx, data_id in enumerate(self.scatter_point_order)}
                        print(f"RESET: Reinitialized scatter_point_order with {len(self.scatter_point_order)} points")
                        print(f"RESET: First 5 points: {self.scatter_point_order[:5]}")
                    
                # Create a mapping from DataFrame index to scatter plot index
                # This is crucial for highlight_point to work correctly
                df_idx_to_scatter_idx = {}
                for df_idx, data_id in enumerate(df_data_ids):
                    if data_id in self.data_id_to_scatter_idx:
                        scatter_idx = self.data_id_to_scatter_idx[data_id]
                        df_idx_to_scatter_idx[df_idx] = scatter_idx
                        
                        # Verify mapping consistency
                        if scatter_idx < len(self.scatter_point_order):
                            expected_data_id = self.scatter_point_order[scatter_idx]
                            if expected_data_id != data_id:
                                print(f"ERROR: Scatter mapping inconsistency for DF index {df_idx}!")
                                print(f"  Data ID {data_id} maps to scatter index {scatter_idx}")
                                print(f"  But scatter index {scatter_idx} should contain {expected_data_id}")
                                
                                # Fix the mapping by finding the correct index
                                correct_scatter_idx = None
                                for i, d_id in enumerate(self.scatter_point_order):
                                    if d_id == data_id:
                                        correct_scatter_idx = i
                                        break
                                        
                                if correct_scatter_idx is not None:
                                    print(f"  FIXING: Updating scatter index for {data_id} to {correct_scatter_idx}")
                                    self.data_id_to_scatter_idx[data_id] = correct_scatter_idx
                                    df_idx_to_scatter_idx[df_idx] = correct_scatter_idx
                
                print(f"DEBUG: Created index mapping table with {len(df_idx_to_scatter_idx)} entries")
                # Sample a few mappings to verify
                sample_size = min(3, len(df_idx_to_scatter_idx))
                sample_indices = list(df_idx_to_scatter_idx.keys())[:sample_size]
                for df_idx in sample_indices:
                    data_id = df_data_ids[df_idx]
                    scatter_idx = df_idx_to_scatter_idx[df_idx]
                    print(f"  DataFrame index {df_idx} (DataID {data_id}) maps to scatter plot index {scatter_idx}")
                
                # Validate that all DataIDs in scatter_point_order exist in the new DataFrame
                missing_data_ids = [data_id for data_id in self.scatter_point_order if data_id not in new_data_id_to_index]
                if missing_data_ids:
                    print(f"WARNING: {len(missing_data_ids)} DataIDs from scatter plot are missing in new DataFrame")
                    self.sorting_applied = False  # Reset sorting flag as mapping is no longer valid
            
            # Log information about the updated data
            print(f"DEBUG: Updated highlight manager with {len(data_df)} data points, Sorting applied: {self.sorting_applied}")
            
            # After all updates, double-check that our scatter_point_order is still valid
            if self.sorting_applied and len(self.scatter_point_order) > 0:
                if set(self.data_id_to_scatter_idx.keys()) != set(self.scatter_point_order):
                    print("ERROR: data_id_to_scatter_idx keys don't match scatter_point_order!")
                    missing_ids = set(self.scatter_point_order) - set(self.data_id_to_scatter_idx.keys())
                    extra_ids = set(self.data_id_to_scatter_idx.keys()) - set(self.scatter_point_order)
                    if missing_ids:
                        print(f"Missing IDs in mapping: {list(missing_ids)[:5]}")
                    if extra_ids:
                        print(f"Extra IDs in mapping: {list(extra_ids)[:5]}")
                    
                    # Fix the mapping
                    print("FIXING: Rebuilding data_id_to_scatter_idx from scatter_point_order")
                    self.data_id_to_scatter_idx = {data_id: idx for idx, data_id in enumerate(self.scatter_point_order)}
                else:
                    print("VERIFIED: scatter_point_order and data_id_to_scatter_idx are in sync")
            
            # Add a test function call to verify mapping
            self._debug_row_mapping()
            
            # Reconnect click detection after plot refresh
            self._setup_click_detection()
            
            # Force enable picking on all plot artists
            self._force_enable_picking()
            
        except Exception as e:
            print(f"Error updating references: {str(e)}")
            import traceback
            traceback.print_exc()
