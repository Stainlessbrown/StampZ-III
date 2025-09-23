import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

class SphereManager:
    """
    Manager class for rendering spheres at centroid coordinates in 3D space.
    
    This class manages the creation and visualization of translucent spheres at
    Centroid_X/Y/Z coordinates. Spheres are only rendered if valid coordinate data exists.
    The colors are taken from the "Sphere" column of the dataframe, with a default of gray
    if no color is specified.
    """
    
    def __init__(self, ax, canvas, data_df: pd.DataFrame):
        """
        Initialize the SphereManager.
        
        Args:
            ax: The matplotlib 3D axis object
            canvas: The matplotlib canvas for rendering
            data_df: Pandas DataFrame containing the data to visualize
        """
        self.ax = ax
        self.canvas = canvas
        self.data_df = data_df
        self.sphere_objects = []  # Store references to sphere objects for later removal
        
        # Constants for sphere rendering
        self.ALPHA = 0.15  # Fixed transparency
        self.DEFAULT_RADIUS = 0.02  # Default radius when none specified
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        
        # Define color mapping (similar to existing color dictionary but with yellow instead of black)
        self.color_map = {
            'red': 'r',
            'green': 'g', 
            'blue': 'b',
            'yellow': 'y',  # Yellow instead of black
            'cyan': 'c',
            'magenta': 'm',
            'orange': 'orange',
            'purple': 'purple',
            'brown': 'brown',
            'pink': 'pink',
            'lime': 'lime',
            'navy': 'navy',
            'teal': 'teal'
        }
        
        # Initialize visibility states (all spheres visible by default)
        self.visibility_states = {color: True for color in self.color_map.values()}
        
        self.logger.info("SphereManager initialized successfully")
    
    def clear_spheres(self) -> None:
        """Remove all sphere objects from the plot."""
        try:
            # Remove all sphere objects from the plot
            # 3D surface objects from plot_surface() are stored differently than regular collections
            sphere_count_before = len(self.sphere_objects)
            for sphere in self.sphere_objects:
                try:
                    # Try to remove the sphere surface object
                    sphere.remove()
                except (ValueError, AttributeError) as e:
                    # If direct removal fails, try to remove from collections
                    if sphere in self.ax.collections:
                        self.ax.collections.remove(sphere)
                    elif hasattr(sphere, '_remove_method'):
                        # Some matplotlib objects have custom removal methods
                        sphere._remove_method()
                except Exception as e:
                    self.logger.warning(f"Could not remove sphere object: {e}")
            
            # Clear the list of sphere objects
            self.sphere_objects = []
            self.logger.info(f"Cleared {sphere_count_before} sphere objects from plot")
        except Exception as e:
            self.logger.error(f"Error clearing spheres: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_references(self, ax, canvas, data_df: pd.DataFrame) -> None:
        """
        Update references to the axis, canvas, and dataframe.
        
        Args:
            ax: The matplotlib 3D axis object
            canvas: The matplotlib canvas for rendering
            data_df: Pandas DataFrame containing the data to visualize
        """
        try:
            # Update references
            self.ax = ax
            self.canvas = canvas
            self.data_df = data_df
            
            # Clear existing spheres
            self.clear_spheres()
            
            self.logger.info(f"Updated SphereManager with {len(data_df)} data points")
        except Exception as e:
            self.logger.error(f"Error updating references: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def toggle_visibility(self, color: str) -> None:
        """
        Toggle the visibility state of spheres of the specified color.
        
        Args:
            color: The color code of the spheres to toggle
        """
        try:
            if color in self.visibility_states:
                self.visibility_states[color] = not self.visibility_states[color]
                self.logger.info(f"Toggled visibility of {color} spheres to {self.visibility_states[color]}")
                self.render_spheres()  # Refresh the spheres with updated visibility
        except Exception as e:
            self.logger.error(f"Error toggling sphere visibility: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_active_colors(self) -> List[str]:
        """
        Get list of colors currently used in the data.
        
        Returns:
            List of color codes that are actually present in the dataset
        """
        try:
            # Get unique sphere data and parse color names
            sphere_data = self.data_df['Sphere'].dropna().unique()
            color_names = set()
            
            for sphere_str in sphere_data:
                if '/' in str(sphere_str):
                    # Format: "color/radius" - extract color part
                    color_name = str(sphere_str).split('/')[0].strip()
                else:
                    # Format: just "color"
                    color_name = str(sphere_str).strip()
                color_names.add(color_name)
            
            # Convert to color codes
            active_colors = [self._get_color(color_name) for color_name in color_names]
            return active_colors
        except Exception as e:
            self.logger.error(f"Error getting active colors: {str(e)}")
            return []
    
    def _create_sphere_mesh(self, center: Tuple[float, float, float], radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a 3D sphere mesh for rendering.
        
        Args:
            center: (x, y, z) coordinates of the sphere center
            radius: Radius of the sphere
            
        Returns:
            Tuple of (x, y, z) mesh grids for the sphere surface
        """
        # Create sphere mesh - use enough points for a smooth sphere but not too many for performance
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        return x, y, z
    
    def _get_color(self, color_name: str) -> str:
        """
        Get color code from color name.
        
        Args:
            color_name: Name of the color
            
        Returns:
            Matplotlib color code
        """
        # If color_name is in our map, return the corresponding code
        if color_name in self.color_map:
            return self.color_map[color_name]
        
        # If it's not in our map but is a valid color string, return it directly
        try:
            if isinstance(color_name, str):
                # Check if it might be a hex color or other valid color name
                return color_name
        except:
            pass
        
        # Default color is gray
        return 'gray'
    
    def render_spheres(self) -> None:
        """
        Render spheres at centroid coordinates using variable radii.
        
        Spheres are only drawn where Centroid_X/Y/Z coordinates exist.
        Colors are taken from the "Sphere" column with a default of gray.
        Radii are taken from the "Radius" column with a default of DEFAULT_RADIUS.
        All spheres have a fixed alpha value of 0.15.
        """
        try:
            # Clear existing spheres first
            self.clear_spheres()
            
            # Filter data to only include rows with valid Centroid coordinates
            # ALL THREE coordinates must be present - no fallback
            valid_mask = (
                self.data_df['Centroid_X'].notna() & 
                self.data_df['Centroid_Y'].notna() & 
                self.data_df['Centroid_Z'].notna()
            )
            centroid_data = self.data_df[valid_mask].copy()
            
            # Log how many valid centroid points we found
            self.logger.info(f"Found {len(centroid_data)} points with valid centroid coordinates for spheres")
            print(f"DEBUG: Found {len(centroid_data)} points with valid centroid coordinates for spheres")
            
            # SAFE DEBUG: Show what sphere data we're working with (doesn't change logic)
            if len(centroid_data) > 0:
                print(f"DEBUG: Sample centroid rows with sphere data:")
                for i, (idx, row) in enumerate(centroid_data.head(5).iterrows()):
                    sphere_val = row.get('Sphere', 'N/A')
                    radius_val = row.get('Radius', 'N/A') 
                    data_id = row.get('DataID', 'N/A')
                    print(f"  Row {idx}: DataID={data_id}, Sphere='{sphere_val}', Radius='{radius_val}' (type: {type(radius_val)})")
                    # Extra debug for radius conversion
                    if pd.notna(radius_val):
                        try:
                            radius_float = float(radius_val)
                            print(f"    Radius is NOT NaN - will use: {radius_float}")
                        except (ValueError, TypeError):
                            print(f"    Radius '{radius_val}' is not convertible to float - using default: {self.DEFAULT_RADIUS}")
                    else:
                        print(f"    Radius IS NaN - will use default: {self.DEFAULT_RADIUS}")
            
            if len(centroid_data) == 0:
                self.logger.info("No valid centroid data found for sphere rendering")
                print("DEBUG: No valid centroid data found for sphere rendering")
                return
            
            # Group by unique centroid positions to avoid duplicate spheres
            # Multiple data points can share the same centroid (same cluster), but we only want one sphere per position
            unique_centroids = {}
            
            for idx, row in centroid_data.iterrows():
                # Create a key from centroid coordinates (rounded to avoid floating point issues)
                centroid_key = (
                    round(float(row['Centroid_X']), 6),
                    round(float(row['Centroid_Y']), 6), 
                    round(float(row['Centroid_Z']), 6)
                )
                
                # Only keep the first occurrence of each unique centroid position
                if centroid_key not in unique_centroids:
                    unique_centroids[centroid_key] = row
            
            print(f"DEBUG: Reduced {len(centroid_data)} rows to {len(unique_centroids)} unique centroid positions")
            
            # Process each unique centroid position
            sphere_count = 0
            for centroid_key, row in unique_centroids.items():
                try:
                    # Parse sphere data which can be in format "color/radius" or just "color"
                    sphere_data = row.get('Sphere')
                    if pd.notna(sphere_data):
                        sphere_str = str(sphere_data).strip()
                        if '/' in sphere_str:
                            # Format: "color/radius"
                            parts = sphere_str.split('/')
                            color_name = parts[0].strip()
                            try:
                                radius = float(parts[1].strip())
                            except (ValueError, IndexError):
                                radius = self.DEFAULT_RADIUS
                                self.logger.warning(f"Invalid radius in sphere data '{sphere_str}', using default")
                        else:
                            # Format: just "color"
                            color_name = sphere_str
                            radius = self.DEFAULT_RADIUS
                    else:
                        # No sphere data
                        color_name = 'gray'
                        radius = self.DEFAULT_RADIUS
                    
                    # Get color code and check visibility
                    color = self._get_color(color_name)
                    
                    # Skip if sphere color is not visible
                    if not self.visibility_states.get(color, True):
                        continue
                        
                    # Ensure radius is positive
                    if not (radius > 0):
                        self.logger.warning(f"Invalid radius value {radius}, using default")
                        radius = self.DEFAULT_RADIUS
                    
                    # Use the exact centroid coordinates (not rounded)
                    center = (
                        float(row['Centroid_X']), 
                        float(row['Centroid_Y']), 
                        float(row['Centroid_Z'])
                    )
                    
                    print(f"DEBUG: Rendering sphere at {center} with color {color} and radius {radius}")
                    
                    # Create sphere mesh with variable radius
                    x, y, z = self._create_sphere_mesh(center, radius)
                    
                    # Render the sphere
                    sphere = self.ax.plot_surface(
                        x, y, z,
                        color=color,
                        alpha=self.ALPHA,
                        linewidth=0,
                        antialiased=True
                    )
                    
                    # Add to list of objects for later removal
                    self.sphere_objects.append(sphere)
                    sphere_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error rendering sphere at position {centroid_key}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully rendered {sphere_count} visible spheres")
            print(f"DEBUG: Successfully rendered {sphere_count} visible spheres")
            
            # Note: Canvas draw is handled by the main refresh_plot method to avoid conflicts
            
        except Exception as e:
            self.logger.error(f"Error rendering spheres: {str(e)}")
            import traceback
            traceback.print_exc()

