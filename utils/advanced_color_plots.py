#!/usr/bin/env python3
"""
Advanced Color Visualization for StampZ
Ternary (2D) and Quaternary (3D/4D) plotting for sophisticated color analysis.

This module provides:
- 2D Ternary plots for RGB ratio analysis (3 variables in 2D space)
- 3D Quaternary plots for L*a*b* + additional variable analysis
- ML-ready feature extraction from both coordinate systems
- Interactive visualization capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from typing import List, Tuple, Dict, Optional, Any
import math
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    # Create dummy classes to prevent NameError
    class _DummyPlotly:
        class Figure:
            pass
    go = _DummyPlotly()
    px = None
    print("Warning: plotly not installed. Install with: pip install plotly")
    print("3D interactive plots will not be available.")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. ML features will be limited.")


@dataclass
class ColorPoint:
    """Represents a color point with both coordinate systems."""
    id: str
    rgb: Tuple[float, float, float]        # Original RGB values
    lab: Tuple[float, float, float]        # L*a*b* values  
    ternary_coords: Tuple[float, float]    # 2D ternary coordinates
    metadata: Dict[str, Any]               # Additional info (stamp name, etc.)


class TernaryPlotter:
    """Creates 2D ternary diagrams for RGB ratio analysis."""
    
    def __init__(self, figsize=(10, 8)):
        """Initialize ternary plotter.
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        self.triangle_coords = self._calculate_triangle_coordinates()
        
    def _calculate_triangle_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Calculate the coordinates of the equilateral triangle vertices."""
        # Standard equilateral triangle with base on x-axis
        height = math.sqrt(3) / 2
        return {
            'red': (0.5, height),      # Top vertex (Red = 100%)
            'green': (0.0, 0.0),       # Bottom left (Green = 100%) 
            'blue': (1.0, 0.0)         # Bottom right (Blue = 100%)
        }
    
    def rgb_to_ternary(self, rgb: Tuple[float, float, float]) -> Tuple[float, float]:
        """Convert RGB values to ternary coordinates.
        
        Args:
            rgb: RGB tuple (0-255 values)
            
        Returns:
            (x, y) coordinates in ternary space
        """
        r, g, b = rgb
        
        # Normalize to percentages (handle zero sum case)
        total = r + g + b
        if total == 0:
            # Pure black - place at center
            r_pct = g_pct = b_pct = 1/3
        else:
            r_pct = r / total
            g_pct = g / total
            b_pct = b / total
        
        # Convert to ternary coordinates
        # x = 0.5 * (2*blue + green) 
        # y = (sqrt(3)/2) * green
        x = 0.5 * (2 * b_pct + g_pct)
        y = (math.sqrt(3) / 2) * g_pct
        
        return (x, y)
    
    def create_ternary_plot(self, color_points: List[ColorPoint],
                           title: str = "RGB Ternary Plot",
                           color_by: str = "auto",
                           show_hull: bool = False) -> Any:
        """Create a 2D ternary plot.
        
        Args:
            color_points: List of ColorPoint objects to plot
            title: Plot title
            color_by: How to color points ("auto", "lightness", "cluster", etc.)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw triangle framework
        self._draw_triangle_framework(ax)
        
        # Convert RGB to ternary coordinates
        ternary_coords = []
        colors = []
        labels = []
        
        for point in color_points:
            x, y = self.rgb_to_ternary(point.rgb)
            ternary_coords.append((x, y))
            labels.append(point.id)
            
            # Determine point color based on color_by parameter
            if color_by == "auto":
                # Use the actual RGB color (normalized)
                colors.append(tuple(c/255.0 for c in point.rgb))
            elif color_by == "lightness":
                # Color by L* value from Lab
                lightness = point.lab[0] / 100.0  # Normalize L* (0-100) to 0-1
                colors.append(plt.cm.viridis(lightness))
            
        # Plot points
        x_coords, y_coords = zip(*ternary_coords)
        scatter = ax.scatter(x_coords, y_coords, c=colors, s=60, alpha=0.7, 
                           edgecolors='black', linewidths=0.5)
        
        # Add convex hull if requested
        if show_hull and len(color_points) >= 3:
            try:
                from scipy.spatial import ConvexHull
                points = np.column_stack((x_coords, y_coords))
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'k-', alpha=0.3)
            except ImportError:
                print("Warning: scipy.spatial.ConvexHull not available for hull plotting")
            except Exception:
                pass  # Ignore if convex hull fails
        
        # Add labels if not too many points
        if len(color_points) <= 20:
            for i, (x, y) in enumerate(ternary_coords):
                ax.annotate(labels[i], (x, y), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')  # Hide axes for clean look
        
        return fig
    
    def save_plot(self, fig: Any, filename: str) -> bool:
        """Save ternary plot to file.
        
        Args:
            fig: matplotlib Figure object
            filename: Output filename
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"Error saving plot to {filename}: {e}")
            return False
    
    def _draw_triangle_framework(self, ax):
        """Draw the ternary triangle framework with labels and grid."""
        coords = self.triangle_coords
        
        # Draw triangle outline
        triangle_x = [coords['green'][0], coords['blue'][0], coords['red'][0], coords['green'][0]]
        triangle_y = [coords['green'][1], coords['blue'][1], coords['red'][1], coords['green'][1]]
        ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)
        
        # Add vertex labels with enhanced styling
        ax.text(coords['red'][0], coords['red'][1] + 0.08, 'RED\n100%', 
               ha='center', va='bottom', fontweight='bold', fontsize=14, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))
        ax.text(coords['green'][0] - 0.08, coords['green'][1], 'GREEN\n100%', 
               ha='right', va='center', fontweight='bold', fontsize=14,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
        ax.text(coords['blue'][0] + 0.08, coords['blue'][1], 'BLUE\n100%', 
               ha='left', va='center', fontweight='bold', fontsize=14,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.2))
        
        # Add edge labels to clarify reading directions
        # Red-Green edge (bottom)
        mid_rg_x = (coords['red'][0] + coords['green'][0]) / 2
        mid_rg_y = (coords['red'][1] + coords['green'][1]) / 2 - 0.06
        ax.text(mid_rg_x, mid_rg_y, 'Blue content increases ↑', 
               ha='center', va='top', fontsize=10, style='italic', alpha=0.8)
        
        # Red-Blue edge (right)
        mid_rb_x = (coords['red'][0] + coords['blue'][0]) / 2 + 0.06
        mid_rb_y = (coords['red'][1] + coords['blue'][1]) / 2
        ax.text(mid_rb_x, mid_rb_y, 'Green\ncontent\nincreases\n←', 
               ha='left', va='center', fontsize=10, style='italic', alpha=0.8)
        
        # Green-Blue edge (left)
        mid_gb_x = (coords['green'][0] + coords['blue'][0]) / 2 - 0.06
        mid_gb_y = (coords['green'][1] + coords['blue'][1]) / 2
        ax.text(mid_gb_x, mid_gb_y, 'Red\ncontent\nincreases\n→', 
               ha='right', va='center', fontsize=10, style='italic', alpha=0.8)
        
        # Draw grid lines (percentage lines)
        self._draw_ternary_grid(ax)
        
        # Set equal aspect and limits
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)
    
    def _draw_ternary_grid(self, ax):
        """Draw percentage grid lines on the ternary plot."""
        # Draw percentage lines (every 20%)
        for pct in [0.2, 0.4, 0.6, 0.8]:
            # Lines parallel to each side of the triangle
            self._draw_percentage_lines(ax, pct)
    
    def _draw_percentage_lines(self, ax, percentage):
        """Draw lines at a specific percentage level."""
        coords = self.triangle_coords
        
        # Lines parallel to green-blue edge (red percentage lines)
        p1 = (coords['green'][0] + percentage * (coords['red'][0] - coords['green'][0]),
              coords['green'][1] + percentage * (coords['red'][1] - coords['green'][1]))
        p2 = (coords['blue'][0] + percentage * (coords['red'][0] - coords['blue'][0]),
              coords['blue'][1] + percentage * (coords['red'][1] - coords['blue'][1]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k--', alpha=0.3, linewidth=0.5)
        
        # Lines parallel to red-blue edge (green percentage lines)  
        p3 = (coords['red'][0] + percentage * (coords['green'][0] - coords['red'][0]),
              coords['red'][1] + percentage * (coords['green'][1] - coords['red'][1]))
        p4 = (coords['blue'][0] + percentage * (coords['green'][0] - coords['blue'][0]),
              coords['blue'][1] + percentage * (coords['green'][1] - coords['blue'][1]))
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], 'k--', alpha=0.3, linewidth=0.5)
        
        # Lines parallel to red-green edge (blue percentage lines)
        p5 = (coords['red'][0] + percentage * (coords['blue'][0] - coords['red'][0]),
              coords['red'][1] + percentage * (coords['blue'][1] - coords['red'][1]))
        p6 = (coords['green'][0] + percentage * (coords['blue'][0] - coords['green'][0]),
              coords['green'][1] + percentage * (coords['blue'][1] - coords['green'][1]))
        ax.plot([p5[0], p6[0]], [p5[1], p6[1]], 'k--', alpha=0.3, linewidth=0.5)


class QuaternaryPlotter:
    """Creates 3D/4D plots for L*a*b* + additional variable analysis."""
    
    def __init__(self):
        """Initialize quaternary plotter."""
        if not HAS_PLOTLY:
            print("Warning: Plotly not available. 3D plots will be limited.")
    
    def create_lab_scatter(self, color_points: List[ColorPoint], 
                         title: str = "L*a*b* Color Space",
                         color_by: str = "actual",
                         fourth_dimension: Optional[str] = None) -> Optional[go.Figure]:
        """Create 3D L*a*b* scatter plot with optional 4th dimension.
        
        Args:
            color_points: List of ColorPoint objects
            title: Plot title
            fourth_dimension: Optional 4th variable for sizing/coloring
            
        Returns:
            Plotly Figure object or None if plotly unavailable
        """
        if not HAS_PLOTLY:
            print("Plotly required for 3D plots")
            return None
        
        # Extract L*a*b* coordinates
        L_values = [point.lab[0] for point in color_points]
        a_values = [point.lab[1] for point in color_points]  
        b_values = [point.lab[2] for point in color_points]
        
        # Create RGB colors for points (normalized)
        colors = [f'rgb({int(r)},{int(g)},{int(b)})' 
                 for r, g, b in [point.rgb for point in color_points]]
        
        labels = [point.id for point in color_points]
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=a_values,  # a* axis (green-red)
            y=b_values,  # b* axis (blue-yellow)
            z=L_values,  # L* axis (lightness) - vertical
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><br>' +
                         'L*: %{z:.1f}<br>' +
                         'a*: %{x:.1f}<br>' +
                         'b*: %{y:.1f}<br>' +
                         '<extra></extra>'
        ))
        
        # Set layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='a* (Green ← → Red)',
                yaxis_title='b* (Blue ← → Yellow)', 
                zaxis_title='L* (Lightness)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def save_plot(self, fig: Any, filename: str) -> bool:
        """Save 3D plot to file.
        
        Args:
            fig: plotly Figure object
            filename: Output filename (.html for interactive plots)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not HAS_PLOTLY or fig is None:
            return False
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            if filename.endswith('.html'):
                fig.write_html(filename)
            elif filename.endswith('.png'):
                # Requires kaleido: pip install kaleido
                fig.write_image(filename, width=800, height=600)
            else:
                # Default to HTML
                fig.write_html(filename + '.html')
            
            return True
        except Exception as e:
            print(f"Error saving plot to {filename}: {e}")
            return False
    
    def create_tetrahedron_plot(self, color_points: List[ColorPoint]) -> Optional[go.Figure]:
        """Create 4D tetrahedron plot for CMYK or L*a*b*+C analysis.
        
        Args:
            color_points: List of ColorPoint objects
            
        Returns:
            Plotly Figure object or None if plotly unavailable
        """
        if not HAS_PLOTLY:
            return None
        
        # This would implement 4-variable tetrahedron visualization
        # For now, return a placeholder
        print("Tetrahedron plotting not yet implemented")
        return None


class ColorClusterAnalyzer:
    """ML-based color clustering and pattern recognition."""
    
    def __init__(self):
        """Initialize cluster analyzer."""
        self.scaler = StandardScaler() if HAS_SKLEARN else None
    
    def extract_features(self, color_points: List[ColorPoint]) -> Dict[str, np.ndarray]:
        """Extract ML features from color points in both coordinate systems.
        
        Args:
            color_points: List of ColorPoint objects
            
        Returns:
            Dictionary of feature arrays for ML analysis
        """
        features = {}
        
        # RGB features
        rgb_array = np.array([point.rgb for point in color_points])
        features['rgb'] = rgb_array
        
        # L*a*b* features  
        lab_array = np.array([point.lab for point in color_points])
        features['lab'] = lab_array
        
        # Ternary coordinate features
        ternary_plotter = TernaryPlotter()
        ternary_coords = []
        for point in color_points:
            x, y = ternary_plotter.rgb_to_ternary(point.rgb)
            ternary_coords.append([x, y])
        features['ternary'] = np.array(ternary_coords)
        
        # Derived features
        features['lightness'] = lab_array[:, 0]  # L* values
        features['chroma'] = np.sqrt(lab_array[:, 1]**2 + lab_array[:, 2]**2)  # Chroma
        features['hue'] = np.arctan2(lab_array[:, 2], lab_array[:, 1])  # Hue angle
        
        return features
    
    def cluster_by_families(self, color_points: List[ColorPoint], 
                          method: str = "ternary") -> Dict[str, Any]:
        """Cluster colors into families using specified method.
        
        Args:
            color_points: List of ColorPoint objects
            method: Clustering method ("ternary", "lab", "combined")
            
        Returns:
            Dictionary with clustering results
        """
        if not HAS_SKLEARN:
            print("Scikit-learn required for clustering")
            return {}
        
        features = self.extract_features(color_points)
        
        if method == "ternary":
            cluster_data = features['ternary']
        elif method == "lab":
            cluster_data = features['lab']
        elif method == "combined":
            # Combine normalized ternary and Lab features
            ternary_norm = self.scaler.fit_transform(features['ternary'])
            lab_norm = self.scaler.fit_transform(features['lab'])
            cluster_data = np.hstack([ternary_norm, lab_norm])
        
        # Perform K-means clustering
        n_clusters = min(5, len(color_points) // 2)  # Adaptive cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data)
        
        # Analyze clusters
        clusters = {}
        cluster_dict_for_return = {}
        for i in range(n_clusters):
            cluster_points = [point for j, point in enumerate(color_points) 
                            if cluster_labels[j] == i]
            
            # Determine cluster family name based on dominant color
            cluster_name = self._determine_cluster_family(cluster_points)
            
            # Store detailed cluster info
            clusters[cluster_name] = {
                'points': cluster_points,
                'center': kmeans.cluster_centers_[i],
                'size': len(cluster_points)
            }
            
            # Store simple name -> points mapping for return
            cluster_dict_for_return[cluster_name] = cluster_points
        
        return {
            'clusters': cluster_dict_for_return,  # Simple name -> points mapping
            'detailed_clusters': clusters,  # Detailed cluster info
            'labels': cluster_labels,
            'method': method,
            'n_clusters': n_clusters
        }
    
    def _determine_cluster_family(self, cluster_points: List[ColorPoint]) -> str:
        """Determine the color family name for a cluster."""
        # Calculate average RGB to determine dominant color
        avg_rgb = np.mean([point.rgb for point in cluster_points], axis=0)
        r, g, b = avg_rgb
        
        # Simple color family classification
        max_component = max(r, g, b)
        
        if max_component == r and r > g * 1.2 and r > b * 1.2:
            return "Red Family"
        elif max_component == g and g > r * 1.2 and g > b * 1.2:
            return "Green Family"
        elif max_component == b and b > r * 1.2 and b > g * 1.2:
            return "Blue Family"
        elif r > 150 and g > 150 and abs(r - g) < 30 and b < r * 0.7:
            return "Yellow Family"
        elif r > 100 and b > 100 and abs(r - b) < 50 and g < r * 0.7:
            return "Purple Family"
        elif g > 100 and b > 100 and abs(g - b) < 50 and r < g * 0.7:
            return "Cyan Family"
        else:
            return "Mixed Colors"
    
    def detect_outliers(self, color_points: List[ColorPoint]) -> List[ColorPoint]:
        """Detect color outliers using DBSCAN clustering.
        
        Args:
            color_points: List of ColorPoint objects
            
        Returns:
            List of ColorPoint objects that are outliers
        """
        if not HAS_SKLEARN:
            return []
        
        features = self.extract_features(color_points)
        
        # Use combined features for outlier detection
        combined_features = np.hstack([
            self.scaler.fit_transform(features['lab']),
            self.scaler.fit_transform(features['ternary'])
        ])
        
        # DBSCAN clustering to find outliers
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = dbscan.fit_predict(combined_features)
        
        # Points labeled as -1 are outliers
        outlier_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        outlier_points = [color_points[i] for i in outlier_indices]
        
        return outlier_points


def demo_advanced_plotting():
    """Demonstrate the advanced plotting capabilities."""
    print("=== StampZ Advanced Color Visualization Demo ===\n")
    
    # Create sample color points
    sample_colors = [
        ColorPoint("Red_Stamp", (200, 50, 50), (45.2, 68.3, 45.1), (0, 0), 
                   {"stamp_id": "F137", "year": "1920"}),
        ColorPoint("Blue_Stamp", (50, 50, 200), (32.1, 22.4, -67.8), (0, 0), 
                   {"stamp_id": "F138", "year": "1920"}),
        ColorPoint("Green_Stamp", (50, 200, 50), (72.4, -68.2, 61.3), (0, 0), 
                   {"stamp_id": "F139", "year": "1921"}),
        ColorPoint("Purple_Mix", (150, 50, 150), (41.8, 58.9, -28.4), (0, 0), 
                   {"stamp_id": "F140", "year": "1921"}),
        ColorPoint("Orange_Mix", (200, 100, 50), (58.3, 42.1, 58.7), (0, 0), 
                   {"stamp_id": "F141", "year": "1922"}),
    ]
    
    # Update ternary coordinates
    ternary_plotter = TernaryPlotter()
    for point in sample_colors:
        point.ternary_coords = ternary_plotter.rgb_to_ternary(point.rgb)
    
    print("📊 Sample Color Analysis:")
    for point in sample_colors:
        print(f"  {point.id}: RGB{point.rgb} → L*a*b*{point.lab}")
        print(f"    Ternary: {point.ternary_coords}")
    
    # Create ternary plot
    print("\n🔺 Creating 2D Ternary Plot (RGB Ratios)...")
    fig = ternary_plotter.create_ternary_plot(sample_colors, 
                                            title="StampZ Color Families - RGB Ratios")
    
    # Create 3D plot if available
    if HAS_PLOTLY:
        print("🎯 Creating 3D L*a*b* Plot...")
        quaternary_plotter = QuaternaryPlotter()
        fig_3d = quaternary_plotter.create_lab_scatter(sample_colors, 
                                                     title="StampZ Colors - L*a*b* Space")
    
    # Demonstrate clustering
    if HAS_SKLEARN:
        print("🤖 Performing ML Color Family Analysis...")
        analyzer = ColorClusterAnalyzer()
        
        # Extract features for ML
        features = analyzer.extract_features(sample_colors)
        print(f"  Extracted {len(features)} feature types:")
        for feat_name, feat_array in features.items():
            print(f"    {feat_name}: {feat_array.shape}")
        
        # Cluster analysis
        cluster_results = analyzer.cluster_by_families(sample_colors, method="combined")
        print(f"  Found {cluster_results.get('n_clusters', 0)} color families")
        
        for family_name, family_points in cluster_results.get('clusters', {}).items():
            print(f"    {family_name}: {len(family_points)} stamps")
    
    print("\n✅ Advanced Plotting Demo Complete!")
    print("\n🚀 Integration Benefits:")
    print("  • 2D Ternary: Quick color family classification")
    print("  • 3D L*a*b*: Precise color space analysis") 
    print("  • ML Features: Pattern recognition & outlier detection")
    print("  • Combined Views: Best of both mathematical worlds")
    print("\n🎯 Perfect foundation for ML-enhanced stamp analysis!")


if __name__ == "__main__":
    demo_advanced_plotting()