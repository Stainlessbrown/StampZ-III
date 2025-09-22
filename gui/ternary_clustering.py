#!/usr/bin/env python3
"""
Ternary Plot Clustering Module

Handles K-means clustering functionality for ternary plots including:
- Cluster computation and management
- Centroid calculations
- Sphere visualization
- Cluster data organization for datasheet integration

This module is extracted from the main ternary_plot_window.py for better modularity.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Optional ML imports
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class TernaryClusterManager:
    """Manages K-means clustering for ternary plots."""
    
    def __init__(self):
        """Initialize the cluster manager."""
        self.clusters = {}
        self.cluster_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
        self.n_clusters = 3
        
    def has_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        return HAS_SKLEARN
    
    def clear_clusters(self):
        """Clear all cluster data."""
        self.clusters = {}
        logger.info("Clusters cleared")
    
    def compute_clusters(self, color_points: List[Any], n_clusters: int = 3) -> Dict[int, List[Any]]:
        """
        Compute K-means clusters on ternary coordinates.
        
        Args:
            color_points: List of ColorPoint objects
            n_clusters: Number of clusters to compute
            
        Returns:
            Dictionary mapping cluster IDs to lists of ColorPoint objects
        """
        if not color_points or not HAS_SKLEARN:
            logger.warning("Cannot compute clusters: no data or scikit-learn unavailable")
            return {}
        
        try:
            # Extract and normalize ternary coordinates to ensure consistent 2D format
            ternary_coords_list = []
            for i, point in enumerate(color_points):
                if hasattr(point, 'ternary_coords') and point.ternary_coords is not None:
                    coords = point.ternary_coords
                    # Ensure we have exactly 2D coordinates (x, y)
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        # Take only first 2 coordinates for ternary (x, y)
                        ternary_coords_list.append([float(coords[0]), float(coords[1])])
                    else:
                        logger.warning(f"Point {i} has invalid ternary_coords: {coords}, skipping")
                        continue
                else:
                    logger.warning(f"Point {i} has no ternary_coords, skipping")
                    continue
            
            if len(ternary_coords_list) < 2:
                logger.warning(f"Only {len(ternary_coords_list)} valid points found, need at least 2 for clustering")
                return {}
            
            # Convert to numpy array - should now be consistent 2D shape
            ternary_coords = np.array(ternary_coords_list)
            
            # Update color_points list to match the valid coordinates
            valid_color_points = []
            coord_index = 0
            for point in color_points:
                if hasattr(point, 'ternary_coords') and point.ternary_coords is not None:
                    coords = point.ternary_coords
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        valid_color_points.append(point)
                        coord_index += 1
            
            color_points = valid_color_points  # Use only points with valid coordinates
            
            # Debug: Show what data we're clustering on
            logger.info(f"DEBUG: K-means input data shape: {ternary_coords.shape}")
            logger.info(f"DEBUG: Using {len(color_points)} valid points out of original set")
            logger.info(f"DEBUG: First 5 ternary coordinates: {ternary_coords[:5]}")
            logger.info(f"DEBUG: Coordinate ranges - X: {ternary_coords[:, 0].min():.4f} to {ternary_coords[:, 0].max():.4f}")
            logger.info(f"DEBUG: Coordinate ranges - Y: {ternary_coords[:, 1].min():.4f} to {ternary_coords[:, 1].max():.4f}")
            
            # Apply K-means
            n_clusters = min(n_clusters, len(color_points) // 2)
            if n_clusters < 2:
                logger.warning(f"Not enough points for clustering: need at least 4 points for 2 clusters, have {len(color_points)}")
                return {}
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(ternary_coords)
            
            # Debug: Show K-means results
            logger.info(f"DEBUG: K-means found {len(np.unique(cluster_labels))} unique clusters")
            logger.info(f"DEBUG: Cluster centers from K-means: {kmeans.cluster_centers_}")
            for i, center in enumerate(kmeans.cluster_centers_):
                logger.info(f"DEBUG: K-means cluster {i} center: ({center[0]:.4f}, {center[1]:.4f})")
            
            # Group points by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(color_points[i])
            
            self.clusters = clusters
            logger.info(f"K-means: {len(clusters)} clusters computed")
            return clusters
            
        except Exception as e:
            logger.exception("K-means clustering failed")
            return {}
    
    def calculate_centroids(self, clusters: Optional[Dict[int, List[Any]]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Calculate cluster centroids in normalized Plot_3D format.
        
        Args:
            clusters: Optional clusters dict, uses self.clusters if None
            
        Returns:
            Dictionary mapping cluster IDs to centroid data
        """
        if clusters is None:
            clusters = self.clusters
            
        if not clusters:
            logger.warning("No clusters available for centroid calculation")
            return {}
        
        try:
            import matplotlib.pyplot as plt
            colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
            
            cluster_centroids = {}
            for cluster_idx, (cluster_id, cluster_points) in enumerate(clusters.items()):
                # Calculate cluster centroid in L*a*b* space
                cluster_l = [cp.lab[0] for cp in cluster_points]
                cluster_a = [cp.lab[1] for cp in cluster_points]
                cluster_b = [cp.lab[2] for cp in cluster_points]
                
                # Convert centroid to normalized 0-1 range (Plot_3D format)
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
                
            return cluster_centroids
            
        except Exception as e:
            logger.exception(f"Failed to calculate centroids: {e}")
            return {}
    
    def get_cluster_assignment(self, point: Any, clusters: Optional[Dict[int, List[Any]]] = None) -> str:
        """
        Get the cluster assignment for a specific point.
        
        Args:
            point: ColorPoint object
            clusters: Optional clusters dict, uses self.clusters if None
            
        Returns:
            Cluster ID as string, empty if not found
        """
        if clusters is None:
            clusters = self.clusters
            
        if not clusters:
            return ''
            
        for cluster_id, cluster_points in clusters.items():
            if point in cluster_points:
                return str(cluster_id)
        
        return ''
    
    def create_sphere_patches(self, clusters: Optional[Dict[int, List[Any]]] = None) -> List[Tuple[Any, str, int]]:
        """
        Create sphere visualization patches for matplotlib.
        
        Args:
            clusters: Optional clusters dict, uses self.clusters if None
            
        Returns:
            List of (circle_patch, color, cluster_label) tuples
        """
        if clusters is None:
            clusters = self.clusters
            
        if not clusters:
            return []
        
        try:
            import matplotlib.pyplot as plt
            patches = []
            
            for i, (label, points) in enumerate(clusters.items()):
                color = self.cluster_colors[i % len(self.cluster_colors)]
                
                # Extract coordinates for this cluster
                x_coords = [p.ternary_coords[0] for p in points]
                y_coords = [p.ternary_coords[1] for p in points]
                
                # Calculate cluster center
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                
                # Set sphere radius to represent ΔE (0.02 for normalized coordinates)
                radius = 0.02
                
                # Create circle patch
                circle = plt.Circle((center_x, center_y), radius, 
                                  color=color, alpha=0.15, zorder=1)
                
                patches.append((circle, color, label))
            
            return patches
            
        except Exception as e:
            logger.exception(f"Failed to create sphere patches: {e}")
            return []
    
    def create_centroid_markers(self, clusters: Optional[Dict[int, List[Any]]] = None) -> List[Dict[str, Any]]:
        """
        Create centroid marker data for plotting.
        
        Args:
            clusters: Optional clusters dict, uses self.clusters if None
            
        Returns:
            List of centroid marker dictionaries
        """
        if clusters is None:
            clusters = self.clusters
            
        if not clusters:
            return []
        
        try:
            markers = []
            
            for i, (label, points) in enumerate(clusters.items()):
                color = self.cluster_colors[i % len(self.cluster_colors)]
                
                # Extract coordinates for this cluster
                x_coords = [p.ternary_coords[0] for p in points]
                y_coords = [p.ternary_coords[1] for p in points]
                
                # Calculate cluster center
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                
                markers.append({
                    'x': center_x,
                    'y': center_y,
                    'color': color,
                    'size': 60,
                    'marker': 'o',
                    'label': f'Cluster {label} ({len(points)} pts)',
                    'cluster_id': label
                })
            
            return markers
            
        except Exception as e:
            logger.exception(f"Failed to create centroid markers: {e}")
            return []
    
    def save_cluster_assignments_to_db(self, database_manager, sample_set_name: str):
        """
        Save cluster assignments to database for persistence.
        
        Args:
            database_manager: Database manager instance
            sample_set_name: Name of the sample set
        """
        if not self.clusters:
            logger.warning("No clusters to save to database")
            return
        
        try:
            for cluster_id, cluster_points in self.clusters.items():
                for point in cluster_points:
                    # Parse point ID to get image_name and coordinate_point
                    if '_pt' in point.id:
                        image_name, pt_part = point.id.rsplit('_pt', 1)
                        coordinate_point = int(pt_part)
                    else:
                        image_name = point.id
                        coordinate_point = 1
                    
                    # Save cluster assignment using database manager
                    success = database_manager.update_cluster_assignment(
                        image_name=image_name,
                        coordinate_point=coordinate_point,
                        cluster_id=int(cluster_id)
                    )
                    
                    if not success:
                        logger.warning(f"Failed to save cluster assignment for {point.id}")
            
            logger.info(f"Saved cluster assignments for {sum(len(points) for points in self.clusters.values())} points")
            
        except Exception as e:
            logger.exception(f"Failed to save cluster assignments: {e}")
    
    def clear_clusters(self):
        """Clear all cluster data."""
        self.clusters = {}
        logger.info("Cleared all cluster data")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        if not self.clusters:
            return {'cluster_count': 0, 'total_points': 0}
        
        stats = {
            'cluster_count': len(self.clusters),
            'total_points': sum(len(points) for points in self.clusters.values()),
            'cluster_sizes': {str(k): len(v) for k, v in self.clusters.items()},
            'average_cluster_size': np.mean([len(points) for points in self.clusters.values()])
        }
        
        return stats