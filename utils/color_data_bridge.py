#!/usr/bin/env python3
"""
Color Data Bridge for StampZ Advanced Plotting
Connects existing L*a*b* databases with the new ternary/quaternary visualization system.

This module:
- Extracts color data from existing .db files
- Converts to ColorPoint objects for advanced plotting
- Provides filtering and grouping capabilities
- Bridges the gap between existing analysis and new ML features
"""

import os
import sqlite3
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

from .advanced_color_plots import ColorPoint, TernaryPlotter, QuaternaryPlotter, ColorClusterAnalyzer
from .color_analysis_db import ColorAnalysisDB


class ColorDataBridge:
    """Bridge between StampZ color databases and advanced plotting system."""
    
    def __init__(self):
        """Initialize the data bridge."""
        self.ternary_plotter = TernaryPlotter()
    
    def get_available_sample_sets(self) -> List[str]:
        """Get list of available color analysis databases.
        
        Returns:
            List of sample set names (database names without .db extension)
        """
        try:
            from .path_utils import get_color_analysis_dir
            analysis_dir = get_color_analysis_dir()
            
            if not os.path.exists(analysis_dir):
                return []
            
            # Find all .db files that are not library files
            db_files = []
            for file in os.listdir(analysis_dir):
                if file.endswith('.db') and not file.endswith('_library.db'):
                    # Extract sample set name (remove .db extension)
                    sample_set = os.path.splitext(file)[0]
                    db_files.append(sample_set)
            
            return sorted(db_files)
            
        except Exception as e:
            print(f"Error getting sample sets: {e}")
            return []
    
    def load_color_points_from_database(self, sample_set_name: str, 
                                      image_filter: Optional[str] = None,
                                      limit: Optional[int] = None) -> List[ColorPoint]:
        """Load color data from database and convert to ColorPoint objects.
        
        Args:
            sample_set_name: Name of the sample set database
            image_filter: Optional filter to only include specific images
            limit: Optional limit on number of points to load
            
        Returns:
            List of ColorPoint objects ready for advanced plotting
        """
        color_points = []
        
        try:
            # Connect to the color analysis database
            db = ColorAnalysisDB(sample_set_name)
            
            # Get all measurements, optionally filtered
            measurements = db.get_all_measurements()
            
            if image_filter:
                # Filter measurements by image name (case-insensitive partial match)
                measurements = [m for m in measurements 
                              if image_filter.lower() in m.get('image_name', '').lower()]
            
            if limit:
                measurements = measurements[:limit]
            
            print(f"Loaded {len(measurements)} measurements from {sample_set_name}")
            
            # Convert each measurement to a ColorPoint
            for i, measurement in enumerate(measurements):
                try:
                    # Extract required data
                    lab = (
                        measurement.get('l_value', 50.0),
                        measurement.get('a_value', 0.0),
                        measurement.get('b_value', 0.0)
                    )
                    
                    rgb = (
                        measurement.get('rgb_r', 128.0),
                        measurement.get('rgb_g', 128.0),
                        measurement.get('rgb_b', 128.0)
                    )
                    
                    # Generate unique ID for this point
                    image_name = measurement.get('image_name', 'unknown')
                    coord_point = measurement.get('coordinate_point', i+1)
                    point_id = f"{image_name}_pt{coord_point}"
                    
                    # Convert RGB to ternary coordinates
                    ternary_coords = self.ternary_plotter.rgb_to_ternary(rgb)
                    
                    # Create metadata dictionary
                    metadata = {
                        'sample_set': sample_set_name,
                        'image_name': image_name,
                        'coordinate_point': coord_point,
                        'x_position': measurement.get('x_position', 0),
                        'y_position': measurement.get('y_position', 0),
                        'measurement_date': measurement.get('measurement_date', ''),
                        'notes': measurement.get('notes', ''),
                        'sample_type': measurement.get('sample_type', 'circle'),
                        'sample_width': measurement.get('sample_width', 20),
                        'sample_height': measurement.get('sample_height', 20),
                        'anchor': measurement.get('anchor', 'center')
                    }
                    
                    # Add cluster information if available
                    if 'cluster_id' in measurement:
                        metadata['cluster_id'] = measurement['cluster_id']
                    
                    # Add Delta E information if available  
                    if 'delta_e' in measurement:
                        metadata['delta_e'] = measurement['delta_e']
                    
                    # Create ColorPoint object
                    color_point = ColorPoint(
                        id=point_id,
                        rgb=rgb,
                        lab=lab,
                        ternary_coords=ternary_coords,
                        metadata=metadata
                    )
                    
                    color_points.append(color_point)
                    
                except Exception as e:
                    print(f"Warning: Failed to convert measurement {i}: {e}")
                    continue
            
            print(f"Successfully converted {len(color_points)} measurements to ColorPoint objects")
            return color_points
            
        except Exception as e:
            print(f"Error loading color points from {sample_set_name}: {e}")
            return []
    
    def load_color_points_from_ods(self, ods_file_path: str, 
                                 sheet_name: Optional[str] = None) -> List[ColorPoint]:
        """Load color data from ODS file and convert to ColorPoint objects.
        
        Args:
            ods_file_path: Path to the ODS file
            sheet_name: Optional specific sheet name to load
            
        Returns:
            List of ColorPoint objects
        """
        color_points = []
        
        try:
            # Import ODS handling utilities
            from .ods_importer import ODSImporter
            
            importer = ODSImporter()
            
            # Load data from ODS file
            ods_data = importer.load_ods_file(ods_file_path)
            if not ods_data:
                print(f"Failed to load ODS file: {ods_file_path}")
                return []
            
            # Determine which sheet to use
            if sheet_name and sheet_name in ods_data:
                sheets_to_process = [sheet_name]
            else:
                # Use all sheets that contain color data
                sheets_to_process = list(ods_data.keys())
            
            # Process each sheet
            for sheet in sheets_to_process:
                sheet_data = ods_data[sheet]
                
                print(f"Processing sheet '{sheet}' with {len(sheet_data)} rows")
                
                for i, row in enumerate(sheet_data):
                    try:
                        # Expect columns: DataID, L*, a*, b*, R, G, B, etc.
                        # This is flexible - adapt to actual ODS structure
                        
                        data_id = str(row.get('DataID', f"ods_row_{i+1}"))
                        
                        # Extract L*a*b* values
                        lab = (
                            float(row.get('L*', row.get('L_star', row.get('L', 50.0)))),
                            float(row.get('a*', row.get('a_star', row.get('a', 0.0)))),
                            float(row.get('b*', row.get('b_star', row.get('b', 0.0))))
                        )
                        
                        # Extract RGB values
                        rgb = (
                            float(row.get('R', row.get('rgb_r', 128.0))),
                            float(row.get('G', row.get('rgb_g', 128.0))),
                            float(row.get('B', row.get('rgb_b', 128.0)))
                        )
                        
                        # Convert to ternary coordinates
                        ternary_coords = self.ternary_plotter.rgb_to_ternary(rgb)
                        
                        # Create metadata
                        metadata = {
                            'source': 'ods_file',
                            'file_path': ods_file_path,
                            'sheet_name': sheet,
                            'row_index': i,
                            'data_id': data_id
                        }
                        
                        # Add any additional columns as metadata
                        for key, value in row.items():
                            if key not in ['DataID', 'L*', 'a*', 'b*', 'R', 'G', 'B', 
                                         'L_star', 'a_star', 'b_star', 'rgb_r', 'rgb_g', 'rgb_b']:
                                metadata[key] = value
                        
                        # Create ColorPoint
                        color_point = ColorPoint(
                            id=data_id,
                            rgb=rgb,
                            lab=lab,
                            ternary_coords=ternary_coords,
                            metadata=metadata
                        )
                        
                        color_points.append(color_point)
                        
                    except Exception as e:
                        print(f"Warning: Failed to convert ODS row {i}: {e}")
                        continue
            
            print(f"Successfully converted {len(color_points)} ODS entries to ColorPoint objects")
            return color_points
            
        except Exception as e:
            print(f"Error loading color points from ODS file: {e}")
            return []
    
    def group_by_image(self, color_points: List[ColorPoint]) -> Dict[str, List[ColorPoint]]:
        """Group color points by image name.
        
        Args:
            color_points: List of ColorPoint objects
            
        Returns:
            Dictionary mapping image names to lists of ColorPoint objects
        """
        groups = {}
        
        for point in color_points:
            image_name = point.metadata.get('image_name', 'unknown')
            if image_name not in groups:
                groups[image_name] = []
            groups[image_name].append(point)
        
        return groups
    
    def group_by_color_family(self, color_points: List[ColorPoint]) -> Dict[str, List[ColorPoint]]:
        """Group color points by detected color families using ML clustering.
        
        Args:
            color_points: List of ColorPoint objects
            
        Returns:
            Dictionary mapping color family names to lists of ColorPoint objects
        """
        if not color_points:
            return {}
        
        # Use ML clustering to identify color families
        analyzer = ColorClusterAnalyzer()
        cluster_results = analyzer.cluster_by_families(color_points, method="combined")
        
        return cluster_results.get('clusters', {})
    
    def filter_by_lightness(self, color_points: List[ColorPoint], 
                          min_l: float = 0.0, max_l: float = 100.0) -> List[ColorPoint]:
        """Filter color points by L* (lightness) values.
        
        Args:
            color_points: List of ColorPoint objects
            min_l: Minimum L* value (0-100)
            max_l: Maximum L* value (0-100)
            
        Returns:
            Filtered list of ColorPoint objects
        """
        return [point for point in color_points 
                if min_l <= point.lab[0] <= max_l]
    
    def filter_by_chroma(self, color_points: List[ColorPoint], 
                        min_chroma: float = 0.0, max_chroma: float = 200.0) -> List[ColorPoint]:
        """Filter color points by chroma (color saturation).
        
        Args:
            color_points: List of ColorPoint objects
            min_chroma: Minimum chroma value
            max_chroma: Maximum chroma value
            
        Returns:
            Filtered list of ColorPoint objects
        """
        filtered = []
        for point in color_points:
            a_star, b_star = point.lab[1], point.lab[2]
            chroma = (a_star**2 + b_star**2)**0.5
            if min_chroma <= chroma <= max_chroma:
                filtered.append(point)
        
        return filtered
    
    def get_summary_stats(self, color_points: List[ColorPoint]) -> Dict[str, Any]:
        """Get summary statistics for a set of color points.
        
        Args:
            color_points: List of ColorPoint objects
            
        Returns:
            Dictionary with summary statistics
        """
        if not color_points:
            return {}
        
        # Extract coordinate arrays
        lab_values = [point.lab for point in color_points]
        rgb_values = [point.rgb for point in color_points]
        
        import numpy as np
        lab_array = np.array(lab_values)
        rgb_array = np.array(rgb_values)
        
        # Calculate statistics
        stats = {
            'total_points': len(color_points),
            'lab_means': {
                'L*': float(np.mean(lab_array[:, 0])),
                'a*': float(np.mean(lab_array[:, 1])),
                'b*': float(np.mean(lab_array[:, 2]))
            },
            'lab_std': {
                'L*': float(np.std(lab_array[:, 0])),
                'a*': float(np.std(lab_array[:, 1])),  
                'b*': float(np.std(lab_array[:, 2]))
            },
            'lab_ranges': {
                'L*': (float(np.min(lab_array[:, 0])), float(np.max(lab_array[:, 0]))),
                'a*': (float(np.min(lab_array[:, 1])), float(np.max(lab_array[:, 1]))),
                'b*': (float(np.min(lab_array[:, 2])), float(np.max(lab_array[:, 2])))
            },
            'rgb_means': {
                'R': float(np.mean(rgb_array[:, 0])),
                'G': float(np.mean(rgb_array[:, 1])),
                'B': float(np.mean(rgb_array[:, 2]))
            }
        }
        
        # Add image distribution if metadata available
        images = set(point.metadata.get('image_name', 'unknown') for point in color_points)
        stats['unique_images'] = len(images)
        stats['images'] = sorted(list(images))
        
        # Add sample set distribution
        sample_sets = set(point.metadata.get('sample_set', 'unknown') for point in color_points)
        stats['unique_sample_sets'] = len(sample_sets)
        stats['sample_sets'] = sorted(list(sample_sets))
        
        return stats


def demo_data_bridge():
    """Demonstrate the data bridge capabilities."""
    print("=== StampZ Color Data Bridge Demo ===\n")
    
    # Initialize data bridge
    bridge = ColorDataBridge()
    
    # Get available sample sets
    sample_sets = bridge.get_available_sample_sets()
    print(f"📊 Available Sample Sets: {len(sample_sets)}")
    for i, sample_set in enumerate(sample_sets[:5]):  # Show first 5
        print(f"  {i+1}. {sample_set}")
    
    if sample_sets:
        # Load data from first available sample set
        first_set = sample_sets[0]
        print(f"\n🔍 Loading data from '{first_set}'...")
        
        color_points = bridge.load_color_points_from_database(first_set, limit=20)
        
        if color_points:
            print(f"✅ Loaded {len(color_points)} color points")
            
            # Show sample data
            print(f"\n📋 Sample Color Points:")
            for i, point in enumerate(color_points[:3]):
                print(f"  {i+1}. {point.id}")
                print(f"     RGB: {point.rgb}")
                print(f"     L*a*b*: {point.lab}")
                print(f"     Ternary: {point.ternary_coords}")
                print(f"     Image: {point.metadata.get('image_name', 'N/A')}")
            
            # Get summary statistics
            print(f"\n📈 Summary Statistics:")
            stats = bridge.get_summary_stats(color_points)
            print(f"  Total Points: {stats['total_points']}")
            print(f"  Unique Images: {stats['unique_images']}")
            print(f"  L*a*b* Means: L*={stats['lab_means']['L*']:.1f}, a*={stats['lab_means']['a*']:.1f}, b*={stats['lab_means']['b*']:.1f}")
            
            # Group by image
            print(f"\n🗂️ Grouping by Image:")
            image_groups = bridge.group_by_image(color_points)
            for image_name, points in list(image_groups.items())[:3]:
                print(f"  {image_name}: {len(points)} points")
            
            # Demonstrate filtering
            print(f"\n🔍 Filtering Demonstrations:")
            
            # Filter by lightness
            bright_points = bridge.filter_by_lightness(color_points, min_l=60)
            print(f"  Bright colors (L* > 60): {len(bright_points)} points")
            
            # Filter by chroma
            saturated_points = bridge.filter_by_chroma(color_points, min_chroma=30)
            print(f"  Saturated colors (Chroma > 30): {len(saturated_points)} points")
            
            print(f"\n✅ Data Bridge Demo Complete!")
            print(f"🔗 Ready to feed existing StampZ data into advanced visualization!")
        
        else:
            print("❌ No color points loaded")
    else:
        print("❌ No sample sets found")
        print("💡 Tip: Run some color analysis first to generate databases")


if __name__ == "__main__":
    demo_data_bridge()