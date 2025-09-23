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
                                      limit: Optional[int] = None) -> Tuple[List[ColorPoint], str]:
        """Load color data from database and convert to normalized ColorPoint objects.
        
        Args:
            sample_set_name: Name of the sample set database
            image_filter: Optional filter to only include specific images
            limit: Optional[int] on number of points to load
            
        Returns:
            Tuple of (List of ColorPoint objects with normalized values, "NORMALIZED" format string)
        """
        color_points = []
        
        try:
            # Connect to the color analysis database
            db = ColorAnalysisDB(sample_set_name)
            
            # Get all measurements, optionally filtered
            all_measurements = db.get_all_measurements()
            
            # Separate CENTROIDS entries from regular measurements
            centroid_entries = [m for m in all_measurements if m.get('image_name') == 'CENTROIDS']
            regular_measurements = [m for m in all_measurements if m.get('image_name') != 'CENTROIDS']
            
            print(f"Found {len(regular_measurements)} regular measurements and {len(centroid_entries)} centroid entries")
            
            # Create lookup table for centroid data by cluster_id
            centroid_lookup = {}
            for centroid in centroid_entries:
                cluster_id = centroid.get('cluster_id')
                if cluster_id is not None:
                    centroid_lookup[cluster_id] = {
                        'centroid_x': centroid.get('centroid_x'),
                        'centroid_y': centroid.get('centroid_y'), 
                        'centroid_z': centroid.get('centroid_z'),
                        'sphere_color': centroid.get('sphere_color'),
                        'sphere_radius': centroid.get('sphere_radius')
                    }
            
            print(f"Created centroid lookup for clusters: {list(centroid_lookup.keys())}")
            
            # Use regular measurements for processing
            measurements = regular_measurements
            
            if image_filter:
                # Filter measurements by image name (case-insensitive partial match)
                measurements = [m for m in measurements 
                              if image_filter.lower() in m.get('image_name', '').lower()]
            
            if limit:
                measurements = measurements[:limit]
            
            print(f"Processing {len(measurements)} measurements from {sample_set_name}")
            print(f"🔧 NORMALIZATION: Converting all data to normalized format for unified Ternary/Plot_3D compatibility")
            
            # Convert each measurement to a normalized ColorPoint
            for i, measurement in enumerate(measurements):
                try:
                    # Generate unique ID for this point
                    image_name = measurement.get('image_name', 'unknown')
                    coord_point = measurement.get('coordinate_point', i+1)
                    point_id = f"{image_name}_pt{coord_point}"
                    
                    # Extract L*a*b* values and normalize them for unified system
                    raw_l = measurement.get('l_value', 50.0)
                    raw_a = measurement.get('a_value', 0.0)
                    raw_b = measurement.get('b_value', 0.0)
                    
                    # Auto-detect and normalize L*a*b* values based on range
                    if (0.0 <= raw_l <= 1.0 and abs(raw_a) <= 1.0 and abs(raw_b) <= 1.0):
                        # Already normalized - use as actual L*a*b*
                        lab = (raw_l * 100.0, (raw_a * 255.0) - 127.5, (raw_b * 255.0) - 127.5)
                        if i < 3:
                            print(f"✅ NORMALIZED INPUT: Point {i} - ({raw_l:.3f}, {raw_a:.3f}, {raw_b:.3f}) → Lab ({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})")
                    else:
                        # Actual L*a*b* values - use as-is
                        lab = (raw_l, raw_a, raw_b)
                        if i < 3:
                            print(f"✅ ACTUAL LAB INPUT: Point {i} - Lab ({raw_l:.1f}, {raw_a:.1f}, {raw_b:.1f})")
                    
                    # Get stored RGB values
                    stored_rgb = (
                        measurement.get('rgb_r', 0.0),
                        measurement.get('rgb_g', 0.0),
                        measurement.get('rgb_b', 0.0)
                    )
                    
                    # Check if RGB values are valid (not all zeros)
                    if stored_rgb == (0.0, 0.0, 0.0) or max(stored_rgb) == 0.0:
                        # RGB values are invalid - calculate from L*a*b* values
                        if i < 5:  # Debug first few points
                            print(f"DEBUG: Point {i} ({point_id}) - Invalid RGB {stored_rgb}, Lab values: {lab}")
                        
                        try:
                            from colorspacious import cspace_convert
                            # Convert L*a*b* to normalized RGB (0-1 range for unified system)
                            rgb = cspace_convert(lab, "CIELab", "sRGB1")
                            # Clamp to valid range
                            rgb = tuple(max(0.0, min(1.0, c)) for c in rgb)
                            
                            if i < 5:  # Debug first few conversions
                                print(f"DEBUG: Colorspacious - Lab{lab} -> Normalized RGB{rgb}")
                        except (ImportError, Exception) as conv_error:
                            if i < 3:
                                print(f"DEBUG: Colorspacious failed ({conv_error}), using fallback conversion")
                            
                            # Improved fallback: Better Lab->RGB approximation that preserves color variety
                            l, a, b_lab = lab
                            
                            # Normalize L* (0-100) to reasonable lightness
                            l_norm = max(0, min(100, l)) / 100.0
                            
                            # Handle a* and b* values more carefully
                            # a* (green-red): negative=green, positive=red  
                            # b* (blue-yellow): negative=blue, positive=yellow
                            a_val = max(-128, min(127, a))  # -128 to +127
                            b_val_lab = max(-128, min(127, b_lab))  # -128 to +127
                            
                            # Convert Lab to normalized RGB (0-1) using approximation
                            # Base lightness component (normalized)
                            base = l_norm  # Already 0-1
                            
                            # a* affects Red-Green balance (normalize influence)
                            a_influence = a_val / 128.0  # -1 to +1
                            b_influence = b_val_lab / 128.0  # -1 to +1
                            
                            if a_influence > 0:  # Positive a* = more red
                                r = base + (a_influence * 0.3)  # Boost red
                                g = base - (a_influence * 0.15)  # Reduce green
                            else:  # Negative a* = more green  
                                r = base + (a_influence * 0.15)  # Reduce red
                                g = base - (a_influence * 0.3)  # Boost green
                            
                            # b* affects Blue-Yellow balance
                            if b_influence > 0:  # Positive b* = more yellow
                                r += (b_influence * 0.15)
                                g += (b_influence * 0.15)  
                                b_val = base - (b_influence * 0.3)  # Reduce blue
                            else:  # Negative b* = more blue
                                r += (b_influence * 0.1)  # Reduce red
                                g += (b_influence * 0.1)  # Reduce green
                                b_val = base - (b_influence * 0.4)  # Boost blue
                            
                            # Clamp to valid normalized RGB range (0-1)
                            r = max(0.0, min(1.0, r))
                            g = max(0.0, min(1.0, g))
                            b_val = max(0.0, min(1.0, b_val))
                            rgb = (r, g, b_val)
                            
                            if i < 3:
                                print(f"DEBUG: Fallback Lab{lab} -> L*:{l:.1f}, a*:{a_val}, b*:{b_val_lab} -> Normalized RGB{rgb}")
                    else:
                        # Normalize stored RGB values (convert from 0-255 to 0-1 if needed)
                        if max(stored_rgb) > 1.0:
                            # Stored as 0-255, normalize to 0-1
                            rgb = tuple(c / 255.0 for c in stored_rgb)
                            if i < 3:
                                print(f"DEBUG: Point {i} normalized stored RGB: {stored_rgb} → {rgb}")
                        else:
                            # Already normalized
                            rgb = stored_rgb
                            if i < 3:
                                print(f"DEBUG: Point {i} using normalized stored RGB: {rgb}")
                    
                    # point_id already created above for consistency
                    
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
                    cluster_id = measurement.get('cluster_id')
                    if cluster_id is not None:
                        metadata['cluster_id'] = cluster_id
                        
                        # Note: Centroid coordinates are NOT assigned here to individual data points.
                        # Centroid data should only appear in dedicated centroid rows (2-5, etc.) 
                        # as determined by the datasheet formatting process.
                        # Each cluster gets exactly ONE centroid entry in its designated row.
                    
                    # Add Delta E information if available  
                    if 'delta_e' in measurement:
                        metadata['delta_e'] = measurement['delta_e']
                    
                    # Store marker/color preferences from database
                    # Database columns: marker_preference, color_preference
                    metadata['marker_preference'] = measurement.get('marker_preference', '.')  # Plot_3D primary
                    metadata['color_preference'] = measurement.get('color_preference', 'blue')  # Plot_3D primary
                    
                    # Create aliases for Ternary plot (same data, different key names)
                    metadata['ternary_marker'] = metadata['marker_preference']       # Ternary alias
                    metadata['ternary_marker_color'] = metadata['color_preference']  # Ternary alias
                    
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
            
            print(f"Successfully converted {len(color_points)} measurements to normalized ColorPoint objects")
            return color_points, "NORMALIZED"
            
        except Exception as e:
            print(f"Error loading color points from {sample_set_name}: {e}")
            return [], 'UNKNOWN'
    
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
    
    def save_color_points_to_database(self, sample_set_name: str, color_points: List[ColorPoint]) -> bool:
        """Save color points back to the internal database.
        
        Args:
            sample_set_name: Name of the sample set database to save to
            color_points: List of ColorPoint objects to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from .color_analysis_db import ColorAnalysisDB
            
            print(f"💾 SAVING TO DATABASE: {sample_set_name}")
            print(f"   Points to save: {len(color_points)}")
            
            # Connect to the database
            db = ColorAnalysisDB(sample_set_name)
            
            # Get all existing measurements to preserve data integrity
            existing_measurements = db.get_all_measurements()
            print(f"   Existing measurements in database: {len(existing_measurements)}")
            
            if not existing_measurements:
                print("❌ No existing measurements found - cannot save to empty database")
                return False
            
            # Create a mapping of point IDs to color points for easy lookup
            point_lookup = {}
            for point in color_points:
                # Extract image name and coordinate from point ID (e.g., "image_pt5" -> image="image", coord=5)
                parts = point.id.split('_pt')
                if len(parts) == 2:
                    image_name = parts[0]
                    try:
                        coord_point = int(parts[1])
                        key = (image_name, coord_point)
                        point_lookup[key] = point
                    except ValueError:
                        print(f"Warning: Could not parse coordinate from point ID: {point.id}")
            
            print(f"   Created lookup table for {len(point_lookup)} points")
            
            # Update existing measurements with new data from Plot_3D
            updated_count = 0
            for measurement in existing_measurements:
                image_name = measurement.get('image_name', '')
                coord_point = measurement.get('coordinate_point', 0)
                key = (image_name, coord_point)
                
                if key in point_lookup:
                    point = point_lookup[key]
                    
                    # Update database measurement with Plot_3D changes
                    # Preserve original L*a*b* and RGB values, but update preferences and analysis data
                    
                    # Extract preferences from ColorPoint metadata (saved by Plot_3D)
                    marker_pref = point.metadata.get('marker_preference', '.')
                    color_pref = point.metadata.get('color_preference', 'blue')
                    cluster_id = point.metadata.get('cluster_id', None)
                    delta_e = point.metadata.get('delta_e', None)
                    
                    # Extract centroid and sphere data
                    centroid_x = point.metadata.get('centroid_x', None)
                    centroid_y = point.metadata.get('centroid_y', None) 
                    centroid_z = point.metadata.get('centroid_z', None)
                    sphere_color = point.metadata.get('sphere_color', None)
                    sphere_radius = point.metadata.get('sphere_radius', None)
                    
                    # Update the measurement record using the correct method
                    try:
                        db.update_plot3d_extended_values(
                            image_name,
                            coord_point,
                            cluster_id=cluster_id,
                            delta_e=delta_e,
                            centroid_x=centroid_x,
                            centroid_y=centroid_y,
                            centroid_z=centroid_z,
                            sphere_color=sphere_color,
                            sphere_radius=sphere_radius,
                            marker=marker_pref,
                            color=color_pref
                        )
                        updated_count += 1
                        
                        if updated_count <= 3:  # Debug first few updates
                            print(f"   Updated {image_name}_pt{coord_point}: marker={marker_pref}, color={color_pref}, cluster={cluster_id}")
                        
                    except Exception as update_error:
                        print(f"Warning: Failed to update measurement {measurement['id']}: {update_error}")
            
            print(f"✅ Successfully updated {updated_count} measurements in database: {sample_set_name}")
            return updated_count > 0
            
        except Exception as e:
            print(f"❌ Error saving to database {sample_set_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _detect_database_format(self, measurements: List[Dict[str, Any]], sample_set_name: str) -> str:
        """Intelligently detect whether database contains normalized (0-1) or actual Lab values.
        
        Args:
            measurements: List of measurement dictionaries from database
            sample_set_name: Name of the sample set for logging
            
        Returns:
            'NORMALIZED', 'ACTUAL_LAB', 'MIXED', or 'UNKNOWN'
        """
        if not measurements:
            return 'UNKNOWN'
        
        # Analyze a sample of measurements to determine format
        sample_size = min(50, len(measurements))  # Check up to 50 measurements
        sample_measurements = measurements[:sample_size]
        
        normalized_count = 0
        actual_lab_count = 0
        
        for measurement in sample_measurements:
            l_val = measurement.get('l_value', 50.0)
            a_val = measurement.get('a_value', 0.0)
            b_val = measurement.get('b_value', 0.0)
            
            # Skip invalid/null values
            if l_val is None or a_val is None or b_val is None:
                continue
                
            # Normalized format indicators:
            # - L* in range 0-1 (typical: 0.2-0.9)
            # - a* and b* in range 0-1 (with 0.5 being neutral)
            if (0.0 <= l_val <= 1.0 and 0.0 <= a_val <= 1.0 and 0.0 <= b_val <= 1.0):
                normalized_count += 1
            
            # Actual Lab format indicators:
            # - L* in range 0-100 (typical: 20-90)
            # - a* and b* in range -127.5 to +127.5 (typical: -50 to +50)
            elif (0.0 <= l_val <= 100.0 and -127.5 <= a_val <= 127.5 and -127.5 <= b_val <= 127.5 and
                  (l_val > 1.0 or abs(a_val) > 1.0 or abs(b_val) > 1.0)):
                actual_lab_count += 1
        
        total_analyzed = normalized_count + actual_lab_count
        
        if total_analyzed == 0:
            return 'UNKNOWN'
        
        # Determine format based on predominant pattern
        normalized_ratio = normalized_count / total_analyzed
        actual_lab_ratio = actual_lab_count / total_analyzed
        
        print(f"🔍 FORMAT ANALYSIS: {sample_set_name}")
        print(f"   Analyzed: {total_analyzed}/{sample_size} measurements")
        print(f"   Normalized pattern: {normalized_count} ({normalized_ratio*100:.1f}%)")
        print(f"   Actual Lab pattern: {actual_lab_count} ({actual_lab_ratio*100:.1f}%)")
        
        # Decision thresholds
        if normalized_ratio >= 0.8:
            return 'NORMALIZED'
        elif actual_lab_ratio >= 0.8:
            return 'ACTUAL_LAB'
        elif normalized_ratio > 0.1 and actual_lab_ratio > 0.1:
            return 'MIXED'
        else:
            return 'UNKNOWN'


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
        
        color_points, db_format = bridge.load_color_points_from_database(first_set, limit=20)
        print(f"📋 Database Format Detected: {db_format}")
        
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