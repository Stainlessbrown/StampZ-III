#!/usr/bin/env python3
"""
Ternary Plot Export Module

Handles all export functionality for ternary plots including:
- Plot image export (PNG, etc.)
- Data export in various formats (CSV, ODS, Excel)
- Plot_3D template format compliance
- Cluster data export with proper structure
- Database export for persistence

This module fixes the Plot_3D format compatibility issues by ensuring:
- Headers match Plot_3D template exactly
- Centroid data is properly placed in rows 2-7 (protected area)  
- Individual point data starts at row 8
- Proper normalized coordinate format (0-1 range)
"""

import logging
import os
import csv
import numpy as np
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class TernaryExportManager:
    """Manages all export functionality for ternary plots."""
    
    # Plot_3D template column headers (exact match required)
    PLOT3D_COLUMNS = [
        "Xnorm", "Ynorm", "Znorm", "DataID", "Cluster", "ΔE", "Marker", "Color",
        "Centroid_X", "Centroid_Y", "Centroid_Z", "Sphere", "Radius"
    ]
    
    def __init__(self, ternary_window_ref=None):
        """
        Initialize the export manager.
        
        Args:
            ternary_window_ref: Reference to the parent ternary window
        """
        self.ternary_window = ternary_window_ref
        
    def export_data(self, color_points: List[Any], clusters: Dict[int, List[Any]] = None, 
                   cluster_manager: Any = None, sample_set_name: str = "Ternary_Analysis") -> bool:
        """
        Export ternary data in multiple formats with Plot_3D compatibility.
        
        Args:
            color_points: List of ColorPoint objects
            clusters: Dictionary of cluster data
            cluster_manager: TernaryClusterManager instance for centroid calculation
            sample_set_name: Name for the exported file
            
        Returns:
            True if export successful
        """
        try:
            if not color_points:
                messagebox.showwarning("No Data", "No data to export.")
                return False
            
            # File selection dialog
            filename = filedialog.asksaveasfilename(
                title="Export Ternary Data",
                defaultextension=".ods",
                filetypes=[
                    ("Plot_3D Template (ODS)", "*.ods"),
                    ("Plot_3D Template (Excel)", "*.xlsx"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ],
                initialfile=f"ternary_export_{sample_set_name.replace(' ', '_')}.ods"
            )
            
            if not filename:
                return False
            
            # Export based on file extension
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in ['.ods', '.xlsx']:
                # Plot_3D template format with proper structure
                success = self._export_plot3d_format(filename, color_points, clusters, cluster_manager)
            elif file_ext == '.csv':
                # CSV format (legacy compatibility)
                success = self._export_csv_format(filename, color_points, clusters, cluster_manager)
            else:
                # Default to CSV for unknown extensions
                success = self._export_csv_format(filename, color_points, clusters, cluster_manager)
            
            if success:
                # Force dialog to appear on top (macOS fix)
                if self.ternary_window and hasattr(self.ternary_window, 'window'):
                    # Create a proper dialog parent
                    root = self.ternary_window.window
                    root.lift()
                    root.focus_force()
                    root.attributes('-topmost', True)
                    root.update()
                    
                    # Use after() to ensure dialog appears on top
                    def show_export_dialog():
                        root.attributes('-topmost', False)
                        messagebox.showinfo("Export Complete", 
                            f"Exported {len(color_points)} data points to:\n{os.path.basename(filename)}\n\n"
                            f"Format: {'Plot_3D Template' if file_ext in ['.ods', '.xlsx'] else 'CSV'}")
                    
                    root.after(200, show_export_dialog)
                else:
                    messagebox.showinfo("Export Complete", 
                        f"Exported {len(color_points)} data points to:\n{os.path.basename(filename)}\n\n"
                        f"Format: {'Plot_3D Template' if file_ext in ['.ods', '.xlsx'] else 'CSV'}")
                
                if self.ternary_window:
                    self.ternary_window._update_status(f"Exported {len(color_points)} rows to {os.path.basename(filename)}")
                
            return success
            
        except Exception as e:
            logger.exception("Export failed")
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
            return False
    
    def save_plot_image(self, figure, sample_set_name: str = "Ternary_Analysis") -> bool:
        """
        Save the current plot as an image file.
        
        Args:
            figure: Matplotlib figure object
            sample_set_name: Name for the saved file
            
        Returns:
            True if save successful
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"ternary_plot_{sample_set_name}_{timestamp}.png"
            
            filename = filedialog.asksaveasfilename(
                title="Save Ternary Plot",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=default_name
            )
            
            if filename:
                figure.savefig(filename, dpi=300, bbox_inches='tight')
                
                if self.ternary_window:
                    self.ternary_window._update_status(f"Plot saved: {os.path.basename(filename)}")
                
                return True
            
            return False
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save plot: {e}")
            return False
    
    def save_as_database(self, color_points: List[Any], sample_set_name: str) -> bool:
        """
        Save current color data as a new Ternary database.
        
        Args:
            color_points: List of ColorPoint objects
            sample_set_name: Current sample set name for default naming
            
        Returns:
            True if save successful
        """
        try:
            if not color_points:
                messagebox.showwarning("No Data", "No color data to save.")
                return False
            
            # Ask for new database name
            import tkinter.simpledialog
            
            # Clean up the sample set name for database naming
            clean_name = sample_set_name.replace('External: ', '').replace('.ods', '').replace('.xlsx', '')
            if clean_name.startswith('Ternary_'):
                clean_name = clean_name[8:]  # Remove existing "Ternary_" prefix
            if clean_name.startswith('ternary_'):  # Handle lowercase version
                clean_name = clean_name[8:]
            
            new_name = tkinter.simpledialog.askstring(
                "Save As Database",
                "Enter name for new Ternary database:",
                initialvalue=f"Ternary_{clean_name}"
            )
            
            if not new_name:
                return False
            
            # Ensure it starts with Ternary_
            if not new_name.startswith("Ternary_"):
                new_name = f"Ternary_{new_name}"
            
            # Create new database and save all current color points
            from utils.color_analysis_db import ColorAnalysisDB
            new_db = ColorAnalysisDB(new_name)
            
            # Save each color point to the new database
            saved_count = 0
            for point in color_points:
                try:
                    # Extract image name and coordinate point from ID
                    if '_pt' in point.id:
                        image_name, pt_part = point.id.rsplit('_pt', 1)
                        coord_point = int(pt_part)
                    else:
                        image_name = point.id
                        coord_point = 1
                    
                    # Create measurement set
                    set_id = new_db.create_measurement_set(image_name)
                    
                    # Save color measurement with proper RGB values
                    success = new_db.save_color_measurement(
                        set_id=set_id,
                        coordinate_point=coord_point,
                        x_pos=0.0,  # Default position
                        y_pos=0.0,  # Default position
                        l_value=point.lab[0],
                        a_value=point.lab[1],
                        b_value=point.lab[2],
                        rgb_r=point.rgb[0],
                        rgb_g=point.rgb[1],
                        rgb_b=point.rgb[2],
                        replace_existing=True
                    )
                    
                    if success:
                        saved_count += 1
                        
                except Exception as point_error:
                    logger.warning(f"Failed to save point {point.id}: {point_error}")
                    continue
            
            # Update parent window if available
            if self.ternary_window:
                self.ternary_window.sample_set_name = new_name
                self.ternary_window.window.title(f"RGB Ternary Analysis - {new_name}")
                
                # Force dialog to appear on top (macOS fix)
                root = self.ternary_window.window
                root.lift()
                root.focus_force()
                root.attributes('-topmost', True)
                root.update()
                
                # Use after() to ensure dialog appears on top
                def show_save_dialog():
                    root.attributes('-topmost', False)
                    messagebox.showinfo(
                        "Database Saved",
                        f"Successfully saved {saved_count} color points to database '{new_name}'.\n\n"
                        "The ternary window is now using the new database.\n"
                        "Any changes you make will be saved to this database only."
                    )
                
                root.after(200, show_save_dialog)
            else:
                messagebox.showinfo(
                    "Database Saved",
                    f"Successfully saved {saved_count} color points to database '{new_name}'.\n\n"
                    "The ternary window is now using the new database.\n"
                    "Any changes you make will be saved to this database only."
                )
            
            if self.ternary_window:
                self.ternary_window._update_status(f"Saved as new database: {new_name} ({saved_count} points)")
            
            return True
            
        except Exception as e:
            logger.exception("Failed to save as database")
            messagebox.showerror("Save Error", f"Failed to save as database: {e}")
            return False
    
    # Private helper methods
    
    def _export_plot3d_format(self, filename: str, color_points: List[Any], 
                             clusters: Dict[int, List[Any]] = None, 
                             cluster_manager: Any = None) -> bool:
        """
        Export data in Plot_3D template format with proper structure.
        
        This fixes the format compatibility issues by ensuring:
        - Row 1: Headers (exactly matching Plot_3D template)
        - Rows 2-7: Cluster centroids (protected area)
        - Rows 8+: Individual data points
        """
        try:
            import pandas as pd
            
            # Calculate cluster centroids if available
            cluster_centroids = {}
            if clusters and cluster_manager:
                cluster_centroids = cluster_manager.calculate_centroids()
            elif clusters:
                # Fallback centroid calculation
                cluster_centroids = self._calculate_legacy_centroids(clusters)
            
            # Use Plot_3D's exact approach: construct data exactly as it should appear in the sheet
            logger.info(f"Creating Plot_3D format with {len(cluster_centroids)} centroids and {len(color_points)} data points")
            
            # Create the complete data structure exactly as it should appear
            sheet_data = []
            
            # Row 0: Headers
            sheet_data.append(self.PLOT3D_COLUMNS)
            
            # Rows 1-6: Cluster centroids (exactly 6 rows for rows 2-7 in display)
            centroid_rows = self._create_centroid_rows(cluster_centroids)
            logger.info(f"Created {len(centroid_rows)} centroid rows for protected area")
            
            # Add centroid rows
            for centroid_row in centroid_rows:
                sheet_data.append(centroid_row)
            
            # Fill remaining centroid slots with empty rows (ensure exactly 6 centroid rows)
            while len(sheet_data) < 7:  # 1 header + 6 centroid rows = 7 total
                empty_row = [''] * len(self.PLOT3D_COLUMNS)
                sheet_data.append(empty_row)
            
            # Rows 7+: Individual data points (starts at display row 8)
            data_rows = self._create_data_rows(color_points, clusters, cluster_manager)
            logger.info(f"Created {len(data_rows)} individual data rows")
            
            # Add data rows
            for data_row in data_rows:
                sheet_data.append(data_row)
            
            logger.info(f"Total sheet structure: {len(sheet_data)} rows (1 header + 6 centroid slots + {len(data_rows)} data rows)")
            
            # DEBUG: Show exact structure
            logger.info(f"Sheet structure verification:")
            for i, row in enumerate(sheet_data[:10]):  # Show first 10 rows
                if i == 0:
                    logger.info(f"  Row {i+1} (Header): {row[3] if len(row) > 3 else 'N/A'}")
                elif i < 7:
                    dataID = row[3] if len(row) > 3 else ''
                    centroid_x = row[8] if len(row) > 8 else ''
                    logger.info(f"  Row {i+1} (Centroid): DataID='{dataID}', Centroid_X='{centroid_x}' {'(FILLED)' if centroid_x else '(EMPTY)'}")
                else:
                    dataID = row[3] if len(row) > 3 else 'N/A'
                    centroid_x = row[8] if len(row) > 8 else 'N/A'
                    logger.info(f"  Row {i+1} (Data): DataID='{dataID}', Centroid_X='{centroid_x}'")
            
            # Create DataFrame exactly like Plot_3D does - using data WITHOUT headers
            df = pd.DataFrame(sheet_data[1:], columns=self.PLOT3D_COLUMNS)  # Skip header row
            
            # Apply Plot_3D's exact processing
            df = df.replace('', np.nan).dropna(how='all').fillna('')
            df.reset_index(drop=True, inplace=True)
            
            # Sort by _row_order to ensure correct sequence (should already be correct, but just in case)
            df = df.sort_values('_row_order').reset_index(drop=True)
            
            # Remove the temporary _row_order column before export
            df = df.drop(columns=['_row_order'])
            
            # DEBUG: Show the order after DataFrame creation
            logger.info(f"Row order after DataFrame creation:")
            for i in range(min(10, len(df))):
                row = df.iloc[i]
                dataID = row['DataID'] if 'DataID' in row else 'N/A'
                centroid_x = row['Centroid_X'] if 'Centroid_X' in row else 'N/A'
                logger.info(f"  Row {i}: DataID='{dataID}', Centroid_X='{centroid_x}'")
            
            # Export based on file extension - CRITICAL: Preserve row order
            file_ext = os.path.splitext(filename)[1].lower()
            
            logger.info(f"Exporting to {file_ext} format, preserving row order...")
            
            if file_ext == '.ods':
                # OpenDocument Spreadsheet
                try:
                    # Ensure no sorting occurs during export
                    df.to_excel(filename, engine='odf', index=False, sort_columns=False)
                    logger.info(f"ODS export successful")
                except Exception as ods_error:
                    logger.warning(f"ODS export failed: {ods_error}, trying openpyxl")
                    # Fallback to xlsx
                    xlsx_filename = filename.replace('.ods', '.xlsx')
                    df.to_excel(xlsx_filename, index=False, engine='openpyxl')
                    filename = xlsx_filename
                    logger.info(f"Fallback XLSX export successful")
                    
            elif file_ext == '.xlsx':
                # Excel format - ensure no sorting occurs
                df.to_excel(filename, index=False, engine='openpyxl')
                logger.info(f"XLSX export successful")
            
            # Log the actual structure for verification
            logger.info(f"=== Final DataFrame Structure ===")
            logger.info(f"Total DataFrame rows: {len(df)}")
            logger.info(f"Headers: {list(df.columns)}")
            
            # Show final structure to verify centroids are in correct positions
            for i in range(min(12, len(df))):  # Show first 12 rows
                row = df.iloc[i]
                dataID = row['DataID'] if 'DataID' in row else ''
                cluster = row['Cluster'] if 'Cluster' in row else ''
                centroid_x = row['Centroid_X'] if 'Centroid_X' in row else ''
                
                row_type = "CENTROID" if (i < 6 and centroid_x != '') else "DATA" if i >= 6 else "EMPTY"
                logger.info(f"  DataFrame row {i} → Excel row {i+2}: [{row_type}] DataID='{dataID}', Cluster='{cluster}', Centroid_X='{centroid_x}'")
            
            logger.info(f"Export complete: {len(cluster_centroids)} centroids (rows 2-7), {len(color_points)} data points (rows 8+)")
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to export Plot_3D format: {e}")
            return False
    
    def _export_csv_format(self, filename: str, color_points: List[Any], 
                          clusters: Dict[int, List[Any]] = None, 
                          cluster_manager: Any = None) -> bool:
        """Export data in legacy CSV format for backward compatibility."""
        try:
            # Legacy CSV headers
            headers = ["DataID", "R", "G", "B", "L*", "a*", "b*", "Ternary_X", "Ternary_Y", "Ternary_Z", "Cluster"]
            
            # Prepare export data
            export_data = [headers]
            
            for point in color_points:
                x = point.ternary_coords[0] if hasattr(point, 'ternary_coords') else ''
                y = point.ternary_coords[1] if hasattr(point, 'ternary_coords') else ''
                z = point.ternary_coords[2] if hasattr(point, 'ternary_coords') and len(point.ternary_coords) > 2 else 0.0
                
                # Find cluster assignment
                cluster = ''
                if clusters:
                    if cluster_manager:
                        cluster = cluster_manager.get_cluster_assignment(point)
                    else:
                        # Fallback cluster lookup
                        for cluster_id, cluster_points in clusters.items():
                            if point in cluster_points:
                                cluster = str(cluster_id)
                                break
                
                row = [
                    point.id,
                    f"{point.rgb[0]:.2f}", f"{point.rgb[1]:.2f}", f"{point.rgb[2]:.2f}",
                    f"{point.lab[0]:.2f}", f"{point.lab[1]:.2f}", f"{point.lab[2]:.2f}",
                    f"{x:.6f}" if x != '' else '', f"{y:.6f}" if y != '' else '', f"{z:.6f}",
                    cluster
                ]
                export_data.append(row)
            
            # Write CSV file
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in export_data:
                    writer.writerow(row)
            
            logger.info(f"Exported CSV format: {len(color_points)} data points")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to export CSV format: {e}")
            return False
    
    def _create_centroid_rows(self, cluster_centroids: Dict[int, Dict[str, Any]]) -> List[List[Any]]:
        """Create centroid rows for the protected area (rows 2-7)."""
        centroid_rows = []
        
        logger.info(f"Creating centroid rows from {len(cluster_centroids)} clusters: {list(cluster_centroids.keys())}")
        
        # Sort cluster IDs for consistent ordering
        sorted_cluster_ids = sorted(cluster_centroids.keys())
        
        for cluster_id in sorted_cluster_ids:
            if len(centroid_rows) >= 6:  # Maximum 6 centroid rows
                logger.warning(f"Skipping cluster {cluster_id} - already have 6 centroid rows")
                break
                
            centroid_data = cluster_centroids[cluster_id]
            
            logger.info(f"Creating centroid row for cluster {cluster_id}: "
                       f"centroid=({centroid_data['centroid_x']:.4f}, {centroid_data['centroid_y']:.4f}, {centroid_data['centroid_z']:.4f}), "
                       f"color={centroid_data['sphere_color']}, points={centroid_data['point_count']}")
            
            # Create centroid row in Plot_3D format
            # CRITICAL: Centroids have coordinates in BOTH main columns (A-C) AND centroid columns (I-K)
            centroid_row = [
                centroid_data['centroid_x'],     # Xnorm (A) - centroid coordinates in main columns
                centroid_data['centroid_y'],     # Ynorm (B) - centroid coordinates in main columns
                centroid_data['centroid_z'],     # Znorm (C) - centroid coordinates in main columns
                f"Cluster_{cluster_id}",         # DataID (D) - cluster identifier
                str(cluster_id),                 # Cluster (E) - cluster ID
                '',                              # ΔE (F) - empty for centroids
                'x',                             # Marker (G) - centroid marker
                centroid_data['sphere_color'],   # Color (H) - cluster color
                centroid_data['centroid_x'],     # Centroid_X (I) - SAME coordinates as main columns
                centroid_data['centroid_y'],     # Centroid_Y (J) - SAME coordinates as main columns
                centroid_data['centroid_z'],     # Centroid_Z (K) - SAME coordinates as main columns
                centroid_data['sphere_color'],   # Sphere (L) - cluster sphere color
                centroid_data['sphere_radius']   # Radius (M) - sphere radius
            ]
            
            logger.debug(f"Centroid row for cluster {cluster_id}: Main coords=({centroid_data['centroid_x']:.4f},{centroid_data['centroid_y']:.4f},{centroid_data['centroid_z']:.4f}), Centroid coords=({centroid_data['centroid_x']:.4f},{centroid_data['centroid_y']:.4f},{centroid_data['centroid_z']:.4f})")
            
            centroid_rows.append(centroid_row)
            
            logger.debug(f"Created centroid row for cluster {cluster_id}: "
                        f"({centroid_data['centroid_x']:.4f}, {centroid_data['centroid_y']:.4f}, {centroid_data['centroid_z']:.4f})")
        
        return centroid_rows
    
    def _create_data_rows(self, color_points: List[Any], clusters: Dict[int, List[Any]] = None,
                         cluster_manager: Any = None) -> List[List[Any]]:
        """Create individual data point rows (starting at row 8)."""
        data_rows = []
        
        for point in color_points:
            # Convert L*a*b* to normalized 0-1 range (Plot_3D format)
            l_norm = max(0.0, min(1.0, point.lab[0] / 100.0))        # L*: 0-100 → 0-1
            a_norm = max(0.0, min(1.0, (point.lab[1] + 127.5) / 255.0))  # a*: -127.5 to +127.5 → 0-1  
            b_norm = max(0.0, min(1.0, (point.lab[2] + 127.5) / 255.0))  # b*: -127.5 to +127.5 → 0-1
            
            # Find cluster assignment with debugging
            cluster_assignment = ''
            if clusters:
                if cluster_manager:
                    cluster_assignment = cluster_manager.get_cluster_assignment(point)
                    logger.debug(f"Cluster manager assigned point {point.id} to cluster: {cluster_assignment}")
                else:
                    # Fallback cluster lookup
                    for cluster_id, cluster_points in clusters.items():
                        if point in cluster_points:
                            cluster_assignment = str(cluster_id)
                            logger.debug(f"Fallback assigned point {point.id} to cluster: {cluster_assignment}")
                            break
                
                if not cluster_assignment:
                    logger.warning(f"Point {point.id} not assigned to any cluster")
            
            # Create data row in Plot_3D format
            # NOTE: Individual points do NOT have centroid data - that goes only in rows 2-7
            data_row = [
                round(l_norm, 6),      # Xnorm (L* normalized)
                round(a_norm, 6),      # Ynorm (a* normalized) 
                round(b_norm, 6),      # Znorm (b* normalized)
                point.id,              # DataID
                cluster_assignment,    # Cluster (just assignment, no centroid data)
                '',                    # ΔE (empty - to be calculated)
                '.',                   # Marker (default)
                'blue',                # Color (default)
                '',                    # Centroid_X (EMPTY - centroids only in rows 2-7)
                '',                    # Centroid_Y (EMPTY - centroids only in rows 2-7)
                '',                    # Centroid_Z (EMPTY - centroids only in rows 2-7)
                '',                    # Sphere (EMPTY - centroids only in rows 2-7)
                ''                     # Radius (EMPTY - centroids only in rows 2-7)
            ]
            
            data_rows.append(data_row)
        
        logger.info(f"Created {len(data_rows)} data rows in Plot_3D format")
        return data_rows
    
    def _calculate_legacy_centroids(self, clusters: Dict[int, List[Any]]) -> Dict[int, Dict[str, Any]]:
        """Calculate cluster centroids using legacy method as fallback."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            cluster_centroids = {}
            colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
            
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
                
                logger.debug(f"Legacy centroid for cluster {cluster_id}: "
                           f"L:{centroid_l_norm:.4f}, a:{centroid_a_norm:.4f}, b:{centroid_b_norm:.4f}")
            
            return cluster_centroids
            
        except Exception as e:
            logger.exception(f"Failed to calculate legacy centroids: {e}")
            return {}