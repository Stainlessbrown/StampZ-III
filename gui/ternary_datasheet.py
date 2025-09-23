#!/usr/bin/env python3
"""
Ternary Plot Datasheet Integration Module

Handles all datasheet communication and integration between ternary plots
and the realtime Plot_3D spreadsheet including:
- Datasheet creation and population  
- Data synchronization (bidirectional)
- Cluster data management in datasheets
- Ternary-specific preference handling
- Format conversion (Lab to Plot_3D normalized)

This module provides a clean interface between ternary plots and datasheets
while maintaining separation from the complex realtime Plot_3D sheet.
"""

import logging
import os
import tkinter as tk
from tkinter import messagebox
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TernaryDatasheetManager:
    """Manages datasheet integration for ternary plots."""
    
    def __init__(self, ternary_window_ref=None):
        """
        Initialize the datasheet manager.
        
        Args:
            ternary_window_ref: Reference to the parent ternary window
        """
        self.ternary_window = ternary_window_ref
        self.datasheet_ref = None
        self.external_file_path = None
        
    def open_realtime_datasheet(self, color_points: List[Any], clusters: Dict[int, List[Any]] = None, 
                              sample_set_name: str = "Ternary_Analysis", external_file_path: str = None) -> Any:
        """
        Open a realtime datasheet with current ternary data.
        
        Args:
            color_points: List of ColorPoint objects
            clusters: Dictionary of cluster data
            sample_set_name: Name for the sample set
            external_file_path: Path to external file if applicable
            
        Returns:
            Reference to created datasheet or None on failure
        """
        try:
            from gui.realtime_plot3d_sheet import RealtimePlot3DSheet
            
            # DEBUG: Check what we received
            logger.info(f"📊 DATASHEET MANAGER DEBUG: Received {len(color_points) if color_points else 0} color points")
            logger.info(f"📊 DATASHEET MANAGER DEBUG: Sample set name: {sample_set_name}")
            logger.info(f"📊 DATASHEET MANAGER DEBUG: Has clusters: {bool(clusters)}")
            
            if not color_points:
                logger.warning("⚠️ DATASHEET MANAGER DEBUG: No color points received! This is the problem.")
                # Still show dialog but with 0 points to help user understand
                self._show_datasheet_ready_dialog(0, "")
                return None
            
            # Create datasheet name based on current data
            if external_file_path:
                datasheet_name = f"Ternary: {os.path.basename(external_file_path, '.ods').replace('.xlsx', '')}"
            else:
                datasheet_name = f"Ternary: {sample_set_name}"
            
            logger.info(f"📊 DATASHEET MANAGER DEBUG: Creating datasheet with name: {datasheet_name}")
            
            # Create realtime datasheet (don't load initial data, we'll populate it)
            datasheet = RealtimePlot3DSheet(
                parent=self.ternary_window.window if self.ternary_window else None,
                sample_set_name=datasheet_name,
                load_initial_data=False
            )
            
            logger.info(f"📊 DATASHEET MANAGER DEBUG: About to populate datasheet with {len(color_points)} points")
            
            # Populate datasheet with current ternary data converted to Plot_3D format
            self.populate_datasheet_with_ternary_data(datasheet, color_points, clusters)
            
            # Store references for bidirectional updates
            self.datasheet_ref = datasheet
            self.external_file_path = external_file_path
            
            # Setup bidirectional integration
            self._setup_datasheet_integration(datasheet)
            
            # Show datasheet ready dialog
            cluster_info = ""
            if clusters:
                cluster_info = f"\\n• {len(clusters)} K-means clusters with centroids in rows 2-7"
            
            self._show_datasheet_ready_dialog(len(color_points), cluster_info)
            
            logger.info(f"Realtime datasheet opened: {datasheet_name}")
            return datasheet
            
        except Exception as e:
            logger.exception(f"Failed to open realtime datasheet: {e}")
            messagebox.showerror("Datasheet Error", f"Failed to open realtime datasheet: {e}")
            return None
    
    def populate_datasheet_with_ternary_data(self, datasheet: Any, color_points: List[Any], 
                                           clusters: Dict[int, List[Any]] = None):
        """
        Populate realtime datasheet with ternary data in Plot_3D format.
        
        Args:
            datasheet: RealtimePlot3DSheet instance
            color_points: List of ColorPoint objects
            clusters: Dictionary of cluster data
        """
        try:
            if not color_points:
                logger.warning("No color points to populate datasheet")
                return
                
            # Calculate cluster centroids if clusters exist
            cluster_centroids = {}
            if clusters:
                cluster_centroids = self._calculate_cluster_centroids_for_datasheet(clusters)
            
            # Convert ColorPoint objects to Plot_3D normalized format
            plot3d_data_rows = self._convert_color_points_to_plot3d_format(color_points, clusters)
            
            # Setup proper Plot_3D sheet structure
            self._setup_datasheet_structure(datasheet, plot3d_data_rows, cluster_centroids)
            
            logger.info(f"Populated datasheet with {len(plot3d_data_rows)} data rows in Plot_3D format")
            
        except Exception as e:
            logger.exception(f"Failed to populate datasheet: {e}")
            raise Exception(f"Datasheet population failed: {e}")
    
    def sync_from_datasheet(self) -> List[Any]:
        """
        Sync data from linked datasheet back to ternary plot.
        
        Returns:
            List of updated ColorPoint objects
        """
        try:
            if not self.datasheet_ref or not hasattr(self.datasheet_ref, 'sheet'):
                logger.warning("🔄 SYNC DEBUG: No datasheet linked for sync")
                return []
                
            logger.info("🔄 SYNC DEBUG: Starting datasheet sync process")
            
            # Get current data from datasheet (skip header and protected rows)
            sheet_data = self.datasheet_ref.sheet.get_sheet_data()
            logger.info(f"🔄 SYNC DEBUG: Retrieved {len(sheet_data)} total rows from datasheet")
            
            # Skip header row (index 0) and protected rows (indices 1-6), start from index 7
            data_rows = sheet_data[7:] if len(sheet_data) > 7 else []
            logger.info(f"🔄 SYNC DEBUG: Processing {len(data_rows)} data rows (excluding header & centroid rows)")
            
            if not data_rows:
                logger.warning("🔄 SYNC DEBUG: No data rows in linked datasheet")
                return []
                
            # Debug: Show first few rows to verify data structure
            for i, row in enumerate(data_rows[:3]):
                logger.info(f"🔄 SYNC DEBUG: Row {i+8}: {row[:8]}...")  # Show first 8 columns
                
            # Convert datasheet data back to ColorPoint objects
            updated_color_points = self._convert_datasheet_to_color_points(data_rows)
            
            logger.info(f"🔄 SYNC DEBUG: Successfully synced {len(updated_color_points)} points from datasheet")
            return updated_color_points
            
        except Exception as e:
            logger.exception(f"🔄 SYNC DEBUG: Failed to sync from datasheet: {e}")
            return []
    
    def refresh_datasheet_cluster_data(self, color_points: List[Any], clusters: Dict[int, List[Any]],
                                     cluster_manager: Any = None):
        """
        Refresh cluster data in the open datasheet.
        
        Args:
            color_points: List of ColorPoint objects
            clusters: Dictionary of cluster data  
            cluster_manager: TernaryClusterManager instance for centroid calculation
        """
        try:
            if not (self.datasheet_ref and hasattr(self.datasheet_ref, 'sheet')):
                return
                
            if not clusters:
                return
                
            # Update cluster column for each data row
            total_rows = self.datasheet_ref.sheet.get_total_rows()
            
            for row_idx in range(7, total_rows):  # Start at row 8 (index 7) where data begins
                try:
                    row_data = self.datasheet_ref.sheet.get_row_data(row_idx)
                    if not row_data or len(row_data) < 5:  # Need at least DataID column
                        continue
                        
                    data_id = row_data[3]  # DataID is column 4 (index 3)
                    if not data_id:
                        continue
                        
                    # Find matching color point
                    matching_point = None
                    for point in color_points:
                        if point.id == data_id:
                            matching_point = point
                            break
                            
                    if matching_point:
                        # Get cluster assignment and centroid data
                        cluster_assignment = ''
                        centroid_x = centroid_y = centroid_z = ''
                        sphere_color = sphere_radius = ''
                        
                        if cluster_manager:
                            # Use cluster manager to get assignment
                            cluster_assignment = cluster_manager.get_cluster_assignment(matching_point)
                            
                            if cluster_assignment:
                                # Get centroid data from cluster manager
                                centroids = cluster_manager.calculate_centroids()
                                cluster_id_int = int(cluster_assignment)
                                
                                if cluster_id_int in centroids:
                                    centroid_data = centroids[cluster_id_int]
                                    centroid_x = centroid_data['centroid_x']
                                    centroid_y = centroid_data['centroid_y']
                                    centroid_z = centroid_data['centroid_z']
                                    sphere_color = centroid_data['sphere_color']
                                    sphere_radius = str(centroid_data['sphere_radius'])
                        else:
                            # Fallback to legacy cluster lookup
                            for cluster_id, cluster_points in clusters.items():
                                if matching_point in cluster_points:
                                    cluster_assignment = str(cluster_id)
                                    
                                    # Calculate cluster centroid (legacy)
                                    cluster_l = [cp.lab[0] for cp in cluster_points]
                                    cluster_a = [cp.lab[1] for cp in cluster_points]
                                    cluster_b = [cp.lab[2] for cp in cluster_points]
                                    
                                    centroid_x = round(sum(cluster_l) / len(cluster_l) / 100.0, 6)
                                    centroid_y = round((sum(cluster_a) / len(cluster_a) + 127.5) / 255.0, 6)
                                    centroid_z = round((sum(cluster_b) / len(cluster_b) + 127.5) / 255.0, 6)
                                    
                                    sphere_color = '#808080'  # Gray fallback
                                    sphere_radius = '0.02'
                                    break
                                
                        # Update the datasheet cells
                        if cluster_assignment:
                            self.datasheet_ref.sheet.set_cell_data(row_idx, 4, cluster_assignment)  # Cluster
                            if centroid_x:
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 8, centroid_x)   # Centroid_X
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 9, centroid_y)   # Centroid_Y
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 10, centroid_z)  # Centroid_Z
                            if sphere_color:
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 11, sphere_color)  # Sphere color
                                self.datasheet_ref.sheet.set_cell_data(row_idx, 12, sphere_radius) # Sphere radius
                            
                except Exception as row_error:
                    logger.warning(f"Failed to update cluster data for row {row_idx}: {row_error}")
                    continue
                    
            # Refresh the datasheet display
            if hasattr(self.datasheet_ref.sheet, 'refresh'):
                self.datasheet_ref.sheet.refresh()
                
            logger.info(f"Updated cluster data in datasheet for {len(clusters)} clusters")
            
        except Exception as e:
            logger.exception(f"Failed to refresh datasheet cluster data: {e}")
    
    def handle_datasheet_changes(self, row_data: List[Any]) -> bool:
        """
        Handle changes from the datasheet and save to ternary-specific columns.
        
        Args:
            row_data: List of row data from datasheet
            
        Returns:
            True if handled successfully
        """
        try:
            if len(row_data) < 8:  # Need at least DataID, Marker, Color columns
                return False
                
            data_id = row_data[3]  # DataID column
            marker = row_data[6] if len(row_data) > 6 else '.'  # Marker column
            color = row_data[7] if len(row_data) > 7 else 'blue'  # Color column
            
            # Check if this is a cluster centroid row (in restricted area)
            if data_id.startswith('Cluster_'):
                cluster_id = data_id.replace('Cluster_', '')
                return self._handle_cluster_centroid_changes(cluster_id, row_data)
            else:
                # Regular data point - save individual preferences
                return self._save_ternary_preferences_to_db(data_id, marker, color)
                
        except Exception as e:
            logger.exception(f"Error handling datasheet changes: {e}")
            return False
    
    def close_datasheet(self):
        """Clean up datasheet references."""
        self.datasheet_ref = None
        self.external_file_path = None
        logger.info("Datasheet references cleared")
    
    def is_datasheet_linked(self) -> bool:
        """Check if a datasheet is currently linked."""
        return self.datasheet_ref is not None
    
    def bring_datasheet_to_front(self):
        """Bring the linked datasheet to front."""
        try:
            if self.datasheet_ref and hasattr(self.datasheet_ref, 'window'):
                self.datasheet_ref.window.lift()
                self.datasheet_ref.window.focus_force()
                self.datasheet_ref.window.attributes('-topmost', True)
                self.datasheet_ref.window.after(100, 
                    lambda: self.datasheet_ref.window.attributes('-topmost', False))
        except Exception as e:
            logger.warning(f"Could not bring datasheet to front: {e}")
    
    # Private helper methods
    
    def _setup_datasheet_integration(self, datasheet: Any):
        """Setup bidirectional integration with the datasheet."""
        try:
            # Add back-reference to datasheet for navigation
            datasheet.ternary_window_ref = self.ternary_window
            
            # Add callback for datasheet to save ternary preferences when changes are made
            datasheet.ternary_save_callback = self.handle_datasheet_changes
            
            # Add "Return to Ternary" button to datasheet toolbar if possible
            if (hasattr(datasheet, 'plot3d_btn') and 
                hasattr(datasheet.plot3d_btn, 'master') and
                self.ternary_window):
                return_btn = tk.Button(
                    datasheet.plot3d_btn.master, 
                    text="Return to Ternary", 
                    command=self.ternary_window._bring_ternary_to_front
                )
                return_btn.pack(side=tk.LEFT, padx=5)
                
        except Exception as e:
            logger.warning(f"Failed to setup complete datasheet integration: {e}")
    
    def _calculate_cluster_centroids_for_datasheet(self, clusters: Dict[int, List[Any]]) -> Dict[int, Dict[str, Any]]:
        """Calculate cluster centroids in Plot_3D format for datasheet population."""
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
                
                logger.debug(f"Cluster {cluster_id} centroid - L:{centroid_l_norm:.4f}, "
                           f"a:{centroid_a_norm:.4f}, b:{centroid_b_norm:.4f} ({len(cluster_points)} points)")
            
            return cluster_centroids
            
        except Exception as e:
            logger.exception(f"Failed to calculate cluster centroids: {e}")
            return {}
    
    def _convert_color_points_to_plot3d_format(self, color_points: List[Any], 
                                             clusters: Dict[int, List[Any]] = None) -> List[List[Any]]:
        """Convert ColorPoint objects to Plot_3D normalized format."""
        plot3d_data_rows = []
        
        for point in color_points:
            # Convert L*a*b* to normalized 0-1 range (Plot_3D format) - CORRECTED
            l_norm = max(0.0, min(1.0, point.lab[0] / 100.0))  # L*: 0-100 → 0-1 (CORRECT)
            a_norm = max(0.0, min(1.0, (point.lab[1] + 127.5) / 255.0))  # a*: -127.5 to +127.5 → 0-1 (CORRECT)
            b_norm = max(0.0, min(1.0, (point.lab[2] + 127.5) / 255.0))  # b*: -127.5 to +127.5 → 0-1 (CORRECT)
            
            # DEBUG: Log the conversion to verify correctness
            logger.debug(f"🔄 LAB CONVERT DEBUG: {point.id} Lab=({point.lab[0]:.2f}, {point.lab[1]:.2f}, {point.lab[2]:.2f}) → Norm=({l_norm:.6f}, {a_norm:.6f}, {b_norm:.6f})")
            
            # Store original data in metadata for accuracy preservation
            if not hasattr(point, 'metadata'):
                point.metadata = {}
            point.metadata['original_rgb'] = point.rgb
            point.metadata['original_ternary_coords'] = point.ternary_coords
            
            # Find cluster assignment
            cluster_assignment = ''
            if clusters:
                for cluster_id, cluster_points in clusters.items():
                    if point in cluster_points:
                        cluster_assignment = str(cluster_id)
                        break
            
            # Create row data in Plot_3D format (no centroid data in individual rows)
            row_data = [
                round(l_norm, 6),      # Xnorm (L*)
                round(a_norm, 6),      # Ynorm (a*) 
                round(b_norm, 6),      # Znorm (b*)
                point.id,              # DataID
                cluster_assignment,    # Cluster (just assignment, no centroid data)
                '',                    # ∆E (empty - to be calculated)
                '.',                   # Marker (default)
                'blue',                # Color (default)
                '',                    # Centroid_X (empty - centroids go in restricted area)
                '',                    # Centroid_Y (empty - centroids go in restricted area)
                '',                    # Centroid_Z (empty - centroids go in restricted area)
                '',                    # Sphere (empty - centroids go in restricted area)
                ''                     # Radius (empty - centroids go in restricted area)
            ]
            
            plot3d_data_rows.append(row_data)
        
        return plot3d_data_rows
    
    def _setup_datasheet_structure(self, datasheet: Any, plot3d_data_rows: List[List[Any]], 
                                  cluster_centroids: Dict[int, Dict[str, Any]]):
        """Setup the proper Plot_3D sheet structure."""
        try:
            if not hasattr(datasheet, 'sheet'):
                logger.warning("Datasheet sheet widget not found")
                return
            
            # Calculate total rows needed: 7 reserved + data + buffer
            total_rows_needed = 7 + len(plot3d_data_rows) + 10
            
            # Clear existing sheet and create proper structure
            current_rows = datasheet.sheet.get_total_rows()
            if current_rows > 0:
                datasheet.sheet.delete_rows(0, current_rows)
            
            # Create empty rows for proper structure
            empty_rows = [[''] * len(datasheet.PLOT3D_COLUMNS)] * total_rows_needed
            datasheet.sheet.insert_rows(rows=empty_rows, idx=0)
            
            # Set headers in row 1 (index 0)
            datasheet.sheet.set_row_data(0, values=datasheet.PLOT3D_COLUMNS)
            
            # Populate rows 2-7 (indices 1-6) with cluster centroids in protected area
            logger.info(f"Populating {len(cluster_centroids)} centroids in rows 2-7")
            centroid_row_index = 1  # Start at row 2 (index 1)
            
            for cluster_id, centroid_data in cluster_centroids.items():
                if centroid_row_index >= 7:  # Don't exceed reserved area
                    logger.warning(f"Too many clusters, skipping cluster {cluster_id}")
                    break
                
                # Create centroid row with cluster data
                centroid_row = [
                    centroid_data['centroid_x'],     # Xnorm (cluster centroid L*)
                    centroid_data['centroid_y'],     # Ynorm (cluster centroid a*)
                    centroid_data['centroid_z'],     # Znorm (cluster centroid b*)
                    f"Cluster_{cluster_id}",         # DataID (cluster identifier)
                    str(cluster_id),                 # Cluster (cluster ID)
                    '',                              # ΔE (empty)
                    'x',                             # Marker (centroid marker)
                    centroid_data['sphere_color'],   # Color (cluster color)
                    centroid_data['centroid_x'],     # Centroid_X (self-reference)
                    centroid_data['centroid_y'],     # Centroid_Y (self-reference)
                    centroid_data['centroid_z'],     # Centroid_Z (self-reference)
                    centroid_data['sphere_color'],   # Sphere (cluster sphere color)
                    centroid_data['sphere_radius']   # Radius (sphere radius)
                ]
                
                datasheet.sheet.set_row_data(centroid_row_index, values=centroid_row)
                centroid_row_index += 1
            
            logger.info(f"Populated {len(cluster_centroids)} cluster centroids in rows 2-{centroid_row_index}")
            
            # Insert individual data points starting at row 8 (index 7)
            for i, row_data in enumerate(plot3d_data_rows):
                row_index = 7 + i  # Start at row 8 (display row 8)
                datasheet.sheet.set_row_data(row_index, values=row_data)
            
            logger.info(f"Populated {len(plot3d_data_rows)} data rows starting at row 8")
            
            # Apply proper formatting and validation
            if hasattr(datasheet, '_apply_formatting'):
                datasheet._apply_formatting()
            if hasattr(datasheet, '_setup_validation'):
                datasheet._setup_validation()
            
            # Force refresh the datasheet display
            if hasattr(datasheet.sheet, 'refresh'):
                datasheet.sheet.refresh()
            elif hasattr(datasheet.sheet, 'update'):
                datasheet.sheet.update()
            
        except Exception as e:
            logger.exception(f"Failed to setup datasheet structure: {e}")
    
    def _convert_datasheet_to_color_points(self, data_rows: List[List[Any]]) -> List[Any]:
        """Convert datasheet data back to ColorPoint objects."""
        from utils.advanced_color_plots import ColorPoint
        updated_color_points = []
        
        logger.info(f"🔄 CONVERT DEBUG: Converting {len(data_rows)} datasheet rows to ColorPoint objects")
        
        for row_idx, row in enumerate(data_rows):
            try:
                # Parse Plot_3D format: [Xnorm, Ynorm, Znorm, DataID, Cluster, ΔE, Marker, Color, ...]
                if len(row) < 4 or not row[3]:  # Skip incomplete rows or rows without DataID
                    logger.debug(f"🔄 CONVERT DEBUG: Skipping incomplete row {row_idx+8}: {len(row)} columns, DataID={row[3] if len(row) > 3 else 'N/A'}")
                    continue
                    
                x_norm = float(row[0]) if row[0] != '' else 0.0  # L* normalized
                y_norm = float(row[1]) if row[1] != '' else 0.0  # a* normalized 
                z_norm = float(row[2]) if row[2] != '' else 0.0  # b* normalized
                data_id = str(row[3]) if row[3] != '' else f"Point_{len(updated_color_points)}"
                
                # Extract marker information from datasheet
                marker_style = str(row[6]) if len(row) > 6 and row[6] != '' else '.'
                marker_color = str(row[7]) if len(row) > 7 and row[7] != '' else 'blue'
                
                # Convert HEX colors to user-friendly names
                if marker_color.startswith('#'):
                    original_color = marker_color
                    marker_color = self._hex_to_color_name(marker_color)
                    logger.debug(f"🔄 CONVERT DEBUG: Converting HEX color {original_color} -> {marker_color} for {data_id}")
                elif marker_color not in ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'black', 'cyan', 'magenta', 'white']:
                    # Fallback for unknown color names
                    logger.debug(f"🔄 CONVERT DEBUG: Unknown color '{marker_color}' for {data_id}, using 'blue'")
                    marker_color = 'blue'
                
                logger.debug(f"🔄 CONVERT DEBUG: Processing point {data_id}: marker={marker_style}, color={marker_color}")
                
                # Save ternary preferences to database for persistence
                self._save_ternary_preferences_to_db(data_id, marker_style, marker_color)
                
                # Convert normalized values back to L*a*b*
                l_star = x_norm * 100.0  # 0-1 → 0-100
                a_star = (y_norm * 255.0) - 127.5  # 0-1 → -127.5 to +127.5
                b_star = (z_norm * 255.0) - 127.5  # 0-1 → -127.5 to +127.5
                
                # Convert normalized Plot_3D values back to actual L*a*b* values
                l_star = x_norm * 100.0  # 0-1 → 0-100
                a_star = (y_norm * 255.0) - 127.5  # 0-1 → -127.5 to +127.5
                b_star = (z_norm * 255.0) - 127.5  # 0-1 → -127.5 to +127.5
                
                # DEBUG: Log the reverse conversion to identify corruption
                logger.debug(f"🔄 REVERSE LAB DEBUG: {data_id} Norm=({x_norm:.6f}, {y_norm:.6f}, {z_norm:.6f}) → Lab=({l_star:.2f}, {a_star:.2f}, {b_star:.2f})")
                
                # Use original RGB if available, otherwise convert from Lab
                rgb = self._get_rgb_from_lab_or_original(l_star, a_star, b_star, len(updated_color_points))
                
                # DEBUG: Log RGB calculation result
                logger.debug(f"🔄 RGB CALC DEBUG: {data_id} RGB=({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})")
                
                # Calculate ternary coordinates
                ternary_coords = self.ternary_window.ternary_plotter.rgb_to_ternary(rgb) if self.ternary_window else [0.333, 0.333]
                
                # DEBUG: Log ternary coordinate calculation
                logger.debug(f"🔄 TERNARY CALC DEBUG: {data_id} Ternary=({ternary_coords[0]:.6f}, {ternary_coords[1]:.6f})")
                
                # Create metadata
                metadata = {
                    'source': 'datasheet_sync',
                    'original_rgb': rgb,
                    'marker': marker_style,
                    'marker_color': marker_color
                }
                
                # Create ColorPoint object
                point = ColorPoint(
                    id=data_id,
                    lab=(l_star, a_star, b_star),
                    rgb=rgb,
                    ternary_coords=ternary_coords,
                    metadata=metadata
                )
                
                logger.debug(f"🔄 CONVERT DEBUG: Created ColorPoint {data_id} with metadata: {point.metadata}")
                
                updated_color_points.append(point)
                
            except Exception as row_error:
                logger.warning(f"Failed to convert datasheet row: {row_error}")
                continue
        
        return updated_color_points
    
    def _get_rgb_from_lab_or_original(self, l_star: float, a_star: float, b_star: float, point_index: int) -> Tuple[float, float, float]:
        """Get RGB from Lab conversion or original if available."""
        # Try to use original RGB from existing points first
        if (self.ternary_window and hasattr(self.ternary_window, 'color_points') and 
            point_index < len(self.ternary_window.color_points)):
            original_point = self.ternary_window.color_points[point_index]
            
            if (hasattr(original_point, 'metadata') and 
                'original_rgb' in original_point.metadata):
                rgb_result = original_point.metadata['original_rgb']
                # SAFETY: Check for corrupted RGB values (all zeros)
                if sum(rgb_result) > 1.0:  # Valid RGB should have some color
                    logger.debug(f"🔄 RGB ORIGINAL DEBUG: Using preserved RGB from metadata: {rgb_result}")
                    return rgb_result
                else:
                    logger.warning(f"🔄 RGB CORRUPTION DEBUG: Detected corrupted RGB in metadata {rgb_result}, falling back to Lab conversion")
            else:
                rgb_result = original_point.rgb
                # SAFETY: Check for corrupted RGB values (all zeros)
                if sum(rgb_result) > 1.0:  # Valid RGB should have some color
                    logger.debug(f"🔄 RGB ORIGINAL DEBUG: Using original RGB from point: {rgb_result}")
                    return rgb_result
                else:
                    logger.warning(f"🔄 RGB CORRUPTION DEBUG: Detected corrupted RGB in original point {rgb_result}, falling back to Lab conversion")
        
        # Convert L*a*b* to RGB
        try:
            from utils.color_conversions import lab_to_rgb
            rgb_result = lab_to_rgb((l_star, a_star, b_star))
            logger.debug(f"🔄 RGB LAB CONVERT DEBUG: Lab=({l_star:.2f}, {a_star:.2f}, {b_star:.2f}) → RGB={rgb_result}")
            return rgb_result
        except ImportError:
            # Fallback approximation
            l_norm = l_star / 100.0
            a_norm = (a_star + 127.5) / 255.0
            b_norm = (b_star + 127.5) / 255.0
            
            r = max(0, min(255, l_norm * 255 + (a_norm - 0.5) * 100))
            g = max(0, min(255, l_norm * 255 - (a_norm - 0.5) * 50 + (b_norm - 0.5) * 20))
            b = max(0, min(255, l_norm * 255 - (b_norm - 0.5) * 100))
            rgb_result = (r, g, b)
            logger.debug(f"🔄 RGB LAB FALLBACK DEBUG: Lab=({l_star:.2f}, {a_star:.2f}, {b_star:.2f}) → RGB={rgb_result}")
            return rgb_result
    
    def _hex_to_color_name(self, hex_color: str) -> str:
        """Convert HEX color codes to user-friendly color names."""
        if not hex_color or not hex_color.startswith('#'):
            return hex_color
            
        # Common HEX to color name mappings
        hex_to_name = {
            '#FF0000': 'red', '#DC143C': 'crimson', '#8B0000': 'darkred',
            '#00FF00': 'lime', '#008000': 'green', '#006400': 'darkgreen', 
            '#0000FF': 'blue', '#000080': 'navy', '#4169E1': 'royalblue',
            '#FFFF00': 'yellow', '#FFD700': 'gold', '#FFA500': 'orange',
            '#FF00FF': 'magenta', '#800080': 'purple', '#4B0082': 'indigo',
            '#00FFFF': 'cyan', '#008080': 'teal', '#20B2AA': 'lightseagreen',
            '#FFC0CB': 'pink', '#FF1493': 'deeppink', '#FF69B4': 'hotpink',
            '#A52A2A': 'brown', '#D2691E': 'chocolate', '#8B4513': 'saddlebrown',
            '#808080': 'gray', '#C0C0C0': 'silver', '#000000': 'black',
            '#FFFFFF': 'white', '#F5F5DC': 'beige', '#DDA0DD': 'plum'
        }
        
        # Try exact match first
        hex_upper = hex_color.upper()
        if hex_upper in hex_to_name:
            return hex_to_name[hex_upper]
            
        # Try to find closest match by converting hex to RGB and comparing
        try:
            # Parse hex color
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16) 
            b = int(hex_color[5:7], 16)
            
            # Simple color classification based on dominant RGB values
            if r > 200 and g < 100 and b < 100:
                return 'red'
            elif r < 100 and g > 200 and b < 100:
                return 'green'
            elif r < 100 and g < 100 and b > 200:
                return 'blue'
            elif r > 200 and g > 200 and b < 100:
                return 'yellow'
            elif r > 200 and g < 100 and b > 200:
                return 'magenta'
            elif r < 100 and g > 200 and b > 200:
                return 'cyan'
            elif r > 150 and g > 100 and b < 100:
                return 'orange'
            elif r > 100 and g < 100 and b > 100:
                return 'purple'
            elif r < 100 and g < 100 and b < 100:
                return 'black'
            elif r > 200 and g > 200 and b > 200:
                return 'white'
            elif abs(r - g) < 50 and abs(g - b) < 50 and abs(r - b) < 50:
                return 'gray'
            else:
                return 'gray'  # Default fallback
                
        except (ValueError, IndexError):
            logger.warning(f"Could not parse HEX color: {hex_color}")
            return 'gray'
    
    def _save_ternary_preferences_to_db(self, point_id: str, marker: str = None, color: str = None) -> bool:
        """Save ternary-specific marker/color preferences to database."""
        try:
            # Parse the point ID to get image_name and coordinate_point
            if '_pt' in point_id:
                image_name, pt_part = point_id.rsplit('_pt', 1)
                coordinate_point = int(pt_part)
            else:
                image_name = point_id
                coordinate_point = 1
            
            # Convert HEX colors to user-friendly names
            if color and color.startswith('#'):
                original_color = color
                color = self._hex_to_color_name(color)
                logger.debug(f"💡 COLOR CONVERT: {original_color} -> {color} for {point_id}")
            
            logger.debug(f"💾 DB SAVE DEBUG: Saving ternary prefs for {point_id} -> {image_name}:{coordinate_point} (marker={marker}, color={color})")
            
            # Use the database's ternary preference update method
            from utils.color_analysis_db import ColorAnalysisDB
            db = ColorAnalysisDB(self.ternary_window.sample_set_name if self.ternary_window else "Default")
            
            success = db.update_ternary_preferences(
                image_name=image_name,
                coordinate_point=coordinate_point,
                marker=marker,
                color=color
            )
            
            if success:
                logger.debug(f"💾 DB SAVE DEBUG: ✅ Saved ternary preferences for {point_id}: marker={marker}, color={color}")
            else:
                logger.warning(f"💾 DB SAVE DEBUG: ❌ Failed to save ternary preferences for {point_id}")
                
            return success
            
        except Exception as e:
            logger.exception(f"Error saving ternary preferences for {point_id}: {e}")
            return False
    
    def _handle_cluster_centroid_changes(self, cluster_id: str, centroid_row_data: List[Any]) -> bool:
        """Handle changes to cluster centroid and propagate to all cluster members."""
        try:
            if not (self.ternary_window and hasattr(self.ternary_window, 'clusters') and self.ternary_window.clusters):
                return False
            
            cluster_id_int = int(cluster_id)
            if cluster_id_int not in self.ternary_window.clusters:
                return False
                
            # Extract centroid changes
            centroid_color = centroid_row_data[7] if len(centroid_row_data) > 7 else None
            
            logger.info(f"Cluster {cluster_id} centroid color changed: {centroid_color}")
            
            # Apply changes to all members of this cluster
            cluster_points = self.ternary_window.clusters[cluster_id_int]
            updated_count = 0
            
            for point in cluster_points:
                success = self._save_ternary_preferences_to_db(
                    point.id,
                    marker=None,  # Don't change individual markers
                    color=centroid_color if centroid_color else None
                )
                
                if success:
                    updated_count += 1
                    # Update point metadata for immediate refresh
                    if not hasattr(point, 'metadata'):
                        point.metadata = {}
                    if centroid_color:
                        point.metadata['marker_color'] = centroid_color
            
            logger.info(f"Updated {updated_count} points in cluster {cluster_id} with centroid changes")
            
            # Refresh the ternary plot to show changes
            if self.ternary_window and hasattr(self.ternary_window, '_refresh_plot_only'):
                self.ternary_window._refresh_plot_only()
            
            return True
            
        except Exception as e:
            logger.exception(f"Error handling cluster centroid changes for cluster {cluster_id}: {e}")
            return False
    
    def _show_datasheet_ready_dialog(self, point_count: int, cluster_info: str):
        """Show the datasheet ready dialog with proper focus."""
        try:
            # Force dialog to appear on top (macOS fix)
            if self.ternary_window and hasattr(self.ternary_window, 'window'):
                root = self.ternary_window.window
                root.lift()
                root.focus_force() 
                root.attributes('-topmost', True)
                root.update()
                
                # Use after() to ensure dialog appears on top
                def show_dialog():
                    root.attributes('-topmost', False)
                    messagebox.showinfo(
                        "Datasheet Ready",
                        f"Realtime datasheet opened with {point_count} color points.\n\n"
                        "• Data converted to Plot_3D normalized format (0-1 range)\n"
                        "• Use 'Launch Plot_3D' button in datasheet for 3D visualization\n"
                        "• Edit data in datasheet and refresh ternary plot to see changes"
                        f"{cluster_info}"
                    )
                
                root.after(200, show_dialog)
            else:
                messagebox.showinfo(
                    "Datasheet Ready",
                    f"Realtime datasheet opened with {point_count} color points.\n\n"
                    "• Data converted to Plot_3D normalized format (0-1 range)\n"
                    "• Use 'Launch Plot_3D' button in datasheet for 3D visualization\n"
                    "• Edit data in datasheet and refresh ternary plot to see changes"
                    f"{cluster_info}"
                )
        except Exception as e:
            logger.exception(f"Error showing datasheet ready dialog: {e}")
