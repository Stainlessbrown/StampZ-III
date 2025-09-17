#!/usr/bin/env python3
"""
Separate database utilities for color analysis data.
Each sample set gets its own database file for perfect data separation.
"""

import sqlite3
import os
import sys
import re
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ColorAnalysisDB:
    """Handle database operations for color analysis data."""
    
    def __init__(self, sample_set_name: str):
        """Initialize database connection for a specific sample set with standardized naming.
        
        Args:
            sample_set_name: Name of the sample set (becomes the database name)
        """
        from .naming_utils import standardize_name
        
        # Standardize the sample set name
        self.sample_set_name = standardize_name(sample_set_name)
        
        # Clean the standardized name for use as filename
        clean_name = self._clean_filename(self.sample_set_name)
        
        # Use STAMPZ_DATA_DIR environment variable if available (for packaged apps)
        stampz_data_dir = os.getenv('STAMPZ_DATA_DIR')
        if stampz_data_dir:
            color_data_dir = os.path.join(stampz_data_dir, "data", "color_analysis")
            print(f"DEBUG: Using persistent color analysis directory: {color_data_dir}")
        else:
            # Check if we're running in a PyInstaller bundle
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                # PyInstaller bundle - use the same directory as the main app (StampZ-III)
                user_data_dir = os.path.expanduser("~/Library/Application Support/StampZ-III")
                color_data_dir = os.path.join(user_data_dir, "data", "color_analysis")
                print(f"DEBUG: Using bundled app color analysis directory: {color_data_dir}")
            else:
                # Running from source - use relative path
                current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                color_data_dir = os.path.join(current_dir, "data", "color_analysis")
                print(f"DEBUG: Using development color analysis directory: {color_data_dir}")
        
        os.makedirs(color_data_dir, exist_ok=True)
        
        self.db_path = os.path.join(color_data_dir, f"{clean_name}.db")
        print(f"DEBUG: Color analysis database path: {self.db_path}")
        print(f"DEBUG: Database file exists: {os.path.exists(self.db_path)}")
        print(f"DEBUG: Database directory writable: {os.access(color_data_dir, os.W_OK)}")
        self._init_db()
    
    def _clean_filename(self, name: str) -> str:
        """Clean a name to be safe for use as a filename."""
        # Replace spaces and special characters with underscores
        clean = re.sub(r'[^\w\-_\.]', '_', name)
        # Remove multiple consecutive underscores
        clean = re.sub(r'_+', '_', clean)
        # Remove leading/trailing underscores
        clean = clean.strip('_')
        return clean
    
    def _init_db(self):
        """Initialize color analysis database tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Table for measurement sets
            conn.execute("""
                CREATE TABLE IF NOT EXISTS measurement_sets (
                    set_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_name TEXT NOT NULL,
                    measurement_date TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    description TEXT
                )
            """)
            
            # Table for color measurements (individual samples only)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS color_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_id INTEGER NOT NULL,
                    coordinate_point INTEGER NOT NULL,
                    x_position REAL NOT NULL,
                    y_position REAL NOT NULL,
                    l_value REAL NOT NULL,
                    a_value REAL NOT NULL,
                    b_value REAL NOT NULL,
                    rgb_r REAL NOT NULL,
                    rgb_g REAL NOT NULL,
                    rgb_b REAL NOT NULL,
                    sample_type TEXT,
                    sample_size TEXT,
                    sample_anchor TEXT,
                    measurement_date TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    notes TEXT,
                    FOREIGN KEY(set_id) REFERENCES measurement_sets(set_id)
                )
            """)
            
            # Add essential columns to existing databases
            cursor = conn.cursor()
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN sample_type TEXT")
                print("Added sample_type column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN sample_size TEXT")
                print("Added sample_size column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN sample_anchor TEXT")
                print("Added sample_anchor column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Add Plot_3D preference columns for marker/color persistence
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN marker_preference TEXT DEFAULT '.'")
                print("Added marker_preference column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN color_preference TEXT DEFAULT 'blue'")
                print("Added color_preference column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Add Plot_3D extended columns for complete integration
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN cluster_id INTEGER")
                print("Added cluster_id column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN delta_e REAL")
                print("Added delta_e column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN centroid_x REAL")
                print("Added centroid_x column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN centroid_y REAL")
                print("Added centroid_y column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN centroid_z REAL")
                print("Added centroid_z column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN sphere_color TEXT")
                print("Added sphere_color column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN sphere_radius REAL")
                print("Added sphere_radius column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN trendline_valid BOOLEAN DEFAULT 1")
                print("Added trendline_valid column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_set_point 
                ON color_measurements(set_id, coordinate_point)
            """)
    
    def create_measurement_set(self, image_name: str, description: str = None) -> int:
        """Create a new measurement set and return its ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if a measurement set with this image_name already exists
                cursor = conn.execute("""
                    SELECT set_id FROM measurement_sets WHERE image_name = ?
                """, (image_name,))
                existing = cursor.fetchone()
                
                if existing:
                    print(f"Using existing measurement set {existing[0]} for image '{image_name}'")
                    return existing[0]
                
                # Create new measurement set
                cursor = conn.execute("""
                    INSERT INTO measurement_sets (image_name, description)
                    VALUES (?, ?)
                """, (image_name, description))
                print(f"Created new measurement set {cursor.lastrowid} for image '{image_name}'")
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating measurement set: {e}")
            return None

    def save_color_measurement(
        self,
        set_id: int,
        coordinate_point: int,
        x_pos: float,
        y_pos: float,
        l_value: float,
        a_value: float,
        b_value: float,
        rgb_r: float,
        rgb_g: float,
        rgb_b: float,
        sample_type: Optional[str] = None,
        sample_size: Optional[str] = None,
        sample_anchor: Optional[str] = None,
        notes: Optional[str] = None,
        replace_existing: bool = True
    ) -> bool:
        """Save a color measurement with deduplication.
        
        Args:
            set_id: ID of the measurement set
            coordinate_point: Which coordinate point (1-based)
            x_pos, y_pos: Position coordinates
            l_value, a_value, b_value: CIE Lab values
            rgb_r, rgb_g, rgb_b: RGB values
            notes: Optional notes
            replace_existing: If True, replace existing measurements for same set/point
            
        Returns:
            True if save was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if replace_existing:
                    # Check if measurement already exists
                    cursor = conn.execute("""
                        SELECT id FROM color_measurements 
                        WHERE set_id = ? AND coordinate_point = ?
                    """, (set_id, coordinate_point))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing measurement
                        conn.execute("""
                            UPDATE color_measurements SET
                                x_position = ?, y_position = ?,
                                l_value = ?, a_value = ?, b_value = ?,
                                rgb_r = ?, rgb_g = ?, rgb_b = ?,
                                sample_type = ?, sample_size = ?, sample_anchor = ?,
                                measurement_date = datetime('now', 'localtime'),
                                notes = ?
                            WHERE set_id = ? AND coordinate_point = ?
                        """, (
                            x_pos, y_pos, l_value, a_value, b_value,
                            rgb_r, rgb_g, rgb_b, sample_type, sample_size, sample_anchor,
                            notes, set_id, coordinate_point
                        ))
                        print(f"Updated existing measurement for point {coordinate_point}")
                    else:
                        # Insert new measurement
                        conn.execute("""
                            INSERT INTO color_measurements (
                                set_id, coordinate_point, x_position, y_position,
                                l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                                sample_type, sample_size, sample_anchor, notes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            set_id, coordinate_point, x_pos, y_pos,
                            l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                            sample_type, sample_size, sample_anchor, notes
                        ))
                        print(f"Inserted new measurement for set {set_id} point {coordinate_point}")
                else:
                    # Always insert (old behavior) 
                    conn.execute("""
                        INSERT INTO color_measurements (
                            set_id, coordinate_point, x_position, y_position,
                            l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                            sample_type, sample_size, sample_anchor, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        set_id, coordinate_point, x_pos, y_pos,
                        l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                        sample_type, sample_size, sample_anchor, notes
                    ))
                    
                return True
        except sqlite3.Error as e:
            print(f"Error saving color measurement: {e}")
            return False
    
    def save_averaged_measurement(
        self,
        set_id: int,
        averaged_lab: tuple,
        averaged_rgb: tuple,
        source_measurements: List[dict],
        image_name: str,
        notes: Optional[str] = None
    ) -> bool:
        """This method is deprecated. Use AveragedColorAnalysisDB instead.
        
        Averaged measurements are now stored in separate databases.
        Use the color analyzer's save_averaged_measurement_from_samples method instead.
        """
        print(f"WARNING: save_averaged_measurement is deprecated. Averaged measurements should be saved to separate _averages database.")
        return False
    
    def get_all_measurements(self) -> List[dict]:
        """Get all color measurements for this sample set.
        
        Returns:
            List of measurement dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, check what columns exist in the table
                cursor = conn.execute("PRAGMA table_info(color_measurements)")
                columns = [row[1] for row in cursor.fetchall()]
                has_averaged_columns = all(col in columns for col in ['is_averaged', 'source_samples_count', 'source_sample_ids'])
                
                if has_averaged_columns:
                    # Query with averaged columns (for averaged databases)
                    cursor = conn.execute("""
                        SELECT 
                            m.id, m.set_id, s.image_name, m.measurement_date,
                            m.coordinate_point, m.x_position, m.y_position,
                            m.l_value, m.a_value, m.b_value, 
                            m.rgb_r, m.rgb_g, m.rgb_b,
                            m.sample_type, m.sample_size, m.sample_anchor,
                            m.notes, m.is_averaged, m.source_samples_count, m.source_sample_ids
                        FROM color_measurements m
                        JOIN measurement_sets s ON m.set_id = s.set_id
                        ORDER BY m.id
                    """)
                    
                    measurements = []
                    for row in cursor:
                        measurements.append({
                            'id': row[0],
                            'set_id': row[1],
                            'image_name': row[2],
                            'measurement_date': row[3],
                            'coordinate_point': row[4],
                            'x_position': row[5],
                            'y_position': row[6],
                            'l_value': row[7],
                            'a_value': row[8],
                            'b_value': row[9],
                            'rgb_r': row[10],
                            'rgb_g': row[11],
                            'rgb_b': row[12],
                            'sample_type': row[13],
                            'sample_size': row[14],
                            'sample_anchor': row[15],
                            'notes': row[16],
                            'is_averaged': bool(row[17]) if row[17] is not None else False,
                            'source_samples_count': row[18],
                            'source_sample_ids': row[19]
                        })
                else:
                    # Query without averaged columns (for main databases) - include all Plot_3D columns
                    cursor = conn.execute("""
                        SELECT 
                            m.id, m.set_id, s.image_name, m.measurement_date,
                            m.coordinate_point, m.x_position, m.y_position,
                            m.l_value, m.a_value, m.b_value, 
                            m.rgb_r, m.rgb_g, m.rgb_b,
                            m.sample_type, m.sample_size, m.sample_anchor,
                            m.notes, m.marker_preference, m.color_preference,
                            m.cluster_id, m.delta_e, m.centroid_x, m.centroid_y, m.centroid_z,
                            m.sphere_color, m.sphere_radius, m.trendline_valid
                        FROM color_measurements m
                        JOIN measurement_sets s ON m.set_id = s.set_id
                        ORDER BY m.id
                    """)
                    
                    measurements = []
                    for row in cursor:
                        measurements.append({
                            'id': row[0],
                            'set_id': row[1],
                            'image_name': row[2],
                            'measurement_date': row[3],
                            'coordinate_point': row[4],
                            'x_position': row[5],
                            'y_position': row[6],
                            'l_value': row[7],
                            'a_value': row[8],
                            'b_value': row[9],
                            'rgb_r': row[10],
                            'rgb_g': row[11],
                            'rgb_b': row[12],
                            'sample_type': row[13],
                            'sample_size': row[14],
                            'sample_anchor': row[15],
                            'notes': row[16],
                            'marker_preference': row[17] if len(row) > 17 and row[17] else '.',
                            'color_preference': row[18] if len(row) > 18 and row[18] else 'blue',
                            'cluster_id': row[19] if len(row) > 19 and row[19] is not None else None,
                            'delta_e': row[20] if len(row) > 20 and row[20] is not None else None,
                            'centroid_x': row[21] if len(row) > 21 and row[21] is not None else None,
                            'centroid_y': row[22] if len(row) > 22 and row[22] is not None else None,
                            'centroid_z': row[23] if len(row) > 23 and row[23] is not None else None,
                            'sphere_color': row[24] if len(row) > 24 and row[24] else '',
                            'sphere_radius': row[25] if len(row) > 25 and row[25] is not None else None,
                            'trendline_valid': bool(row[26]) if len(row) > 26 and row[26] is not None else True,
                            'is_averaged': False,  # Main DB only contains individual measurements
                            'source_samples_count': None,
                            'source_sample_ids': None
                        })
                
                return measurements
        except sqlite3.Error as e:
            print(f"Error retrieving measurements: {e}")
            return []
    
    def get_measurements_for_image(self, image_name: str) -> List[dict]:
        """Get all measurements for a specific image.
        
        Args:
            image_name: Name of the image
            
        Returns:
            List of measurement dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        m.id, m.set_id, s.image_name, m.measurement_date,
                        m.coordinate_point, m.x_position, m.y_position,
                        m.l_value, m.a_value, m.b_value,
                        m.rgb_r, m.rgb_g, m.rgb_b, m.notes
                    FROM color_measurements m
                    JOIN measurement_sets s ON m.set_id = s.set_id
                    WHERE s.image_name = ?
                    ORDER BY m.coordinate_point, m.measurement_date
                """, (image_name,))
                
                measurements = []
                for row in cursor:
                    measurements.append({
                        'id': row[0],
                        'set_id': row[1],
                        'image_name': row[2],
                        'measurement_date': row[3],
                        'coordinate_point': row[4],
                        'x_position': row[5],
                        'y_position': row[6],
                        'l_value': row[7],
                        'a_value': row[8],
                        'b_value': row[9],
                        'rgb_r': row[10],
                        'rgb_g': row[11],
                        'rgb_b': row[12],
                        'notes': row[13]
                    })
                
                return measurements
        except sqlite3.Error as e:
            print(f"Error retrieving measurements for image: {e}")
            return []
    
    def clear_all_measurements(self) -> bool:
        """Clear all measurements from this sample set's database.
        
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM color_measurements")
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error clearing measurements: {e}")
            return False
    
    def cleanup_duplicates(self) -> int:
        """Remove duplicate measurements, keeping only the latest for each image/coordinate point.
        
        Returns:
            Number of duplicate measurements removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, count total duplicates
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM color_measurements
                """)
                total_before = cursor.fetchone()[0]
                
                # Remove duplicates, keeping only the latest measurement_date for each image/point
                conn.execute("""
                    DELETE FROM color_measurements
                    WHERE id NOT IN (
                        SELECT id FROM (
                            SELECT id, 
                                   ROW_NUMBER() OVER (
                                       PARTITION BY set_id, coordinate_point 
                                       ORDER BY measurement_date DESC
                                   ) as rn
                            FROM color_measurements
                        ) ranked
                        WHERE rn = 1
                    )
                """)
                
                # Count remaining
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM color_measurements
                """)
                total_after = cursor.fetchone()[0]
                
                duplicates_removed = total_before - total_after
                print(f"Removed {duplicates_removed} duplicate measurements from {self.sample_set_name}")
                return duplicates_removed
                
        except sqlite3.Error as e:
            print(f"Error cleaning duplicates: {e}")
            return 0
    
    def update_marker_color_preferences(self, image_name: str, coordinate_point: int, 
                                       marker: str = None, color: str = None) -> bool:
        """Update marker and color preferences for a specific measurement.
        
        Args:
            image_name: Name of the image
            coordinate_point: Which coordinate point (1-based)
            marker: Marker preference (e.g., '.', 'o', '*')
            color: Color preference (e.g., 'blue', 'red', 'green')
            
        Returns:
            True if update was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build update query for only provided values
                update_parts = []
                values = []
                
                if marker is not None:
                    update_parts.append("marker_preference = ?")
                    values.append(marker)
                
                if color is not None:
                    update_parts.append("color_preference = ?")
                    values.append(color)
                
                if not update_parts:
                    return True  # Nothing to update
                
                # Add WHERE clause values
                values.extend([image_name, coordinate_point])
                
                query = f"""
                    UPDATE color_measurements 
                    SET {', '.join(update_parts)}
                    WHERE set_id = (
                        SELECT set_id FROM measurement_sets WHERE image_name = ?
                    ) AND coordinate_point = ?
                """
                
                cursor = conn.execute(query, values)
                updated_rows = cursor.rowcount
                
                if updated_rows > 0:
                    print(f"Updated preferences for {image_name} point {coordinate_point}: marker={marker}, color={color}")
                    return True
                else:
                    print(f"No measurement found for {image_name} point {coordinate_point}")
                    return False
                    
        except sqlite3.Error as e:
            print(f"Error updating marker/color preferences: {e}")
            return False
    
    def insert_new_measurement(self, image_name: str, coordinate_point: int,
                              x_pos: float, y_pos: float, l_value: float, a_value: float, b_value: float,
                              rgb_r: float = 0.0, rgb_g: float = 0.0, rgb_b: float = 0.0,
                              cluster_id: int = None, delta_e: float = None,
                              centroid_x: float = None, centroid_y: float = None, centroid_z: float = None,
                              sphere_color: str = None, sphere_radius: float = None,
                              marker: str = '.', color: str = 'blue',
                              sample_type: str = None, sample_size: str = None, sample_anchor: str = None,
                              notes: str = None, trendline_valid: bool = True) -> bool:
        """Insert a completely new measurement with all Plot_3D extended values.
        
        Args:
            image_name: Name of the image
            coordinate_point: Which coordinate point (1-based)
            x_pos, y_pos: Position coordinates
            l_value, a_value, b_value: CIE Lab values (required)
            rgb_r, rgb_g, rgb_b: RGB values (optional, defaults to 0)
            cluster_id: K-means cluster assignment
            delta_e: ΔE value
            centroid_x, centroid_y, centroid_z: Cluster centroid coordinates
            sphere_color: Sphere visualization color
            sphere_radius: Sphere visualization radius
            marker: Marker preference
            color: Color preference
            sample_type, sample_size, sample_anchor: Sample metadata
            notes: Optional notes
            trendline_valid: Whether this point is valid for trendlines
            
        Returns:
            True if insertion was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # First, ensure the measurement set exists
                set_id = self.create_measurement_set(image_name)
                if not set_id:
                    return False
                
                # Check if this measurement already exists
                cursor = conn.execute("""
                    SELECT id FROM color_measurements 
                    WHERE set_id = ? AND coordinate_point = ?
                """, (set_id, coordinate_point))
                
                if cursor.fetchone():
                    logger.debug(f"Measurement {image_name} point {coordinate_point} already exists - use update_plot3d_extended_values instead")
                    return False
                
                # Insert the new measurement with ALL extended values
                conn.execute("""
                    INSERT INTO color_measurements (
                        set_id, coordinate_point, x_position, y_position,
                        l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                        sample_type, sample_size, sample_anchor, notes,
                        marker_preference, color_preference,
                        cluster_id, delta_e, centroid_x, centroid_y, centroid_z,
                        sphere_color, sphere_radius, trendline_valid
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    set_id, coordinate_point, x_pos, y_pos,
                    l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                    sample_type, sample_size, sample_anchor, notes,
                    marker, color,
                    cluster_id, delta_e, centroid_x, centroid_y, centroid_z,
                    sphere_color, sphere_radius, int(trendline_valid) if trendline_valid is not None else 1
                ))
                
                print(f"✅ INSERTED new measurement: {image_name} point {coordinate_point} with Plot_3D data")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error inserting new measurement: {e}")
            return False
    
    def insert_or_update_centroid_data(self, cluster_id: int, centroid_x: float, centroid_y: float, centroid_z: float,
                                      sphere_color: str = None, sphere_radius: float = None,
                                      marker: str = '.', color: str = 'blue',
                                      image_name_override: str = None) -> bool:
        """Insert or update centroid data for sphere plotting.
        
        This method handles centroid data that may not be tied to specific sample measurements.
        It creates special 'centroid' entries that can be visualized as spheres in Plot_3D.
        
        Args:
            cluster_id: The cluster ID this centroid represents
            centroid_x, centroid_y, centroid_z: Centroid coordinates
            sphere_color: Color for sphere visualization
            sphere_radius: Radius for sphere visualization
            marker: Marker preference for plotting
            color: Color preference for plotting
            image_name_override: Override image name (defaults to 'CENTROIDS')
            
        Returns:
            True if insertion/update was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Use a special image name for centroid data
                centroid_image_name = image_name_override or 'CENTROIDS'
                
                # Ensure the measurement set exists
                set_id = self.create_measurement_set(centroid_image_name, description="Cluster centroids for sphere plotting")
                if not set_id:
                    return False
                
                # Use cluster_id as coordinate_point for centroids (unique per cluster)
                coordinate_point = cluster_id
                
                # Check if this centroid already exists
                cursor = conn.execute("""
                    SELECT id FROM color_measurements 
                    WHERE set_id = ? AND coordinate_point = ?
                """, (set_id, coordinate_point))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing centroid
                    print(f"    💾 CENTROID UPDATE: Updating existing centroid for cluster {cluster_id}")
                    print(f"    Data: ({centroid_x}, {centroid_y}, {centroid_z}), sphere={sphere_color}/{sphere_radius}")
                    
                    cursor = conn.execute("""
                        UPDATE color_measurements SET
                            centroid_x = ?, centroid_y = ?, centroid_z = ?,
                            sphere_color = ?, sphere_radius = ?,
                            marker_preference = ?, color_preference = ?,
                            cluster_id = ?,
                            measurement_date = datetime('now', 'localtime')
                        WHERE set_id = ? AND coordinate_point = ?
                    """, (
                        centroid_x, centroid_y, centroid_z,
                        sphere_color, sphere_radius,
                        marker, color, cluster_id,
                        set_id, coordinate_point
                    ))
                    
                    updated_rows = cursor.rowcount
                    print(f"    📋 CENTROID UPDATE: Updated {updated_rows} rows for cluster {cluster_id}")
                    print(f"✅ UPDATED centroid for cluster {cluster_id}: ({centroid_x:.3f}, {centroid_y:.3f}, {centroid_z:.3f})")
                else:
                    # Insert new centroid (using centroid coords as Lab values for compatibility)
                    # Handle None values gracefully - Plot_3D will filter out NaN entries
                    x_pos = centroid_x if centroid_x is not None else 0.0
                    y_pos = centroid_y if centroid_y is not None else 0.0
                    l_val = centroid_x if centroid_x is not None else 0.0
                    a_val = centroid_y if centroid_y is not None else 0.0
                    b_val = centroid_z if centroid_z is not None else 0.0
                    
                    print(f"    🆕 CENTROID INSERT: Creating new centroid for cluster {cluster_id}")
                    print(f"    Data: ({centroid_x}, {centroid_y}, {centroid_z}), sphere={sphere_color}/{sphere_radius}")
                    print(f"    Lab values: L={l_val}, a={a_val}, b={b_val}")
                    
                    cursor = conn.execute("""
                        INSERT INTO color_measurements (
                            set_id, coordinate_point, x_position, y_position,
                            l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                            sample_type, marker_preference, color_preference,
                            cluster_id, centroid_x, centroid_y, centroid_z,
                            sphere_color, sphere_radius, trendline_valid
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        set_id, coordinate_point, x_pos, y_pos,  # Use centroid coords as x,y position (or 0.0)
                        l_val, a_val, b_val,  # Centroid coords as Lab values (or 0.0)
                        0.0, 0.0, 0.0,  # RGB not applicable for centroids
                        'centroid', marker, color,
                        cluster_id, centroid_x, centroid_y, centroid_z,  # Store original values (may be None)
                        sphere_color, sphere_radius, 1  # Centroids are always trendline-valid
                    ))
                    
                    inserted_rows = cursor.rowcount
                    print(f"    📋 CENTROID INSERT: Inserted {inserted_rows} rows for cluster {cluster_id}")
                    print(f"✅ INSERTED new centroid for cluster {cluster_id}: ({centroid_x:.3f}, {centroid_y:.3f}, {centroid_z:.3f})")
                
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error inserting/updating centroid data: {e}")
            return False
    
    def update_plot3d_extended_values(self, image_name: str, coordinate_point: int,
                                     cluster_id: int = None, delta_e: float = None,
                                     centroid_x: float = None, centroid_y: float = None, centroid_z: float = None,
                                     sphere_color: str = None, sphere_radius: float = None,
                                     marker: str = None, color: str = None,
                                     trendline_valid: bool = None) -> bool:
        """Update all Plot_3D extended values for a specific measurement.
        
        Args:
            image_name: Name of the image
            coordinate_point: Which coordinate point (1-based)
            cluster_id: K-means cluster assignment
            delta_e: ΔE value
            centroid_x: X coordinate of cluster centroid
            centroid_y: Y coordinate of cluster centroid 
            centroid_z: Z coordinate of cluster centroid
            sphere_color: Sphere visualization color
            sphere_radius: Sphere visualization radius
            marker: Marker preference
            color: Color preference
            trendline_valid: Whether this point is valid for trendlines
            
        Returns:
            True if update was successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Build update query for only provided values
                update_parts = []
                values = []
                
                if cluster_id is not None:
                    update_parts.append("cluster_id = ?")
                    values.append(cluster_id)
                    
                if delta_e is not None:
                    update_parts.append("delta_e = ?")
                    values.append(delta_e)
                    
                if centroid_x is not None:
                    update_parts.append("centroid_x = ?")
                    values.append(centroid_x)
                    
                if centroid_y is not None:
                    update_parts.append("centroid_y = ?")
                    values.append(centroid_y)
                    
                if centroid_z is not None:
                    update_parts.append("centroid_z = ?")
                    values.append(centroid_z)
                    
                if sphere_color is not None:
                    update_parts.append("sphere_color = ?")
                    values.append(sphere_color)
                    
                if sphere_radius is not None:
                    update_parts.append("sphere_radius = ?")
                    values.append(sphere_radius)
                    
                if marker is not None:
                    update_parts.append("marker_preference = ?")
                    values.append(marker)
                    
                if color is not None:
                    update_parts.append("color_preference = ?")
                    values.append(color)
                    
                if trendline_valid is not None:
                    update_parts.append("trendline_valid = ?")
                    values.append(int(trendline_valid))  # Convert bool to int for SQLite
                
                if not update_parts:
                    return True  # Nothing to update
                
                # Add WHERE clause values
                values.extend([image_name, coordinate_point])
                
                query = f"""
                    UPDATE color_measurements 
                    SET {', '.join(update_parts)}
                    WHERE set_id = (
                        SELECT set_id FROM measurement_sets WHERE image_name = ?
                    ) AND coordinate_point = ?
                """
                
                print(f"    💾 DB UPDATE: Executing query for {image_name} pt{coordinate_point}")
                print(f"    Query: {query}")
                print(f"    Values: {values}")
                print(f"    Database path: {self.db_path}")
                
                # First, let's see if the measurement exists at all
                check_cursor = conn.execute("""
                    SELECT COUNT(*) FROM color_measurements 
                    WHERE set_id = (SELECT set_id FROM measurement_sets WHERE image_name = ?) 
                    AND coordinate_point = ?
                """, (image_name, coordinate_point))
                existing_count = check_cursor.fetchone()[0]
                print(f"    🔍 DB CHECK: Found {existing_count} existing measurements for {image_name} pt{coordinate_point}")
                
                cursor = conn.execute(query, values)
                updated_rows = cursor.rowcount
                
                # CRITICAL FIX: Explicitly commit the transaction
                conn.commit()
                print(f"    📋 DB UPDATE: Updated {updated_rows} rows for {image_name} pt{coordinate_point}")
                print(f"    ✅ DB COMMIT: Changes committed to database")
                
                if updated_rows > 0:
                    logger.debug(f"Updated Plot_3D values for {image_name} point {coordinate_point}: {len(update_parts)} fields")
                    return True
                else:
                    logger.debug(f"No measurement found for {image_name} point {coordinate_point}")
                    print(f"    ⚠️ DB UPDATE: No rows found to update for {image_name} pt{coordinate_point}")
                    return False
                    
        except sqlite3.Error as e:
            logger.error(f"Error updating Plot_3D extended values: {e}")
            return False
    
    def get_database_path(self) -> str:
        """Get the path to this sample set's database file."""
        return self.db_path
    
    @staticmethod
    def get_all_sample_set_databases(data_dir: str = None) -> List[str]:
        """Get all sample set database names.
        
        Args:
            data_dir: Optional data directory path
            
        Returns:
            List of sample set names (without .db extension)
        """
        if data_dir is None:
            # Use STAMPZ_DATA_DIR environment variable if available (for packaged apps)
            stampz_data_dir = os.getenv('STAMPZ_DATA_DIR')
            if stampz_data_dir:
                data_dir = os.path.join(stampz_data_dir, "data", "color_analysis")
            else:
                # Check if we're running in a PyInstaller bundle
                if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                    # PyInstaller bundle - use the same directory as the main app (StampZ-III)
                    user_data_dir = os.path.expanduser("~/Library/Application Support/StampZ-III")
                    data_dir = os.path.join(user_data_dir, "data", "color_analysis")
                else:
                    # Running from source - use relative path
                    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    data_dir = os.path.join(current_dir, "data", "color_analysis")
        
        if not os.path.exists(data_dir):
            return []
        
        sample_sets = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.db'):
                sample_sets.append(filename[:-3])  # Remove .db extension
        
        return sorted(sample_sets)


class AveragedColorAnalysisDB(ColorAnalysisDB):
    """Specialized database class for averaged color measurements."""
    
    def __init__(self, sample_set_name: str):
        """Initialize averaged database with _averages suffix.
        
        Args:
            sample_set_name: Base name of the sample set (will be suffixed with _averages)
        """
        # Ensure we're working with the _averages version
        if not sample_set_name.endswith('_averages'):
            sample_set_name = f"{sample_set_name}_averages"
        
        super().__init__(sample_set_name)
    
    def _init_db(self):
        """Initialize averaged color analysis database tables with averaged measurement support."""
        with sqlite3.connect(self.db_path) as conn:
            # Table for measurement sets
            conn.execute("""
                CREATE TABLE IF NOT EXISTS measurement_sets (
                    set_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_name TEXT NOT NULL,
                    measurement_date TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    description TEXT
                )
            """)
            
            # Table for averaged color measurements with extra columns
            conn.execute("""
                CREATE TABLE IF NOT EXISTS color_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    set_id INTEGER NOT NULL,
                    coordinate_point INTEGER NOT NULL,
                    x_position REAL NOT NULL,
                    y_position REAL NOT NULL,
                    l_value REAL NOT NULL,
                    a_value REAL NOT NULL,
                    b_value REAL NOT NULL,
                    rgb_r REAL NOT NULL,
                    rgb_g REAL NOT NULL,
                    rgb_b REAL NOT NULL,
                    sample_type TEXT,
                    sample_size TEXT,
                    sample_anchor TEXT,
                    measurement_date TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    notes TEXT,
                    is_averaged BOOLEAN DEFAULT 1,
                    source_samples_count INTEGER,
                    source_sample_ids TEXT,
                    FOREIGN KEY(set_id) REFERENCES measurement_sets(set_id)
                )
            """)
            
            # Add averaged measurement columns to existing databases if needed
            cursor = conn.cursor()
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN is_averaged BOOLEAN DEFAULT 1")
                print("Added is_averaged column to averaged database")
            except sqlite3.OperationalError:
                pass  # Column already exists
                
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN source_samples_count INTEGER")
                print("Added source_samples_count column to averaged database")
            except sqlite3.OperationalError:
                pass  # Column already exists
                
            try:
                cursor.execute("ALTER TABLE color_measurements ADD COLUMN source_sample_ids TEXT")
                print("Added source_sample_ids column to averaged database")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_set_point 
                ON color_measurements(set_id, coordinate_point)
            """)
    
    def save_averaged_measurement(
        self,
        set_id: int,
        averaged_lab: tuple,
        averaged_rgb: tuple,
        source_measurements: List[dict],
        image_name: str,
        notes: Optional[str] = None
    ) -> bool:
        """Save an averaged color measurement to the averages database.
        
        Args:
            set_id: ID of the measurement set
            averaged_lab: Averaged L*a*b* values as (L, a, b)
            averaged_rgb: Averaged RGB values as (R, G, B)
            source_measurements: List of individual measurements that were averaged
            image_name: Name of the image being analyzed
            notes: Optional notes about the averaging
            
        Returns:
            True if save was successful
        """
        print(f"DEBUG AveragedDB: save_averaged_measurement called")
        print(f"DEBUG AveragedDB: set_id={set_id}, image_name={image_name}")
        print(f"DEBUG AveragedDB: averaged_lab={averaged_lab}, averaged_rgb={averaged_rgb}")
        print(f"DEBUG AveragedDB: source_measurements count={len(source_measurements)}")
        print(f"DEBUG AveragedDB: notes={notes}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Calculate average position from source measurements
                if source_measurements:
                    avg_x = sum(m.get('x_position', 0) for m in source_measurements) / len(source_measurements)
                    avg_y = sum(m.get('y_position', 0) for m in source_measurements) / len(source_measurements)
                else:
                    avg_x = avg_y = 0.0
                
                # Create source sample IDs string for reference
                source_ids = ','.join(str(m.get('id', '')) for m in source_measurements if m.get('id'))
                
                # Use coordinate_point = 999 to indicate this is an averaged measurement
                coordinate_point = 999
                
                # Create notes that include averaging information
                avg_notes = f"Averaged from {len(source_measurements)} samples"
                if notes:
                    avg_notes += f": {notes}"
                
                # Analyze sample parameters from source measurements
                sample_types = [m.get('sample_type', '') for m in source_measurements if m.get('sample_type')]
                sample_sizes = [m.get('sample_size', '') for m in source_measurements if m.get('sample_size')]
                sample_anchors = [m.get('sample_anchor', '') for m in source_measurements if m.get('sample_anchor')]
                
                # Determine aggregated values
                if sample_types:
                    unique_types = set(sample_types)
                    avg_sample_type = sample_types[0] if len(unique_types) == 1 else 'various'
                else:
                    avg_sample_type = 'averaged'
                
                if sample_sizes:
                    unique_sizes = set(sample_sizes)
                    avg_sample_size = sample_sizes[0] if len(unique_sizes) == 1 else 'various'
                else:
                    avg_sample_size = '20'
                
                if sample_anchors:
                    unique_anchors = set(sample_anchors)
                    avg_sample_anchor = sample_anchors[0] if len(unique_anchors) == 1 else 'various'
                else:
                    avg_sample_anchor = 'center'
                
                print(f"DEBUG AveragedDB: Averaging sample parameters - types: {set(sample_types)}, sizes: {set(sample_sizes)}, anchors: {set(sample_anchors)}")
                print(f"DEBUG AveragedDB: Result - type: {avg_sample_type}, size: {avg_sample_size}, anchor: {avg_sample_anchor}")
                
                # Insert averaged measurement
                conn.execute("""
                    INSERT INTO color_measurements (
                        set_id, coordinate_point, x_position, y_position,
                        l_value, a_value, b_value, rgb_r, rgb_g, rgb_b,
                        sample_type, sample_size, sample_anchor, notes,
                        is_averaged, source_samples_count, source_sample_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    set_id, coordinate_point, avg_x, avg_y,
                    averaged_lab[0], averaged_lab[1], averaged_lab[2],
                    averaged_rgb[0], averaged_rgb[1], averaged_rgb[2],
                    avg_sample_type, avg_sample_size, avg_sample_anchor, avg_notes,
                    1, len(source_measurements), source_ids
                ))
                
                print(f"AveragedDB: Saved averaged measurement from {len(source_measurements)} samples for image '{image_name}'")
                return True
                
        except sqlite3.Error as e:
            print(f"Error saving averaged measurement to averages database: {e}")
            return False
