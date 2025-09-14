#!/usr/bin/env python3
"""
External Data Importer for StampZ Realtime Datasheet
Import CSV/ODS data and convert to realtime datasheet format.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ImportResult:
    """Result of an import operation."""
    success: bool
    data: Optional[List[List[str]]] = None
    centroid_data: Optional[List[Tuple[int, List[str]]]] = None  # (cluster_id, row_data) pairs
    rows_imported: int = 0
    rows_skipped: int = 0
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

class ExternalDataImporter:
    """Import external CSV/ODS data into realtime datasheet format."""
    
    # Expected realtime datasheet columns in order (must match PLOT3D_COLUMNS exactly)
    REALTIME_COLUMNS = [
        'Xnorm', 'Ynorm', 'Znorm', 'DataID', 'Cluster', 
        '∆E', 'Marker', 'Color', 'Centroid_X', 'Centroid_Y', 
        'Centroid_Z', 'Sphere', 'Radius'
    ]
    
    # Common column name mappings for various input formats
    COLUMN_MAPPINGS = {
        # Standard Plot_3D format
        'Xnorm': ['Xnorm', 'X_norm', 'X', 'L*', 'L_star', 'Lightness'],
        'Ynorm': ['Ynorm', 'Y_norm', 'Y', 'a*', 'a_star', 'a'],
        'Znorm': ['Znorm', 'Z_norm', 'Z', 'b*', 'b_star', 'b'],
        'DataID': ['DataID', 'Data_ID', 'ID', 'Sample_ID', 'SampleID', 'Name'],
        'Cluster': ['Cluster', 'Group', 'Class', 'Category'],
        'Centroid_X': ['Centroid_X', 'CentroidX', 'Cent_X'],
        'Centroid_Y': ['Centroid_Y', 'CentroidY', 'Cent_Y'],
        'Centroid_Z': ['Centroid_Z', 'CentroidZ', 'Cent_Z'],
        'Marker': ['Marker', 'Symbol', 'Shape'],
        'Color': ['Color', 'Colour'],
        '∆E': ['DeltaE', 'Delta_E', '∆E', 'ΔE', 'dE'],
        'Sphere': ['Sphere', 'Highlight', 'Selection']
    }
    
    # Default values for missing data
    DEFAULTS = {
        'Xnorm': '',
        'Ynorm': '',
        'Znorm': '',
        'DataID': '',
        'Cluster': '',
        '∆E': '',
        'Marker': '.',
        'Color': 'blue',
        'Centroid_X': '',
        'Centroid_Y': '',
        'Centroid_Z': '',
        'Sphere': '',
        'Radius': ''
    }
    
    def __init__(self):
        """Initialize the importer."""
        self.last_column_mapping = {}
        self.last_centroid_data = []
    
    def detect_file_format(self, file_path: str) -> str:
        """Detect the file format based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File format: 'csv', 'ods', 'xlsx', or 'unknown'
        """
        ext = os.path.splitext(file_path)[1].lower()
        format_map = {
            '.csv': 'csv',
            '.ods': 'ods', 
            '.xlsx': 'xlsx',
            '.xls': 'xlsx'
        }
        return format_map.get(ext, 'unknown')
    
    def read_external_file(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Read external file and return DataFrame with warnings.
        
        Args:
            file_path: Path to the external file
            
        Returns:
            Tuple of (DataFrame, list of warnings)
            
        Raises:
            Exception: If file cannot be read
        """
        warnings = []
        file_format = self.detect_file_format(file_path)
        
        try:
            if file_format == 'csv':
                # Try different CSV reading strategies
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    # Try different encoding
                    df = pd.read_csv(file_path, encoding='latin1')
                    warnings.append("Used latin1 encoding for CSV file")
                except pd.errors.ParserError:
                    # Try with different separator
                    df = pd.read_csv(file_path, sep=';')
                    warnings.append("Used semicolon separator for CSV file")
                    
            elif file_format == 'ods':
                df = pd.read_excel(file_path, engine='odf')
                
            elif file_format == 'xlsx':
                df = pd.read_excel(file_path)
                
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise Exception(f"Failed to read file: {e}")
        
        # Basic validation
        if df.empty:
            raise Exception("File contains no data")
            
        if len(df.columns) == 0:
            raise Exception("File contains no columns")
        
        # Handle K-means centroid rows (rows 2-7 in spreadsheet, which are rows 1-6 in DataFrame)
        # These are reserved but if they contain data, they should be preserved as centroid data
        centroid_rows = []
        data_rows = []
        
        if len(df) > 6:  # Only process if we have enough rows
            # Check rows 0-5 (DataFrame index, which are rows 2-7 in spreadsheet) for valid centroid data
            original_df = df.copy()  # Keep original for centroid processing
            for i in range(6):
                if i < len(df):
                    row = df.iloc[i]
                    # Check if this row has meaningful centroid data
                    has_centroid_data = self._is_valid_centroid_row(row)
                    if has_centroid_data:
                        # Store as dictionary for easier processing later
                        row_dict = row.to_dict()
                        centroid_rows.append((i, row_dict))
                        warnings.append(f"Found valid K-means centroid data in row {i+2}")
            
            # Data rows start from row 7 (DataFrame index 6)
            data_rows_df = df.iloc[6:].reset_index(drop=True)
            if not data_rows_df.empty:
                warnings.append(f"Imported {len(data_rows_df)} data rows starting from row 8")
            else:
                warnings.append("No data rows found after centroid area")
        else:
            # If we have 6 or fewer rows, treat them all as data
            data_rows_df = df
            warnings.append("File has 6 or fewer rows, treating all as data rows")
        
        # Store centroid info for later use
        self.last_centroid_data = centroid_rows
        
        # Return only the data rows for normal processing
        df = data_rows_df if 'data_rows_df' in locals() else df
        
        logger.info(f"Read {len(df)} rows and {len(df.columns)} columns from {file_path}")
        return df, warnings
    
    def map_columns(self, df: pd.DataFrame, custom_mappings: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], List[str]]:
        """Map input columns to realtime datasheet columns.
        
        Args:
            df: Input DataFrame
            custom_mappings: Optional custom column mappings
            
        Returns:
            Tuple of (column mapping dict, list of warnings)
        """
        warnings = []
        input_columns = df.columns.tolist()
        mapping = {}
        
        # Use custom mappings if provided
        if custom_mappings:
            for target_col, source_col in custom_mappings.items():
                if source_col in input_columns and target_col in self.REALTIME_COLUMNS:
                    mapping[target_col] = source_col
        
        # Auto-detect remaining columns
        for target_col in self.REALTIME_COLUMNS:
            if target_col in mapping:
                continue
                
            # Try to find matching column
            possible_names = self.COLUMN_MAPPINGS.get(target_col, [target_col])
            
            for possible_name in possible_names:
                # Case-insensitive search
                for input_col in input_columns:
                    if input_col.lower().strip() == possible_name.lower().strip():
                        mapping[target_col] = input_col
                        break
                if target_col in mapping:
                    break
        
        # Report mapping results
        mapped_cols = len(mapping)
        total_target_cols = len(self.REALTIME_COLUMNS)
        
        logger.info(f"Column mapping: {mapped_cols}/{total_target_cols} target columns mapped")
        
        if mapped_cols < total_target_cols:
            unmapped = [col for col in self.REALTIME_COLUMNS if col not in mapping]
            warnings.append(f"Unmapped columns will use defaults: {', '.join(unmapped)}")
        
        # Store for potential reuse
        self.last_column_mapping = mapping.copy()
        
        return mapping, warnings
    
    def validate_data_compatibility(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> List[str]:
        """Validate that the data is compatible with realtime datasheet format.
        
        Args:
            df: Input DataFrame
            column_mapping: Column mapping dictionary
            
        Returns:
            List of warnings
        """
        warnings = []
        
        # Check for essential columns
        essential_columns = ['DataID']
        for col in essential_columns:
            if col not in column_mapping:
                warnings.append(f"Essential column '{col}' not found - rows may not be identifiable")
        
        # Check coordinate columns
        coord_columns = ['Xnorm', 'Ynorm', 'Znorm']
        mapped_coords = [col for col in coord_columns if col in column_mapping]
        
        if len(mapped_coords) == 0:
            warnings.append("No coordinate columns found - this may not be valid plot data")
        elif len(mapped_coords) < 3:
            warnings.append(f"Only {len(mapped_coords)}/3 coordinate columns found")
        
        # Check data types for numeric columns
        numeric_columns = ['Xnorm', 'Ynorm', 'Znorm', 'Centroid_X', 'Centroid_Y', 'Centroid_Z', '∆E', 'Radius']
        for col in numeric_columns:
            if col in column_mapping:
                source_col = column_mapping[col]
                try:
                    pd.to_numeric(df[source_col], errors='coerce')
                except Exception:
                    warnings.append(f"Column '{source_col}' (mapped to {col}) may not be numeric")
        
        return warnings
    
    def convert_to_realtime_format(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Tuple[List[List[str]], List[str]]:
        """Convert DataFrame to realtime datasheet format.
        
        Args:
            df: Input DataFrame
            column_mapping: Column mapping dictionary
            
        Returns:
            Tuple of (list of rows, list of warnings)
        """
        warnings = []
        rows = []
        
        # Process each row
        for idx, row in df.iterrows():
            realtime_row = []
            
            # Convert each column
            for target_col in self.REALTIME_COLUMNS:
                if target_col in column_mapping:
                    source_col = column_mapping[target_col]
                    value = row[source_col]
                    
                    # Handle NaN/None values
                    if pd.isna(value) or value is None:
                        value = self.DEFAULTS[target_col]
                    else:
                        # Convert to string and clean
                        value = str(value).strip()
                        
                        # Special handling for numeric columns
                        if target_col in ['Xnorm', 'Ynorm', 'Znorm', 'Centroid_X', 'Centroid_Y', 'Centroid_Z', '∆E', 'Radius']:
                            try:
                                # Convert to float to validate, then back to string for consistency
                                float_val = float(value)
                                value = str(round(float_val, 4))
                            except (ValueError, TypeError):
                                if value != '':  # Only warn if not already empty
                                    warnings.append(f"Row {idx+1}: Invalid numeric value '{value}' in {target_col}, using default")
                                value = self.DEFAULTS[target_col]
                else:
                    # Use default value for unmapped columns
                    value = self.DEFAULTS[target_col]
                
                realtime_row.append(value)
            
            rows.append(realtime_row)
        
        logger.info(f"Converted {len(rows)} rows to realtime format")
        return rows, warnings
    
    def import_file(self, file_path: str, custom_mappings: Optional[Dict[str, str]] = None) -> ImportResult:
        """Import external file and convert to realtime datasheet format.
        
        Args:
            file_path: Path to the external file
            custom_mappings: Optional custom column mappings
            
        Returns:
            ImportResult object with results and any warnings/errors
        """
        result = ImportResult(success=False)
        
        try:
            # Read the file
            df, read_warnings = self.read_external_file(file_path)
            result.warnings.extend(read_warnings)
            
            # Map columns
            column_mapping, mapping_warnings = self.map_columns(df, custom_mappings)
            result.warnings.extend(mapping_warnings)
            
            # Validate compatibility
            validation_warnings = self.validate_data_compatibility(df, column_mapping)
            result.warnings.extend(validation_warnings)
            
            # Convert to realtime format
            rows, conversion_warnings = self.convert_to_realtime_format(df, column_mapping)
            result.warnings.extend(conversion_warnings)
            
            # Process any centroid data found (stored during file reading)
            centroid_results = []
            for cluster_idx, centroid_row in self.last_centroid_data:
                # Convert centroid row to realtime format using column mapping
                centroid_data = []
                for target_col in self.REALTIME_COLUMNS:
                    if target_col in column_mapping:
                        source_col = column_mapping[target_col]
                        value = centroid_row[source_col] if source_col in centroid_row else self.DEFAULTS[target_col]
                        if pd.isna(value) or value is None:
                            value = self.DEFAULTS[target_col]
                        else:
                            value = str(value).strip()
                            # Handle numeric columns
                            if target_col in ['Xnorm', 'Ynorm', 'Znorm', 'Centroid_X', 'Centroid_Y', 'Centroid_Z', '∆E', 'Radius']:
                                try:
                                    if value and value != '':
                                        float_val = float(value)
                                        value = str(round(float_val, 4))
                                    else:
                                        value = self.DEFAULTS[target_col]
                                except (ValueError, TypeError):
                                    value = self.DEFAULTS[target_col]
                    else:
                        value = self.DEFAULTS[target_col]
                    centroid_data.append(value)
                
                # Extract cluster ID from the original data (not row index)
                cluster_id = cluster_idx  # Default fallback to row index
                
                # Try to get actual cluster ID from the data
                # Check Cluster column first
                if 'Cluster' in centroid_row and pd.notna(centroid_row['Cluster']):
                    try:
                        cluster_id = int(float(centroid_row['Cluster']))
                    except (ValueError, TypeError):
                        pass
                # Check DataID column as backup (in case data is shifted)
                elif 'DataID' in centroid_row and pd.notna(centroid_row['DataID']):
                    try:
                        test_val = float(centroid_row['DataID'])
                        if 0 <= test_val <= 5:
                            cluster_id = int(test_val)
                    except (ValueError, TypeError):
                        pass
                
                centroid_results.append((cluster_id, centroid_data))
            
            # Set results
            result.data = rows
            result.centroid_data = centroid_results
            result.rows_imported = len(rows)
            result.success = True
            
            logger.info(f"Successfully imported {result.rows_imported} data rows and {len(centroid_results)} centroid rows from {file_path}")
            
        except Exception as e:
            error_msg = f"Import failed: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg)
        
        return result
    
    def get_sample_mappings(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get sample column mappings for user reference.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary showing possible mappings for each target column
        """
        input_columns = df.columns.tolist()
        suggestions = {}
        
        for target_col in self.REALTIME_COLUMNS:
            possible_matches = []
            possible_names = self.COLUMN_MAPPINGS.get(target_col, [target_col])
            
            for possible_name in possible_names:
                for input_col in input_columns:
                    if possible_name.lower() in input_col.lower():
                        possible_matches.append(input_col)
            
            suggestions[target_col] = list(set(possible_matches))  # Remove duplicates
        
        return suggestions
    
    def _is_valid_centroid_row(self, row) -> bool:
        """Check if a row contains valid K-means centroid data.
        
        Args:
            row: pandas Series representing a row
            
        Returns:
            bool: True if row contains valid centroid data
        """
        try:
            # Look for cluster ID in both Cluster column and DataID column (CSV structure can be messy)
            cluster_id = None
            
            # Check standard Cluster column (index 4)
            if len(row) > 4 and pd.notna(row.iloc[4]):
                try:
                    test_id = int(float(row.iloc[4]))
                    if 0 <= test_id <= 5:
                        cluster_id = test_id
                except (ValueError, TypeError):
                    pass
            
            # Also check DataID column (index 3) in case data shifted
            if cluster_id is None and len(row) > 3 and pd.notna(row.iloc[3]):
                try:
                    test_id = int(float(row.iloc[3]))
                    if 0 <= test_id <= 5:
                        cluster_id = test_id
                except (ValueError, TypeError):
                    pass
            
            if cluster_id is None:
                return False
            
            # Check if there's centroid coordinate data (Centroid_X, Y, Z columns)
            centroid_indices = [8, 9, 10]  # Centroid_X, Y, Z
            has_coords = False
            
            for coord_idx in centroid_indices:
                if len(row) > coord_idx and pd.notna(row.iloc[coord_idx]):
                    coord_val = str(row.iloc[coord_idx]).strip()
                    if coord_val and coord_val != '' and coord_val.lower() not in ['nan', 'none']:
                        try:
                            float(coord_val)  # Valid numeric coordinate
                            has_coords = True
                            break
                        except (ValueError, TypeError):
                            continue
            
            return has_coords
            
        except Exception:
            return False
