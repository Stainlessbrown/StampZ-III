#!/usr/bin/env python3
"""
Unified Data Logger for StampZ
Consolidates all analysis data (measurements, color analysis, black ink extraction, etc.) 
into a single text file per image for comprehensive documentation.
"""

import os
from datetime import datetime
from pathlib import Path


class UnifiedDataLogger:
    """Logs all StampZ analysis data to a single file per image."""
    
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.data_file_path = self._get_data_file_path()
        
    def _get_data_file_path(self):
        """Generate data file path based on image filename."""
        # Create data file alongside image with _StampZ_Data.txt suffix
        stem = self.image_path.stem
        directory = self.image_path.parent
        return directory / f"{stem}_StampZ_Data.txt"
        
    def log_section(self, tool_name, data_dict):
        """
        Log data from any StampZ tool.
        
        Args:
            tool_name (str): Name of the tool (e.g., "Precision Measurements", "Black Ink Extractor")
            data_dict (dict): Dictionary containing all data to log
        """
        try:
            # Create or append to data file
            mode = 'a' if self.data_file_path.exists() else 'w'
            
            with open(self.data_file_path, mode, encoding='utf-8') as f:
                # Add header for new file
                if mode == 'w':
                    f.write("STAMPZ COMPREHENSIVE ANALYSIS DATA\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Image: {self.image_path.name}\n")
                    f.write(f"Path: {self.image_path}\n")
                    f.write(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Add separator for new tool section
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"{tool_name.upper()} ANALYSIS\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 50 + "\n\n")
                
                # Write data
                for key, value in data_dict.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\n")
                
            return self.data_file_path
            
        except Exception as e:
            print(f"Error logging to unified data file: {e}")
            return None
            
    def log_measurements(self, measurements, dpi, calibration_source, precision_um):
        """Log precision measurement data."""
        data = {
            "DPI": f"{dpi:.1f}",
            "Calibration Source": calibration_source,
            "Precision": f"±{precision_um:.1f}µm",
            "Number of Measurements": len(measurements),
            "Measurements": ""
        }
        
        # Add individual measurements
        measurement_lines = []
        for i, measurement in enumerate(measurements, 1):
            if hasattr(measurement, 'calculate_distance_mm') and hasattr(measurement, 'measurement_engine'):
                distance_mm = measurement.calculate_distance_mm(dpi / 25.4)
                measurement_lines.append(
                    f"  {i}. {measurement.label}: {distance_mm:.2f}mm ({measurement.measurement_type})"
                )
            else:
                measurement_lines.append(f"  {i}. {measurement.label} ({measurement.measurement_type})")
                
        data["Measurements"] = "\n" + "\n".join(measurement_lines) if measurement_lines else "None"
        
        return self.log_section("Precision Measurements", data)
        
    def log_black_ink_extraction(self, extraction_params, results):
        """Log black ink extraction data."""
        data = {
            "Black Threshold": extraction_params.get('black_threshold', 'Unknown'),
            "Saturation Threshold": extraction_params.get('saturation_threshold', 'Unknown'),
            "Extraction Method": extraction_params.get('method', 'Standard'),
            "Output File": results.get('output_file', 'None'),
            "Processing Time": f"{results.get('processing_time', 0):.2f} seconds",
            "Success": "Yes" if results.get('success', False) else "No"
        }
        
        return self.log_section("Black Ink Extraction", data)
        
    def log_color_analysis(self, color_data):
        """Log color analysis data."""
        data = {
            "Analysis Type": color_data.get('analysis_type', 'Unknown'),
            "Color Space": color_data.get('color_space', 'RGB'),
            "Number of Samples": color_data.get('sample_count', 0),
            "Average Color": color_data.get('average_color', 'Not calculated'),
            "Color Range": color_data.get('color_range', 'Not calculated')
        }
        
        return self.log_section("Color Analysis", data)
        
    def log_individual_color_measurements(self, measurements, sample_set_name, image_name):
        """Log individual color measurements from analysis.
        
        Args:
            measurements: List of measurement dictionaries with color data
            sample_set_name: Name of the sample set
            image_name: Name of the analyzed image
        """
        try:
            data = {
                "Sample Set Name": sample_set_name,
                "Image Name": image_name,
                "Number of Measurements": len(measurements),
                "Analysis Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Individual Measurements": ""
            }
            
            # Format individual measurements
            measurement_lines = []
            for i, measurement in enumerate(measurements, 1):
                l_val = measurement.get('l_value', 0)
                a_val = measurement.get('a_value', 0) 
                b_val = measurement.get('b_value', 0)
                r_val = measurement.get('rgb_r', 0)
                g_val = measurement.get('rgb_g', 0)
                b_rgb_val = measurement.get('rgb_b', 0)
                x_pos = measurement.get('x_position', 0)
                y_pos = measurement.get('y_position', 0)
                
                measurement_lines.append(
                    f"  Sample {i}: L*a*b*=({l_val:.2f}, {a_val:.2f}, {b_val:.2f}) | "
                    f"RGB=({r_val:.1f}, {g_val:.1f}, {b_rgb_val:.1f}) | "
                    f"Position=({x_pos:.1f}, {y_pos:.1f})"
                )
            
            data["Individual Measurements"] = "\n" + "\n".join(measurement_lines) if measurement_lines else "None"
            
            return self.log_section("Individual Color Measurements", data)
            
        except Exception as e:
            print(f"Error logging individual color measurements: {e}")
            return None
            
    def log_averaged_color_measurement(self, averaged_data, sample_set_name, image_name, source_count):
        """Log averaged color measurement.
        
        Args:
            averaged_data: Dictionary with averaged L*a*b* and RGB values
            sample_set_name: Name of the sample set
            image_name: Name of the analyzed image 
            source_count: Number of individual samples used in the average
        """
        try:
            l_avg = averaged_data.get('l_value', 0)
            a_avg = averaged_data.get('a_value', 0)
            b_avg = averaged_data.get('b_value', 0)
            r_avg = averaged_data.get('rgb_r', 0)
            g_avg = averaged_data.get('rgb_g', 0)
            b_rgb_avg = averaged_data.get('rgb_b', 0)
            
            data = {
                "Sample Set Name": sample_set_name,
                "Image Name": image_name,
                "Source Sample Count": source_count,
                "Analysis Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Averaged L*a*b*": f"L*={l_avg:.2f}, a*={a_avg:.2f}, b*={b_avg:.2f}",
                "Averaged RGB": f"R={r_avg:.1f}, G={g_avg:.1f}, B={b_rgb_avg:.1f}",
                "Quality Notes": averaged_data.get('notes', 'High-quality average measurement')
            }
            
            return self.log_section("Averaged Color Measurement", data)
            
        except Exception as e:
            print(f"Error logging averaged color measurement: {e}")
            return None
        
    def log_perforation_analysis(self, perf_data):
        """Log comprehensive perforation analysis data from gauge measurements."""
        # Core perforation information
        data = {
            "Perforation Type": perf_data.get('perf_type', 'Unknown'),
            "Catalog Format": perf_data.get('catalog_format', 'Not measured'),
            "Gauge Measurement": perf_data.get('gauge', 'Not measured'),
        }
        
        # Detailed measurements if available
        horizontal = perf_data.get('horizontal_gauge')
        vertical = perf_data.get('vertical_gauge')
        
        if horizontal and horizontal != "Not measured":
            data["Horizontal Gauge"] = f"{horizontal}"
            
        if vertical and vertical != "Not measured":
            data["Vertical Gauge"] = f"{vertical}"
        
        # Technical details
        data["Measurement Method"] = perf_data.get('measurement_method', 'Unknown')
        data["Measurement Tool"] = perf_data.get('measurement_tool', 'Unknown')
        data["DPI Used"] = perf_data.get('dpi_used', 'Unknown')
        data["Color Scheme"] = perf_data.get('color_scheme', 'Default')
        data["Regularity Assessment"] = perf_data.get('regularity', 'Not analyzed')
        data["Notes"] = perf_data.get('notes', 'None')
        
        return self.log_section("Gauge Perforation Analysis", data)
        
    def get_data_file_path(self):
        """Return the path to the unified data file."""
        return self.data_file_path
        
    def file_exists(self):
        """Check if data file already exists."""
        return self.data_file_path.exists()


# Example usage:
if __name__ == "__main__":
    # Demo of unified logging
    logger = UnifiedDataLogger("/path/to/stamp.tif")
    
    # Log measurements
    measurements_data = {
        "DPI": "800.0",
        "Calibration Source": "Manual calibration",
        "Precision": "±15.8µm",
        "Measurements": "\n  1. Stamp Width: 22.45mm (horizontal)\n  2. Stamp Height: 26.78mm (vertical)"
    }
    logger.log_section("Precision Measurements", measurements_data)
    
    # Log color analysis
    color_data = {
        "Analysis Type": "Sample comparison",
        "Color Space": "RGB",
        "Average Color": "RGB(156, 45, 32)"
    }
    logger.log_section("Color Analysis", color_data)
    
    print(f"Data logged to: {logger.get_data_file_path()}")