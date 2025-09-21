#!/usr/bin/env python3
"""
StampZ Precision Measurement Engine
Core engine for architectural-style precision measurements with automatic DPI calibration
Perfect for fraud detection, authentication, and plate studies
"""

from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import json
from datetime import datetime
from pathlib import Path
import math

class MeasurementEngine:
    """Core measurement engine with DPI auto-detection and precision calculations"""
    
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.image = None
        self.dpi = None
        self.pixels_per_mm = None
        self.measurements = []
        self.calibration_source = None
        self.precision_um = None
        
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Load image and extract DPI information automatically"""
        self.image_path = Path(image_path)
        self.image = Image.open(image_path)
        
        # Extract DPI from image metadata
        self.dpi = self._extract_dpi()
        self.pixels_per_mm = self.dpi / 25.4 if self.dpi else None
        self.precision_um = (1000 / self.pixels_per_mm / 2) if self.pixels_per_mm else None
        
        print(f"📷 Loaded: {self.image_path.name}")
        print(f"📐 Resolution: {self.image.size[0]}x{self.image.size[1]} pixels")
        print(f"🔍 DPI: {self.dpi} ({self.calibration_source})")
        if self.precision_um:
            print(f"⚡ Precision: ±{self.precision_um:.1f}µm")
        
        return self._analyze_measurement_capabilities()
    
    def _extract_dpi(self):
        """Extract DPI from image metadata with multiple fallback methods"""
        dpi = None
        
        # Method 1: PIL DPI info (most common)
        if hasattr(self.image, 'info') and 'dpi' in self.image.info:
            dpi_info = self.image.info['dpi']
            if isinstance(dpi_info, tuple):
                dpi = max(dpi_info)  # Use the higher of X/Y DPI
            else:
                dpi = dpi_info
            self.calibration_source = "PIL DPI tag"
            print(f"Found PIL DPI: {dpi}")
        
        # Method 2: EXIF data (VueScan, digital cameras)
        if (dpi is None or dpi < 50) and hasattr(self.image, '_getexif'):
            try:
                exif = self.image._getexif()
                if exif:
                    # Look for resolution tags
                    if 282 in exif and 283 in exif:  # X/Y Resolution tags
                        x_res = exif[282]
                        y_res = exif[283]
                        if isinstance(x_res, tuple):
                            x_dpi = x_res[0] / x_res[1] if x_res[1] != 0 else x_res[0]
                        else:
                            x_dpi = x_res
                        if isinstance(y_res, tuple):
                            y_dpi = y_res[0] / y_res[1] if y_res[1] != 0 else y_res[0]
                        else:
                            y_dpi = y_res
                        dpi = max(x_dpi, y_dpi)
                        self.calibration_source = "EXIF metadata"
                        print(f"Found EXIF DPI: {dpi}")
            except:
                pass
                
        # Method 3: Try PIL's more detailed info
        if (dpi is None or dpi < 50) and hasattr(self.image, 'info'):
            for key in ['resolution', 'Resolution', 'jfif_density', 'tiff_resolution']:
                if key in self.image.info:
                    res_info = self.image.info[key]
                    if isinstance(res_info, (list, tuple)) and len(res_info) >= 2:
                        dpi = max(res_info[:2])
                        self.calibration_source = f"PIL {key}"
                        print(f"Found {key}: {dpi}")
                        break
        
        # Method 4: TIFF-specific DPI extraction (VueScan and scanner TIFF files)
        if (dpi is None or dpi < 50) and self.image.format == 'TIFF':
            try:
                # Try different TIFF tag approaches
                if hasattr(self.image, 'tag_v2'):
                    # Method 4a: XResolution tag (282)
                    if 282 in self.image.tag_v2:
                        x_res = self.image.tag_v2[282]
                        if isinstance(x_res, tuple) and len(x_res) >= 2:
                            dpi = float(x_res[0]) / float(x_res[1]) if x_res[1] != 0 else float(x_res[0])
                        else:
                            dpi = float(x_res)
                        self.calibration_source = "TIFF XResolution tag"
                        print(f"Found TIFF XRes: {dpi} from {str(x_res)}")
                        
                    # Method 4b: Resolution unit tag (296) + resolution
                    elif 296 in self.image.tag_v2 and 282 in self.image.tag_v2:
                        unit = self.image.tag_v2[296]
                        x_res = self.image.tag_v2[282]
                        if unit == 2:  # inches
                            if isinstance(x_res, tuple) and len(x_res) >= 2:
                                dpi = float(x_res[0]) / float(x_res[1]) if x_res[1] != 0 else float(x_res[0])
                            else:
                                dpi = float(x_res)
                            self.calibration_source = "TIFF Resolution + Unit tags"
                            print(f"Found TIFF Res+Unit: {dpi} from {str(x_res)}, unit={unit}")
                            
                # Method 4c: Try tag dictionary directly
                if (dpi is None or dpi < 50) and hasattr(self.image, 'tag'):
                    for tag_id in [282, 283]:  # X and Y resolution
                        if tag_id in self.image.tag:
                            tag_value = self.image.tag[tag_id]
                            if isinstance(tag_value, tuple) and len(tag_value) >= 2:
                                potential_dpi = float(tag_value[0]) / float(tag_value[1]) if tag_value[1] != 0 else float(tag_value[0])
                                if potential_dpi >= 50:  # Reasonable DPI value
                                    dpi = potential_dpi
                                    self.calibration_source = "TIFF tag dictionary"
                                    print(f"Found TIFF tag {tag_id}: {dpi} from {str(tag_value)}")
                                    break
            except Exception as e:
                print(f"TIFF DPI extraction error: {e}")
                
        # Method 5: Check for common scan DPI values from filename
        if (dpi is None or dpi < 50) and self.image_path:
            filename = self.image_path.name.lower()
            if '1200' in filename or '1200dpi' in filename:
                dpi = 1200
                self.calibration_source = "Filename heuristic"
            elif '800' in filename or '800dpi' in filename:
                dpi = 800
                self.calibration_source = "Filename heuristic"
            elif '600' in filename or '600dpi' in filename:
                dpi = 600
                self.calibration_source = "Filename heuristic"
            elif '300' in filename or '300dpi' in filename:
                dpi = 300
                self.calibration_source = "Filename heuristic"
        
        # Method 4: Default for unknown or suspicious values
        if dpi is None or dpi < 10:  # DPI of 1 is clearly wrong
            dpi = 300  # Conservative default
            self.calibration_source = "Default (needs manual calibration)"
        
        return dpi
    
    def _analyze_measurement_capabilities(self):
        """Analyze what measurement types are possible at current DPI"""
        if not self.precision_um:
            return {"error": "No DPI information available"}
        
        capabilities = {
            "dpi": self.dpi,
            "precision_um": self.precision_um,
            "pixels_per_mm": self.pixels_per_mm,
            "capabilities": {
                "basic_measurements": True,  # Always possible
                "plate_varieties": self.precision_um < 50,      # 0.1-0.5mm differences
                "fraud_detection": self.precision_um < 30,       # 0.1-0.3mm differences  
                "overprint_analysis": self.precision_um < 100,   # 0.2-1.0mm variations
                "perforation_spacing": self.precision_um < 25,   # 0.05-0.2mm variations
                "micro_measurements": self.precision_um < 10     # <0.05mm precision
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if self.dpi >= 1200:
            capabilities["recommendations"].append("Excellent: All precision measurements supported")
        elif self.dpi >= 600:
            capabilities["recommendations"].append("Very Good: All standard philatelic measurements supported")
        elif self.dpi >= 300:
            capabilities["recommendations"].append("Fair: Basic measurements only, consider rescanning at 600+ DPI")
        else:
            capabilities["recommendations"].append("Warning: Insufficient resolution for precision measurements")
        
        return capabilities
    
    def calibrate_manually(self, pixel_distance, known_mm):
        """Manual calibration using known reference measurement"""
        self.pixels_per_mm = pixel_distance / known_mm
        self.dpi = self.pixels_per_mm * 25.4
        self.precision_um = 1000 / self.pixels_per_mm / 2
        self.calibration_source = "Manual calibration"
        
        print(f"🔧 Manual calibration: {self.dpi:.1f} DPI, ±{self.precision_um:.1f}µm precision")
        return self._analyze_measurement_capabilities()

class ArchitecturalMeasurement:
    """Single architectural-style measurement with extension lines"""
    
    def __init__(self, start_point, end_point, measurement_type="distance", label="", color="red"):
        self.id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17]  # Unique ID
        self.start_point = start_point  # (x, y) in pixels
        self.end_point = end_point      # (x, y) in pixels
        self.measurement_type = measurement_type  # "distance", "horizontal", "vertical"
        self.label = label
        self.color = color
        self.created = datetime.now()
    
    def calculate_distance_pixels(self):
        """Calculate distance in pixels"""
        dx = self.end_point[0] - self.start_point[0]
        dy = self.end_point[1] - self.start_point[1]
        return math.sqrt(dx**2 + dy**2)
    
    def calculate_distance_mm(self, pixels_per_mm):
        """Calculate distance in millimeters"""
        if not pixels_per_mm:
            return None
        return self.calculate_distance_pixels() / pixels_per_mm
    
    def get_dimension_line_geometry(self, offset=25):
        """Calculate geometry for architectural dimension lines"""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Determine if primarily horizontal or vertical
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > dy:  # Horizontal measurement
            dim_y = max(y1, y2) + offset
            extension_lines = [
                ((x1, y1), (x1, dim_y + 5)),  # Start extension line
                ((x2, y2), (x2, dim_y + 5))   # End extension line
            ]
            dimension_line = ((x1, dim_y), (x2, dim_y))
            text_position = ((x1 + x2) / 2, dim_y + 12)
            text_rotation = 0
        else:  # Vertical measurement
            dim_x = max(x1, x2) + offset
            extension_lines = [
                ((x1, y1), (dim_x + 5, y1)),  # Start extension line
                ((x2, y2), (dim_x + 5, y2))   # End extension line
            ]
            dimension_line = ((dim_x, y1), (dim_x, y2))
            text_position = (dim_x + 15, (y1 + y2) / 2)
            text_rotation = 90
        
        return {
            "extension_lines": extension_lines,
            "dimension_line": dimension_line,
            "text_position": text_position,
            "text_rotation": text_rotation
        }

class MeasurementRenderer:
    """Renders architectural-style measurements on matplotlib plots"""
    
    @staticmethod
    def draw_measurement(ax, measurement, pixels_per_mm, precision=2, offset=25):
        """Draw a single architectural measurement on matplotlib axis"""
        
        # Get dimension line geometry
        geom = measurement.get_dimension_line_geometry(offset)
        
        # Draw extension lines (|)
        for ext_line in geom["extension_lines"]:
            start, end = ext_line
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color=measurement.color, linewidth=1, linestyle='-')
        
        # Draw dimension line with arrows (<--->)
        dim_start, dim_end = geom["dimension_line"]
        arrow = FancyArrowPatch(dim_start, dim_end,
                               arrowstyle='<->', mutation_scale=15,
                               color=measurement.color, linewidth=1.5)
        ax.add_patch(arrow)
        
        # Calculate and display measurement text
        distance_mm = measurement.calculate_distance_mm(pixels_per_mm)
        if distance_mm is not None:
            if measurement.label:
                text = f"{measurement.label}: {distance_mm:.{precision}f}mm"
            else:
                text = f"{distance_mm:.{precision}f}mm"
        else:
            text = f"{measurement.calculate_distance_pixels():.0f}px"
        
        # Draw measurement text
        text_pos = geom["text_position"]
        ax.text(text_pos[0], text_pos[1], text, 
                ha='center', va='center', color=measurement.color, 
                fontsize=9, rotation=geom["text_rotation"], weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    @staticmethod
    def draw_all_measurements(ax, measurements, pixels_per_mm, precision=2):
        """Draw all measurements with automatic offset staggering"""
        for i, measurement in enumerate(measurements):
            # Stagger offsets to avoid overlapping dimension lines
            offset = 25 + (i * 20)
            MeasurementRenderer.draw_measurement(ax, measurement, pixels_per_mm, precision, offset)

class MeasurementExporter:
    """Export measurements for documentation and certification"""
    
    @staticmethod
    def export_report(engine, measurements, output_path=None):
        """Export detailed measurement report"""
        
        report = {
            "metadata": {
                "image_file": str(engine.image_path) if engine.image_path else "Unknown",
                "export_date": datetime.now().isoformat(),
                "image_size": engine.image.size if engine.image else None,
                "dpi": engine.dpi,
                "calibration_source": engine.calibration_source,
                "precision_um": engine.precision_um,
                "pixels_per_mm": engine.pixels_per_mm
            },
            "measurements": []
        }
        
        # Add each measurement
        for measurement in measurements:
            measurement_data = {
                "id": measurement.id,
                "label": measurement.label,
                "type": measurement.measurement_type,
                "start_point": measurement.start_point,
                "end_point": measurement.end_point,
                "distance_pixels": measurement.calculate_distance_pixels(),
                "distance_mm": measurement.calculate_distance_mm(engine.pixels_per_mm),
                "color": measurement.color,
                "created": measurement.created.isoformat()
            }
            report["measurements"].append(measurement_data)
        
        # Generate text report
        text_report = MeasurementExporter._generate_text_report(report)
        
        if output_path:
            # Save JSON
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save text report
            with open(output_path.with_suffix('.txt'), 'w') as f:
                f.write(text_report)
        
        return report, text_report
    
    @staticmethod
    def _generate_text_report(report_data):
        """Generate human-readable text report"""
        
        lines = []
        lines.append("STAMPZ PRECISION MEASUREMENT REPORT")
        lines.append("=" * 50)
        lines.append("")
        
        # Metadata
        meta = report_data["metadata"]
        lines.append("IMAGE INFORMATION:")
        lines.append(f"  File: {Path(meta['image_file']).name}")
        lines.append(f"  Size: {meta['image_size'][0]}x{meta['image_size'][1]} pixels")
        lines.append(f"  DPI: {meta['dpi']} ({meta['calibration_source']})")
        lines.append(f"  Precision: ±{meta['precision_um']:.1f}µm")
        lines.append(f"  Export Date: {meta['export_date'][:19]}")
        lines.append("")
        
        # Measurements
        measurements = report_data["measurements"]
        if measurements:
            lines.append(f"MEASUREMENTS ({len(measurements)} total):")
            lines.append("")
            
            for i, measurement in enumerate(measurements, 1):
                lines.append(f"  {i}. {measurement['label'] if measurement['label'] else f'Measurement {i}'}")
                if measurement['distance_mm']:
                    lines.append(f"     Distance: {measurement['distance_mm']:.3f}mm")
                else:
                    lines.append(f"     Distance: {measurement['distance_pixels']:.1f} pixels")
                lines.append(f"     From: ({measurement['start_point'][0]:.0f}, {measurement['start_point'][1]:.0f})")
                lines.append(f"     To: ({measurement['end_point'][0]:.0f}, {measurement['end_point'][1]:.0f})")
                lines.append(f"     Type: {measurement['type']}")
                lines.append("")
        else:
            lines.append("No measurements recorded.")
            lines.append("")
        
        # Capabilities
        if meta['precision_um']:
            lines.append("MEASUREMENT CAPABILITIES:")
            if meta['precision_um'] < 10:
                lines.append("  ✅ Micro-measurements (<0.05mm)")
            if meta['precision_um'] < 25:
                lines.append("  ✅ Perforation spacing analysis")
            if meta['precision_um'] < 30:
                lines.append("  ✅ Fraud detection")
            if meta['precision_um'] < 50:
                lines.append("  ✅ Plate variety identification")
            if meta['precision_um'] < 100:
                lines.append("  ✅ Overprint analysis")
            lines.append("  ✅ Basic measurements")
        
        return "\n".join(lines)

# Demo and test functions
def demo_measurement_engine():
    """Demonstrate the measurement engine capabilities"""
    
    print("🔧 StampZ Precision Measurement Engine Demo")
    print("=" * 50)
    
    # Create a simulated measurement scenario
    print("\n📊 Testing DPI detection scenarios:")
    
    # Simulate different DPI scenarios
    test_scenarios = [
        {"dpi": 600, "source": "VueScan EXIF"},
        {"dpi": 1200, "source": "High-res scan"},
        {"dpi": 300, "source": "Basic scan"}
    ]
    
    for scenario in test_scenarios:
        print(f"\n  📷 Scenario: {scenario['source']}")
        print(f"     DPI: {scenario['dpi']}")
        
        # Calculate capabilities
        pixels_per_mm = scenario['dpi'] / 25.4
        precision_um = 1000 / pixels_per_mm / 2
        
        print(f"     Precision: ±{precision_um:.1f}µm")
        print(f"     Pixels/mm: {pixels_per_mm:.1f}")
        
        # Show measurement capabilities
        capabilities = {
            "Fraud Detection": precision_um < 30,
            "Plate Studies": precision_um < 50,
            "Overprint Analysis": precision_um < 100,
            "Perforation Spacing": precision_um < 25
        }
        
        print("     Capabilities:")
        for capability, supported in capabilities.items():
            status = "✅" if supported else "❌"
            print(f"       {status} {capability}")
    
    # Create sample measurements
    print(f"\n📐 Sample Measurement Creation:")
    
    measurement1 = ArchitecturalMeasurement(
        (100, 100), (200, 100), 
        measurement_type="horizontal",
        label="Overprint Width",
        color="red"
    )
    
    measurement2 = ArchitecturalMeasurement(
        (150, 80), (150, 120),
        measurement_type="vertical", 
        label="Overprint Height",
        color="blue"
    )
    
    # Test measurement calculations
    pixels_per_mm = 600 / 25.4  # 600 DPI
    
    print(f"  Measurement 1: {measurement1.calculate_distance_mm(pixels_per_mm):.2f}mm")
    print(f"  Measurement 2: {measurement2.calculate_distance_mm(pixels_per_mm):.2f}mm")
    
    print(f"\n✅ Measurement engine ready for StampZ integration!")

if __name__ == "__main__":
    demo_measurement_engine()