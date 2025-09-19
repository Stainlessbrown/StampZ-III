#!/usr/bin/env python3
"""
StampZ Precision Measurement Tool Concept
Architectural-style dimension lines with calibration for real measurements
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np

class PhilatelicMeasurementTool:
    def __init__(self, parent=None):
        if parent is None:
            self.root = tk.Tk()
            self.root.title("StampZ Precision Measurement Tool")
            self.root.geometry("1000x700")
        else:
            self.root = tk.Toplevel(parent)
            self.root.title("Precision Measurement")
            self.root.geometry("1000x700")
        
        self.measurements = []  # Store all measurements
        self.pixels_per_mm = 100  # Default calibration (will be user-set)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the measurement tool interface"""
        
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill="x")
        
        # Calibration section
        cal_frame = ttk.LabelFrame(control_frame, text="Calibration", padding="5")
        cal_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(cal_frame, text="Pixels per mm:").grid(row=0, column=0, sticky="w", padx=5)
        self.cal_var = tk.DoubleVar(value=self.pixels_per_mm)
        cal_spinbox = ttk.Spinbox(cal_frame, from_=10, to=500, increment=1, 
                                 textvariable=self.cal_var, width=10,
                                 command=self.update_calibration)
        cal_spinbox.grid(row=0, column=1, padx=5)
        
        ttk.Button(cal_frame, text="Auto-Calibrate", 
                  command=self.auto_calibrate).grid(row=0, column=2, padx=10)
        
        # Measurement tools
        tools_frame = ttk.LabelFrame(control_frame, text="Measurement Tools", padding="5")
        tools_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(tools_frame, text="Add Horizontal Measurement", 
                  command=lambda: self.add_measurement_mode("horizontal")).grid(row=0, column=0, padx=5)
        ttk.Button(tools_frame, text="Add Vertical Measurement", 
                  command=lambda: self.add_measurement_mode("vertical")).grid(row=0, column=1, padx=5)
        ttk.Button(tools_frame, text="Clear All", 
                  command=self.clear_measurements).grid(row=0, column=2, padx=5)
        
        # Precision settings
        precision_frame = ttk.LabelFrame(control_frame, text="Precision", padding="5")
        precision_frame.pack(fill="x")
        
        ttk.Label(precision_frame, text="Decimal places:").grid(row=0, column=0, sticky="w", padx=5)
        self.precision_var = tk.IntVar(value=2)
        precision_spinbox = ttk.Spinbox(precision_frame, from_=0, to=4, 
                                       textvariable=self.precision_var, width=5)
        precision_spinbox.grid(row=0, column=1, padx=5)
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 6))\n        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)\n        self.canvas.draw()\n        self.canvas.get_tk_widget().pack(fill=\"both\", expand=True)\n        \n        # Load demo image\n        self.create_demo_stamp()\n        \n    def create_demo_stamp(self):\n        \"\"\"Create a demo stamp for measurement demonstration\"\"\"\n        self.ax.clear()\n        \n        # Simulate a stamp with overprint (classic fraud detection scenario)\n        stamp_rect = patches.Rectangle((100, 100), 220, 260, \n                                      linewidth=2, edgecolor='black', \n                                      facecolor='lightblue', alpha=0.3)\n        self.ax.add_patch(stamp_rect)\n        \n        # Add stamp elements\n        vignette = patches.Rectangle((130, 200), 160, 120, \n                                   linewidth=1, edgecolor='navy', \n                                   facecolor='lightcyan', alpha=0.5)\n        self.ax.add_patch(vignette)\n        \n        # Suspicious overprint\n        overprint = patches.Rectangle((160, 250), 80, 30, \n                                    linewidth=2, edgecolor='red', \n                                    facecolor='pink', alpha=0.8)\n        self.ax.add_patch(overprint)\n        \n        self.ax.text(210, 140, \"DEMO STAMP\", ha='center', fontsize=10, weight='bold')\n        self.ax.text(200, 265, \"OVERPRINT\", ha='center', fontsize=8, weight='bold', color='red')\n        \n        self.ax.set_xlim(50, 400)\n        self.ax.set_ylim(50, 400)\n        self.ax.set_aspect('equal')\n        self.ax.grid(True, alpha=0.3)\n        self.ax.set_title(\"Click two points to create measurements\", fontsize=12)\n        \n        self.canvas.draw()\n    \n    def draw_dimension_line(self, start, end, offset=20, color='red', label=\"\"):\n        \"\"\"Draw architectural-style dimension line\"\"\"\n        x1, y1 = start\n        x2, y2 = end\n        \n        # Convert pixels to mm for display\n        distance_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)\n        distance_mm = distance_pixels / self.pixels_per_mm\n        \n        # Determine if horizontal or vertical measurement\n        if abs(x2 - x1) > abs(y2 - y1):  # Horizontal\n            dim_y = max(y1, y2) + offset\n            # Extension lines\n            self.ax.plot([x1, x1], [y1, dim_y + 5], color=color, linewidth=1)\n            self.ax.plot([x2, x2], [y2, dim_y + 5], color=color, linewidth=1)\n            # Dimension line with arrows\n            arrow = FancyArrowPatch((x1, dim_y), (x2, dim_y),\n                                   arrowstyle='<->', mutation_scale=15,\n                                   color=color, linewidth=1.5)\n            self.ax.add_patch(arrow)\n            # Text\n            text_pos = ((x1 + x2) / 2, dim_y + 12)\n            rotation = 0\n        else:  # Vertical\n            dim_x = max(x1, x2) + offset\n            # Extension lines\n            self.ax.plot([x1, dim_x + 5], [y1, y1], color=color, linewidth=1)\n            self.ax.plot([x2, dim_x + 5], [y2, y2], color=color, linewidth=1)\n            # Dimension line with arrows\n            arrow = FancyArrowPatch((dim_x, y1), (dim_x, y2),\n                                   arrowstyle='<->', mutation_scale=15,\n                                   color=color, linewidth=1.5)\n            self.ax.add_patch(arrow)\n            # Text\n            text_pos = (dim_x + 15, (y1 + y2) / 2)\n            rotation = 90\n        \n        # Display measurement with precision\n        precision = self.precision_var.get()\n        text = label if label else f\"{distance_mm:.{precision}f}mm\"\n        \n        self.ax.text(text_pos[0], text_pos[1], text, \n                    ha='center', va='center', color=color, fontsize=9, \n                    rotation=rotation, weight='bold',\n                    bbox=dict(boxstyle=\"round,pad=0.3\", facecolor='white', alpha=0.9))\n    \n    def update_calibration(self):\n        \"\"\"Update calibration and redraw measurements\"\"\"\n        self.pixels_per_mm = self.cal_var.get()\n        self.redraw_measurements()\n    \n    def auto_calibrate(self):\n        \"\"\"Auto-calibrate using known stamp dimension\"\"\"\n        # This would normally use user input or known references\n        # For demo, assume the stamp is 22mm wide\n        stamp_width_pixels = 220  # From our demo stamp\n        known_width_mm = 22.0\n        \n        self.pixels_per_mm = stamp_width_pixels / known_width_mm\n        self.cal_var.set(self.pixels_per_mm)\n        \n        print(f\"Auto-calibrated: {self.pixels_per_mm:.1f} pixels/mm\")\n        self.redraw_measurements()\n    \n    def add_measurement_mode(self, direction):\n        \"\"\"Add measurement interactively\"\"\"\n        print(f\"Click two points to add {direction} measurement\")\n        # In real implementation, this would enable click handlers\n        \n        # Demo: add a measurement\n        if direction == \"horizontal\":\n            self.measurements.append({\n                'start': (160, 250),\n                'end': (240, 250),\n                'type': 'horizontal',\n                'color': 'red',\n                'label': 'Overprint Width'\n            })\n        else:\n            self.measurements.append({\n                'start': (200, 250),\n                'end': (200, 280),\n                'type': 'vertical',\n                'color': 'blue',\n                'label': 'Overprint Height'\n            })\n        \n        self.redraw_measurements()\n    \n    def clear_measurements(self):\n        \"\"\"Clear all measurements\"\"\"\n        self.measurements.clear()\n        self.create_demo_stamp()\n    \n    def redraw_measurements(self):\n        \"\"\"Redraw all measurements with current calibration\"\"\"\n        self.create_demo_stamp()\n        \n        for i, measurement in enumerate(self.measurements):\n            offset = 25 + (i * 15)  # Stagger multiple measurements\n            self.draw_dimension_line(\n                measurement['start'], \n                measurement['end'],\n                offset=offset,\n                color=measurement['color'],\n                label=measurement.get('label', '')\n            )\n        \n        self.canvas.draw()\n    \n    def export_measurements(self):\n        \"\"\"Export measurements for documentation\"\"\"\n        report = \"PHILATELIC MEASUREMENT REPORT\\n\"\n        report += \"=\" * 35 + \"\\n\\n\"\n        report += f\"Calibration: {self.pixels_per_mm:.1f} pixels/mm\\n\\n\"\n        \n        for i, measurement in enumerate(self.measurements, 1):\n            start = measurement['start']\n            end = measurement['end']\n            distance_pixels = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)\n            distance_mm = distance_pixels / self.pixels_per_mm\n            precision = self.precision_var.get()\n            \n            report += f\"Measurement {i}: {distance_mm:.{precision}f}mm\\n\"\n            report += f\"  Type: {measurement['type']}\\n\"\n            report += f\"  From: ({start[0]:.0f}, {start[1]:.0f})\\n\"\n            report += f\"  To: ({end[0]:.0f}, {end[1]:.0f})\\n\\n\"\n        \n        return report

def demo_measurement_tool():
    \"\"\"Run the measurement tool demo\"\"\"\n    print(\"🔧 StampZ Measurement Tool Concept\")\n    print(\"=\" * 35)\n    print(\"Features:\")\n    print(\"  ✅ Architectural-style dimension lines with |<--->| format\")\n    print(\"  ✅ Extension lines for precise endpoint definition\")\n    print(\"  ✅ Real-world calibration (pixels to millimeters)\")\n    print(\"  ✅ Multiple simultaneous measurements\")\n    print(\"  ✅ Adjustable precision (0.01mm capability)\")\n    print(\"  ✅ Perfect for fraud detection and plate studies\")\n    print()\n    print(\"Applications:\")\n    print(\"  🔍 Overprint positioning analysis\")\n    print(\"  🔍 Plate variety identification\")\n    print(\"  🔍 Authentication measurements\")\n    print(\"  🔍 Perforation spacing analysis\")\n    \n    # Create the tool\n    tool = PhilatelicMeasurementTool()\n    tool.root.mainloop()\n\nif __name__ == \"__main__\":\n    demo_measurement_tool()