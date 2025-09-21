#!/usr/bin/env python3
"""
Test the integrated gauge perforation measurement system.
"""

import tkinter as tk
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gauge_integration():
    """Test the gauge perforation dialog integration."""
    
    root = tk.Tk()
    root.title("StampZ Gauge Integration Test")
    root.geometry("300x200")
    
    # Test button to launch gauge dialog
    def launch_gauge():
        try:
            from gui.gauge_perforation_ui import GaugePerforationDialog
            dialog = GaugePerforationDialog(root)
            print("✓ Gauge perforation dialog launched successfully")
        except Exception as e:
            print(f"✗ Error launching gauge dialog: {e}")
            import traceback
            traceback.print_exc()
    
    # Test button to check final gauge system
    def test_gauge_system():
        try:
            from final_perforation_gauge import FinalPerforationGauge
            gauge = FinalPerforationGauge(dpi=800)
            overlay = gauge.create_gauge_overlay(800, 600)
            print(f"✓ Gauge system works - overlay shape: {overlay.shape}")
        except Exception as e:
            print(f"✗ Error with gauge system: {e}")
            import traceback
            traceback.print_exc()
    
    # UI for testing
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    tk.Label(frame, text="StampZ Gauge Integration Test", 
             font=("Arial", 12, "bold")).pack(pady=10)
    
    tk.Button(frame, text="Test Gauge System", 
              command=test_gauge_system, width=20).pack(pady=5)
    
    tk.Button(frame, text="Launch Gauge Dialog", 
              command=launch_gauge, width=20).pack(pady=5)
    
    tk.Button(frame, text="Close", 
              command=root.destroy, width=20).pack(pady=10)
    
    # Run initial tests
    print("Running integration tests...")
    print("=" * 40)
    test_gauge_system()
    print("=" * 40)
    print("Ready to test dialog. Click 'Launch Gauge Dialog' button.")
    
    root.mainloop()


if __name__ == "__main__":
    test_gauge_integration()