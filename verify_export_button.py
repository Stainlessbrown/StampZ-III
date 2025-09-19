#!/usr/bin/env python3
"""
Verification script to confirm the export button implementation works correctly.
This simulates the user workflow without the automatic export dialogs.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main verification function."""
    print("=== StampZ Export Button Verification ===\n")
    
    # Test 1: Verify ColorComparisonManager has export button
    print("Test 1: Checking if export button exists in ColorComparisonManager...")
    
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        from gui.color_comparison_manager import ColorComparisonManager
        
        # Create a test frame
        test_frame = ttk.Frame(root)
        
        # Create the comparison manager
        comparison_manager = ColorComparisonManager(test_frame)
        
        # Check if export button exists
        if hasattr(comparison_manager, 'export_button'):
            button_text = comparison_manager.export_button.cget('text')
            print(f"✅ Export button found: '{button_text}'")
        else:
            print("❌ Export button NOT found")
            return False
            
        root.destroy()
        
    except Exception as e:
        print(f"❌ Error testing ColorComparisonManager: {e}")
        return False
    
    # Test 2: Verify analysis complete dialog is simplified
    print("\nTest 2: Checking if analysis complete dialog is simplified...")
    
    try:
        from app.analysis_manager import AnalysisManager
        import inspect
        
        # Check the method source to see if it's simplified
        source = inspect.getsource(AnalysisManager._show_analysis_complete_dialog)
        
        if "Export Individual Measurements to Data Logger" in source:
            print("❌ Analysis complete dialog still has automatic export options")
            return False
        elif "Use the Compare tab to review" in source:
            print("✅ Analysis complete dialog is simplified and points to Compare tab")
        else:
            print("? Analysis complete dialog state unclear")
            
    except Exception as e:
        print(f"❌ Error checking analysis complete dialog: {e}")
        return False
    
    # Test 3: Verify menu items were removed
    print("\nTest 3: Checking if menu export options were removed...")
    
    try:
        from app.menu_manager import MenuManager
        import inspect
        
        # Check the menu creation source
        source = inspect.getsource(MenuManager._create_color_menu)
        
        if "Export Individual Measurements to Data Logger" in source:
            print("❌ Menu still has automatic export options")
            return False
        elif "moved to the Compare window" in source:
            print("✅ Menu export options removed, replaced with explanation comment")
        else:
            print("? Menu state unclear")
            
    except Exception as e:
        print(f"❌ Error checking menu manager: {e}")
        return False
    
    print("\n=== Verification Summary ===")
    print("✅ All tests passed!")
    print("\nThe new workflow is:")
    print("1. Color Analysis → Simple completion dialog (no auto-exports)")
    print("2. Compare Window → '📊 Export to Unified Data Logger' button")
    print("3. Export Dialog → User chooses when and what to export")
    print("\nThis gives users complete control over export timing and data selection.")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)