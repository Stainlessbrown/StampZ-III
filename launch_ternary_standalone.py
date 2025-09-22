#!/usr/bin/env python3
"""
Standalone Ternary Plot Launcher

This script allows you to directly launch the ternary plot window to visualize
existing color analysis databases without going through the full StampZ-III workflow.

Usage:
  python3 launch_ternary_standalone.py

This will show a dialog to select from available databases and launch the ternary plot.
"""

import sys
import os
sys.path.append('/Users/stanbrown/Desktop/StampZ-III')

import tkinter as tk
from tkinter import ttk, messagebox
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def select_and_launch_ternary():
    """Show database selection dialog and launch ternary plot."""
    try:
        # Import required modules
        from utils.color_data_bridge import ColorDataBridge
        from gui.ternary_plot_window import TernaryPlotWindow
        
        # Get available databases
        bridge = ColorDataBridge()
        sample_sets = bridge.get_available_sample_sets()
        
        if not sample_sets:
            messagebox.showinfo(
                "No Databases Found",
                "No color analysis databases were found.\n\n"
                "Run color analysis in StampZ-III first to create databases."
            )
            return
        
        # Create selection dialog
        root = tk.Tk()
        root.withdraw()  # Hide root window
        
        dialog = tk.Toplevel(root)
        dialog.title("Select Database for Ternary Analysis")
        dialog.geometry("500x400")
        dialog.resizable(True, True)
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Make it modal
        dialog.transient(root)
        dialog.grab_set()
        
        # Header
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=15, pady=(15, 10))
        
        ttk.Label(
            header_frame,
            text="🎨 Ternary Plot Database Launcher",
            font=('Arial', 14, 'bold')
        ).pack()
        
        ttk.Label(
            header_frame,
            text="Select a color analysis database to visualize in ternary plot:",
            font=('Arial', 10)
        ).pack(pady=(5, 0))
        
        # Database list
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Listbox with scrollbar
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Populate list
        for sample_set in sample_sets:
            listbox.insert(tk.END, sample_set)
        
        # Select first item by default
        if sample_sets:
            listbox.selection_set(0)
        
        # Info display
        info_frame = ttk.LabelFrame(list_frame, text="Database Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        info_text = tk.Text(info_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        info_text.pack(fill=tk.BOTH, expand=True)
        
        def update_info(event=None):
            """Update info display when selection changes."""
            selection = listbox.curselection()
            if selection:
                selected_set = sample_sets[selection[0]]
                try:
                    # Load basic info about the database
                    points, db_format = bridge.load_color_points_from_database(selected_set)
                    
                    # Count points with preferences
                    points_with_prefs = sum(1 for p in points 
                                          if hasattr(p, 'metadata') and 
                                             (p.metadata.get('marker', '.') != '.' or 
                                              p.metadata.get('marker_color', 'blue') != 'blue'))
                    
                    info_text.config(state=tk.NORMAL)
                    info_text.delete(1.0, tk.END)
                    info_text.insert(tk.END, 
                        f"Database: {selected_set}\\n"
                        f"Format: {db_format}\\n"
                        f"Total Points: {len(points)}\\n"
                        f"Custom Preferences: {points_with_prefs} points"
                    )
                    info_text.config(state=tk.DISABLED)
                except Exception as e:
                    info_text.config(state=tk.NORMAL)
                    info_text.delete(1.0, tk.END)
                    info_text.insert(tk.END, f"Error loading info: {e}")
                    info_text.config(state=tk.DISABLED)
        
        listbox.bind('<<ListboxSelect>>', update_info)
        update_info()  # Initial update
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=15, pady=(10, 15))
        
        def on_cancel():
            dialog.destroy()
            root.quit()
        
        def on_launch():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Selection Required", "Please select a database.")
                return
            
            selected_set = sample_sets[selection[0]]
            dialog.destroy()
            
            try:
                # Load the data
                points, db_format = bridge.load_color_points_from_database(selected_set)
                
                if not points:
                    messagebox.showerror("Load Error", f"No data found in database: {selected_set}")
                    root.quit()
                    return
                
                logger.info(f"Launching ternary plot for {selected_set} with {len(points)} points")
                
                # Create ternary window
                ternary_window = TernaryPlotWindow(
                    parent=root,
                    sample_set_name=selected_set,
                    color_points=points,
                    db_format=db_format
                )
                
                # Show root window and start main loop
                root.deiconify()
                root.withdraw()  # Keep root hidden
                root.mainloop()
                
            except Exception as e:
                logger.exception("Failed to launch ternary window")
                messagebox.showerror("Launch Error", f"Failed to launch ternary plot:\\n\\n{e}")
                root.quit()
        
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Launch Ternary Plot", command=on_launch).pack(side=tk.RIGHT, padx=(0, 10))
        
        # Start the dialog
        dialog.mainloop()
        
    except Exception as e:
        logger.exception("Failed to create database selection dialog")
        messagebox.showerror("Error", f"Failed to start database launcher:\\n\\n{e}")

def main():
    """Main entry point."""
    print("🎨 StampZ-III Standalone Ternary Plot Launcher")
    print("=" * 50)
    
    # Check if we have any databases
    try:
        from utils.color_data_bridge import ColorDataBridge
        bridge = ColorDataBridge()
        sample_sets = bridge.get_available_sample_sets()
        print(f"Found {len(sample_sets)} available databases:")
        for i, db in enumerate(sample_sets, 1):
            print(f"  {i}. {db}")
        print()
    except Exception as e:
        print(f"Error checking databases: {e}")
        return
    
    if sample_sets:
        select_and_launch_ternary()
    else:
        print("❌ No color analysis databases found.")
        print("💡 Run color analysis in StampZ-III first to create databases.")

if __name__ == "__main__":
    main()