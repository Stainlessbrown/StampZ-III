#!/usr/bin/env python3
"""
StampZ-III Launch Mode Selector

Shows a simple dialog allowing users to choose between:
1. Full StampZ-III Application (complete image analysis workflow)
2. Plot_3D Only Mode (advanced 3D analysis and visualization)

This provides a user-friendly way to access Plot_3D functionality without
requiring separate Python installations or terminal usage.
"""

import tkinter as tk
from tkinter import ttk
import os
import sys


class LaunchSelector:
    """Launch mode selector dialog for StampZ-III."""
    
    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
            self.own_root = True
        else:
            self.root = root
            self.own_root = False
            
        self.root.title("StampZ-III Launcher")
        self.selected_mode = None
        
        self._setup_window()
        self._create_interface()
        
    def _setup_window(self):
        """Configure the launch selector window."""
        # Set window size and make it non-resizable for clean appearance
        window_width = 600
        window_height = 650  # Further increased height to show both buttons
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(False, False)
        
        # Center window on screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Configure styling
        self.root.configure(bg='white')
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
    def _create_interface(self):
        """Create the launch selector interface."""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(
            title_frame, 
            text="StampZ-III", 
            font=("Arial", 24, "bold")
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Choose your analysis mode",
            font=("Arial", 14),
            foreground="gray"
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Options section
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        # Option 1: Full StampZ-III
        full_frame = ttk.LabelFrame(options_frame, text="Complete Analysis Workflow", padding=12)
        full_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            full_frame,
            text="🔬 Full StampZ-III Application",
            font=("Arial", 16, "bold")
        ).pack(anchor='w')
        
        ttk.Label(
            full_frame,
            text="• Complete image analysis workflow\n"
                 "• Color measurement and comparison\n"
                 "• Database management and exports",
            font=("Arial", 11),
            foreground="gray"
        ).pack(anchor='w', pady=(8, 12))
        
        full_button = ttk.Button(
            full_frame,
            text="Open Full StampZ-III",
            command=self._launch_full_app
        )
        full_button.pack(anchor='w')
        
        # Option 2: Plot_3D Only
        plot3d_frame = ttk.LabelFrame(options_frame, text="Advanced 3D Analysis", padding=12)
        plot3d_frame.pack(fill=tk.X)
        
        ttk.Label(
            plot3d_frame,
            text="📊 Plot_3D Only Mode",
            font=("Arial", 16, "bold")
        ).pack(anchor='w')
        
        ttk.Label(
            plot3d_frame,
            text="• Advanced 3D color space analysis\n"
                 "• K-means clustering and ΔE calculations\n"
                 "• Interactive visualization and statistics",
            font=("Arial", 11),
            foreground="gray"
        ).pack(anchor='w', pady=(8, 12))
        
        plot3d_button = ttk.Button(
            plot3d_frame,
            text="Open Plot_3D Only",
            command=self._launch_plot3d_only
        )
        plot3d_button.pack(anchor='w')
        
        # Footer with workflow note
        footer_frame = ttk.Frame(main_frame)
        footer_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Label(
            footer_frame,
            text="💡 Tip: Switch modes using Tools menu inside application",
            font=("Arial", 9),
            foreground="blue"
        ).pack()
        
        # Set initial focus
        full_button.focus_set()
        
    def _launch_full_app(self):
        """Launch the full StampZ-III application."""
        self.selected_mode = "full"
        self.root.quit()  # Stop the mainloop
        
    def _launch_plot3d_only(self):
        """Launch Plot_3D only mode."""
        self.selected_mode = "plot3d"
        self.root.quit()  # Stop the mainloop
        
    def _on_cancel(self):
        """Handle window closing."""
        self.selected_mode = None
        self.root.quit()  # Stop the mainloop
        
    def show(self):
        """Show the launch selector and return the selected mode."""
        # Show the window
        self.root.deiconify()
        
        try:
            self.root.mainloop()
        except:
            pass  # Ignore errors during mainloop
        finally:
            # Hide the window but don't destroy if we don't own it
            try:
                if self.root and self.root.winfo_exists():
                    self.root.withdraw()  # Hide instead of destroy
                    if self.own_root:
                        self.root.quit()
                        self.root.destroy()
            except:
                pass
        return self.selected_mode


def main():
    """Main function for testing the launch selector."""
    selector = LaunchSelector()
    mode = selector.show()
    
    if mode == "full":
        print("User selected: Full StampZ-III Application")
    elif mode == "plot3d":
        print("User selected: Plot_3D Only Mode")
    else:
        print("User cancelled or closed dialog")
    
    return mode


if __name__ == "__main__":
    main()
