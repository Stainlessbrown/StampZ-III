#!/usr/bin/env python3
"""
Standalone Plot_3D Launcher

Allows Plot_3D to be used independently of StampZ for final analysis work:
- ΔE calculations
- K-means clustering 
- Centroid manipulation
- Data sorting and filtering
- Outlier highlighting/elimination

This addresses the workflow where StampZ does initial color analysis,
then Plot_3D becomes the primary tool for advanced analysis.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os

class StandalonePlot3DLauncher:
    """Launcher for standalone Plot_3D usage."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Plot_3D Standalone Launcher")
        self.root.geometry("600x400")
        self.root.resizable(True, True)
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self._create_interface()
    
    def _create_interface(self):
        """Create the launcher interface."""
        
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Label(title_frame, text="Plot_3D Standalone", 
                 font=("Arial", 18, "bold")).pack()
        ttk.Label(title_frame, text="Advanced 3D Color Analysis", 
                 font=("Arial", 12), foreground="blue").pack(pady=5)
        
        # Description
        desc_frame = ttk.LabelFrame(self.root, text="Capabilities", padding=15)
        desc_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        capabilities = [
            "✅ ΔE calculations and analysis",
            "✅ K-means clustering with customizable parameters", 
            "✅ Centroid manipulation and optimization",
            "✅ Advanced data sorting and filtering",
            "✅ Outlier detection and elimination",
            "✅ Interactive 3D visualization with zoom/rotation",
            "✅ Export results back to StampZ format"
        ]
        
        for capability in capabilities:
            ttk.Label(desc_frame, text=capability, font=("Arial", 10)).pack(anchor='w', pady=1)
        
        # Launch options
        options_frame = ttk.LabelFrame(self.root, text="Launch Options", padding=15)
        options_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Option 1: Load existing file
        file_frame = ttk.Frame(options_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="📊 Load Plot_3D File", 
                  command=self._load_existing_file, width=20).pack(side=tk.LEFT)
        ttk.Label(file_frame, text="Open existing .ods/.xlsx file with Plot_3D data", 
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Option 2: Create new analysis
        new_frame = ttk.Frame(options_frame)
        new_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(new_frame, text="🆕 New Analysis", 
                  command=self._create_new_analysis, width=20).pack(side=tk.LEFT)
        ttk.Label(new_frame, text="Start with empty Plot_3D workspace", 
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Option 3: Import from StampZ
        import_frame = ttk.Frame(options_frame)
        import_frame.pack(fill=tk.X)
        
        ttk.Button(import_frame, text="📥 Import StampZ Data", 
                  command=self._import_stampz_data, width=20).pack(side=tk.LEFT)
        ttk.Label(import_frame, text="Import data from StampZ analysis database", 
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Workflow note
        workflow_frame = ttk.LabelFrame(self.root, text="Typical Workflow", padding=15)
        workflow_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        workflow_text = """1. Complete initial color analysis in StampZ
2. Export data to Plot_3D format or use real-time spreadsheet
3. Launch Plot_3D standalone for advanced analysis:
   • Apply K-means clustering to identify color groups
   • Calculate ΔE values between samples and centroids
   • Manipulate cluster centroids for optimal grouping  
   • Identify and handle outliers
   • Fine-tune visualization parameters
4. Save final results back to Plot_3D format"""
        
        ttk.Label(workflow_frame, text=workflow_text, font=("Arial", 9), 
                 justify=tk.LEFT, wraplength=550).pack(anchor='w')
        
        # Close button
        ttk.Button(self.root, text="Exit", command=self.root.quit).pack(pady=10)
    
    def _load_existing_file(self):
        """Load existing Plot_3D file."""
        file_path = filedialog.askopenfilename(
            title="Load Plot_3D Data File",
            filetypes=[
                ('OpenDocument Spreadsheet', '*.ods'),
                ('Excel Workbook', '*.xlsx'),
                ('All files', '*.*')
            ]
        )
        
        if file_path:
            self._launch_plot3d(data_path=file_path)
    
    def _create_new_analysis(self):
        """Create new empty analysis."""
        self._launch_plot3d()
    
    def _import_stampz_data(self):
        """Import data from StampZ analysis database."""
        messagebox.showinfo(
            "Import StampZ Data",
            "This feature will allow you to browse and import data\n"
            "directly from StampZ analysis databases.\n\n"
            "For now, please use the StampZ real-time spreadsheet\n"
            "to export data, then use 'Load Plot_3D File'."
        )
    
    def _launch_plot3d(self, data_path=None):
        """Launch Plot_3D application."""
        try:
            from plot3d.Plot_3D import Plot3DApp
            
            # Close launcher window
            self.root.destroy()
            
            # Create new root for Plot_3D
            plot3d_root = tk.Tk()
            
            # Launch Plot_3D
            if data_path:
                plot3d_app = Plot3DApp(parent=plot3d_root, data_path=data_path)
                print(f"Launched Plot_3D standalone with file: {data_path}")
            else:
                plot3d_app = Plot3DApp(parent=plot3d_root)
                print("Launched Plot_3D standalone with empty workspace")
            
            # Run Plot_3D
            plot3d_root.mainloop()
            
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch Plot_3D:\n{e}")
    
    def run(self):
        """Run the launcher."""
        self.root.mainloop()


def main():
    """Main entry point for standalone Plot_3D."""
    print("🚀 Starting Plot_3D Standalone Launcher...")
    
    launcher = StandalonePlot3DLauncher()
    launcher.run()
    
    print("👋 Plot_3D Standalone session ended")


if __name__ == "__main__":
    main()
