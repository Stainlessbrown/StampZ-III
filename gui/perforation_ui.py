"""
Perforation Measurement UI for StampZ

Provides a user interface for measuring stamp perforations with visual overlay
and data logging integration.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import os
from typing import Optional, Tuple, List
import threading

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perforation_measurement_system import (
    PerforationMeasurementEngine, 
    PerforationAnalysis, 
    PerforationHole
)


class PerforationMeasurementDialog:
    """Dialog for measuring stamp perforations."""
    
    def __init__(self, parent, image_array: np.ndarray = None, image_filename: str = ""):
        self.parent = parent
        self.image_array = image_array
        self.image_filename = image_filename
        self.engine = PerforationMeasurementEngine()
        self.analysis_result = None
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("StampZ - Perforation Measurement")
        self.dialog.geometry("900x700")
        self.dialog.resizable(True, True)
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Create UI components
        self._create_widgets()
        self._setup_layout()
        
        # If we have an image, display it
        if image_array is not None:
            self._display_image(image_array)
    
    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container
        self.main_frame = ttk.Frame(self.dialog)
        
        # Left panel for image display
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Stamp Image")
        self.image_canvas = tk.Canvas(self.image_frame, bg='white', width=500, height=400)
        self.image_scrollbar_v = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_scrollbar_h = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=self.image_scrollbar_v.set, xscrollcommand=self.image_scrollbar_h.set)
        
        # Right panel for controls and results
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Measurement Controls")
        
        # DPI Settings
        self.dpi_frame = ttk.Frame(self.control_frame)
        ttk.Label(self.dpi_frame, text="Image DPI:").pack(side=tk.LEFT)
        
        # Get default DPI and background color from preferences
        default_dpi = "600"  # fallback
        default_bg_color = "black"  # fallback
        try:
            from utils.user_preferences import get_preferences_manager
            prefs_manager = get_preferences_manager()
            default_dpi = str(prefs_manager.get_default_dpi())
            default_bg_color = prefs_manager.get_default_background_color()
        except:
            pass  # Use fallback if preferences not available
            
        self.dpi_var = tk.StringVar(value=default_dpi)
        self.dpi_entry = ttk.Entry(self.dpi_frame, textvariable=self.dpi_var, width=8)
        self.dpi_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(self.dpi_frame, text="Set", command=self._update_dpi).pack(side=tk.LEFT, padx=5)
        
        # Add button to save as default
        ttk.Button(self.dpi_frame, text="Save as Default", command=self._save_dpi_as_default).pack(side=tk.LEFT, padx=5)
        
        # Background color selection
        self.bg_color_frame = ttk.Frame(self.control_frame)
        ttk.Label(self.bg_color_frame, text="Scan Background:").pack(side=tk.LEFT)
        
        self.bg_color_var = tk.StringVar(value=default_bg_color)
        bg_combo = ttk.Combobox(
            self.bg_color_frame,
            textvariable=self.bg_color_var,
            values=['black', 'dark_gray', 'white', 'light_gray'],
            state='readonly',
            width=12
        )
        bg_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(
            self.bg_color_frame,
            text="(color around stamp)",
            font=("TkDefaultFont", 9),
            foreground="gray"
        ).pack(side=tk.LEFT, padx=5)
        
        # Measurement buttons
        self.button_frame = ttk.Frame(self.control_frame)
        self.measure_btn = ttk.Button(
            self.button_frame, 
            text="🔍 Measure Perforations",
            command=self._start_measurement
        )
        self.load_image_btn = ttk.Button(
            self.button_frame,
            text="📁 Load Image", 
            command=self._load_image
        )
        self.save_data_btn = ttk.Button(
            self.button_frame,
            text="💾 Save Data",
            command=self._save_data,
            state='disabled'
        )
        
        # Results display
        self.results_frame = ttk.LabelFrame(self.control_frame, text="Measurement Results")
        self.results_text = tk.Text(self.results_frame, height=15, width=40, wrap=tk.WORD)
        self.results_scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=self.results_scrollbar.set)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(self.control_frame, textvariable=self.progress_var)
        self.progress_bar = ttk.Progressbar(self.control_frame, mode='indeterminate')
        
        # Bottom buttons
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.close_btn = ttk.Button(self.bottom_frame, text="Close", command=self._close_dialog)
        self.help_btn = ttk.Button(self.bottom_frame, text="Help", command=self._show_help)
    
    def _setup_layout(self):
        """Arrange widgets in the dialog."""
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - image display
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Right side - controls
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # DPI controls
        self.dpi_frame.pack(fill=tk.X, pady=5)
        
        # Background color controls
        self.bg_color_frame.pack(fill=tk.X, pady=5)
        
        # Buttons
        self.button_frame.pack(fill=tk.X, pady=5)
        self.measure_btn.pack(fill=tk.X, pady=2)
        self.load_image_btn.pack(fill=tk.X, pady=2)
        self.save_data_btn.pack(fill=tk.X, pady=2)
        
        # Results
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Progress
        self.progress_label.pack(fill=tk.X, pady=2)
        self.progress_bar.pack(fill=tk.X, pady=2)
        
        # Bottom buttons
        self.bottom_frame.pack(fill=tk.X, pady=(10, 0))
        self.help_btn.pack(side=tk.LEFT)
        self.close_btn.pack(side=tk.RIGHT)
    
    def _display_image(self, image_array: np.ndarray):
        """Display the image in the canvas."""
        try:
            # Validate input image
            if image_array is None:
                return
                
            if len(image_array.shape) < 2 or image_array.shape[0] == 0 or image_array.shape[1] == 0:
                return
            
            # Convert from BGR to RGB if needed
            if len(image_array.shape) == 3:
                if image_array.shape[2] == 3:
                    # Assume it's BGR from OpenCV
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
            else:
                image_rgb = image_array
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb.astype('uint8'))
            
            # Ensure canvas has been realized
            self.dialog.update_idletasks()
            
            # Get actual canvas dimensions after it's been rendered
            canvas_width = max(self.image_canvas.winfo_width(), 500)
            canvas_height = max(self.image_canvas.winfo_height(), 400)
            
            # Ensure minimum dimensions
            if canvas_width < 100:
                canvas_width = 500
            if canvas_height < 100:
                canvas_height = 400
            
            # Calculate scaling factor
            scale_w = (canvas_width - 20) / pil_image.width  # Leave some margin
            scale_h = (canvas_height - 20) / pil_image.height
            scale = min(scale_w, scale_h, 1.0)  # Don't scale up
            
            new_width = max(int(pil_image.width * scale), 1)
            new_height = max(int(pil_image.height * scale), 1)
            
            if scale < 1.0:
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            self.photo = ImageTk.PhotoImage(pil_image)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(10, 10, anchor=tk.NW, image=self.photo)
            self.image_canvas.configure(scrollregion=(0, 0, new_width + 20, new_height + 20))
            
            # Store the scaled image for overlay
            self.display_image = pil_image
            self.scale_factor = scale
            
        except Exception as e:
            print(f"Error displaying image: {e}")  # Log but don't show error dialog
    
    def _update_dpi(self):
        """Update the DPI setting for measurements."""
        try:
            dpi = int(self.dpi_var.get())
            if dpi < 72 or dpi > 2400:
                messagebox.showwarning("Invalid DPI", "DPI must be between 72 and 2400")
                return
            self.engine.set_image_dpi(dpi)
            self.progress_var.set(f"DPI set to {dpi}")
        except ValueError:
            messagebox.showerror("Invalid DPI", "Please enter a valid number for DPI")
    
    def _save_dpi_as_default(self):
        """Save the current DPI setting as the default in preferences."""
        try:
            dpi = int(self.dpi_var.get())
            if dpi < 72 or dpi > 2400:
                messagebox.showwarning("Invalid DPI", "DPI must be between 72 and 2400")
                return
                
            from utils.user_preferences import get_preferences_manager
            prefs_manager = get_preferences_manager()
            
            if prefs_manager.set_default_dpi(dpi):
                messagebox.showinfo(
                    "DPI Saved", 
                    f"Default DPI set to {dpi}.\n\nThis will be used as the default for all measurement features."
                )
            else:
                messagebox.showerror("Error", "Failed to save DPI setting.")
                
        except ValueError:
            messagebox.showerror("Invalid DPI", "Please enter a valid number for DPI")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save DPI setting: {str(e)}")
    
    def _load_image(self):
        """Load an image file for analysis."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Stamp Image",
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Load image using OpenCV
                image = cv2.imread(filename)
                if image is None:
                    messagebox.showerror("Error", "Could not load image file")
                    return
                
                self.image_array = image
                self.image_filename = filename
                self._display_image(image)
                self.progress_var.set(f"Loaded: {os.path.basename(filename)}")
                
                # Clear previous results
                self.results_text.delete(1.0, tk.END)
                self.save_data_btn.configure(state='disabled')
                self.analysis_result = None
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _start_measurement(self):
        """Start the perforation measurement process."""
        if self.image_array is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        # Update DPI from entry
        self._update_dpi()
        
        # Update background color
        self.engine.set_background_color(self.bg_color_var.get())
        
        # Start measurement in background thread
        self.measure_btn.configure(state='disabled')
        self.progress_bar.start()
        self.progress_var.set("Analyzing perforations...")
        
        thread = threading.Thread(target=self._run_measurement)
        thread.daemon = True
        thread.start()
    
    def _run_measurement(self):
        """Run the measurement in a background thread."""
        try:
            # Perform the measurement using improved hole-based detection
            analysis = self.engine.measure_perforation(self.image_array, use_hole_detection=True)
            
            # Update UI in main thread
            self.dialog.after(0, self._measurement_complete, analysis)
            
        except Exception as e:
            self.dialog.after(0, self._measurement_error, str(e))
    
    def _measurement_complete(self, analysis: PerforationAnalysis):
        """Handle completed measurement."""
        self.analysis_result = analysis
        
        # Stop progress bar
        self.progress_bar.stop()
        self.measure_btn.configure(state='normal')
        self.save_data_btn.configure(state='normal')
        
        # Display results
        self._display_results(analysis)
        
        # Draw overlay if possible
        self._draw_perforation_overlay(analysis)
        
        # Update status
        if len(analysis.edges) > 0:
            self.progress_var.set(f"✅ Analysis complete - {len(analysis.edges)} edges detected")
        else:
            self.progress_var.set(f"⚠️ Analysis complete - no perforations detected")
    
    def _measurement_error(self, error_msg: str):
        """Handle measurement error."""
        self.progress_bar.stop()
        self.measure_btn.configure(state='normal')
        self.progress_var.set("❌ Analysis failed")
        messagebox.showerror("Measurement Error", f"Analysis failed: {error_msg}")
    
    def _display_results(self, analysis: PerforationAnalysis):
        """Display measurement results in the text widget."""
        self.results_text.delete(1.0, tk.END)
        
        # Main results
        self.results_text.insert(tk.END, "📏 PERFORATION MEASUREMENT\n", "header")
        self.results_text.insert(tk.END, "="*30 + "\n\n")
        
        self.results_text.insert(tk.END, f"Catalog Gauge: {analysis.catalog_gauge}\n", "important")
        self.results_text.insert(tk.END, f"Precise Value: {analysis.overall_gauge:.3f}\n")
        
        if analysis.is_compound_perforation:
            self.results_text.insert(tk.END, f"Type: {analysis.compound_description}\n", "compound")
        else:
            self.results_text.insert(tk.END, f"Type: Uniform perforation\n")
        
        # Edge analysis
        if analysis.edges:
            self.results_text.insert(tk.END, "\n📊 EDGE ANALYSIS\n", "header")
            self.results_text.insert(tk.END, "-"*20 + "\n")
            
            for edge in analysis.edges:
                gauge_str = self.engine.format_gauge_for_catalog(edge.gauge_measurement)
                self.results_text.insert(tk.END, f"{edge.edge_type.title()}: {gauge_str} ")
                self.results_text.insert(tk.END, f"({len(edge.holes)} holes, {edge.measurement_confidence:.0%})\n")
        
        # Warnings
        if analysis.warnings:
            self.results_text.insert(tk.END, "\n⚠️  WARNINGS\n", "warning")
            self.results_text.insert(tk.END, "-"*15 + "\n")
            for warning in analysis.warnings:
                self.results_text.insert(tk.END, f"• {warning}\n")
        
        # Technical notes
        if analysis.technical_notes:
            self.results_text.insert(tk.END, "\n🔧 TECHNICAL DETAILS\n", "header")
            self.results_text.insert(tk.END, "-"*20 + "\n")
            for note in analysis.technical_notes:
                self.results_text.insert(tk.END, f"• {note}\n")
        
        # Configure text tags for formatting
        self.results_text.tag_configure("header", font=("TkDefaultFont", 10, "bold"))
        self.results_text.tag_configure("important", font=("TkDefaultFont", 10, "bold"), foreground="blue")
        self.results_text.tag_configure("compound", foreground="purple")
        self.results_text.tag_configure("warning", foreground="red")
    
    def _draw_perforation_overlay(self, analysis: PerforationAnalysis):
        """Draw perforation holes overlay on the image."""
        if not hasattr(self, 'display_image') or not analysis.edges:
            return
        
        try:
            # Create a copy of the display image for overlay
            overlay_image = self.display_image.copy()
            draw = ImageDraw.Draw(overlay_image)
            
            # Draw holes for each edge
            colors = ['red', 'blue', 'green', 'orange']
            for i, edge in enumerate(analysis.edges):
                color = colors[i % len(colors)]
                
                for hole in edge.holes:
                    # Scale hole coordinates to display image
                    x = hole.center_x * self.scale_factor
                    y = hole.center_y * self.scale_factor
                    r = (hole.diameter / 2) * self.scale_factor
                    
                    # Draw circle outline
                    draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=2)
                    
                    # Draw center point
                    draw.ellipse([x-2, y-2, x+2, y+2], fill=color)
            
            # Update canvas with overlay
            self.photo_overlay = ImageTk.PhotoImage(overlay_image)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_overlay)
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            print(f"Failed to draw overlay: {e}")
    
    def _save_data(self):
        """Save the measurement data to a file."""
        if not self.analysis_result:
            messagebox.showwarning("No Data", "No measurement data to save")
            return
        
        # Generate default filename
        if self.image_filename:
            base_name = os.path.splitext(os.path.basename(self.image_filename))[0]
            default_name = f"{base_name}_perforation_data.txt"
        else:
            default_name = "perforation_data.txt"
        
        # Ask user where to save
        filename = filedialog.asksaveasfilename(
            title="Save Perforation Data",
            initialname=default_name,
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Use the export method from the engine
                output_dir = os.path.dirname(filename)
                image_name = self.image_filename if self.image_filename else "unknown_image"
                
                # Temporarily change the filename for export
                saved_log = self.engine.export_to_data_logger(
                    self.analysis_result, 
                    image_name, 
                    output_dir
                )
                
                # Rename to user's choice if different
                if saved_log and saved_log != filename:
                    import shutil
                    shutil.move(saved_log, filename)
                
                messagebox.showinfo("Success", f"Data saved to:\n{filename}")
                self.progress_var.set(f"💾 Data saved: {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
PERFORATION MEASUREMENT HELP

🎯 Purpose:
Measure stamp perforation gauge automatically using computer vision.

📋 Steps:
1. Load a high-quality stamp image
2. Set the correct DPI (600 recommended for scans)
3. Click "Measure Perforations" 
4. Review results and save data if needed

🔍 Results:
• Catalog Gauge: Standard format (11¼, 12, etc.)
• Precise Value: Exact measurement for research
• Quality: Measurement confidence level
• Edge Analysis: Individual edge measurements
• Warnings: Potential issues or anomalies

⚠️ Important Notes:
• Use high-resolution images (600 DPI+)
• Ensure stamp edges are clearly visible
• Check DPI setting matches your scan
• Eighth-fraction measurements may indicate forgeries
• Irregular spacing suggests reperforations

💡 Tips:
• Clean, sharp images give best results
• Avoid images with heavy cancellations on edges
• Compare results with catalog specifications
• Save data for comprehensive stamp documentation
"""
        
        help_dialog = tk.Toplevel(self.dialog)
        help_dialog.title("Perforation Measurement Help")
        help_dialog.geometry("600x500")
        
        text_widget = tk.Text(help_dialog, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(help_dialog, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.insert(1.0, help_text)
        text_widget.configure(state='disabled')
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        close_btn = ttk.Button(help_dialog, text="Close", command=help_dialog.destroy)
        close_btn.pack(pady=10)
    
    def _close_dialog(self):
        """Close the dialog."""
        self.dialog.destroy()


def test_perforation_ui():
    """Test the perforation measurement UI."""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Create test dialog
    dialog = PerforationMeasurementDialog(root)
    
    root.mainloop()


if __name__ == "__main__":
    test_perforation_ui()