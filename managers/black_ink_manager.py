#!/usr/bin/env python3
"""
Black Ink Manager for StampZ
Handles black ink extraction functionality - extracted from analysis_manager.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os


class BlackInkManager:
    """Manages black ink extraction functionality."""
    
    def __init__(self, main_app):
        self.app = main_app
        self.root = main_app.root if hasattr(main_app, 'root') else None
        
    def open_black_ink_extractor(self):
        """Open the Black Ink Extractor for cancellation extraction."""
        if not self.app.current_file:
            messagebox.showwarning(
                "No Image",
                "Please open a stamp image first.\\n\\n"
                "The Black Ink Extractor works best with colored stamps containing black cancellations."
            )
            return
            
        # Import the extraction functionality
        try:
            from black_ink_extractor import extract_black_ink, extract_colored_cancellation_rgb_only, safe_pil_fromarray
        except ImportError:
            messagebox.showerror(
                "Module Error",
                "Black Ink Extractor module not found.\\n\\n"
                "Please ensure 'black_ink_extractor.py' is in the StampZ directory."
            )
            return
            
        try:
            # Create extraction dialog (orphaned for multi-screen use)
            dialog = tk.Toplevel()
            dialog.title("Black Ink Extractor")
            dialog.geometry("650x700")  # Wider for better layout
            
            # Center dialog
            dialog.update_idletasks()
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = (screen_width // 2) - (dialog.winfo_reqwidth() // 2)
            y = (screen_height // 2) - (dialog.winfo_reqheight() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            self._setup_extraction_dialog(dialog, extract_black_ink, safe_pil_fromarray)
            
        except Exception as e:
            messagebox.showerror(
                "Black Ink Extractor Error",
                f"Failed to open Black Ink Extractor:\\n\\n{str(e)}"
            )
            
    def _setup_extraction_dialog(self, dialog, extract_black_ink, safe_pil_fromarray):
        """Setup the black ink extraction dialog interface."""
        
        # Header
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(
            header_frame,
            text="Black Ink Cancellation Extractor",
            font=("Arial", 14, "bold")
        ).pack()
        
        ttk.Label(
            header_frame,
            text=f"Current Image: {os.path.basename(self.app.current_file)}",
            font=("Arial", 10)
        ).pack(pady=5)
        
        # Description
        desc_frame = ttk.Frame(dialog)
        desc_frame.pack(fill="x", padx=20, pady=10)
        
        description = (
            "This tool extracts black ink cancellations from colored stamps.\\n\\n"
            "Perfect for isolating postmarks, cancellations, and overprints from \\n"
            "colored stamp backgrounds. The extracted black ink appears on a clean \\n"
            "white background for easy study and documentation.\\n\\n"
            "Works excellent with:\\n"
            "• Red cancellations on Penny Blacks and other dark stamps\\n"
            "• Black postmarks on colored stamps\\n"
            "• Overprints and surcharges\\n\\n"
            "Supports 48-bit TIFF files from VueScan and all image formats."
        )
        
        ttk.Label(
            desc_frame,
            text=description,
            font=("Arial", 9),
            justify="left"
        ).pack(anchor="w")
        
        # Focus message
        focus_frame = ttk.Frame(dialog)
        focus_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(
            focus_frame,
            text="Extracting Black Ink (cancellations, postmarks, overprints)",
            font=("Arial", 12, "bold"),
            foreground="#2E8B57"
        ).pack()
        
        # Settings
        self._setup_extraction_settings(dialog, extract_black_ink, safe_pil_fromarray)
        
    def _setup_extraction_settings(self, dialog, extract_black_ink, safe_pil_fromarray):
        """Setup the extraction settings controls."""
        
        # Settings frame
        settings_frame = ttk.LabelFrame(dialog, text="Extraction Settings", padding=10)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        # Black threshold setting
        ttk.Label(settings_frame, text="Black Threshold (0-255):").grid(row=0, column=0, sticky="w", pady=2)
        black_threshold = tk.IntVar(value=60)
        black_threshold_scale = ttk.Scale(
            settings_frame,
            from_=10, to=150,
            variable=black_threshold,
            orient="horizontal",
            length=200
        )
        black_threshold_scale.grid(row=0, column=1, padx=10, pady=2)
        black_threshold_label = ttk.Label(settings_frame, text="60")
        black_threshold_label.grid(row=0, column=2, pady=2)
        
        def update_black_threshold(event):
            black_threshold_label.config(text=str(int(black_threshold.get())))
        black_threshold_scale.bind("<Motion>", update_black_threshold)
        
        # Saturation threshold setting
        ttk.Label(settings_frame, text="Saturation Threshold (0-255):").grid(row=1, column=0, sticky="w", pady=2)
        saturation_threshold = tk.IntVar(value=30)
        saturation_threshold_scale = ttk.Scale(
            settings_frame,
            from_=10, to=100,
            variable=saturation_threshold,
            orient="horizontal",
            length=200
        )
        saturation_threshold_scale.grid(row=1, column=1, padx=10, pady=2)
        saturation_threshold_label = ttk.Label(settings_frame, text="30")
        saturation_threshold_label.grid(row=1, column=2, pady=2)
        
        def update_saturation_threshold(event):
            saturation_threshold_label.config(text=str(int(saturation_threshold.get())))
        saturation_threshold_scale.bind("<Motion>", update_saturation_threshold)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=20, pady=20)
        
        # Extract function
        def extract_cancellation():
            try:
                print(f"Extracting black ink from: {self.app.current_file}")
                print(f"Black threshold: {black_threshold.get()}")
                print(f"Saturation threshold: {saturation_threshold.get()}")
                
                # Load and convert image to numpy array
                from PIL import Image
                import numpy as np
                
                print(f"DEBUG: Loading image from {self.app.current_file}")
                pil_image = Image.open(self.app.current_file)
                
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    print(f"DEBUG: Converting from {pil_image.mode} to RGB")
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array
                img_array = np.array(pil_image)
                print(f"DEBUG: Image loaded as numpy array - shape: {img_array.shape}, dtype: {img_array.dtype}")
                
                # Perform extraction
                results, mask, analysis = extract_black_ink(
                    img_array,
                    black_threshold=black_threshold.get(),
                    saturation_threshold=saturation_threshold.get()
                )
                
                if results is not None and len(results) > 0:
                    # Use the 'pure_black' result for the main output
                    extracted_image = results.get('pure_black', results[list(results.keys())[0]])
                    
                    # Save extracted image
                    input_path = self.app.current_file
                    base_name = os.path.splitext(input_path)[0]
                    output_path = f"{base_name}_black_ink_extracted.png"
                    
                    # Convert and save
                    pil_image = safe_pil_fromarray(extracted_image)
                    pil_image.save(output_path, format='PNG')
                    
                    # Save additional versions if desired
                    if 'enhanced' in results:
                        enhanced_path = f"{base_name}_black_ink_enhanced.png"
                        enhanced_pil = safe_pil_fromarray(results['enhanced'])
                        enhanced_pil.save(enhanced_path, format='PNG')
                    
                    # Log to unified data file with analysis results
                    self._log_extraction_data(
                        output_path, 
                        black_threshold.get(), 
                        saturation_threshold.get(),
                        analysis
                    )
                    
                    # Create detailed success message
                    coverage_pct = analysis.get('coverage_percentage', 0)
                    success_message = (
                        f"Black ink extracted successfully!\\n\\n"
                        f"Saved to: {os.path.basename(output_path)}\\n"
                        f"Coverage: {coverage_pct:.1f}% of image\\n"
                        f"Cancellation pixels: {analysis.get('cancellation_pixels', 0):,}\\n\\n"
                        f"Data logged to unified file for comprehensive documentation."
                    )
                    
                    messagebox.showinfo("Extraction Complete", success_message)
                    
                    dialog.destroy()
                else:
                    messagebox.showerror("Extraction Failed", "Could not extract black ink from image.")
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"DETAILED ERROR: {error_details}")
                
                messagebox.showerror(
                    "Extraction Error",
                    f"Failed to extract ink:\\n\\n{str(e)}\\n\\nFull error details printed to terminal."
                )
        
        ttk.Button(
            button_frame,
            text="Extract Black Ink",
            command=extract_cancellation,
            style="Accent.TButton"
        ).pack(side="left", padx=5)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy
        ).pack(side="right", padx=5)
        
    def _log_extraction_data(self, output_path, black_threshold, saturation_threshold, analysis=None):
        """Log extraction data to unified data file."""
        try:
            # Import unified logger
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from utils.unified_data_logger import UnifiedDataLogger
            
            # Create logger and log extraction data
            logger = UnifiedDataLogger(self.app.current_file)
            
            extraction_params = {
                'black_threshold': black_threshold,
                'saturation_threshold': saturation_threshold,
                'method': 'Standard black ink extraction'
            }
            
            # Include analysis data if available
            results = {
                'output_file': os.path.basename(output_path),
                'success': True,
                'processing_time': 0  # Could be measured if needed
            }
            
            # Add analysis data if available
            if analysis:
                results.update({
                    'coverage_percentage': analysis.get('coverage_percentage', 0),
                    'cancellation_pixels': analysis.get('cancellation_pixels', 0),
                    'total_pixels': analysis.get('total_pixels', 0),
                    'red_median': analysis.get('red_median', 0),
                    'avg_cancellation_brightness': analysis.get('avg_cancellation_brightness', 0)
                })
            
            logger.log_black_ink_extraction(extraction_params, results)
            
        except Exception as e:
            print(f"Warning: Could not log to unified data file: {e}")
