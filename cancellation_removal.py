#!/usr/bin/env python3
"""
Advanced Philatelic Cancellation Removal Tool

This script provides multiple sophisticated methods for separating and removing
black cancellations from stamp images, going beyond simple color space conversion.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, restoration, segmentation, filters
from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb
import argparse
import os
from pathlib import Path

class CancellationRemover:
    def __init__(self, image_path):
        """Initialize with image path and load the image."""
        self.image_path = image_path
        
        # Try different loading methods for different image formats
        self.original = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.original is None:
            # Try with IMREAD_UNCHANGED for special formats
            self.original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if self.original is None:
                raise ValueError(f"Could not load image: {image_path}")
        
        # Ensure we have proper format and data type
        if self.original.dtype != np.uint8:
            # Normalize to 8-bit
            if self.original.dtype == np.uint16:
                self.original = (self.original / 256).astype(np.uint8)
            else:
                self.original = cv2.convertScaleAbs(self.original)
        
        # Ensure 3-channel color image
        if len(self.original.shape) == 2:
            # Grayscale to color
            self.original = cv2.cvtColor(self.original, cv2.COLOR_GRAY2BGR)
        elif len(self.original.shape) == 3 and self.original.shape[2] == 4:
            # RGBA to RGB
            self.original = cv2.cvtColor(self.original, cv2.COLOR_BGRA2BGR)
        
        self.original_rgb = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.original_rgb.shape[:2]
        
    def enhanced_color_separation(self):
        """
        Enhanced color space analysis focusing on cancellation isolation.
        """
        results = {}
        
        # LAB color space - often best for separating ink types
        lab = rgb2lab(self.original_rgb / 255.0)
        results['lab_l'] = lab[:, :, 0]  # Lightness
        results['lab_a'] = lab[:, :, 1]  # Green-Red
        results['lab_b'] = lab[:, :, 2]  # Blue-Yellow
        
        # HSV color space
        hsv = rgb2hsv(self.original_rgb / 255.0)
        results['hsv_h'] = hsv[:, :, 0]  # Hue
        results['hsv_s'] = hsv[:, :, 1]  # Saturation
        results['hsv_v'] = hsv[:, :, 2]  # Value
        
        # Custom channel combinations
        rgb_float = self.original_rgb / 255.0
        results['red_minus_blue'] = rgb_float[:, :, 0] - rgb_float[:, :, 2]
        results['green_minus_red'] = rgb_float[:, :, 1] - rgb_float[:, :, 0]
        
        return results
    
    def adaptive_thresholding_mask(self):
        """
        Create cancellation mask using adaptive thresholding techniques.
        """
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Multiple threshold techniques
        masks = {}
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        masks['adaptive'] = adaptive
        
        # Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks['otsu'] = otsu
        
        # Multi-level threshold for cancellation detection
        # Typically cancellations are much darker than stamp colors
        _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        masks['dark_regions'] = dark_mask
        
        return masks
    
    def morphological_cancellation_detection(self):
        """
        Use morphological operations to detect cancellation patterns.
        """
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Create structuring elements for typical cancellation patterns
        # Lines (for postmark lines)
        kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # Detect horizontal and vertical lines
        h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_line_h)
        v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_line_v)
        
        # Combine line detections
        lines_combined = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0)
        
        # Text-like patterns (for postmark text)
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_patterns = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_text)
        
        return {
            'horizontal_lines': h_lines,
            'vertical_lines': v_lines,
            'combined_lines': lines_combined,
            'text_patterns': text_patterns
        }
    
    def frequency_domain_separation(self):
        """
        Use frequency domain analysis to separate cancellation from stamp design.
        """
        gray = cv2.cvtColor(self.original_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create frequency filters
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # High pass filter - enhances cancellation edges
        high_pass_mask = np.ones((rows, cols), np.uint8)
        cv2.circle(high_pass_mask, (ccol, crow), 30, 0, -1)
        
        # Apply high pass filter
        f_shift_hp = f_shift * high_pass_mask
        f_ishift_hp = np.fft.ifftshift(f_shift_hp)
        img_hp = np.fft.ifft2(f_ishift_hp)
        img_hp = np.abs(img_hp)
        
        # Band pass filter - targets specific cancellation frequencies
        band_pass_mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(band_pass_mask, (ccol, crow), 80, 1, -1)
        cv2.circle(band_pass_mask, (ccol, crow), 20, 0, -1)
        
        f_shift_bp = f_shift * band_pass_mask
        f_ishift_bp = np.fft.ifftshift(f_shift_bp)
        img_bp = np.fft.ifft2(f_ishift_bp)
        img_bp = np.abs(img_bp)
        
        return {
            'high_pass': img_hp,
            'band_pass': img_bp,
            'magnitude_spectrum': np.log(np.abs(f_shift) + 1)
        }
    
    def intelligent_inpainting(self, mask):
        """
        Use advanced inpainting to reconstruct areas under cancellations.
        """
        # Convert mask to proper format
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Ensure mask is binary
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Multiple inpainting methods
        results = {}
        
        # Fast marching method
        inpainted_fm = cv2.inpaint(self.original, mask_binary, 3, cv2.INPAINT_TELEA)
        results['fast_marching'] = cv2.cvtColor(inpainted_fm, cv2.COLOR_BGR2RGB)
        
        # Navier-Stokes method
        inpainted_ns = cv2.inpaint(self.original, mask_binary, 3, cv2.INPAINT_NS)
        results['navier_stokes'] = cv2.cvtColor(inpainted_ns, cv2.COLOR_BGR2RGB)
        
        return results
    
    def create_composite_mask(self):
        """
        Create an intelligent composite mask combining multiple detection methods.
        """
        # Get various detection results
        color_channels = self.enhanced_color_separation()
        morph_results = self.morphological_cancellation_detection()
        freq_results = self.frequency_domain_separation()
        thresh_masks = self.adaptive_thresholding_mask()
        
        # Start with dark regions as base
        base_mask = thresh_masks['dark_regions']
        
        # Enhance with morphological line detection
        line_enhanced = cv2.bitwise_or(base_mask, morph_results['combined_lines'])
        
        # Use frequency domain high-pass for fine details
        freq_norm = (freq_results['high_pass'] / freq_results['high_pass'].max() * 255).astype(np.uint8)
        _, freq_thresh = cv2.threshold(freq_norm, 50, 255, cv2.THRESH_BINARY)
        
        # Combine all masks
        composite = cv2.bitwise_or(line_enhanced, freq_thresh)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        composite = cv2.morphologyEx(composite, cv2.MORPH_CLOSE, kernel)
        composite = cv2.morphologyEx(composite, cv2.MORPH_OPEN, kernel)
        
        return composite
    
    def process_image(self, output_dir=None):
        """
        Main processing function that applies all techniques and saves results.
        """
        if output_dir is None:
            output_dir = Path(self.image_path).parent / "cancellation_removal_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        base_name = Path(self.image_path).stem
        
        results = {}
        
        print(f"Processing {self.image_path}...")
        
        # 1. Enhanced color separation
        print("  - Analyzing color channels...")
        color_results = self.enhanced_color_separation()
        
        # 2. Create composite cancellation mask
        print("  - Creating cancellation mask...")
        composite_mask = self.create_composite_mask()
        results['cancellation_mask'] = composite_mask
        
        # 3. Apply intelligent inpainting
        print("  - Applying inpainting...")
        inpainted = self.intelligent_inpainting(composite_mask)
        results.update(inpainted)
        
        # 4. Frequency domain analysis
        print("  - Frequency domain analysis...")
        freq_results = self.frequency_domain_separation()
        
        # Save all results
        print("  - Saving results...")
        
        # Save original
        cv2.imwrite(str(output_dir / f"{base_name}_original.png"), self.original)
        
        # Save mask
        cv2.imwrite(str(output_dir / f"{base_name}_mask.png"), composite_mask)
        
        # Save inpainted results
        for method, result in inpainted.items():
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{base_name}_{method}.png"), result_bgr)
        
        # Save interesting color channels
        for channel, data in color_results.items():
            if channel in ['lab_a', 'red_minus_blue']:  # Most promising channels
                normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
                cv2.imwrite(str(output_dir / f"{base_name}_{channel}.png"), normalized)
        
        # Create comparison plot
        self.create_comparison_plot(results, str(output_dir / f"{base_name}_comparison.png"))
        
        print(f"Results saved to: {output_dir}")
        return results
    
    def create_comparison_plot(self, results, output_path):
        """Create a comparison plot showing original and processed results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Cancellation Removal Results', fontsize=16)
        
        # Original
        axes[0, 0].imshow(self.original_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Mask
        axes[0, 1].imshow(results['cancellation_mask'], cmap='gray')
        axes[0, 1].set_title('Detected Cancellation')
        axes[0, 1].axis('off')
        
        # Fast marching inpaint
        axes[0, 2].imshow(results['fast_marching'])
        axes[0, 2].set_title('Fast Marching Inpaint')
        axes[0, 2].axis('off')
        
        # Navier-Stokes inpaint
        axes[1, 0].imshow(results['navier_stokes'])
        axes[1, 0].set_title('Navier-Stokes Inpaint')
        axes[1, 0].axis('off')
        
        # Difference visualization
        diff = np.abs(self.original_rgb.astype(float) - results['fast_marching'].astype(float))
        axes[1, 1].imshow(diff.astype(np.uint8))
        axes[1, 1].set_title('Difference (Original vs Inpainted)')
        axes[1, 1].axis('off')
        
        # Hide last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Advanced Philatelic Cancellation Removal')
    parser.add_argument('input_path', help='Path to input image or directory')
    parser.add_argument('--output', '-o', help='Output directory (default: input_path + "_results")')
    parser.add_argument('--batch', '-b', action='store_true', 
                       help='Process all images in directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if args.batch and input_path.is_dir():
        # Process all images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process...")
        
        for image_file in image_files:
            try:
                remover = CancellationRemover(str(image_file))
                remover.process_image(args.output)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                
    elif input_path.is_file():
        # Process single image
        try:
            remover = CancellationRemover(str(input_path))
            remover.process_image(args.output)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    else:
        print(f"Invalid input path: {input_path}")

if __name__ == "__main__":
    main()