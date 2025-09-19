#!/usr/bin/env python3
"""
Robust Philatelic Cancellation Extractor for 48-bit TIFFs

This tool works directly with 48-bit TIFF files from VueScan, 
extracting cancellation layers without format conversion issues.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse
import sys

def load_48bit_tiff(image_path):
    """Load 48-bit TIFF using PIL, which handles it better than OpenCV."""
    print(f"Loading 48-bit TIFF: {image_path}")
    
    try:
        # Use PIL to load the image properly
        pil_image = Image.open(image_path)
        print(f"PIL format: {pil_image.mode}, size: {pil_image.size}")
        
        # Convert PIL image to numpy array
        img_array = np.array(pil_image)
        print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"Value range: {img_array.min()} - {img_array.max()}")
        
        # Handle different bit depths
        if img_array.dtype == np.uint16:
            # 16-bit per channel - scale to 8-bit for processing
            # but keep precision by using float32 intermediate
            img_float = img_array.astype(np.float32) / 65535.0  # Normalize to 0-1
            img_8bit = (img_float * 255).astype(np.uint8)
            print(f"Converted to 8-bit: {img_8bit.min()} - {img_8bit.max()}")
        else:
            # Already 8-bit or other format
            img_8bit = img_array.astype(np.uint8)
        
        # Ensure we have RGB format (PIL might give us RGB, OpenCV expects BGR)
        if len(img_8bit.shape) == 3 and img_8bit.shape[2] == 3:
            # PIL gives RGB, convert to BGR for OpenCV compatibility
            img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
            img_rgb = img_8bit  # PIL already gave us RGB
        elif len(img_8bit.shape) == 2:
            # Grayscale
            img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
            img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {img_8bit.shape}")
        
        return img_bgr, img_rgb, img_array  # Return original high-bit version too
        
    except Exception as e:
        print(f"PIL loading failed: {e}")
        # Fallback to OpenCV with error handling
        return load_with_opencv_fallback(image_path)

def load_with_opencv_fallback(image_path):
    """Fallback OpenCV loading with robust error handling."""
    print("Trying OpenCV fallback...")
    
    try:
        # Try IMREAD_UNCHANGED first
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("OpenCV could not load image")
        
        print(f"OpenCV shape: {img.shape}, dtype: {img.dtype}")
        
        # Handle 16-bit images
        if img.dtype == np.uint16:
            # Convert to 8-bit
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img)
        
        # Ensure 3-channel BGR
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img_bgr = img[:, :, :3]  # Drop alpha
        else:
            img_bgr = img
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        return img_bgr, img_rgb, img_bgr
        
    except Exception as e:
        print(f"OpenCV fallback also failed: {e}")
        raise ValueError(f"Could not load image with any method: {image_path}")

def create_cancellation_mask(img_bgr):
    """Create cancellation detection mask using multiple methods."""
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale range: {gray.min()} - {gray.max()}")
    
    masks = {}
    
    # Method 1: Dark regions threshold
    # Cancellations are typically darker than stamp colors
    threshold_value = 100  # Adjust based on your images
    _, dark_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    masks['dark_regions'] = dark_mask
    
    # Method 2: Adaptive threshold to catch varying lighting
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    masks['adaptive'] = adaptive_mask
    
    # Method 3: Edge detection for postmark lines
    edges = cv2.Canny(gray, 30, 100)
    # Dilate to connect broken lines
    kernel = np.ones((2,2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    masks['edges'] = edges_dilated
    
    # Method 4: Morphological operations to detect lines
    # Horizontal lines (common in postmarks)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
    _, horizontal = cv2.threshold(horizontal, 30, 255, cv2.THRESH_BINARY)
    
    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
    _, vertical = cv2.threshold(vertical, 30, 255, cv2.THRESH_BINARY)
    
    # Combine line detections
    lines = cv2.bitwise_or(horizontal, vertical)
    masks['lines'] = lines
    
    # Method 5: Text-like patterns (for postmark text)
    text_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    text_patterns = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, text_kernel)
    _, text_patterns = cv2.threshold(text_patterns, 20, 255, cv2.THRESH_BINARY)
    masks['text_patterns'] = text_patterns
    
    # Combine all masks with different weights
    print("Combining detection masks...")
    combined = np.zeros_like(gray, dtype=np.uint8)
    
    # Give more weight to dark regions and adaptive threshold
    combined = cv2.bitwise_or(combined, dark_mask)
    combined = cv2.bitwise_or(combined, adaptive_mask)
    
    # Add line and edge detection
    combined = cv2.bitwise_or(combined, edges_dilated)
    combined = cv2.bitwise_or(combined, lines)
    
    # Clean up the combined mask
    # Close gaps
    kernel = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Remove small noise
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    masks['combined_final'] = combined
    
    return combined, masks

def extract_cancellation_layers(img_bgr, img_rgb, mask):
    """Extract cancellation in multiple formats for study."""
    results = {}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Cancellation on white background (most useful for reading)
    cancellation_on_white = np.where(mask > 0, gray, 255)
    results['cancellation_on_white'] = cancellation_on_white
    
    # 2. Pure black cancellation on white (high contrast)
    binary_cancellation = np.where(mask > 0, 0, 255)
    results['binary_cancellation'] = binary_cancellation
    
    # 3. Enhanced contrast version
    cancellation_only = cv2.bitwise_and(gray, gray, mask=mask)
    if np.any(cancellation_only > 0):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(cancellation_only)
        enhanced_on_white = np.where(mask > 0, enhanced, 255)
        results['enhanced_cancellation'] = enhanced_on_white
    else:
        results['enhanced_cancellation'] = cancellation_on_white
    
    # 4. Inverted cancellation (white text on black background)
    inverted = 255 - cancellation_on_white
    results['inverted_cancellation'] = inverted
    
    # 5. Color-preserved cancellation
    mask_3d = np.stack([mask, mask, mask], axis=2) > 0
    color_cancellation = np.where(mask_3d, img_rgb, [255, 255, 255])
    results['color_cancellation'] = color_cancellation
    
    # 6. Edge-enhanced cancellation
    edges = cv2.Canny(gray, 30, 100)
    edge_cancellation = cv2.bitwise_and(edges, mask)
    edge_on_white = np.where(edge_cancellation > 0, 0, 255)
    results['edge_cancellation'] = edge_on_white
    
    return results

def analyze_cancellation(mask):
    """Analyze the detected cancellation characteristics."""
    total_pixels = mask.size
    cancellation_pixels = np.count_nonzero(mask)
    coverage_percentage = (cancellation_pixels / total_pixels) * 100
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    analysis = {
        'coverage_percentage': coverage_percentage,
        'total_pixels': total_pixels,
        'cancellation_pixels': cancellation_pixels,
        'num_components': num_labels - 1,  # Subtract background
    }
    
    if num_labels > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        analysis['component_areas'] = component_areas.tolist()
        analysis['largest_component'] = int(np.max(component_areas))
        analysis['average_component'] = int(np.mean(component_areas))
    
    return analysis

def create_results_plot(img_rgb, mask, extraction_results, component_masks, analysis, base_name):
    """Create comprehensive results visualization."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle(f'Cancellation Extraction Results - {base_name}', fontsize=14, y=0.98)
    
    # Row 1: Original and detection
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original 48-bit TIFF')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Final Detection Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(extraction_results['cancellation_on_white'], cmap='gray')
    axes[0, 2].set_title('Cancellation on White')
    axes[0, 2].axis('off')
    
    # Row 2: Different extraction methods
    axes[1, 0].imshow(extraction_results['binary_cancellation'], cmap='gray')
    axes[1, 0].set_title('Binary Cancellation')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(extraction_results['enhanced_cancellation'], cmap='gray')
    axes[1, 1].set_title('Enhanced Contrast')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(extraction_results['inverted_cancellation'], cmap='gray')
    axes[1, 2].set_title('Inverted Cancellation')
    axes[1, 2].axis('off')
    
    # Row 3: Color and edge versions
    axes[2, 0].imshow(extraction_results['color_cancellation'])
    axes[2, 0].set_title('Color Cancellation')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(extraction_results['edge_cancellation'], cmap='gray')
    axes[2, 1].set_title('Edge Enhanced')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(component_masks['dark_regions'], cmap='gray')
    axes[2, 2].set_title('Dark Regions Mask')
    axes[2, 2].axis('off')
    
    # Row 4: Component masks and analysis
    axes[3, 0].imshow(component_masks['adaptive'], cmap='gray')
    axes[3, 0].set_title('Adaptive Threshold')
    axes[3, 0].axis('off')
    
    axes[3, 1].imshow(component_masks['lines'], cmap='gray')
    axes[3, 1].set_title('Line Detection')
    axes[3, 1].axis('off')
    
    # Analysis text
    axes[3, 2].axis('off')
    analysis_text = f"""Cancellation Analysis:

Coverage: {analysis['coverage_percentage']:.1f}%
Total pixels: {analysis['total_pixels']:,}
Cancellation: {analysis['cancellation_pixels']:,}
Components: {analysis['num_components']}

Largest component: {analysis.get('largest_component', 0):,} px
Average component: {analysis.get('average_component', 0):,} px"""
    
    axes[3, 2].text(0.05, 0.95, analysis_text, transform=axes[3, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    return fig

def process_48bit_image(image_path, output_dir=None):
    """Main processing function for 48-bit TIFF images."""
    
    print(f"Processing 48-bit TIFF: {image_path}")
    print("-" * 50)
    
    # Load the image
    img_bgr, img_rgb, original_array = load_48bit_tiff(image_path)
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(image_path).parent / "cancellation_extraction_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    base_name = Path(image_path).stem
    
    # Create cancellation mask
    print("\nCreating cancellation detection mask...")
    final_mask, component_masks = create_cancellation_mask(img_bgr)
    
    # Extract cancellation layers
    print("Extracting cancellation layers...")
    extraction_results = extract_cancellation_layers(img_bgr, img_rgb, final_mask)
    
    # Analyze results
    print("Analyzing cancellation characteristics...")
    analysis = analyze_cancellation(final_mask)
    
    # Save all results
    print("\nSaving results...")
    
    # Save the processing-resolution original
    cv2.imwrite(str(output_dir / f"{base_name}_original_8bit.png"), img_bgr)
    
    # Save final mask
    cv2.imwrite(str(output_dir / f"{base_name}_final_mask.png"), final_mask)
    
    # Save component masks
    for name, mask in component_masks.items():
        cv2.imwrite(str(output_dir / f"{base_name}_mask_{name}.png"), mask)
    
    # Save extraction results
    for name, result in extraction_results.items():
        if len(result.shape) == 3:  # Color image
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{base_name}_{name}.png"), result_bgr)
        else:  # Grayscale
            cv2.imwrite(str(output_dir / f"{base_name}_{name}.png"), result)
    
    # Create and save comprehensive plot
    print("Creating results visualization...")
    fig = create_results_plot(img_rgb, final_mask, extraction_results, 
                            component_masks, analysis, base_name)
    fig.savefig(str(output_dir / f"{base_name}_complete_analysis.png"), 
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    # Save analysis report
    with open(output_dir / f"{base_name}_analysis_report.txt", 'w') as f:
        f.write(f"48-bit TIFF Cancellation Analysis Report\n")
        f.write(f"Image: {Path(image_path).name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Original image shape: {original_array.shape}\n")
        f.write(f"Original image dtype: {original_array.dtype}\n")
        f.write(f"Processing resolution: {img_rgb.shape}\n\n")
        
        for key, value in analysis.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write(f"\nExtraction methods generated:\n")
        for method in extraction_results.keys():
            f.write(f"  - {method.replace('_', ' ').title()}\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Coverage: {analysis['coverage_percentage']:.1f}% of image")
    print(f"Components detected: {analysis['num_components']}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description='Extract cancellations from 48-bit TIFF stamp images'
    )
    parser.add_argument('image_path', help='Path to 48-bit TIFF image')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all TIFF files in directory')
    
    args = parser.parse_args()
    
    input_path = Path(args.image_path)
    
    if args.batch and input_path.is_dir():
        # Process all TIFF files
        tiff_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
        print(f"Found {len(tiff_files)} TIFF files to process")
        
        for tiff_file in tiff_files:
            try:
                print(f"\n{'='*60}")
                output_dir = process_48bit_image(str(tiff_file), args.output)
                print(f"✓ Completed: {tiff_file.name}")
            except Exception as e:
                print(f"✗ Error processing {tiff_file.name}: {e}")
                
    elif input_path.is_file():
        try:
            output_dir = process_48bit_image(str(input_path), args.output)
            
            # Open results
            results_plot = output_dir / f"{input_path.stem}_complete_analysis.png"
            if results_plot.exists():
                print(f"\nOpening results visualization...")
                import subprocess
                subprocess.run(['open', str(results_plot)], check=False)
                
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print(f"Invalid path: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()