#!/usr/bin/env python3
"""
Simple Philatelic Cancellation Extractor

This tool extracts cancellation layers from stamp images with robust image format handling.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_image_robust(image_path):
    """Load image with robust format handling."""
    # Try loading with different flags
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"Loaded image: {img.shape}, dtype: {img.dtype}, range: {img.min()}-{img.max()}")
    
    # Convert to 8-bit if needed
    if img.dtype == np.uint16:
        # Scale 16-bit to 8-bit
        img = (img / 256).astype(np.uint8)
        print(f"Converted to 8-bit: range {img.min()}-{img.max()}")
    elif img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
        print(f"Converted to 8-bit: range {img.min()}-{img.max()}")
    
    # Handle different channel configurations
    if len(img.shape) == 2:
        # Grayscale - convert to 3-channel
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            # RGBA - drop alpha channel
            img_bgr = img[:, :, :3]
        elif img.shape[2] == 3:
            # Already BGR
            img_bgr = img.copy()
        else:
            raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    # Convert to RGB for processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_bgr, img_rgb

def create_cancellation_mask(img_bgr):
    """Create a mask that identifies cancellation areas."""
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Multiple thresholding approaches
    masks = []
    
    # 1. Dark regions (cancellations are typically dark)
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    masks.append(dark_mask)
    
    # 2. Adaptive threshold
    adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    masks.append(adaptive_mask)
    
    # 3. Edge detection for postmark lines
    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges to make them more connected
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    masks.append(edges_dilated)
    
    # 4. Morphological line detection
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    _, horizontal_lines = cv2.threshold(horizontal_lines, 50, 255, cv2.THRESH_BINARY)
    
    # Vertical lines  
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
    _, vertical_lines = cv2.threshold(vertical_lines, 50, 255, cv2.THRESH_BINARY)
    
    # Combine line detections
    lines_combined = cv2.bitwise_or(horizontal_lines, vertical_lines)
    masks.append(lines_combined)
    
    # Combine all masks
    combined_mask = np.zeros_like(gray)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    return combined_mask, {
        'dark_regions': dark_mask,
        'adaptive': adaptive_mask, 
        'edges': edges_dilated,
        'lines': lines_combined
    }

def extract_cancellation_layers(img_bgr, img_rgb, mask):
    """Extract cancellation in multiple enhanced formats."""
    results = {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Cancellation on white background
    cancellation_on_white = np.where(mask == 255, gray, 255)
    results['cancellation_on_white'] = cancellation_on_white
    
    # 2. High contrast black cancellation on white
    binary_cancellation = np.where(mask == 255, 0, 255)
    results['binary_cancellation'] = binary_cancellation
    
    # 3. Enhanced contrast cancellation
    cancellation_only = cv2.bitwise_and(gray, gray, mask=mask)
    # Apply histogram equalization only to cancellation areas
    if np.any(cancellation_only > 0):
        enhanced = cv2.equalizeHist(cancellation_only)
        enhanced_on_white = np.where(mask == 255, enhanced, 255)
        results['enhanced_cancellation'] = enhanced_on_white
    else:
        results['enhanced_cancellation'] = cancellation_on_white
    
    # 4. Inverted cancellation
    results['inverted_cancellation'] = 255 - cancellation_on_white
    
    # 5. Color cancellation extraction
    color_mask_3d = np.stack([mask, mask, mask], axis=2)
    color_cancellation = np.where(color_mask_3d == 255, img_rgb, [255, 255, 255])
    results['color_cancellation'] = color_cancellation
    
    return results

def analyze_cancellation(mask):
    """Analyze cancellation characteristics."""
    total_pixels = mask.shape[0] * mask.shape[1]
    cancellation_pixels = np.sum(mask == 255)
    coverage_percentage = (cancellation_pixels / total_pixels) * 100
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    analysis = {
        'coverage_percentage': coverage_percentage,
        'num_components': num_labels - 1,  # Subtract background
        'total_pixels': total_pixels,
        'cancellation_pixels': cancellation_pixels
    }
    
    if num_labels > 1:  # If there are components besides background
        component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        analysis['component_sizes'] = component_sizes.tolist()
        analysis['largest_component'] = int(np.max(component_sizes)) if len(component_sizes) > 0 else 0
    
    return analysis

def process_image(image_path, output_dir=None):
    """Process a single image and extract cancellations."""
    
    print(f"Processing: {image_path}")
    
    # Load image
    img_bgr, img_rgb = load_image_robust(image_path)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(image_path).parent / "cancellation_extraction_results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    base_name = Path(image_path).stem
    
    # Create cancellation mask
    print("  - Creating cancellation mask...")
    final_mask, component_masks = create_cancellation_mask(img_bgr)
    
    # Extract cancellation layers
    print("  - Extracting cancellation layers...")
    extraction_results = extract_cancellation_layers(img_bgr, img_rgb, final_mask)
    
    # Analyze cancellation
    print("  - Analyzing cancellation...")
    analysis = analyze_cancellation(final_mask)
    
    # Save results
    print("  - Saving results...")
    
    # Save original
    cv2.imwrite(str(output_dir / f"{base_name}_original.png"), img_bgr)
    
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
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Cancellation Extraction Results - {base_name}', fontsize=14)
    
    # Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Final mask
    axes[0, 1].imshow(final_mask, cmap='gray')
    axes[0, 1].set_title('Final Mask')
    axes[0, 1].axis('off')
    
    # Cancellation on white
    axes[0, 2].imshow(extraction_results['cancellation_on_white'], cmap='gray')
    axes[0, 2].set_title('Cancellation on White')
    axes[0, 2].axis('off')
    
    # Binary cancellation
    axes[1, 0].imshow(extraction_results['binary_cancellation'], cmap='gray')
    axes[1, 0].set_title('Binary Cancellation')
    axes[1, 0].axis('off')
    
    # Enhanced cancellation
    axes[1, 1].imshow(extraction_results['enhanced_cancellation'], cmap='gray')
    axes[1, 1].set_title('Enhanced Cancellation')
    axes[1, 1].axis('off')
    
    # Inverted cancellation
    axes[1, 2].imshow(extraction_results['inverted_cancellation'], cmap='gray')
    axes[1, 2].set_title('Inverted Cancellation')
    axes[1, 2].axis('off')
    
    # Color cancellation
    axes[2, 0].imshow(extraction_results['color_cancellation'])
    axes[2, 0].set_title('Color Cancellation')
    axes[2, 0].axis('off')
    
    # Component masks
    axes[2, 1].imshow(component_masks['dark_regions'], cmap='gray')
    axes[2, 1].set_title('Dark Regions Mask')
    axes[2, 1].axis('off')
    
    # Analysis text
    axes[2, 2].axis('off')
    analysis_text = f"""Analysis:
Coverage: {analysis['coverage_percentage']:.1f}%
Components: {analysis['num_components']}
Pixels: {analysis['cancellation_pixels']:,}
Largest: {analysis.get('largest_component', 0):,} px"""
    axes[2, 2].text(0.1, 0.5, analysis_text, transform=axes[2, 2].transAxes, 
                   fontsize=12, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(str(output_dir / f"{base_name}_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Save analysis text
    with open(output_dir / f"{base_name}_analysis.txt", 'w') as f:
        f.write(f"Cancellation Analysis for {Path(image_path).name}\n")
        f.write("=" * 50 + "\n\n")
        for key, value in analysis.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to: {output_dir}")
    print(f"Coverage: {analysis['coverage_percentage']:.1f}% | Components: {analysis['num_components']}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Extract Philatelic Cancellations')
    parser.add_argument('image_path', help='Path to stamp image')
    parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        output_dir = process_image(args.image_path, args.output)
        
        # Open the comparison image
        comparison_file = output_dir / f"{Path(args.image_path).stem}_comparison.png"
        if comparison_file.exists():
            print(f"\nOpening comparison plot...")
            import subprocess
            subprocess.run(['open', str(comparison_file)], check=False)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()