#!/usr/bin/env python3
"""
Black Ink Cancellation Extractor for Philatelic Images

Extracts black cancellation ink from colored stamp images, creating clean
cancellation-only images on white backgrounds for study and documentation.

Works best with:
- Light to medium colored stamps (reds, blues, greens, yellows)
- Good contrast between black cancellation and stamp colors
- 48-bit TIFF files from VueScan (or any image format)

Author: AI Assistant for StampZ-III Project
"""

from PIL import Image
import numpy as np

def safe_pil_fromarray(array, mode=None):
    """Safely convert numpy array to PIL Image with comprehensive dtype handling."""
    print(f"DEBUG: safe_pil_fromarray input - dtype: {array.dtype}, shape: {array.shape}")
    
    # Ensure proper uint8 format for PIL
    if array.dtype != np.uint8:
        print(f"DEBUG: Converting array from {array.dtype} to uint8")
        if array.dtype in [np.uint16, np.int16, np.int32, np.int64]:
            # Scale down from higher bit depths
            if array.max() > 255:
                array = (array.astype(np.float64) / array.max() * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        elif array.dtype in [np.float32, np.float64]:
            # Handle float arrays (may be 0-1 or 0-255 range)
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            # Fallback for any other types
            array = np.clip(array.astype(np.float64), 0, 255).astype(np.uint8)
    
    # Ensure values are in valid range
    array = np.clip(array, 0, 255)
    
    print(f"DEBUG: safe_pil_fromarray converted - dtype: {array.dtype}, range: {array.min()}-{array.max()}")
    
    try:
        return Image.fromarray(array, mode)
    except Exception as e:
        print(f"ERROR: PIL conversion still failed: {e}")
        print(f"Array details: shape={array.shape}, dtype={array.dtype}, contiguous={array.flags['C_CONTIGUOUS']}")
        # Try making contiguous
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)
            return Image.fromarray(array, mode)
        raise
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

def extract_black_ink(img_array, black_threshold=60, saturation_threshold=30, red_offset=40):
    """
    Extract black ink (cancellations) from colored stamp images.
    
    Args:
        img_array: RGB image as numpy array
        black_threshold: Maximum brightness for black ink detection (0-255)
        saturation_threshold: Maximum color saturation for black ink (0-255)  
        red_offset: How much darker than median red channel to consider black
    
    Returns:
        dict with extracted cancellation images and analysis
    """
    
    # Separate color channels
    red = img_array[:, :, 0].astype(np.float32)
    green = img_array[:, :, 1].astype(np.float32)  
    blue = img_array[:, :, 2].astype(np.float32)
    
    # Calculate pixel characteristics
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)
    brightness = (red + green + blue) / 3
    color_range = max_channel - min_channel
    
    # Method 1: True black ink detection
    # Black ink should be dark in ALL channels with low color saturation
    is_truly_black = (brightness < black_threshold) & (color_range < saturation_threshold)
    
    # Method 2: Red channel analysis (often best for cancellation visibility)
    red_channel = img_array[:, :, 0]
    red_median = np.median(red_channel)
    red_dark_threshold = max(0, red_median - red_offset)
    is_dark_in_red = red_channel < red_dark_threshold
    
    # Combine both detection methods
    black_ink_mask = is_truly_black | is_dark_in_red
    
    # Create extraction results
    results = {}
    
    # Pure black on white (high contrast for reading)
    results['pure_black'] = np.where(black_ink_mask, 0, 255).astype(np.uint8)
    
    # Grayscale cancellation (preserves ink density variations)
    results['grayscale'] = np.where(black_ink_mask, brightness.astype(np.uint8), 255)
    
    # Inverted version (white text on black background)
    results['inverted'] = 255 - results['pure_black']
    
    # Enhanced contrast version
    cancellation_values = np.where(black_ink_mask, brightness, 255)
    # Stretch contrast of cancellation areas
    if np.any(black_ink_mask):
        cancel_pixels = cancellation_values[black_ink_mask]
        if len(cancel_pixels) > 0:
            min_val, max_val = cancel_pixels.min(), cancel_pixels.max()
            if max_val > min_val:
                enhanced = np.where(
                    black_ink_mask,
                    ((cancellation_values - min_val) / (max_val - min_val) * 255).astype(np.uint8),
                    255
                )
                results['enhanced'] = enhanced
            else:
                results['enhanced'] = results['grayscale']
        else:
            results['enhanced'] = results['grayscale']
    else:
        results['enhanced'] = results['grayscale']
    
    # Analysis
    total_pixels = black_ink_mask.size
    cancellation_pixels = np.sum(black_ink_mask)
    coverage_percentage = (cancellation_pixels / total_pixels) * 100
    
    analysis = {
        'coverage_percentage': coverage_percentage,
        'total_pixels': total_pixels,
        'cancellation_pixels': cancellation_pixels,
        'red_median': red_median,
        'red_threshold': red_dark_threshold,
        'black_threshold_used': black_threshold,
        'saturation_threshold_used': saturation_threshold,
        'avg_cancellation_brightness': np.mean(brightness[black_ink_mask]) if np.any(black_ink_mask) else 0
    }
    
    return results, black_ink_mask, analysis

def extract_colored_cancellation(img_array, cancellation_color='red', color_threshold=40, saturation_threshold=50):
    """
    Extract colored cancellation ink (red, blue, green) from stamp images.
    
    Args:
        img_array: RGB image as numpy array
        cancellation_color: 'red', 'blue', or 'green'
        color_threshold: How strong the target color should be (0-255)
        saturation_threshold: Minimum saturation to avoid picking up neutral areas
    
    Returns:
        dict with extracted cancellation images and analysis
    """
    # Always use RGB-only method to avoid HSV conversion issues
    print(f"DEBUG: Using RGB-only extraction for {cancellation_color}")
    return extract_colored_cancellation_rgb_only(img_array, cancellation_color, color_threshold)

def extract_black_stamp_background(img_array, black_threshold=60):
    """
    Extract the black stamp design, leaving non-black areas (like red cancellations) visible.
    This is especially useful for Penny Blacks where we want to isolate red cancellations.
    """
    print(f"DEBUG: Extracting black background with threshold {black_threshold}")
    
    # Calculate brightness for each pixel
    if len(img_array.shape) == 3:
        brightness = np.mean(img_array, axis=2)
    else:
        brightness = img_array
    
    # Create mask for black areas
    black_mask = brightness < black_threshold
    
    # Create result where black areas become white, preserving other colors
    if len(img_array.shape) == 3:
        result = img_array.copy()
        result[black_mask] = [255, 255, 255]  # Make black areas white
    else:
        result = np.where(black_mask, 255, img_array)
    
    print(f"DEBUG: Removed {np.sum(black_mask)} black pixels ({np.sum(black_mask)/black_mask.size*100:.1f}%)")
    
    return result, black_mask

def extract_colored_cancellation_rgb_only(img_array, cancellation_color='red', color_threshold=40, dark_background_mode=True, strictness=15, black_subtraction=True):
    """
    RGB-only fallback method for colored cancellation extraction.
    
    This method avoids HSV conversion issues by using only RGB channel analysis.
    """
    print(f"DEBUG: Using RGB-only extraction for {cancellation_color}")
    print(f"DEBUG: Input array dtype: {img_array.dtype}, shape: {img_array.shape}")
    
    # Validate input array
    if img_array is None or img_array.size == 0:
        print("ERROR: Empty or None input array")
        return None, None, None
    
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        print(f"ERROR: Invalid array shape {img_array.shape}, expected (height, width, 3)")
        return None, None, None
    
    if img_array.shape[0] < 1 or img_array.shape[1] < 1:
        print(f"ERROR: Array too small: {img_array.shape}")
        return None, None, None
    
    # Robust data type handling
    if img_array.dtype != np.uint8:
        print(f"DEBUG: Converting from {img_array.dtype} to uint8")
        if img_array.dtype in [np.uint16, np.int16, np.int32, np.int64]:
            # Scale down from higher bit depths
            if img_array.max() > 255:
                img_array = (img_array.astype(np.float64) / img_array.max() * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        elif img_array.dtype in [np.float32, np.float64]:
            # Handle float arrays (may be 0-1 or 0-255 range)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        else:
            # Fallback for any other types
            img_array = np.clip(img_array.astype(np.float64), 0, 255).astype(np.uint8)
    
    # Ensure values are in valid range
    img_array = np.clip(img_array, 0, 255)
    print(f"DEBUG: Final array dtype: {img_array.dtype}, range: {img_array.min()}-{img_array.max()}")
    
    red = img_array[:, :, 0].astype(np.float32)
    green = img_array[:, :, 1].astype(np.float32)
    blue = img_array[:, :, 2].astype(np.float32)
    
    # Enhanced RGB-based color detection with dark background handling
    if cancellation_color == 'red':
        # Special case: If this is a black stamp (like Penny Black) with red cancellation,
        # ANY red tint should be considered cancellation since the stamp has no red design
        overall_brightness = (red + green + blue) / 3
        is_black_stamp = np.mean(overall_brightness) < 100  # Very dark stamp overall
        
        if is_black_stamp and black_subtraction:
            print("DEBUG: Detected black stamp (like Penny Black) - using black subtraction approach")
            
            # Step 1: Remove the black stamp design first
            black_threshold_for_removal = min(80, 50 + color_threshold)  # Adjust based on sensitivity
            img_without_black, removed_black_mask = extract_black_stamp_background(img_array, black_threshold_for_removal)
            
            # Step 2: Now detect red in the remaining image (which should mostly be red cancellation)
            red_remaining = img_without_black[:, :, 0].astype(np.float32)
            green_remaining = img_without_black[:, :, 1].astype(np.float32)
            blue_remaining = img_without_black[:, :, 2].astype(np.float32)
            
            # In the remaining image, look for any red tint (since black stamp is removed)
            # This should be much cleaner now
            non_white_mask = (red_remaining < 250) | (green_remaining < 250) | (blue_remaining < 250)
            red_bias = (red_remaining > green_remaining + 5) & (red_remaining > blue_remaining + 5)
            
            # Combine: non-white areas that have red bias
            color_mask = non_white_mask & red_bias
            
            print(f"DEBUG: After black removal, found {np.sum(color_mask)} red pixels")
            
        elif is_black_stamp:
            print("DEBUG: Detected black stamp - using direct red detection (no black subtraction)")
            # Fall back to the previous direct detection method
            # Calculate background red level
            background_red = np.percentile(red, 50)
            background_green = np.percentile(green, 50) 
            background_blue = np.percentile(blue, 50)
            
            print(f"DEBUG: Background levels - R:{background_red:.1f}, G:{background_green:.1f}, B:{background_blue:.1f}")
            
            # Look for pixels significantly redder than background
            red_enhancement_threshold = max(8, color_threshold // 3)
            red_enhanced = red > (background_red + red_enhancement_threshold)
            
            # Also ensure red is dominant
            red_dominance = (red > green + 3) & (red > blue + 3)
            
            # Combine
            color_mask = red_enhanced & red_dominance
            
            print(f"DEBUG: Direct detection found {np.sum(color_mask)} pixels")
            
        else:
            # Method 1: Traditional dominance for bright areas (colored stamps)
            bright_mask = red > 100
            dominance_mask = (red > green + color_threshold) & (red > blue + color_threshold)
            traditional_red = bright_mask & dominance_mask
            
            if dark_background_mode:
                # Method 2: Enhanced detection for dark backgrounds (colored stamps)
                dark_areas = overall_brightness < 80  # Very dark areas
                
                # In dark areas, look for any red bias, even if subtle
                red_bias_threshold = max(5, color_threshold // 4)  # Much more sensitive
                red_tint = (red > green + red_bias_threshold) & (red > blue + red_bias_threshold)
                
                # Additional check: red should be the strongest channel in dark areas
                red_strongest_dark = (red >= green) & (red >= blue)
                
                # For dark areas, combine red tint with red being strongest channel
                dark_red = dark_areas & red_tint & red_strongest_dark
                
                # Method 3: Relative red enhancement detection
                if overall_brightness.max() > 0:
                    red_ratio = red / (overall_brightness + 1)
                    enhanced_red_areas = red_ratio > 1.1
                    moderate_dark = (overall_brightness < 120) & (overall_brightness > 30)
                    relative_red = moderate_dark & enhanced_red_areas
                else:
                    relative_red = np.zeros_like(red, dtype=bool)
                
                # Combine all detection methods
                color_mask = traditional_red | dark_red | relative_red
                print(f"DEBUG: Red detection - Traditional: {np.sum(traditional_red)}, Dark: {np.sum(dark_red)}, Relative: {np.sum(relative_red)}")
            else:
                # Standard detection only
                color_mask = traditional_red
                print(f"DEBUG: Red detection - Traditional only: {np.sum(traditional_red)}")
        
        # For black stamps (Penny Black), skip aggressive filtering since any red is cancellation
        if is_black_stamp:
            print("DEBUG: Skipping strict filtering for black stamp - any red is cancellation")
            # Just do basic cleanup
            try:
                import cv2
                mask_uint8 = (color_mask * 255).astype(np.uint8)
                kernel_small = np.ones((2,2), np.uint8)
                # Just remove single pixel noise
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_small)
                color_mask = mask_uint8 > 0
            except ImportError:
                pass  # Use original mask if no cv2
        elif np.any(color_mask):
            # Import cv2 for advanced morphological operations (colored stamps only)
            try:
                import cv2
            except ImportError:
                print("WARNING: OpenCV not available, skipping advanced filtering")
                color_mask = color_mask  # Use unfiltered mask
            else:
                # Apply morphological operations to clean up the mask
                kernel_small = np.ones((3,3), np.uint8)
                
                # Convert to uint8 for morphological operations
                mask_uint8 = (color_mask * 255).astype(np.uint8)
                
                # Close small gaps in cancellation lines
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_small)
                
                # Remove isolated small spots (likely stamp noise, not cancellation)
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_small)
                
                # Additional filtering: Remove very large connected areas
                # (cancellations are usually lines/text, not large solid areas)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                
                filtered_mask = np.zeros_like(mask_uint8)
                total_image_area = mask_uint8.shape[0] * mask_uint8.shape[1]
                
                for i in range(1, num_labels):  # Skip background (label 0)
                    area = stats[i, cv2.CC_STAT_AREA]
                    # Remove components that are too large (likely stamp design, not cancellation)
                    # or too small (likely noise)
                    if 10 < area < (total_image_area * 0.15):  # Between 10 pixels and 15% of image
                        filtered_mask[labels == i] = 255
                
                color_mask = filtered_mask > 0
    elif cancellation_color == 'blue':
        # Blue should dominate over red and green  
        color_mask = (blue > red + color_threshold) & (blue > green + color_threshold)
        color_mask = color_mask & (blue > 100)
    elif cancellation_color == 'green':
        # Green should dominate over red and blue
        color_mask = (green > red + color_threshold) & (green > blue + color_threshold)
        color_mask = color_mask & (green > 100)
    else:
        color_mask = np.zeros_like(red, dtype=bool)
    
    # Convert to uint8 mask
    mask_uint8 = (color_mask * 255).astype(np.uint8)
    
    # Clean up the mask if OpenCV is available
    try:
        import cv2
        kernel = np.ones((3,3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    except ImportError:
        pass
    
    # Convert back to boolean
    final_mask = mask_uint8 > 127
    
    # Create extraction results
    results = {}
    gray = np.mean(img_array, axis=2).astype(np.uint8)
    
    # If we used black subtraction on a black stamp, save the intermediate result
    if is_black_stamp and black_subtraction and 'img_without_black' in locals():
        results['black_removed'] = img_without_black.astype(np.uint8)
    
    # 1. Pure colored cancellation on white
    results['pure_colored'] = np.where(final_mask, 0, 255).astype(np.uint8)
    
    # 2. Preserve original colors of the cancellation
    try:
        color_preserved = np.where(
            np.stack([final_mask, final_mask, final_mask], axis=2),
            img_array,
            [255, 255, 255]
        )
        # Ensure PIL-compatible dtype
        if color_preserved.dtype != np.uint8:
            color_preserved = color_preserved.astype(np.uint8)
        results['color_preserved'] = color_preserved
    except Exception as e:
        print(f"WARNING: Could not create color-preserved result: {e}")
        # Fallback: create white image with same dimensions
        results['color_preserved'] = np.full_like(img_array, 255, dtype=np.uint8)
    
    # 3. Grayscale version
    try:
        grayscale = np.where(final_mask, gray, 255)
        results['grayscale'] = grayscale.astype(np.uint8)
    except Exception as e:
        print(f"WARNING: Could not create grayscale result: {e}")
        results['grayscale'] = np.full((img_array.shape[0], img_array.shape[1]), 255, dtype=np.uint8)
    
    # 4. Enhanced contrast version
    if np.any(final_mask):
        cancellation_values = gray[final_mask]
        if len(cancellation_values) > 0:
            min_val, max_val = cancellation_values.min(), cancellation_values.max()
            if max_val > min_val:
                enhanced = np.where(
                    final_mask,
                    ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8),
                    255
                )
                results['enhanced'] = enhanced
            else:
                results['enhanced'] = results['grayscale']
        else:
            results['enhanced'] = results['grayscale']
    else:
        results['enhanced'] = results['grayscale']
    
    # Analysis
    total_pixels = final_mask.size
    cancellation_pixels = np.sum(final_mask)
    coverage_percentage = (cancellation_pixels / total_pixels) * 100
    
    analysis = {
        'coverage_percentage': coverage_percentage,
        'total_pixels': total_pixels,
        'cancellation_pixels': cancellation_pixels,
        'cancellation_color': cancellation_color,
        'color_threshold_used': color_threshold,
        'saturation_threshold_used': 'N/A (RGB-only mode)',
        'avg_cancellation_brightness': np.mean(gray[final_mask]) if cancellation_pixels > 0 else 0,
        'extraction_method': f'RGB-only with dark background mode: {dark_background_mode}',
        'dark_background_mode': dark_background_mode
    }
    
    return results, mask_uint8, analysis

def process_image(image_path, output_dir=None, black_threshold=60, saturation_threshold=30, red_offset=40):
    """Process a single image and extract black ink cancellations."""
    
    print(f"Processing: {Path(image_path).name}")
    print("-" * 50)
    
    # Load image
    try:
        pil_image = Image.open(image_path)
        print(f"Format: {pil_image.mode}, Size: {pil_image.size}")
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        img_array = np.array(pil_image)
        print(f"Array: {img_array.shape}, {img_array.dtype}, range: {img_array.min()}-{img_array.max()}")
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(image_path).parent / "black_ink_extraction"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    base_name = Path(image_path).stem
    
    # Extract black ink
    print("Extracting black ink cancellations...")
    results, mask, analysis = extract_black_ink(
        img_array, black_threshold, saturation_threshold, red_offset
    )
    
    # Save results
    print("Saving extraction results...")
    
    # Save original for reference
    pil_image.save(output_dir / f"{base_name}_original.png")
    
    # Save mask
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        output_dir / f"{base_name}_detection_mask.png"
    )
    
    # Save all extraction variants
    for variant, image_data in results.items():
        Image.fromarray(image_data).save(
            output_dir / f"{base_name}_{variant}_cancellation.png"
        )
    
    # Create comparison visualization
    create_comparison_plot(img_array, results, analysis, 
                         output_dir / f"{base_name}_comparison.png", base_name)
    
    # Save analysis report
    save_analysis_report(analysis, output_dir / f"{base_name}_analysis.txt", 
                        Path(image_path).name, black_threshold, saturation_threshold, red_offset)
    
    print(f"Results saved to: {output_dir}")
    print(f"Black ink coverage: {analysis['coverage_percentage']:.1f}%")
    print(f"Cancellation pixels: {analysis['cancellation_pixels']:,}")
    
    return output_dir

def create_comparison_plot(original, results, analysis, save_path, base_name):
    """Create a comparison plot showing original and all extraction results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Black Ink Extraction Results - {base_name}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Stamp')
    axes[0, 0].axis('off')
    
    # Pure black extraction
    axes[0, 1].imshow(results['pure_black'], cmap='gray')
    axes[0, 1].set_title('Pure Black Extraction')
    axes[0, 1].axis('off')
    
    # Grayscale extraction
    axes[0, 2].imshow(results['grayscale'], cmap='gray')
    axes[0, 2].set_title('Grayscale Extraction')
    axes[0, 2].axis('off')
    
    # Enhanced extraction
    axes[1, 0].imshow(results['enhanced'], cmap='gray')
    axes[1, 0].set_title('Enhanced Contrast')
    axes[1, 0].axis('off')
    
    # Inverted extraction
    axes[1, 1].imshow(results['inverted'], cmap='gray')
    axes[1, 1].set_title('Inverted (White on Black)')
    axes[1, 1].axis('off')
    
    # Analysis text
    axes[1, 2].axis('off')
    analysis_text = f"""Black Ink Analysis:

Coverage: {analysis['coverage_percentage']:.1f}%
Total pixels: {analysis['total_pixels']:,}
Cancellation pixels: {analysis['cancellation_pixels']:,}

Detection Parameters:
Black threshold: {analysis['black_threshold_used']}
Saturation threshold: {analysis['saturation_threshold_used']}
Red median: {analysis['red_median']:.1f}
Red threshold: {analysis['red_threshold']:.1f}

Avg cancellation brightness: {analysis['avg_cancellation_brightness']:.1f}"""
    
    axes[1, 2].text(0.05, 0.95, analysis_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

def save_analysis_report(analysis, save_path, filename, black_thresh, sat_thresh, red_offset):
    """Save detailed analysis report."""
    
    with open(save_path, 'w') as f:
        f.write("BLACK INK CANCELLATION EXTRACTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Image: {filename}\n")
        f.write(f"Processing date: {Path().cwd()}\n\n")
        
        f.write("EXTRACTION PARAMETERS:\n")
        f.write(f"  Black threshold: {black_thresh} (0-255)\n")
        f.write(f"  Saturation threshold: {sat_thresh} (0-255)\n") 
        f.write(f"  Red channel offset: {red_offset}\n\n")
        
        f.write("ANALYSIS RESULTS:\n")
        f.write(f"  Total image pixels: {analysis['total_pixels']:,}\n")
        f.write(f"  Black ink pixels: {analysis['cancellation_pixels']:,}\n")
        f.write(f"  Coverage percentage: {analysis['coverage_percentage']:.2f}%\n")
        f.write(f"  Red channel median: {analysis['red_median']:.1f}\n")
        f.write(f"  Red threshold used: {analysis['red_threshold']:.1f}\n")
        f.write(f"  Avg cancellation brightness: {analysis['avg_cancellation_brightness']:.1f}\n\n")
        
        f.write("OUTPUT FILES GENERATED:\n")
        f.write("  - *_pure_black_cancellation.png (High contrast black on white)\n")
        f.write("  - *_grayscale_cancellation.png (Preserves ink density)\n")
        f.write("  - *_enhanced_cancellation.png (Enhanced contrast)\n")
        f.write("  - *_inverted_cancellation.png (White on black)\n")
        f.write("  - *_detection_mask.png (Detection mask)\n")
        f.write("  - *_comparison.png (Side-by-side comparison)\n")

def main():
    """Main command line interface."""
    
    parser = argparse.ArgumentParser(
        description='Extract black ink cancellations from colored stamp images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image with default settings
  python3 black_ink_extractor.py stamp.tif
  
  # Process with custom sensitivity 
  python3 black_ink_extractor.py stamp.tif --black-threshold 70 --red-offset 50
  
  # Batch process all TIFF files in directory
  python3 black_ink_extractor.py /path/to/stamps/ --batch
  
  # Custom output directory
  python3 black_ink_extractor.py stamp.tif --output /path/to/results/
        """
    )
    
    parser.add_argument('input_path', help='Path to image file or directory')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all image files in directory')
    parser.add_argument('--black-threshold', type=int, default=60,
                       help='Black ink brightness threshold (0-255, default: 60)')
    parser.add_argument('--saturation-threshold', type=int, default=30,
                       help='Color saturation threshold (0-255, default: 30)')
    parser.add_argument('--red-offset', type=int, default=40,
                       help='Red channel detection offset (default: 40)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if args.batch and input_path.is_dir():
        # Batch processing
        image_extensions = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {input_path}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image files to process")
        print("=" * 60)
        
        success_count = 0
        for image_file in image_files:
            try:
                result_dir = process_image(
                    str(image_file), args.output, 
                    args.black_threshold, args.saturation_threshold, args.red_offset
                )
                if result_dir:
                    success_count += 1
                print(f"✓ Completed: {image_file.name}\n")
            except Exception as e:
                print(f"✗ Error processing {image_file.name}: {e}\n")
        
        print(f"Batch processing complete: {success_count}/{len(image_files)} successful")
        
    elif input_path.is_file():
        # Single file processing
        try:
            result_dir = process_image(
                str(input_path), args.output,
                args.black_threshold, args.saturation_threshold, args.red_offset
            )
            
            if result_dir:
                # Open comparison plot
                comparison_file = result_dir / f"{input_path.stem}_comparison.png"
                if comparison_file.exists():
                    print(f"\nOpening results visualization...")
                    import subprocess
                    subprocess.run(['open', str(comparison_file)], check=False)
            
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print(f"Invalid path: {input_path}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()