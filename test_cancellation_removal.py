#!/usr/bin/env python3
"""
Interactive test script for the cancellation removal tool.
Run this to test the tool on your stamp images.
"""

import sys
from pathlib import Path
from cancellation_removal import CancellationRemover
import matplotlib.pyplot as plt

def interactive_test():
    """Interactive testing function."""
    print("Advanced Philatelic Cancellation Removal - Interactive Test")
    print("=" * 50)
    
    # Get image path from user
    while True:
        image_path = input("Enter path to stamp image (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            break
            
        if not Path(image_path).exists():
            print(f"File not found: {image_path}")
            continue
            
        try:
            print(f"\nProcessing: {image_path}")
            
            # Initialize the remover
            remover = CancellationRemover(image_path)
            
            # Process the image
            results = remover.process_image()
            
            # Ask user what they want to see
            print("\nProcessing complete! Available results:")
            print("1. View comparison plot")
            print("2. Show individual color channels")
            print("3. Show detection masks")
            print("4. Process another image")
            print("5. Quit")
            
            while True:
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == '1':
                    # Show the comparison plot that was saved
                    base_name = Path(image_path).stem
                    output_dir = Path(image_path).parent / "cancellation_removal_results"
                    comparison_path = output_dir / f"{base_name}_comparison.png"
                    
                    if comparison_path.exists():
                        print(f"Opening comparison plot: {comparison_path}")
                        # On macOS, use 'open' command to display image
                        import subprocess
                        subprocess.run(['open', str(comparison_path)])
                    else:
                        print("Comparison plot not found")
                        
                elif choice == '2':
                    # Show color channel analysis
                    print("Analyzing color channels...")
                    color_results = remover.enhanced_color_separation()
                    
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle('Color Channel Analysis', fontsize=16)
                    
                    channels = ['lab_l', 'lab_a', 'lab_b', 'hsv_h', 'hsv_s', 'hsv_v']
                    titles = ['LAB L (Lightness)', 'LAB A (Green-Red)', 'LAB B (Blue-Yellow)',
                             'HSV H (Hue)', 'HSV S (Saturation)', 'HSV V (Value)']
                    
                    for i, (channel, title) in enumerate(zip(channels, titles)):
                        row, col = i // 3, i % 3
                        if channel in color_results:
                            data = color_results[channel]
                            axes[row, col].imshow(data, cmap='gray')
                            axes[row, col].set_title(title)
                            axes[row, col].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                elif choice == '3':
                    # Show detection masks
                    print("Creating detection masks...")
                    
                    # Get different mask components
                    thresh_masks = remover.adaptive_thresholding_mask()
                    morph_results = remover.morphological_cancellation_detection()
                    composite_mask = remover.create_composite_mask()
                    
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle('Cancellation Detection Masks', fontsize=16)
                    
                    axes[0, 0].imshow(thresh_masks['dark_regions'], cmap='gray')
                    axes[0, 0].set_title('Dark Regions')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(morph_results['combined_lines'], cmap='gray')
                    axes[0, 1].set_title('Line Detection')
                    axes[0, 1].axis('off')
                    
                    axes[1, 0].imshow(composite_mask, cmap='gray')
                    axes[1, 0].set_title('Composite Mask')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(remover.original_rgb)
                    axes[1, 1].set_title('Original')
                    axes[1, 1].axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                    
                elif choice == '4':
                    break
                    
                elif choice == '5':
                    return
                    
                else:
                    print("Invalid choice. Please enter 1-5.")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            print("Make sure the required packages are installed:")
            print("pip install opencv-python numpy matplotlib scipy scikit-image pillow")

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'cv2', 'numpy', 'matplotlib', 'scipy', 'skimage', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall with:")
        print("pip install opencv-python numpy matplotlib scipy scikit-image pillow")
        return False
    
    return True

if __name__ == "__main__":
    if check_dependencies():
        interactive_test()
    else:
        sys.exit(1)