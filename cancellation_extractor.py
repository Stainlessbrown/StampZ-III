#!/usr/bin/env python3
"""
Philatelic Cancellation Extractor

This tool extracts and enhances cancellations (postmarks/obliterations) from stamp images,
making them clearly visible for study and documentation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from cancellation_removal import CancellationRemover

class CancellationExtractor(CancellationRemover):
    """Extends CancellationRemover to focus on extracting and enhancing cancellations."""
    
    def extract_cancellation_layer(self):
        """
        Extract the cancellation as a clear, enhanced layer.
        """
        # Get the composite mask (this identifies cancellation areas)
        cancellation_mask = self.create_composite_mask()
        
        # Create multiple enhanced cancellation extractions
        results = {}
        
        # Method 1: Direct mask application with enhancement
        # Ensure we have proper image format
        if len(self.original.shape) == 3:
            gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original.copy()
        
        # Ensure proper data type
        if gray.dtype != np.uint8:
            gray = cv2.convertScaleAbs(gray)
        
        # Apply mask to extract only cancellation areas
        cancellation_only = cv2.bitwise_and(gray, gray, mask=cancellation_mask)
        
        # Enhance contrast of the extracted cancellation
        enhanced_cancellation = cv2.equalizeHist(cancellation_only)
        
        # Make non-cancellation areas white for better visibility
        cancellation_on_white = np.where(cancellation_mask == 255, enhanced_cancellation, 255)
        results['cancellation_on_white'] = cancellation_on_white
        
        # Method 2: Inverted mask for better contrast
        cancellation_inverted = 255 - cancellation_on_white
        results['cancellation_inverted'] = cancellation_inverted
        
        # Method 3: Color-enhanced cancellation
        # Apply mask to original color image
        color_cancellation = self.original_rgb.copy()
        mask_3d = np.stack([cancellation_mask, cancellation_mask, cancellation_mask], axis=2)
        
        # Make non-cancellation areas white
        color_cancellation = np.where(mask_3d == 255, color_cancellation, [255, 255, 255])
        results['color_cancellation'] = color_cancellation
        
        # Method 4: High contrast black and white
        _, binary_mask = cv2.threshold(cancellation_mask, 127, 255, cv2.THRESH_BINARY)
        binary_cancellation = np.where(binary_mask == 255, 0, 255)  # Black cancellation on white
        results['binary_cancellation'] = binary_cancellation
        
        # Method 5: Edge-enhanced cancellation
        edges = cv2.Canny(gray, 50, 150)
        edge_cancellation = cv2.bitwise_and(edges, edges, mask=cancellation_mask)
        edge_enhanced = np.where(cancellation_mask == 255, edge_cancellation, 255)
        results['edge_enhanced'] = edge_enhanced
        
        return results, cancellation_mask
    
    def analyze_cancellation_characteristics(self, mask):
        """
        Analyze the characteristics of the detected cancellation.
        """
        analysis = {}
        
        # Calculate coverage area
        total_pixels = mask.shape[0] * mask.shape[1]
        cancellation_pixels = np.sum(mask == 255)
        coverage_percentage = (cancellation_pixels / total_pixels) * 100
        analysis['coverage_percentage'] = coverage_percentage
        
        # Find cancellation components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        analysis['num_components'] = num_labels - 1  # Subtract background
        analysis['component_sizes'] = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        analysis['component_centroids'] = centroids[1:]  # Skip background
        
        # Analyze shape characteristics
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            analysis['largest_contour_area'] = cv2.contourArea(largest_contour)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            analysis['bounding_box'] = (x, y, w, h)
            analysis['aspect_ratio'] = w / h if h > 0 else 0
        
        return analysis
    
    def create_detailed_analysis_plot(self, extraction_results, mask, analysis):
        """
        Create a comprehensive analysis plot showing all extraction methods.
        """
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Cancellation Extraction Analysis', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(self.original_rgb)
        axes[0, 0].set_title('Original Stamp')
        axes[0, 0].axis('off')
        
        # Detection mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Detected Cancellation Mask')
        axes[0, 1].axis('off')
        
        # Cancellation on white background
        axes[0, 2].imshow(extraction_results['cancellation_on_white'], cmap='gray')
        axes[0, 2].set_title('Cancellation on White')
        axes[0, 2].axis('off')
        
        # Inverted cancellation
        axes[1, 0].imshow(extraction_results['cancellation_inverted'], cmap='gray')
        axes[1, 0].set_title('Inverted Cancellation')
        axes[1, 0].axis('off')
        
        # Color cancellation
        axes[1, 1].imshow(extraction_results['color_cancellation'])
        axes[1, 1].set_title('Color Cancellation')
        axes[1, 1].axis('off')
        
        # Binary cancellation
        axes[1, 2].imshow(extraction_results['binary_cancellation'], cmap='gray')
        axes[1, 2].set_title('Binary Cancellation')
        axes[1, 2].axis('off')
        
        # Edge enhanced
        axes[2, 0].imshow(extraction_results['edge_enhanced'], cmap='gray')
        axes[2, 0].set_title('Edge Enhanced')
        axes[2, 0].axis('off')
        
        # Analysis text
        axes[2, 1].axis('off')
        analysis_text = f"""Cancellation Analysis:
        
Coverage: {analysis['coverage_percentage']:.1f}%
Components: {analysis['num_components']}
        
Largest Component:
Area: {analysis.get('largest_contour_area', 0):.0f} pixels
        
Bounding Box:
{analysis.get('bounding_box', 'N/A')}
        
Aspect Ratio: {analysis.get('aspect_ratio', 0):.2f}
"""
        axes[2, 1].text(0.1, 0.9, analysis_text, transform=axes[2, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Component visualization
        if analysis['num_components'] > 0:
            component_vis = mask.copy()
            # Add component centers
            for centroid in analysis['component_centroids']:
                cv2.circle(component_vis, (int(centroid[0]), int(centroid[1])), 5, 128, -1)
            axes[2, 2].imshow(component_vis, cmap='gray')
            axes[2, 2].set_title('Component Centers')
        else:
            axes[2, 2].axis('off')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def extract_and_analyze(self, output_dir=None):
        """
        Main extraction function that processes the image and saves all results.
        """
        if output_dir is None:
            output_dir = Path(self.image_path).parent / "cancellation_extraction_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        base_name = Path(self.image_path).stem
        
        print(f"Extracting cancellations from {self.image_path}...")
        
        # Extract cancellation layers
        print("  - Extracting cancellation layers...")
        extraction_results, mask = self.extract_cancellation_layer()
        
        # Analyze cancellation characteristics
        print("  - Analyzing cancellation characteristics...")
        analysis = self.analyze_cancellation_characteristics(mask)
        
        # Save all extraction results
        print("  - Saving extraction results...")
        
        # Save original for reference
        cv2.imwrite(str(output_dir / f"{base_name}_original.png"), self.original)
        
        # Save mask
        cv2.imwrite(str(output_dir / f"{base_name}_detection_mask.png"), mask)
        
        # Save all extraction methods
        for method, result in extraction_results.items():
            if len(result.shape) == 3:  # Color image
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_dir / f"{base_name}_{method}.png"), result_bgr)
            else:  # Grayscale image
                cv2.imwrite(str(output_dir / f"{base_name}_{method}.png"), result)
        
        # Create and save analysis plot
        print("  - Creating analysis plot...")
        fig = self.create_detailed_analysis_plot(extraction_results, mask, analysis)
        fig.savefig(str(output_dir / f"{base_name}_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save analysis as text file
        with open(output_dir / f"{base_name}_analysis.txt", 'w') as f:
            f.write(f"Cancellation Analysis for {Path(self.image_path).name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Coverage Percentage: {analysis['coverage_percentage']:.2f}%\n")
            f.write(f"Number of Components: {analysis['num_components']}\n")
            f.write(f"Component Sizes: {analysis['component_sizes'].tolist()}\n")
            if 'largest_contour_area' in analysis:
                f.write(f"Largest Component Area: {analysis['largest_contour_area']:.0f} pixels\n")
                f.write(f"Bounding Box (x,y,w,h): {analysis['bounding_box']}\n")
                f.write(f"Aspect Ratio: {analysis['aspect_ratio']:.3f}\n")
        
        print(f"Extraction results saved to: {output_dir}")
        print(f"Coverage: {analysis['coverage_percentage']:.1f}% of image")
        print(f"Components detected: {analysis['num_components']}")
        
        return extraction_results, analysis

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Extract and Enhance Philatelic Cancellations')
    parser.add_argument('input_path', help='Path to input image or directory')
    parser.add_argument('--output', '-o', help='Output directory')
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
                extractor = CancellationExtractor(str(image_file))
                extractor.extract_and_analyze(args.output)
                print()  # Add blank line between files
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                
    elif input_path.is_file():
        # Process single image
        try:
            extractor = CancellationExtractor(str(input_path))
            extractor.extract_and_analyze(args.output)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    else:
        print(f"Invalid input path: {input_path}")

if __name__ == "__main__":
    main()