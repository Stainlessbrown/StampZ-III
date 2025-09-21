#!/usr/bin/env python3
"""
Fixed Monotone Stamp Perforation Detection

Addresses issues with:
- Missing monotone visualization
- Averaging opposite sides instead of using best-side selection
- Under-counting perforations due to too-conservative tic detection
"""

import cv2
import numpy as np
import math
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PerforationTic:
    """A perforation indentation point."""
    x: float
    y: float
    depth: float
    edge_type: str


class MonotonePerforationDetectorFixed:
    """Fixed perforation detection with visualization and improved tic detection."""
    
    def __init__(self, dpi: int = 800, save_debug_images: bool = True, save_faux_reverse: bool = True):
        self.dpi = dpi
        self.save_debug = save_debug_images
        self.save_faux_reverse = save_faux_reverse
    
    def detect_perforations(self, image: np.ndarray, stamp_name: str = "stamp") -> Dict:
        """Detect perforations with visualization and debugging."""
        
        print("=== FIXED Monotone Perforation Detection ===")
        
        # Step 1: Create faux reverse stamp with expanded background
        faux_reverse = self._create_faux_reverse_stamp(image, stamp_name)
        
        # Step 2: Create monotone stamp from faux reverse
        monotone_stamp = self._create_monotone_stamp(faux_reverse, stamp_name)
        
        # Step 2: Enhance structural features  
        enhanced = self._enhance_structural_features(monotone_stamp, stamp_name)
        
        # Step 3: Detect perforation lines using improved detection
        results = self._detect_perforations_improved(enhanced, stamp_name)
        
        return results
    
    def _create_monotone_stamp(self, image: np.ndarray, stamp_name: str) -> np.ndarray:
        """Convert stamp to monotone while preserving structural features."""
        
        print("Creating monotone stamp...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # 1. Detect image format and create appropriate stamp mask
        mean_intensity = np.mean(gray)
        
        if mean_intensity < 80:  # Faux reverse: mostly black background
            print(f"   Processing faux reverse format (mean intensity: {mean_intensity:.1f})")
            # In faux reverse: white (255) = stamp, black (0) = background/perforations
            _, stamp_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            print(f"   Processing normal format (mean intensity: {mean_intensity:.1f})")
            # Normal format: use multiple thresholding methods
            _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Adaptive threshold to catch details
            adaptive_mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
            )
            
            # Combine masks - stamp is where either method detects content
            stamp_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)
        
        # Clean up the mask  
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_OPEN, kernel)
        
        # 2. Create monotone stamp
        monotone = np.zeros_like(gray)
        monotone[stamp_mask > 0] = 128  # Uniform gray for stamp area
        
        # 3. Preserve perforations based on image format
        stamp_interior = stamp_mask > 0
        
        if mean_intensity < 80:  # Faux reverse format
            # In faux reverse: perforations are already black (0) within white stamp areas (255)
            perforations = (gray < 50) & stamp_interior  # Black pixels within stamp
        else:
            # Normal format: multiple perforation detection methods
            very_dark = gray < 60  # Dark pixels (potential perforations)
            
            # Also use edge detection to find perforation boundaries
            edges = cv2.Canny(gray, 20, 50)
            edge_perforations = edges > 0
            
            # Combine perforation detection methods
            perforations = (very_dark & stamp_interior) | (edge_perforations & stamp_interior)
        
        monotone[perforations] = 0  # Keep perforations black
        
        # Save debug image
        if self.save_debug:
            debug_image = np.zeros((monotone.shape[0], monotone.shape[1] * 3), dtype=np.uint8)
            debug_image[:, :monotone.shape[1]] = gray  # Original
            debug_image[:, monotone.shape[1]:monotone.shape[1]*2] = stamp_mask  # Mask
            debug_image[:, monotone.shape[1]*2:] = monotone  # Monotone result
            
            debug_path = f"{stamp_name}_monotone_debug.jpg"
            cv2.imwrite(debug_path, debug_image)
            print(f"   Saved debug image: {debug_path}")
        
        print(f"Monotone stamp created: {np.unique(monotone)} unique values")
        return monotone
    
    def _create_faux_reverse_stamp(self, image: np.ndarray, stamp_name: str) -> np.ndarray:
        """Create faux reverse stamp: unprinted paper against expanded black background.
        
        This simulates scanning from the back with expanded borders to eliminate
        close-cropping issues and give perforations room to be fully detected.
        """
        print("Creating faux reverse stamp with expanded background...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Calculate expansion: 1-2mm each side
        pixels_per_mm = self.dpi / 25.4
        expansion_mm = 1.5  # 1.5mm expansion each side
        expansion_pixels = int(expansion_mm * pixels_per_mm)
        
        print(f"   Expanding background by {expansion_pixels}px ({expansion_mm}mm) each side")
        
        # Create expanded canvas
        h, w = gray.shape
        new_h = h + (2 * expansion_pixels)
        new_w = w + (2 * expansion_pixels)
        
        # Start with black background (simulating scanner bed)
        faux_reverse = np.zeros((new_h, new_w), dtype=np.uint8)
        
        # Create stamp mask to identify stamp vs background areas
        _, stamp_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_OPEN, kernel)
        
        # Create expanded stamp mask
        expanded_mask = np.zeros((new_h, new_w), dtype=np.uint8)
        expanded_mask[expansion_pixels:expansion_pixels+h, expansion_pixels:expansion_pixels+w] = stamp_mask
        
        # In the faux reverse:
        # - Stamp areas become white (unprinted paper)
        # - Background stays black
        # - Perforations (holes) stay black
        faux_reverse[expanded_mask > 0] = 255  # White paper where stamp exists
        
        # Preserve perforation holes as black
        original_dark = gray < 60  # Dark areas in original (perforations, cancellations, etc.)
        stamp_interior = stamp_mask > 0
        perforations_in_original = original_dark & stamp_interior
        
        # Map perforations to expanded image
        expanded_perforations = np.zeros((new_h, new_w), dtype=bool)
        expanded_perforations[expansion_pixels:expansion_pixels+h, expansion_pixels:expansion_pixels+w] = perforations_in_original
        
        faux_reverse[expanded_perforations] = 0  # Keep perforations black
        
        # Optional: save faux reverse image
        if self.save_faux_reverse:
            faux_path = f"{stamp_name}_faux_reverse.jpg"
            cv2.imwrite(faux_path, faux_reverse)
            print(f"   Saved faux reverse image: {faux_path}")
        
        # Debug: check pixel value distribution
        unique_values = np.unique(faux_reverse)
        mean_val = np.mean(faux_reverse)
        print(f"   Faux reverse: unique values {unique_values}, mean {mean_val:.1f}")
        
        return faux_reverse
    
    def _enhance_structural_features(self, monotone: np.ndarray, stamp_name: str) -> np.ndarray:
        """Enhance structural features like edges and perforations."""
        
        print("Enhancing structural features...")
        
        # 1. Edge detection with multiple scales
        edges1 = cv2.Canny(monotone, 20, 60)  # Sensitive 
        edges2 = cv2.Canny(monotone, 40, 100)  # Standard
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, edges2)
        
        # 2. Morphological operations to connect perforation features
        kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line)
        
        # 3. Combine original monotone with enhanced edges
        enhanced = monotone.copy()
        enhanced[edges > 0] = 255  # Bright edges for visibility
        
        # Save debug image
        if self.save_debug:
            debug_path = f"{stamp_name}_enhanced_debug.jpg"
            cv2.imwrite(debug_path, enhanced)
            print(f"   Saved enhanced debug image: {debug_path}")
        
        return enhanced
    
    def _detect_perforations_improved(self, enhanced: np.ndarray, stamp_name: str) -> Dict:
        """Improved perforation detection with better tic finding."""
        
        print("Detecting perforations with improved algorithm...")
        
        results = {}
        h, w = enhanced.shape
        
        # Use standard edge regions - boundary detection was too complex
        edge_width = min(25, min(h, w) // 18)  # Slightly smaller regions
        print(f"Using edge width: {edge_width}")
        
        edge_regions = {
            'top': enhanced[:edge_width, :],
            'bottom': enhanced[-edge_width:, :],
            'left': enhanced[:, :edge_width],
            'right': enhanced[:, -edge_width:]
        }
        
        for edge_type, region in edge_regions.items():
            if region.size == 0:
                continue
                
            print(f"\\nProcessing {edge_type} edge region: {region.shape}")
            
            # Find perforation line
            perforation_line = self._find_perforation_line_improved(region, edge_type)
            
            if len(perforation_line) < 10:
                print(f"   {edge_type}: Too few line points ({len(perforation_line)})")
                continue
            
            # Adjust coordinates to full image
            if edge_type == 'bottom':
                perforation_line = [(x, y + h - edge_width) for x, y in perforation_line]
            elif edge_type == 'right':
                perforation_line = [(x + w - edge_width, y) for x, y in perforation_line]
            
            # Find tics with improved detection
            raw_tics = self._find_improved_tics(perforation_line, edge_type)
            
            # Cluster nearby tics and keep the best from each cluster
            clustered_tics = self._cluster_and_filter_tics(raw_tics, edge_type)
            
            print(f"   {edge_type}: Found {len(raw_tics)} raw tics -> {len(clustered_tics)} clustered tics")
            
            if len(clustered_tics) >= 3:
                gauge = self._calculate_gauge_from_tics(clustered_tics)
                results[edge_type] = {
                    'gauge': gauge,
                    'tics': len(clustered_tics),
                    'tic_positions': [(t.x, t.y) for t in clustered_tics]
                }
                print(f"   {edge_type}: Gauge = {gauge:.2f}")
        
        # Select best side from each axis (opposite sides should be identical)
        final_results = self._select_best_sides(results)
        
        # Create perforation detection visualization
        if self.save_debug:
            self._save_perforation_visualization(enhanced, results, stamp_name)
        
        return final_results
    
    def _find_perforation_line_improved(self, region: np.ndarray, edge_type: str) -> List[Tuple[int, int]]:
        """Improved perforation line detection."""
        
        h, w = region.shape
        line_points = []
        
        if edge_type in ['top', 'bottom']:
            # Horizontal edge - scan vertically
            for x in range(0, w, 2):  # Every 2 pixels
                col = region[:, x]
                
                # Look for stamp boundary (transition to gray=128)
                if edge_type == 'top':
                    # Find first stamp pixel (128) from top
                    stamp_pixels = np.where(col == 128)[0]
                    if len(stamp_pixels) > 0:
                        # But also check for perforations (0) that might be deeper
                        perf_pixels = np.where(col == 0)[0]
                        if len(perf_pixels) > 0 and perf_pixels[-1] > stamp_pixels[0]:
                            # Use deepest perforation
                            line_points.append((x, perf_pixels[-1]))
                        else:
                            # Use stamp boundary
                            line_points.append((x, stamp_pixels[0]))
                else:  # bottom
                    # Find last stamp pixel from bottom
                    stamp_pixels = np.where(col == 128)[0]
                    if len(stamp_pixels) > 0:
                        # Check for perforations above the boundary
                        perf_pixels = np.where(col == 0)[0]
                        if len(perf_pixels) > 0 and perf_pixels[0] < stamp_pixels[-1]:
                            # Use highest perforation
                            line_points.append((x, perf_pixels[0]))
                        else:
                            # Use stamp boundary
                            line_points.append((x, stamp_pixels[-1]))
        else:
            # Vertical edge - scan horizontally
            for y in range(0, h, 2):  # Every 2 pixels
                row = region[y, :]
                
                if edge_type == 'left':
                    # Find first stamp pixel from left
                    stamp_pixels = np.where(row == 128)[0]
                    if len(stamp_pixels) > 0:
                        # Check for deeper perforations
                        perf_pixels = np.where(row == 0)[0]
                        if len(perf_pixels) > 0 and perf_pixels[-1] > stamp_pixels[0]:
                            line_points.append((perf_pixels[-1], y))
                        else:
                            line_points.append((stamp_pixels[0], y))
                else:  # right
                    # Find last stamp pixel from right
                    stamp_pixels = np.where(row == 128)[0]
                    if len(stamp_pixels) > 0:
                        # Check for perforations extending outward
                        perf_pixels = np.where(row == 0)[0]
                        if len(perf_pixels) > 0 and perf_pixels[0] < stamp_pixels[-1]:
                            line_points.append((perf_pixels[0], y))
                        else:
                            line_points.append((stamp_pixels[-1], y))
        
        return line_points
    
    def _find_improved_tics(self, line_points: List[Tuple[int, int]], edge_type: str) -> List[PerforationTic]:
        """Improved tic detection that catches more perforations."""
        
        if len(line_points) < 10:
            return []
        
        # Sort line points
        if edge_type in ['top', 'bottom']:
            line_points.sort(key=lambda p: p[0])  # Sort by x
        else:
            line_points.sort(key=lambda p: p[1])  # Sort by y
        
        tics = []
        
        # Use selective detection parameters for major perforations only
        window = 6  # Slightly larger window for stability
        # 18-gauge threshold: finest perforation ever used
        min_tic_distance_18gauge = int((20.0 / 18.0) * (self.dpi / 25.4))  # 35px at 800 DPI
        min_tic_distance = max(min_tic_distance_18gauge, 30)  # Use 18-gauge as minimum
        
        print(f"   Using 18-gauge minimum spacing: {min_tic_distance}px ({min_tic_distance/(self.dpi/25.4):.2f}mm)")
        
        for i in range(window, len(line_points) - window):
            current_point = line_points[i]
            neighborhood = line_points[i-window:i+window+1]
            
            is_tic = False
            
            if edge_type == 'top':
                # For top edge, tics are local maxima in y (deeper into stamp)
                current_y = current_point[1]
                neighbor_ys = [p[1] for p in neighborhood]
                mean_y = np.mean(neighbor_ys)
                # High selectivity - only major perforation indentations
                depth_threshold = 3.5  # Higher threshold for major indentations
                if current_y >= max(neighbor_ys) and current_y > mean_y + depth_threshold:
                    is_tic = True
            elif edge_type == 'bottom':
                current_y = current_point[1]
                neighbor_ys = [p[1] for p in neighborhood]
                mean_y = np.mean(neighbor_ys)
                depth_threshold = 3.5
                if current_y <= min(neighbor_ys) and current_y < mean_y - depth_threshold:
                    is_tic = True
            elif edge_type == 'left':
                current_x = current_point[0]
                neighbor_xs = [p[0] for p in neighborhood]
                mean_x = np.mean(neighbor_xs)
                depth_threshold = 3.5
                if current_x >= max(neighbor_xs) and current_x > mean_x + depth_threshold:
                    is_tic = True
            else:  # right
                current_x = current_point[0]
                neighbor_xs = [p[0] for p in neighborhood]
                mean_x = np.mean(neighbor_xs)
                depth_threshold = 3.5
                if current_x <= min(neighbor_xs) and current_x < mean_x - depth_threshold:
                    is_tic = True
            
            if is_tic:
                # Check minimum distance from existing tics
                too_close = False
                for existing_tic in tics:
                    dist = math.sqrt((current_point[0] - existing_tic.x)**2 + (current_point[1] - existing_tic.y)**2)
                    if dist < min_tic_distance:
                        too_close = True
                        break
                
                if not too_close:
                    # Calculate depth
                    if edge_type in ['top', 'bottom']:
                        depth = abs(current_point[1] - np.mean([p[1] for p in neighborhood]))
                    else:
                        depth = abs(current_point[0] - np.mean([p[0] for p in neighborhood]))
                    
                    tic = PerforationTic(
                        x=current_point[0],
                        y=current_point[1],
                        depth=depth,
                        edge_type=edge_type
                    )
                    tics.append(tic)
        
        return tics
    
    def _cluster_and_filter_tics(self, tics: List[PerforationTic], edge_type: str) -> List[PerforationTic]:
        """Cluster nearby tics and keep the best one from each cluster.
        
        This reduces over-detection by grouping tics that are likely 
        from the same perforation and keeping only the strongest.
        """
        if len(tics) < 2:
            return tics
            
        # Estimate cluster radius based on expected perforation spacing
        # For 14 gauge: ~1.43mm spacing, so cluster within ~0.5mm
        pixels_per_mm = self.dpi / 25.4
        cluster_radius_mm = 0.4  # Conservative cluster radius
        cluster_radius_pixels = cluster_radius_mm * pixels_per_mm
        
        print(f"   Clustering with radius {cluster_radius_pixels:.1f}px ({cluster_radius_mm}mm)")
        
        # Sort tics by position for processing
        if edge_type in ['top', 'bottom']:
            sorted_tics = sorted(tics, key=lambda t: t.x)
        else:
            sorted_tics = sorted(tics, key=lambda t: t.y)
        
        clusters = []
        current_cluster = [sorted_tics[0]]
        
        for i in range(1, len(sorted_tics)):
            current_tic = sorted_tics[i]
            last_tic = current_cluster[-1]
            
            # Calculate distance between tics
            if edge_type in ['top', 'bottom']:
                distance = abs(current_tic.x - last_tic.x)
            else:
                distance = abs(current_tic.y - last_tic.y)
            
            if distance <= cluster_radius_pixels:
                # Add to current cluster
                current_cluster.append(current_tic)
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = [current_tic]
        
        # Don't forget the last cluster
        clusters.append(current_cluster)
        
        # From each cluster, select the tic with the greatest depth
        best_tics = []
        for cluster in clusters:
            if len(cluster) == 1:
                best_tics.append(cluster[0])
            else:
                # Select tic with maximum depth (strongest indentation)
                best_tic = max(cluster, key=lambda t: t.depth)
                best_tics.append(best_tic)
        
        print(f"   Clustered {len(tics)} tics into {len(clusters)} clusters")
        
        return best_tics
    
    def _save_perforation_visualization(self, enhanced: np.ndarray, results: Dict, stamp_name: str):
        """Save visualization showing detected perforations and edge regions."""
        
        # Create color visualization
        h, w = enhanced.shape
        vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Define edge regions
        edge_width = min(25, min(h, w) // 18)
        
        # Draw edge regions in different colors
        colors = {
            'top': (0, 255, 0),     # Green
            'bottom': (0, 255, 255), # Yellow 
            'left': (255, 0, 0),    # Blue
            'right': (255, 0, 255)  # Magenta
        }
        
        # Highlight edge regions
        cv2.rectangle(vis, (0, 0), (w, edge_width), colors['top'], 2)  # Top
        cv2.rectangle(vis, (0, h-edge_width), (w, h), colors['bottom'], 2)  # Bottom
        cv2.rectangle(vis, (0, 0), (edge_width, h), colors['left'], 2)  # Left
        cv2.rectangle(vis, (w-edge_width, 0), (w, h), colors['right'], 2)  # Right
        
        # Draw detected tics
        for edge_type, data in results.items():
            if edge_type not in ['top', 'bottom', 'left', 'right']:
                continue
                
            if 'tic_positions' not in data:
                continue
                
            color = colors[edge_type]
            positions = data['tic_positions']
            gauge = data['gauge']
            
            # Draw tic positions as circles
            for x, y in positions:
                cv2.circle(vis, (int(x), int(y)), 3, color, -1)
                
            # Add gauge text
            if edge_type == 'top':
                text_pos = (10, 20)
            elif edge_type == 'bottom':
                text_pos = (10, h - 10)
            elif edge_type == 'left':
                text_pos = (5, h // 2)
            else:  # right
                text_pos = (w - 100, h // 2)
                
            text = f"{edge_type}: {gauge:.1f} ({len(positions)} tics)"
            cv2.putText(vis, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        debug_path = f"{stamp_name}_perforation_detection.jpg"
        cv2.imwrite(debug_path, vis)
        print(f"   Saved perforation detection visualization: {debug_path}")
    
    def _calculate_gauge_from_tics(self, tics: List[PerforationTic]) -> float:
        """Calculate perforation gauge using center-to-center spacing method.
        
        Uses minimum 3 perforations to determine consistent spacing,
        then divides 20mm by that spacing to get gauge.
        """
        if len(tics) < 3:
            print(f"   Need minimum 3 tics for spacing calculation, got {len(tics)}")
            return 0.0
        
        # Sort tics by position
        edge_type = tics[0].edge_type
        if edge_type in ['top', 'bottom']:
            sorted_tics = sorted(tics, key=lambda t: t.x)
        else:
            sorted_tics = sorted(tics, key=lambda t: t.y)
        
        # Calculate center-to-center distances between consecutive tics
        distances_pixels = []
        for i in range(len(sorted_tics) - 1):
            t1, t2 = sorted_tics[i], sorted_tics[i + 1]
            if edge_type in ['top', 'bottom']:
                distance = abs(t2.x - t1.x)
            else:
                distance = abs(t2.y - t1.y)
            distances_pixels.append(distance)
        
        if not distances_pixels:
            return 0.0
        
        # Use median spacing to avoid outliers from defective perforations
        median_spacing_pixels = np.median(distances_pixels)
        
        # Convert to mm
        pixels_per_mm = self.dpi / 25.4  # 25.4 mm per inch
        spacing_mm = median_spacing_pixels / pixels_per_mm
        
        # Gauge = 20mm / center-to-center spacing
        gauge = 20.0 / spacing_mm if spacing_mm > 0 else 0.0
        
        # Quality check: calculate standard deviation of spacings
        std_spacing = np.std(distances_pixels)
        cv = std_spacing / median_spacing_pixels if median_spacing_pixels > 0 else 1.0
        
        print(f"   Gauge calc: {len(tics)} tics, median spacing {median_spacing_pixels:.1f}px ({spacing_mm:.3f}mm)")
        print(f"   20mm ÷ {spacing_mm:.3f}mm = {gauge:.2f} gauge (CV={cv:.2f})")
        
        return gauge
    
    def _select_best_sides(self, results: Dict) -> Dict:
        """Select best side from each axis since opposite sides should be identical."""
        if not results:
            return {}
        
        final_results = {}
        
        # Select best horizontal side (top/bottom)
        horizontal_sides = {side: data for side, data in results.items() if side in ['top', 'bottom']}
        if horizontal_sides:
            # Select side with more tics (better detection)
            best_h_side = max(horizontal_sides.items(), key=lambda x: x[1]['tics'])
            final_results[best_h_side[0]] = best_h_side[1]
            final_results['horizontal_gauge'] = best_h_side[1]['gauge']
            print(f"\\nSelected {best_h_side[0]} as best horizontal side ({best_h_side[1]['tics']} tics)")
        
        # Select best vertical side (left/right)
        vertical_sides = {side: data for side, data in results.items() if side in ['left', 'right']}
        if vertical_sides:
            # Select side with more tics (better detection)
            best_v_side = max(vertical_sides.items(), key=lambda x: x[1]['tics'])
            final_results[best_v_side[0]] = best_v_side[1]
            final_results['vertical_gauge'] = best_v_side[1]['gauge']
            print(f"Selected {best_v_side[0]} as best vertical side ({best_v_side[1]['tics']} tics)")
        
        # Add all individual side results for debugging
        for side in ['top', 'bottom', 'left', 'right']:
            if side in results:
                final_results[side] = results[side]
        
        return final_results


def test_fixed_monotone():
    """Test the fixed monotone approach."""
    
    # Test with a single stamp first
    stamp_path = os.path.expanduser("~/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif")
    
    if os.path.exists(stamp_path):
        image = cv2.imread(stamp_path)
        if image is not None:
            stamp_name = "138-S10"
            detector = MonotonePerforationDetectorFixed(dpi=800, save_debug_images=True)
            results = detector.detect_perforations(image, stamp_name)
            
            print("\\n=== FINAL RESULTS ===")
            if results:
                # Show individual sides
                for side in ['top', 'bottom', 'left', 'right']:
                    if side in results:
                        data = results[side]
                        print(f"  {side.upper()}: {data['gauge']:.2f} ({data['tics']} tics)")
                
                # Show best side results  
                h_gauge = results.get('horizontal_gauge', 0)
                v_gauge = results.get('vertical_gauge', 0)
                
                if h_gauge > 0 and v_gauge > 0:
                    compound_gauge = (h_gauge + v_gauge) / 2
                    print(f"\\nBEST SIDE RESULTS:")
                    print(f"  Horizontal: {h_gauge:.2f}")
                    print(f"  Vertical: {v_gauge:.2f}")
                    print(f"  Compound: {compound_gauge:.2f}")
                    print(f"  Notation: {h_gauge:.1f} × {v_gauge:.1f}")
                    print(f"\\nExpected: 14.0 × 13.5")
                    print(f"Error: {abs(h_gauge - 14.0):.1f} × {abs(v_gauge - 13.5):.1f}")
            else:
                print("No perforations detected")
        else:
            print("Could not load image")
    else:
        print(f"Image not found: {stamp_path}")


if __name__ == "__main__":
    test_fixed_monotone()