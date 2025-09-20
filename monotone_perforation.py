#!/usr/bin/env python3
"""
Monotone Stamp Perforation Detection

This approach converts the stamp to monotone (uniform color) while preserving
structural features like perforations and edges. This eliminates ink patterns,
cancellation marks, and text that interfere with perforation detection.

Key insight: Perforations are STRUCTURAL features, not visual/color features.
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PerforationTic:
    """A perforation indentation point."""
    x: float
    y: float
    depth: float
    edge_type: str


class MonotonePerforationDetector:
    """Perforation detection using monotone stamp preprocessing."""
    
    def __init__(self, dpi: int = 800):
        self.dpi = dpi
    
    def detect_perforations(self, image: np.ndarray) -> Dict:
        """Detect perforations using monotone preprocessing approach."""
        
        print("=== Monotone Perforation Detection ===")
        
        # Step 1: Create monotone stamp
        monotone_stamp = self._create_monotone_stamp(image)
        
        # Step 2: Enhance structural features  
        enhanced = self._enhance_structural_features(monotone_stamp)
        
        # Step 3: Detect perforation lines using structural analysis
        results = self._detect_structural_perforations(enhanced)
        
        return results
    
    def _create_monotone_stamp(self, image: np.ndarray) -> np.ndarray:
        """Convert stamp to monotone while preserving structural features."""
        
        print("Creating monotone stamp...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # 1. Detect stamp vs background using adaptive thresholding
        # This separates structural features from background
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        
        # 2. Create stamp mask (non-background areas)
        # Use Otsu thresholding to find stamp region
        _, stamp_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_CLOSE, kernel)
        stamp_mask = cv2.morphologyEx(stamp_mask, cv2.MORPH_OPEN, kernel)
        
        # 4. Create monotone stamp
        # Everything inside stamp becomes uniform gray, background stays black
        monotone = np.zeros_like(gray)
        monotone[stamp_mask > 0] = 128  # Uniform gray for stamp area
        
        # 5. Preserve perforations (they should remain dark/black)
        # Perforations are the darkest areas within the stamp region
        very_dark = gray < 50  # Very dark pixels (likely perforations)
        stamp_interior = stamp_mask > 0
        perforations = very_dark & stamp_interior
        monotone[perforations] = 0  # Keep perforations black
        
        print(f"Monotone stamp created: {np.unique(monotone)} unique values")
        return monotone
    
    def _enhance_structural_features(self, monotone: np.ndarray) -> np.ndarray:
        """Enhance structural features like edges and perforations."""
        
        print("Enhancing structural features...")
        
        # 1. Edge detection to find structural boundaries
        edges = cv2.Canny(monotone, 30, 100)
        
        # 2. Morphological operations to connect perforation features
        kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line)
        
        # 3. Combine original monotone with enhanced edges
        enhanced = monotone.copy()
        enhanced[edges > 0] = 255  # Bright edges for visibility
        
        return enhanced
    
    def _detect_structural_perforations(self, enhanced: np.ndarray) -> Dict:
        """Detect perforations based on structural analysis using stamp boundary detection."""
        
        print("Detecting structural perforations using boundary analysis...")
        
        results = {}
        h, w = enhanced.shape
        
        # First, find the actual stamp boundaries in the monotone image
        stamp_boundaries = self._find_stamp_boundaries(enhanced)
        
        if not stamp_boundaries:
            print("Could not detect stamp boundaries, falling back to edge regions")
            edge_width = min(30, min(h, w) // 15)
        else:
            print(f"Detected stamp boundaries: {stamp_boundaries}")
            # Use boundary-based detection instead of fixed edge regions
        
        # Create boundary-based regions or fall back to edge regions
        if stamp_boundaries:
            edge_regions = self._create_boundary_regions(enhanced, stamp_boundaries)
        else:
            edge_width = min(30, min(h, w) // 15)
            edge_regions = {
                'top': enhanced[:edge_width, :],
                'bottom': enhanced[-edge_width:, :],
                'left': enhanced[:, :edge_width],
                'right': enhanced[:, -edge_width:]
            }
        
        for edge_type, region_info in edge_regions.items():
            if isinstance(region_info, dict):
                region = region_info['region']
                boundary_line = region_info.get('boundary_line', [])
                region_offset = region_info.get('offset', (0, 0))
            else:
                region = region_info
                boundary_line = []
                region_offset = (0, 0)
                
            if region.size == 0:
                continue
                
            print(f"\\nProcessing {edge_type} edge region: {region.shape}")
            
            # Find perforation line using boundary-aware approach
            if boundary_line:
                perforation_line = boundary_line  # Use detected boundary as perforation line
                print(f"   Using detected boundary line with {len(perforation_line)} points")
            else:
                perforation_line = self._find_perforation_line_structural(region, edge_type)
            
            if len(perforation_line) < 10:
                print(f"{edge_type}: Too few line points ({len(perforation_line)})")
                continue
            
            # Adjust coordinates to full image based on region offset
            if region_offset != (0, 0):
                offset_x, offset_y = region_offset
                perforation_line = [(x + offset_x, y + offset_y) for x, y in perforation_line]
            else:
                # Legacy adjustment for non-boundary regions
                if edge_type == 'bottom':
                    perforation_line = [(x, y + h - min(30, min(h, w) // 15)) for x, y in perforation_line]
                elif edge_type == 'right':
                    perforation_line = [(x + w - min(30, min(h, w) // 15), y) for x, y in perforation_line]
            
            # Find tics along the line using structural analysis
            tics = self._find_structural_tics(enhanced, perforation_line, edge_type)
            
            # If very few tics found, try a more sensitive approach for close-cropped images
            if len(tics) < 8 and len(perforation_line) > 20:
                print(f"   {edge_type}: Low tic count ({len(tics)}), trying sensitive detection...")
                sensitive_tics = self._find_sensitive_tics(perforation_line, edge_type)
                if len(sensitive_tics) > len(tics) * 1.5:  # Significantly more tics found
                    tics = sensitive_tics
                    print(f"   {edge_type}: Using sensitive detection ({len(tics)} tics)")
            
            print(f"{edge_type}: Found {len(tics)} structural tics")
            
            if len(tics) >= 3:
                gauge = self._calculate_gauge_from_tics(tics)
                results[edge_type] = {
                    'gauge': gauge,
                    'tics': len(tics),
                    'tic_positions': [(t.x, t.y) for t in tics],
                    'quality_score': self._calculate_quality_score(tics)
                }
                print(f"{edge_type}: Gauge = {gauge:.2f}")
        
        # Select best side from each axis (horizontal and vertical)
        best_results = self._select_best_sides(results)
        
        return best_results
    
    def _find_perforation_line_structural(self, region: np.ndarray, edge_type: str) -> List[Tuple[int, int]]:
        """Find the perforation line using structural features."""
        
        h, w = region.shape
        line_points = []
        
        if edge_type in ['top', 'bottom']:
            # Horizontal edge - scan vertically for structural features
            for x in range(0, w, 3):  # Every 3 pixels for efficiency
                col = region[:, x]
                
                # Look for the transition from background (0) to stamp (128) to perforation (0)
                # This creates the scalloped perforation line
                
                if edge_type == 'top':
                    # Find the deepest perforation point in the top region
                    perforation_points = np.where(col == 0)[0]  # Black pixels (perforations)
                    if len(perforation_points) > 0:
                        # Use the deepest (highest y) perforation point
                        deepest_y = perforation_points[-1] if len(perforation_points) > 0 else 0
                        line_points.append((x, deepest_y))
                    else:
                        # No perforation found, use stamp boundary
                        stamp_points = np.where(col == 128)[0]
                        if len(stamp_points) > 0:
                            line_points.append((x, stamp_points[0]))
                else:  # bottom
                    # Find the deepest perforation point in the bottom region
                    perforation_points = np.where(col == 0)[0]
                    if len(perforation_points) > 0:
                        # Use the highest (lowest y) perforation point for bottom edge
                        highest_y = perforation_points[0]
                        line_points.append((x, highest_y))
                    else:
                        # No perforation found, use stamp boundary
                        stamp_points = np.where(col == 128)[0]
                        if len(stamp_points) > 0:
                            line_points.append((x, stamp_points[-1]))
        else:
            # Vertical edge - scan horizontally
            for y in range(0, h, 3):  # Every 3 pixels
                row = region[y, :]
                
                if edge_type == 'left':
                    perforation_points = np.where(row == 0)[0]
                    if len(perforation_points) > 0:
                        deepest_x = perforation_points[-1]
                        line_points.append((deepest_x, y))
                    else:
                        stamp_points = np.where(row == 128)[0]
                        if len(stamp_points) > 0:
                            line_points.append((stamp_points[0], y))
                else:  # right
                    perforation_points = np.where(row == 0)[0]
                    if len(perforation_points) > 0:
                        highest_x = perforation_points[0]
                        line_points.append((highest_x, y))
                    else:
                        stamp_points = np.where(row == 128)[0]
                        if len(stamp_points) > 0:
                            line_points.append((stamp_points[-1], y))
        
        return line_points
    
    def _find_structural_tics(self, enhanced: np.ndarray, line_points: List[Tuple[int, int]], edge_type: str) -> List[PerforationTic]:
        """Find tics using structural analysis of the perforation line."""
        
        if len(line_points) < 10:
            return []
        
        # Sort line points
        if edge_type in ['top', 'bottom']:
            line_points.sort(key=lambda p: p[0])  # Sort by x
        else:
            line_points.sort(key=lambda p: p[1])  # Sort by y
        
        # Find structural indentations (where line deviates inward)
        tics = []
        
        # Use a sliding window to find local extrema
        window = 6  # Moderate smoothing
        min_tic_distance = max(16, int(self.dpi / 32))  # Moderate minimum distance
        
        for i in range(window, len(line_points) - window):
            current_point = line_points[i]
            
            # Get local neighborhood
            neighborhood = line_points[i-window:i+window+1]
            
            # Calculate if this is a local extremum (indentation)
            is_tic = False
            
            if edge_type == 'top':
                # For top edge, tics are local maxima in y (deeper into stamp)
                current_y = current_point[1]
                neighbor_ys = [p[1] for p in neighborhood]
                if current_y == max(neighbor_ys) and current_y > np.mean(neighbor_ys) + 2.5:
                    is_tic = True
            elif edge_type == 'bottom':
                # For bottom edge, tics are local minima in y (higher up into stamp)
                current_y = current_point[1]
                neighbor_ys = [p[1] for p in neighborhood]
                if current_y == min(neighbor_ys) and current_y < np.mean(neighbor_ys) - 2.5:
                    is_tic = True
            elif edge_type == 'left':
                # For left edge, tics are local maxima in x (deeper into stamp)
                current_x = current_point[0]
                neighbor_xs = [p[0] for p in neighborhood]
                if current_x == max(neighbor_xs) and current_x > np.mean(neighbor_xs) + 2.5:
                    is_tic = True
            else:  # right
                # For right edge, tics are local minima in x (deeper into stamp)
                current_x = current_point[0]
                neighbor_xs = [p[0] for p in neighborhood]
                if current_x == min(neighbor_xs) and current_x < np.mean(neighbor_xs) - 2.5:
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
                    # Calculate depth of indentation
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
    
    def _calculate_gauge_from_tics(self, tics: List[PerforationTic]) -> float:
        """Calculate perforation gauge from tic spacing."""
        if len(tics) < 2:
            return 0.0
        
        # Calculate distances between consecutive tics
        distances = []
        for i in range(len(tics) - 1):
            t1, t2 = tics[i], tics[i + 1]
            distance = math.sqrt((t2.x - t1.x)**2 + (t2.y - t1.y)**2)
            distances.append(distance)
        
        if not distances:
            return 0.0
            
        # Average spacing
        avg_spacing_pixels = np.mean(distances)
        
        # Convert to gauge  
        pixels_per_mm = self.dpi / 25.4  # 25.4 mm per inch
        spacing_mm = avg_spacing_pixels / pixels_per_mm
        gauge = 20.0 / spacing_mm if spacing_mm > 0 else 0.0
        
        print(f"   Gauge calc: {len(tics)} tics, {avg_spacing_pixels:.1f}px spacing = {spacing_mm:.3f}mm = gauge {gauge:.2f}")
        
        return gauge
        
    def _calculate_quality_score(self, tics: List[PerforationTic]) -> float:
        """Calculate quality score for a set of tics.
        
        Higher score means better quality tics for gauge measurement.
        Considers: number of tics, uniformity of spacing, and depth of tics.
        """
        if len(tics) < 3:
            return 0.0
            
        # 1. Number of tics (more is better)
        num_tics_score = min(1.0, len(tics) / 20.0)  # Max at 20 tics
        
        # 2. Spacing uniformity
        distances = []
        for i in range(len(tics) - 1):
            t1, t2 = tics[i], tics[i + 1]
            distance = math.sqrt((t2.x - t1.x)**2 + (t2.y - t1.y)**2)
            distances.append(distance)
            
        std_dev = np.std(distances)
        mean_distance = np.mean(distances)
        if mean_distance > 0:
            coefficient_of_variation = std_dev / mean_distance
            uniformity_score = max(0.0, 1.0 - coefficient_of_variation)
        else:
            uniformity_score = 0.0
            
        # 3. Tic depth (deeper is better)
        depths = [t.depth for t in tics]
        avg_depth = np.mean(depths) if depths else 0
        depth_score = min(1.0, avg_depth / 5.0)  # Normalize, max at depth of 5
        
        # Combine scores with weights
        quality_score = (0.4 * num_tics_score) + (0.4 * uniformity_score) + (0.2 * depth_score)
        return quality_score
        
    def _select_best_sides(self, results: Dict) -> Dict:
        """Select the best side from each axis based on quality score.
        
        Since opposite sides (top/bottom and left/right) should have the same perforation,
        we select the best quality side from each axis.
        """
        if not results:
            return {}
            
        best_results = {}
        
        # Horizontal axis (top/bottom)
        horizontal_sides = {side: data for side, data in results.items() 
                          if side in ['top', 'bottom']}
        if horizontal_sides:
            best_h_side = max(horizontal_sides.items(), 
                             key=lambda x: x[1]['quality_score'])  
            best_results[best_h_side[0]] = best_h_side[1]
            best_results['horizontal_gauge'] = best_h_side[1]['gauge']
            print(f"\nSelected {best_h_side[0]} as best horizontal side")
            
        # Vertical axis (left/right)
        vertical_sides = {side: data for side, data in results.items() 
                        if side in ['left', 'right']}
        if vertical_sides:
            best_v_side = max(vertical_sides.items(), 
                           key=lambda x: x[1]['quality_score'])
            best_results[best_v_side[0]] = best_v_side[1]
            best_results['vertical_gauge'] = best_v_side[1]['gauge']
            print(f"Selected {best_v_side[0]} as best vertical side")
            
        return best_results
        
    def _find_stamp_boundaries(self, enhanced: np.ndarray) -> Dict:
        """Find the actual boundaries of the stamp in the monotone image.
        
        Returns boundary coordinates for each edge.
        """
        h, w = enhanced.shape
        boundaries = {}
        
        # Find stamp pixels (gray = 128)
        stamp_mask = (enhanced == 128)
        
        if not np.any(stamp_mask):
            return {}  # No stamp found
            
        # Find the bounding box of the stamp
        stamp_coords = np.where(stamp_mask)
        if len(stamp_coords[0]) == 0:
            return {}
            
        min_y, max_y = np.min(stamp_coords[0]), np.max(stamp_coords[0])
        min_x, max_x = np.min(stamp_coords[1]), np.max(stamp_coords[1])
        
        print(f"   Stamp bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        
        # For each edge, find the actual boundary line
        search_depth = min(20, min(h, w) // 20)  # How deep to search from boundary
        
        boundaries = {
            'top': {'min_y': min_y, 'max_y': min(min_y + search_depth, h-1), 'x_range': (min_x, max_x)},
            'bottom': {'min_y': max(max_y - search_depth, 0), 'max_y': max_y, 'x_range': (min_x, max_x)},
            'left': {'min_x': min_x, 'max_x': min(min_x + search_depth, w-1), 'y_range': (min_y, max_y)},
            'right': {'min_x': max(max_x - search_depth, 0), 'max_x': max_x, 'y_range': (min_y, max_y)}
        }
        
        return boundaries
        
    def _create_boundary_regions(self, enhanced: np.ndarray, boundaries: Dict) -> Dict:
        """Create edge regions based on detected stamp boundaries."""
        h, w = enhanced.shape
        regions = {}
        
        for edge_type, bounds in boundaries.items():
            if edge_type in ['top', 'bottom']:
                min_y, max_y = bounds['min_y'], bounds['max_y']
                min_x, max_x = bounds['x_range']
                
                region = enhanced[min_y:max_y+1, min_x:max_x+1]
                
                # Create the perforation line by finding the stamp boundary in this region
                boundary_line = []
                for x in range(0, region.shape[1], 2):  # Sample every 2 pixels
                    col = region[:, x]
                    
                    if edge_type == 'top':
                        # Find first stamp pixel from top
                        stamp_pixels = np.where(col == 128)[0]
                        if len(stamp_pixels) > 0:
                            boundary_y = stamp_pixels[0]
                            boundary_line.append((x, boundary_y))
                    else:  # bottom
                        # Find last stamp pixel from bottom
                        stamp_pixels = np.where(col == 128)[0]
                        if len(stamp_pixels) > 0:
                            boundary_y = stamp_pixels[-1]
                            boundary_line.append((x, boundary_y))
                            
                regions[edge_type] = {
                    'region': region,
                    'boundary_line': boundary_line,
                    'offset': (min_x, min_y)
                }
                
            else:  # left, right
                min_x, max_x = bounds['min_x'], bounds['max_x']
                min_y, max_y = bounds['y_range']
                
                region = enhanced[min_y:max_y+1, min_x:max_x+1]
                
                boundary_line = []
                for y in range(0, region.shape[0], 2):  # Sample every 2 pixels
                    row = region[y, :]
                    
                    if edge_type == 'left':
                        # Find first stamp pixel from left
                        stamp_pixels = np.where(row == 128)[0]
                        if len(stamp_pixels) > 0:
                            boundary_x = stamp_pixels[0]
                            boundary_line.append((boundary_x, y))
                    else:  # right
                        # Find last stamp pixel from right
                        stamp_pixels = np.where(row == 128)[0]
                        if len(stamp_pixels) > 0:
                            boundary_x = stamp_pixels[-1]
                            boundary_line.append((boundary_x, y))
                            
                regions[edge_type] = {
                    'region': region,
                    'boundary_line': boundary_line,
                    'offset': (min_x, min_y)
                }
                
        return regions
        
    def _find_sensitive_tics(self, line_points: List[Tuple[int, int]], edge_type: str) -> List[PerforationTic]:
        """More sensitive tic detection for close-cropped images.
        
        Uses smaller window and lower thresholds to catch tics that might be
        partially cut off by cropping.
        """
        if len(line_points) < 10:
            return []
            
        # Sort line points
        if edge_type in ['top', 'bottom']:
            line_points.sort(key=lambda p: p[0])  # Sort by x
        else:
            line_points.sort(key=lambda p: p[1])  # Sort by y
            
        tics = []
        window = 4  # Smaller window for sensitivity
        min_tic_distance = max(12, int(self.dpi / 40))  # Closer spacing allowed
        
        for i in range(window, len(line_points) - window):
            current_point = line_points[i]
            neighborhood = line_points[i-window:i+window+1]
            
            is_tic = False
            
            if edge_type == 'top':
                current_y = current_point[1]
                neighbor_ys = [p[1] for p in neighborhood]
                if current_y == max(neighbor_ys) and current_y > np.mean(neighbor_ys) + 1.5:  # Lower threshold
                    is_tic = True
            elif edge_type == 'bottom':
                current_y = current_point[1]
                neighbor_ys = [p[1] for p in neighborhood]
                if current_y == min(neighbor_ys) and current_y < np.mean(neighbor_ys) - 1.5:
                    is_tic = True
            elif edge_type == 'left':
                current_x = current_point[0]
                neighbor_xs = [p[0] for p in neighborhood]
                if current_x == max(neighbor_xs) and current_x > np.mean(neighbor_xs) + 1.5:
                    is_tic = True
            else:  # right
                current_x = current_point[0]
                neighbor_xs = [p[0] for p in neighborhood]
                if current_x == min(neighbor_xs) and current_x < np.mean(neighbor_xs) - 1.5:
                    is_tic = True
                    
            if is_tic:
                # Check minimum distance
                too_close = False
                for existing_tic in tics:
                    dist = math.sqrt((current_point[0] - existing_tic.x)**2 + (current_point[1] - existing_tic.y)**2)
                    if dist < min_tic_distance:
                        too_close = True
                        break
                        
                if not too_close:
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
        
    def _calculate_adaptive_edge_width(self, enhanced: np.ndarray, base_width: int) -> int:
        """Calculate adaptive edge width based on image characteristics.
        
        For close-cropped images or pairs/blocks, we need wider edge regions
        to capture the complete perforation pattern.
        """
        h, w = enhanced.shape
        
        # Check for potential close cropping by examining edge content
        edge_content_scores = []
        
        # Sample the four edges to see how much perforation content they have
        test_width = min(base_width, 20)
        edges_to_test = [
            enhanced[:test_width, :],      # top
            enhanced[-test_width:, :],     # bottom  
            enhanced[:, :test_width],      # left
            enhanced[:, -test_width:]      # right
        ]
        
        for edge in edges_to_test:
            if edge.size == 0:
                continue
                
            # Count perforation pixels (black = 0) vs stamp pixels (gray = 128)
            perforation_pixels = np.sum(edge == 0)
            stamp_pixels = np.sum(edge == 128)
            edge_pixels = np.sum(edge == 255)  # Enhanced edges
            
            total_pixels = edge.size
            perforation_ratio = perforation_pixels / total_pixels if total_pixels > 0 else 0
            
            edge_content_scores.append({
                'perforation_ratio': perforation_ratio,
                'edge_pixels': edge_pixels,
                'has_structure': perforation_ratio > 0.05 or edge_pixels > total_pixels * 0.1
            })
        
        # If edges have high perforation content, likely close-cropped or pair/block
        high_content_edges = sum(1 for score in edge_content_scores if score['has_structure'])
        avg_perforation_ratio = np.mean([s['perforation_ratio'] for s in edge_content_scores])
        
        # Adaptive width calculation - be more conservative
        if high_content_edges >= 3 and avg_perforation_ratio > 0.20:
            # Very likely close-cropped or pair/block - use moderately wider edge region
            adaptive_width = min(int(base_width * 1.2), min(h, w) // 10)
            print(f"   Detected close-crop/pair pattern: using wider edge width {adaptive_width}")
        elif avg_perforation_ratio > 0.25:
            # High perforation density - slightly wider
            adaptive_width = min(int(base_width * 1.1), min(h, w) // 12)
            print(f"   High perforation density: using slightly wider edge width {adaptive_width}")
        else:
            # Normal stamp with good margins
            adaptive_width = base_edge_width
            print(f"   Using standard edge width {adaptive_width}")
            
        return adaptive_width


def test_monotone_method(stamp_path=None):
    """Test the monotone stamp approach."""
    import os
    
    # Default stamp if none provided
    if stamp_path is None:
        stamp_path = os.path.expanduser("~/Desktop/2025 Color Analysis/138 - 10c red/138-S10.tif")
    
    # Expand user path if needed
    stamp_path = os.path.expanduser(stamp_path)
    
    print(f"Testing stamp: {stamp_path}")
    
    if os.path.exists(stamp_path):
        image = cv2.imread(stamp_path)
        if image is not None:
            detector = MonotonePerforationDetector(dpi=800)
            results = detector.detect_perforations(image)
            
            print("\\n=== FINAL RESULTS ===")
            if results:
                # Print individual sides that were selected as best
                for edge_type, data in results.items():
                    if edge_type in ['top', 'bottom', 'left', 'right']:
                        print(f"  {edge_type.upper()}: {data['gauge']:.2f} ({data['tics']} tics)")
                
                # Get the best horizontal and vertical gauges
                h_gauge = results.get('horizontal_gauge', 0)
                v_gauge = results.get('vertical_gauge', 0)
                
                if h_gauge > 0 and v_gauge > 0:
                    # Calculate compound gauge using best sides only
                    compound_gauge = (h_gauge + v_gauge) / 2
                    print(f"\\nCompound gauge (best sides): {compound_gauge:.2f}")
                    print(f"Catalog notation: {h_gauge:.1f} × {v_gauge:.1f}")
            else:
                print("No perforations detected")
        else:
            print("Could not load image")
    else:
        print(f"Image not found: {stamp_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use command line argument for stamp path
        test_monotone_method(sys.argv[1])
    else:
        # Use default stamp
        test_monotone_method()
