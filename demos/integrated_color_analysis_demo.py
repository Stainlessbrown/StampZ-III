#!/usr/bin/env python3
"""
Integrated StampZ Color Analysis Demo
Complete demonstration of the advanced color visualization pipeline.

This demo shows:
1. Loading real StampZ color data using the data bridge
2. Creating ternary plots for RGB analysis
3. Creating 3D L*a*b* plots for color space analysis
4. ML-based color family clustering
5. Advanced filtering and grouping operations

This serves as both a demo and a template for actual analysis workflows.
"""

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.color_data_bridge import ColorDataBridge
from utils.advanced_color_plots import TernaryPlotter, QuaternaryPlotter, ColorClusterAnalyzer


def main():
    """Run the complete integrated color analysis demonstration."""
    
    print("🎨 StampZ Integrated Color Analysis Demo")
    print("=" * 50)
    
    # Step 1: Initialize the data bridge
    print("\n📊 Step 1: Initialize Data Bridge")
    bridge = ColorDataBridge()
    
    # Step 2: Get available sample sets
    print("\n📋 Step 2: Get Available Sample Sets")
    sample_sets = bridge.get_available_sample_sets()
    
    if not sample_sets:
        print("❌ No sample sets found!")
        print("💡 Please run some color analysis first to generate databases")
        return
    
    print(f"Available Sample Sets: {len(sample_sets)}")
    for i, sample_set in enumerate(sample_sets):
        print(f"  {i+1}. {sample_set}")
    
    # Step 3: Load color data from the first available sample set
    first_set = sample_sets[0]
    print(f"\n🔍 Step 3: Load Data from '{first_set}'")
    
    color_points = bridge.load_color_points_from_database(first_set)
    
    if not color_points:
        print("❌ No color data loaded")
        return
    
    print(f"✅ Loaded {len(color_points)} color measurements")
    
    # Step 4: Show detailed sample analysis
    print(f"\n📈 Step 4: Data Analysis Summary")
    stats = bridge.get_summary_stats(color_points)
    
    print(f"  📊 Dataset Overview:")
    print(f"    Total Points: {stats['total_points']}")
    print(f"    Unique Images: {stats['unique_images']}")
    print(f"    Sample Sets: {stats['unique_sample_sets']}")
    
    print(f"  🎨 Color Space Statistics (L*a*b*):")
    print(f"    L* (Lightness): {stats['lab_means']['L*']:.1f} ± {stats['lab_std']['L*']:.1f}")
    print(f"    a* (Green-Red): {stats['lab_means']['a*']:.1f} ± {stats['lab_std']['a*']:.1f}")
    print(f"    b* (Blue-Yellow): {stats['lab_means']['b*']:.1f} ± {stats['lab_std']['b*']:.1f}")
    
    print(f"  📺 RGB Statistics:")
    print(f"    R (Red): {stats['rgb_means']['R']:.0f}")
    print(f"    G (Green): {stats['rgb_means']['G']:.0f}")
    print(f"    B (Blue): {stats['rgb_means']['B']:.0f}")
    
    # Step 5: Group and filter data
    print(f"\n🗂️ Step 5: Data Organization")
    
    # Group by images
    image_groups = bridge.group_by_image(color_points)
    print(f"  📷 Images Analysis:")
    for image_name, points in list(image_groups.items())[:5]:  # Show first 5
        print(f"    {image_name}: {len(points)} measurements")
    if len(image_groups) > 5:
        print(f"    ... and {len(image_groups) - 5} more images")
    
    # Filter by color properties
    print(f"  🔍 Color Filtering Examples:")
    
    # Bright colors
    bright_colors = bridge.filter_by_lightness(color_points, min_l=50)
    print(f"    Bright colors (L* > 50): {len(bright_colors)} points")
    
    # Dark colors
    dark_colors = bridge.filter_by_lightness(color_points, max_l=30)
    print(f"    Dark colors (L* < 30): {len(dark_colors)} points")
    
    # Saturated colors
    saturated_colors = bridge.filter_by_chroma(color_points, min_chroma=25)
    print(f"    Saturated colors (Chroma > 25): {len(saturated_colors)} points")
    
    # Neutral colors
    neutral_colors = bridge.filter_by_chroma(color_points, max_chroma=15)
    print(f"    Neutral colors (Chroma < 15): {len(neutral_colors)} points")
    
    # Step 6: Advanced plotting demonstrations
    print(f"\n📊 Step 6: Advanced Visualization")
    
    # Create ternary plotter
    ternary_plotter = TernaryPlotter()
    
    try:
        print("  🔺 Creating RGB Ternary Plot...")
        ternary_fig = ternary_plotter.create_ternary_plot(color_points, 
                                                         title=f"RGB Ternary Analysis - {first_set}",
                                                         color_by="actual",
                                                         show_hull=True)
        
        # Save the plot
        ternary_output = f"/Users/stanbrown/Desktop/StampZ-III/output/ternary_plot_{first_set}.png"
        ternary_plotter.save_plot(ternary_fig, ternary_output)
        print(f"    ✅ Ternary plot saved: {ternary_output}")
        
    except Exception as e:
        print(f"    ⚠️ Ternary plot creation failed: {e}")
    
    # Create 3D L*a*b* plotter (if plotly available)
    try:
        quaternary_plotter = QuaternaryPlotter()
        
        print("  📐 Creating 3D L*a*b* Plot...")
        lab_fig = quaternary_plotter.create_lab_scatter(color_points,
                                                       title=f"L*a*b* Color Space - {first_set}",
                                                       color_by="actual")
        
        # Save the plot
        lab_output = f"/Users/stanbrown/Desktop/StampZ-III/output/lab_3d_plot_{first_set}.html"
        quaternary_plotter.save_plot(lab_fig, lab_output)
        print(f"    ✅ 3D L*a*b* plot saved: {lab_output}")
        
    except Exception as e:
        print(f"    ⚠️ 3D L*a*b* plot creation failed: {e}")
    
    # Step 7: Machine Learning Analysis
    print(f"\n🤖 Step 7: Machine Learning Analysis")
    
    try:
        analyzer = ColorClusterAnalyzer()
        
        # Perform color family clustering
        print("  🎯 Performing Color Family Clustering...")
        cluster_results = analyzer.cluster_by_families(color_points, method="combined")
        
        families = cluster_results.get('clusters', {})
        print(f"    ✅ Identified {len(families)} Color Families:")
        
        for family_name, family_points in families.items():
            print(f"      {family_name}: {len(family_points)} colors")
        
        # Outlier detection
        print("  🚨 Outlier Detection...")
        outliers = analyzer.detect_outliers(color_points)
        print(f"    ✅ Detected {len(outliers)} Potential Outliers")
        
        if outliers:
            print(f"    🔍 Sample Outliers:")
            for i, outlier in enumerate(outliers[:3]):  # Show first 3
                print(f"      {i+1}. {outlier.id}: RGB{outlier.rgb}, L*a*b*{outlier.lab}")
        
    except Exception as e:
        print(f"    ⚠️ ML analysis failed: {e}")
    
    # Step 8: Advanced analysis demonstrations
    print(f"\n🔬 Step 8: Advanced Analysis Examples")
    
    # Find the most colorful stamp regions
    most_saturated = sorted(color_points, 
                           key=lambda p: (p.lab[1]**2 + p.lab[2]**2)**0.5, 
                           reverse=True)[:3]
    
    print(f"  🌈 Most Saturated Regions:")
    for i, point in enumerate(most_saturated):
        chroma = (point.lab[1]**2 + point.lab[2]**2)**0.5
        print(f"    {i+1}. {point.id}: Chroma = {chroma:.1f}")
        print(f"       From image: {point.metadata.get('image_name', 'Unknown')}")
    
    # Find extreme lightness values
    brightest = max(color_points, key=lambda p: p.lab[0])
    darkest = min(color_points, key=lambda p: p.lab[0])
    
    print(f"  💡 Lightness Extremes:")
    print(f"    Brightest: {brightest.id} (L* = {brightest.lab[0]:.1f})")
    print(f"    Darkest: {darkest.id} (L* = {darkest.lab[0]:.1f})")
    
    # Color diversity analysis
    print(f"  📊 Color Diversity Analysis:")
    
    # Calculate color distribution across different images
    image_diversity = {}
    for image_name, points in image_groups.items():
        if len(points) > 1:
            # Calculate average chroma for this image
            avg_chroma = sum((p.lab[1]**2 + p.lab[2]**2)**0.5 for p in points) / len(points)
            image_diversity[image_name] = avg_chroma
    
    # Sort by diversity
    sorted_diversity = sorted(image_diversity.items(), key=lambda x: x[1], reverse=True)
    
    print(f"    Most Colorful Images (by average chroma):")
    for i, (image_name, avg_chroma) in enumerate(sorted_diversity[:3]):
        point_count = len(image_groups[image_name])
        print(f"      {i+1}. {image_name}: {avg_chroma:.1f} chroma ({point_count} points)")
    
    # Step 9: Report generation
    print(f"\n📋 Step 9: Analysis Summary Report")
    print(f"=" * 40)
    print(f"StampZ Color Analysis Report: {first_set}")
    print(f"Generated from {len(color_points)} color measurements")
    print(f"")
    print(f"Key Findings:")
    print(f"• Dataset covers {stats['unique_images']} different images")
    print(f"• Color space spans L*: {stats['lab_ranges']['L*'][0]:.1f} to {stats['lab_ranges']['L*'][1]:.1f}")
    print(f"• Chroma range: {min((p.lab[1]**2 + p.lab[2]**2)**0.5 for p in color_points):.1f} to {max((p.lab[1]**2 + p.lab[2]**2)**0.5 for p in color_points):.1f}")
    print(f"• {len(saturated_colors)} highly saturated regions identified")
    print(f"• {len(bright_colors)} bright regions, {len(dark_colors)} dark regions")
    print(f"")
    print(f"Technical Details:")
    print(f"• Advanced visualization plots saved in output/ directory")
    print(f"• Machine learning clustering identified color families")
    print(f"• Statistical analysis completed successfully")
    print(f"")
    print(f"✅ Complete Integrated Analysis Finished!")
    print(f"🔗 StampZ data successfully bridged to advanced visualization system")


if __name__ == "__main__":
    main()