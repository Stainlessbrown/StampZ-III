#!/usr/bin/env python3
"""
Philatelic Measurement Tool - Architectural Style Dimension Lines
Demonstrates precise measurement with extension lines and dimension annotations
Perfect for detecting fraud and analyzing plate differences.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np

def draw_dimension_line(ax, start, end, offset=20, text="", color='red', precision=2):
    """
    Draw architectural-style dimension line with extension lines and measurement text.
    
    Args:
        ax: Matplotlib axis
        start: (x, y) start point
        end: (x, y) end point  
        offset: Distance of dimension line from measured edge
        text: Custom text (if empty, shows calculated distance)
        color: Color of dimension lines
        precision: Decimal places for measurement
    """
    x1, y1 = start
    x2, y2 = end
    
    # Calculate dimension line position
    if abs(x2 - x1) > abs(y2 - y1):  # Horizontal measurement
        dim_y = max(y1, y2) + offset
        ext_start1 = (x1, y1)
        ext_end1 = (x1, dim_y + 5)
        ext_start2 = (x2, y2)
        ext_end2 = (x2, dim_y + 5)
        dim_start = (x1, dim_y)
        dim_end = (x2, dim_y)
        text_pos = ((x1 + x2) / 2, dim_y + 8)
        text_rotation = 0
    else:  # Vertical measurement
        dim_x = max(x1, x2) + offset
        ext_start1 = (x1, y1)
        ext_end1 = (dim_x + 5, y1)
        ext_start2 = (x2, y2)
        ext_end2 = (dim_x + 5, y2)
        dim_start = (dim_x, y1)
        dim_end = (dim_x, y2)
        text_pos = (dim_x + 12, (y1 + y2) / 2)
        text_rotation = 90
    
    # Draw extension lines (|)
    ax.plot([ext_start1[0], ext_end1[0]], [ext_start1[1], ext_end1[1]], 
            color=color, linewidth=1, linestyle='-')
    ax.plot([ext_start2[0], ext_end2[0]], [ext_start2[1], ext_end2[1]], 
            color=color, linewidth=1, linestyle='-')
    
    # Draw dimension line with arrows (<--->)
    arrow = FancyArrowPatch(dim_start, dim_end,
                           arrowstyle='<->', mutation_scale=15,
                           color=color, linewidth=1.5)
    ax.add_patch(arrow)
    
    # Calculate and display measurement
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if not text:
        text = f"{distance:.{precision}f}mm"
    
    ax.text(text_pos[0], text_pos[1], text, 
            ha='center', va='center', color=color, fontsize=9, 
            rotation=text_rotation, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

def demonstrate_philatelic_measurements():
    """Demonstrate precise measurements on a simulated stamp with overprint."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Simulate a stamp outline (let's say 22mm x 26mm)
    stamp_rect = patches.Rectangle((50, 50), 220, 260, 
                                  linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3)
    ax.add_patch(stamp_rect)
    
    # Simulate stamp design elements
    # Central vignette
    vignette = patches.Rectangle((80, 120), 160, 120, 
                               linewidth=1, edgecolor='navy', facecolor='lightcyan', alpha=0.5)
    ax.add_patch(vignette)
    
    # Overprint (could be suspicious)
    overprint = patches.Rectangle((120, 180), 80, 40, 
                                linewidth=1, edgecolor='red', facecolor='pink', alpha=0.7)
    ax.add_patch(overprint)
    
    # Text elements
    ax.text(160, 80, "STAMP DESIGN", ha='center', fontsize=8, weight='bold')
    ax.text(160, 200, "OVERPRINT", ha='center', fontsize=10, weight='bold', color='red')
    
    # === PRECISE MEASUREMENTS ===
    
    # 1. Overall stamp dimensions
    draw_dimension_line(ax, (50, 50), (270, 50), offset=-30, text="22.0mm", color='red')
    draw_dimension_line(ax, (50, 50), (50, 310), offset=-30, text="26.0mm", color='red')
    
    # 2. Overprint position measurements (critical for fraud detection)
    draw_dimension_line(ax, (50, 180), (120, 180), offset=15, text="7.0mm", color='orange')
    draw_dimension_line(ax, (120, 50), (120, 180), offset=15, text="13.0mm", color='orange')
    
    # 3. Overprint dimensions
    draw_dimension_line(ax, (120, 180), (200, 180), offset=-20, text="8.0mm", color='purple')
    draw_dimension_line(ax, (200, 180), (200, 220), offset=10, text="4.0mm", color='purple')
    
    # 4. Vignette measurements (for plate studies)
    draw_dimension_line(ax, (80, 120), (240, 120), offset=-40, text="16.0mm", color='green')
    
    # Add title and annotations
    ax.set_title("Philatelic Precision Measurement Tool\nArchitectural-Style Dimension Lines", 
                fontsize=14, weight='bold', pad=20)
    
    # Add measurement legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='Stamp Dimensions'),
        plt.Line2D([0], [0], color='orange', lw=2, label='Overprint Position'),
        plt.Line2D([0], [0], color='purple', lw=2, label='Overprint Size'),
        plt.Line2D([0], [0], color='green', lw=2, label='Design Elements')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add notes
    ax.text(320, 280, "FRAUD DETECTION NOTES:", fontsize=10, weight='bold')
    ax.text(320, 260, "• Overprint 0.5mm off-center", fontsize=9, color='red')
    ax.text(320, 240, "• Compare with known authentic", fontsize=9)
    ax.text(320, 220, "• Plate variations: ±0.1mm typical", fontsize=9)
    ax.text(320, 200, "• Extension lines ensure precise", fontsize=9)
    ax.text(320, 185, "  placement of measurement", fontsize=9)
    
    # Set axis properties
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 350)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Position (0.1mm precision)", fontsize=10)
    ax.set_ylabel("Position (0.1mm precision)", fontsize=10)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🔍 Philatelic Precision Measurement Demo")
    print("=" * 45)
    
    # Create the demonstration
    fig = demonstrate_philatelic_measurements()
    
    # Save the demo
    output_path = "/tmp/philatelic_measurements_demo.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Demo saved to: {output_path}")
    
    # Show advantages of this approach
    print("\n🎯 ADVANTAGES OF ARCHITECTURAL-STYLE MEASUREMENTS:")
    print("   ✅ Extension lines clearly show measurement boundaries")
    print("   ✅ No ambiguity about measurement endpoints")
    print("   ✅ Professional technical drawing appearance")
    print("   ✅ Multiple measurements can be shown simultaneously")
    print("   ✅ Perfect for fraud detection (precise positioning)")
    print("   ✅ Ideal for plate studies (small dimensional differences)")
    print("   ✅ Clear documentation for certification/authentication")
    
    print("\n📏 TYPICAL PHILATELIC APPLICATIONS:")
    print("   • Overprint positioning analysis")
    print("   • Perforation measurement and spacing")
    print("   • Design element positioning")
    print("   • Plate variety identification")
    print("   • Authentication and fraud detection")
    print("   • Centering analysis")
    
    plt.show()