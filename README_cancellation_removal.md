# Advanced Philatelic Cancellation Removal

This tool provides sophisticated methods for separating and removing black cancellations (obliterations) from stamp images using advanced computer vision techniques.

## Features

### Multiple Detection Methods
- **Enhanced Color Space Analysis**: LAB, HSV, and custom channel combinations
- **Morphological Pattern Detection**: Specifically targets postmark lines and text patterns
- **Frequency Domain Filtering**: Uses FFT to separate cancellation frequencies from stamp design
- **Adaptive Thresholding**: Multiple threshold techniques for robust detection
- **Intelligent Composite Masking**: Combines all methods for optimal results

### Advanced Reconstruction
- **Fast Marching Inpainting**: Efficient reconstruction of removed areas
- **Navier-Stokes Inpainting**: Physics-based reconstruction for natural results
- **Content-Aware Processing**: Preserves underlying stamp design while removing cancellations

## Installation

Install the required Python packages:

```bash
pip install opencv-python numpy matplotlib scipy scikit-image pillow
```

## Usage

### Interactive Mode
For testing and experimentation:

```bash
python test_cancellation_removal.py
```

This provides an interactive interface where you can:
- Process individual images
- View color channel analysis
- Examine detection masks
- Compare results side-by-side

### Command Line Mode
For batch processing:

```bash
# Process a single image
python cancellation_removal.py path/to/stamp.jpg

# Process all images in a directory
python cancellation_removal.py path/to/images/ --batch

# Specify custom output directory
python cancellation_removal.py path/to/stamp.jpg --output /path/to/results/
```

### Programmatic Usage

```python
from cancellation_removal import CancellationRemover

# Initialize with image path
remover = CancellationRemover("path/to/stamp.jpg")

# Process the image
results = remover.process_image()

# Access individual components
color_channels = remover.enhanced_color_separation()
cancellation_mask = remover.create_composite_mask()
inpainted_results = remover.intelligent_inpainting(cancellation_mask)
```

## Output Files

The tool generates several output files for each processed image:

- `*_original.png` - Original image for reference
- `*_mask.png` - Detected cancellation mask (white = cancellation areas)
- `*_fast_marching.png` - Fast marching inpainting result
- `*_navier_stokes.png` - Navier-Stokes inpainting result
- `*_lab_a.png` - LAB A-channel (often shows good separation)
- `*_red_minus_blue.png` - Custom color channel combination
- `*_comparison.png` - Side-by-side comparison plot

## Technical Approach

### 1. Color Space Analysis
The tool analyzes multiple color spaces to find the best separation between cancellations and stamp designs:

- **LAB Color Space**: Often provides the best separation for different ink types
- **HSV Color Space**: Useful for hue-based separation
- **Custom Combinations**: Red-minus-blue and other combinations that enhance contrast

### 2. Pattern Detection
Morphological operations detect specific cancellation patterns:

- **Line Detection**: Horizontal and vertical postmark lines
- **Text Pattern Detection**: Postmark text and date stamps
- **Structural Analysis**: Different kernel sizes for various cancellation types

### 3. Frequency Domain Processing
FFT-based filtering separates spatial frequencies:

- **High-pass Filtering**: Enhances sharp cancellation edges
- **Band-pass Filtering**: Targets specific cancellation frequencies
- **Magnitude Spectrum Analysis**: Identifies frequency characteristics

### 4. Intelligent Masking
Combines all detection methods into a robust composite mask:

- Weights different detection methods based on reliability
- Applies morphological cleanup operations
- Adapts to different image characteristics

### 5. Advanced Inpainting
Reconstructs removed areas using sophisticated algorithms:

- **Fast Marching Method**: Efficient for large areas
- **Navier-Stokes Method**: Physics-based for natural textures
- **Context-aware Processing**: Preserves stamp design patterns

## Best Results Tips

1. **High Resolution Images**: Works best with images ≥300 DPI
2. **Good Lighting**: Even lighting reduces false detections
3. **Color Stamps**: More effective on colored stamps than monochrome
4. **Clean Scans**: Remove dust and artifacts before processing
5. **Parameter Tuning**: Adjust thresholds for specific stamp types

## Limitations

- Works best on black/dark cancellations over colored stamps
- May struggle with very heavy cancellations that completely obscure design
- Light or colored cancellations may require parameter adjustment
- Results depend on image quality and resolution

## Integration with StampZ

This tool is designed to work alongside your existing StampZ workflow:

```python
# Can be integrated into StampZ processing pipeline
def process_stamp_with_cancellation_removal(image_path):
    # Remove cancellations
    remover = CancellationRemover(image_path)
    results = remover.process_image()
    
    # Use cleaned image for further StampZ processing
    cleaned_image = results['fast_marching']  # or 'navier_stokes'
    
    # Continue with StampZ analysis...
```

## Advanced Usage

For researchers and advanced users, individual components can be accessed:

```python
# Analyze specific color channels
color_results = remover.enhanced_color_separation()
best_channel = color_results['lab_a']  # Often best for cancellation separation

# Custom frequency filtering
freq_results = remover.frequency_domain_separation()
custom_filtered = freq_results['band_pass']

# Fine-tune detection parameters
custom_mask = remover.adaptive_thresholding_mask()
adjusted_mask = custom_mask['dark_regions']  # Adjust threshold as needed
```

## Future Enhancements

Planned improvements include:
- Machine learning-based cancellation detection
- Automatic parameter optimization
- Support for colored cancellations
- Integration with OCR for postmark reading
- Batch processing with progress tracking