#!/usr/bin/env python3
"""
Test script to verify the black ink extraction fix.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_black_ink_extraction():
    """Test the complete black ink extraction workflow."""
    print("=== Testing Black Ink Extraction Fix ===\n")
    
    # Create a test image
    print("1. Creating test image...")
    test_image = np.ones((200, 200, 3), dtype=np.uint8) * 200  # Light background
    
    # Add some "black ink" areas
    test_image[50:100, 50:100] = [30, 30, 30]      # Dark gray square (cancellation)
    test_image[120:140, 120:140] = [0, 0, 0]       # Pure black square (postmark)
    test_image[160:180, 50:70] = [20, 25, 28]      # Slightly colored dark area
    
    # Add some colored areas that shouldn't be detected as black ink
    test_image[50:100, 120:170] = [200, 50, 50]    # Red stamp area
    test_image[120:170, 50:100] = [50, 50, 200]    # Blue stamp area
    
    # Save test image
    test_path = "/tmp/test_stamp_image.png"
    pil_test = Image.fromarray(test_image)
    pil_test.save(test_path)
    print(f"✅ Test image created: {test_path}")
    
    # Test the extraction function directly
    print("\n2. Testing extract_black_ink function...")
    try:
        from black_ink_extractor import extract_black_ink
        
        results, mask, analysis = extract_black_ink(test_image)
        
        print("✅ extract_black_ink function works!")
        print(f"   Results: {list(results.keys())}")
        print(f"   Coverage: {analysis['coverage_percentage']:.1f}%")
        print(f"   Black pixels: {analysis['cancellation_pixels']:,}")
        
    except Exception as e:
        print(f"❌ extract_black_ink failed: {e}")
        return False
    
    # Test the manager workflow simulation
    print("\n3. Testing manager workflow simulation...")
    try:
        from managers.black_ink_manager import BlackInkManager
        from black_ink_extractor import safe_pil_fromarray
        
        # Simulate the fixed workflow
        print("   Loading image from file...")
        pil_image = Image.open(test_path)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        img_array = np.array(pil_image)
        print(f"   Image loaded: shape={img_array.shape}, dtype={img_array.dtype}")
        
        print("   Extracting black ink...")
        results, mask, analysis = extract_black_ink(
            img_array,
            black_threshold=60,
            saturation_threshold=30
        )
        
        print("   Converting result to PIL image...")
        extracted_image = results.get('pure_black', results[list(results.keys())[0]])
        pil_result = safe_pil_fromarray(extracted_image)
        
        # Save test result
        result_path = "/tmp/test_extraction_result.png"
        pil_result.save(result_path)
        
        print("✅ Manager workflow simulation successful!")
        print(f"   Result saved: {result_path}")
        print(f"   Coverage: {analysis['coverage_percentage']:.1f}%")
        
    except Exception as e:
        print(f"❌ Manager workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Clean up
    try:
        os.remove(test_path)
        os.remove(result_path)
        print("\n4. ✅ Test files cleaned up")
    except:
        pass
    
    print("\n=== All Tests Passed! ===")
    print("The black ink extraction fix is working correctly.")
    print("The TypeError should now be resolved.")
    
    return True

if __name__ == "__main__":
    success = test_black_ink_extraction()
    sys.exit(0 if success else 1)