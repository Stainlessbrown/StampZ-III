#!/usr/bin/env python3

"""Test template selector dialog in isolation."""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot3d.template_selector import TemplateSelector

logging.basicConfig(level=logging.DEBUG)

def test_template_selector():
    """Test the template selector dialog."""
    print("Testing template selector...")
    
    try:
        selector = TemplateSelector(parent=None)
        
        print("Template selector result:")
        print(f"  Database mode: {selector.database_mode}")
        print(f"  Selected database: {selector.selected_database}")
        print(f"  File path: {selector.file_path}")
        
    except Exception as e:
        print(f"Error testing template selector: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_template_selector()