#!/usr/bin/env python3
"""
macOS UI Fixes for StampZ-III

Addresses common macOS UI issues:
- Button text visibility in both light and dark modes
- Proper theme handling for dialogs and windows
- Color consistency across different macOS versions
"""
import tkinter as tk
from tkinter import ttk
import sys


def configure_macos_button_styles():
    """Configure button styles for better visibility on macOS."""
    try:
        style = ttk.Style()
        
        # Get current theme name
        current_theme = style.theme_use()
        
        # Configure button styles to ensure text is always visible
        style.configure('TButton', 
                       focuscolor='systemSelectedTextBackgroundColor')
        
        # Configure specific button styles for better contrast
        style.configure('Action.TButton',
                       foreground='systemControlTextColor',
                       focuscolor='systemSelectedTextBackgroundColor')
        
        style.configure('Warning.TButton',
                       foreground='systemRedColor',
                       focuscolor='systemSelectedTextBackgroundColor')
        
        style.configure('Success.TButton', 
                       foreground='systemGreenColor',
                       focuscolor='systemSelectedTextBackgroundColor')
        
        print(f"✅ Configured macOS button styles for theme: {current_theme}")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not configure macOS button styles: {e}")
        return False


def fix_tk_button_colors(widget, recursive=True):
    """
    Fix color issues with old-style tk.Button widgets.
    
    Args:
        widget: The widget to scan and fix
        recursive: Whether to check child widgets
    """
    fixed_count = 0
    
    try:
        # Check if this widget is a tk.Button with problematic colors
        if isinstance(widget, tk.Button):
            current_bg = widget.cget('bg')
            current_fg = widget.cget('fg')
            
            # Check for problematic color combinations
            if (current_bg and current_fg and 
                current_bg.lower() == current_fg.lower()):
                
                print(f"🔧 Fixing button with same bg/fg color: '{widget.cget('text')}'")
                
                # Use system colors for better compatibility
                widget.configure(
                    bg='systemControlBackgroundColor',
                    fg='systemControlTextColor',
                    highlightbackground='systemControlAccentColor',
                    activebackground='systemSelectedContentBackgroundColor',
                    activeforeground='systemSelectedTextColor'
                )
                fixed_count += 1
        
        # Recursively check child widgets
        if recursive:
            try:
                for child in widget.winfo_children():
                    fixed_count += fix_tk_button_colors(child, recursive=True)
            except:
                pass  # Some widgets might not have children
                
    except Exception as e:
        # Silently ignore widgets that don't support color configuration
        pass
    
    return fixed_count


def apply_macos_dark_mode_fixes(root_window):
    """
    Apply comprehensive dark mode fixes for macOS.
    
    Args:
        root_window: The root tkinter window or toplevel window
    """
    try:
        # Configure appearance for macOS
        if sys.platform == 'darwin':
            try:
                # Try to detect and handle dark mode
                root_window.tk.call('tk::unsupported::MacWindowStyle', 
                                  root_window._w, 'unified')
            except:
                pass  # Not available on all macOS versions
        
        # Configure button styles
        configure_macos_button_styles()
        
        # Fix any problematic tk.Button widgets
        fixed_buttons = fix_tk_button_colors(root_window)
        if fixed_buttons > 0:
            print(f"✅ Fixed {fixed_buttons} button color issues")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error applying macOS fixes: {e}")
        return False


def ensure_dialog_visibility(dialog_window):
    """
    Ensure dialog windows are properly visible on macOS.
    
    Args:
        dialog_window: The dialog tkinter window
    """
    try:
        # Bring to front and focus
        dialog_window.lift()
        dialog_window.attributes('-topmost', True)
        dialog_window.after(100, lambda: dialog_window.attributes('-topmost', False))
        dialog_window.focus_force()
        
        # Apply UI fixes
        apply_macos_dark_mode_fixes(dialog_window)
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not ensure dialog visibility: {e}")
        return False


# Auto-apply fixes when module is imported on macOS
if sys.platform == 'darwin':
    print("🍎 macOS UI fixes loaded")
