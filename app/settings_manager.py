"""
Settings Manager for StampZ Application

Handles application preferences, about dialog, and dependency checking.
"""

import tkinter as tk
from tkinter import messagebox
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.stampz_app import StampZApp

logger = logging.getLogger(__name__)


class SettingsManager:
    """Manages application settings and preferences."""
    
    def __init__(self, app: 'StampZApp'):
        self.app = app
        self.root = app.root
        
    def show_about(self):
        """Display the About dialog."""
        try:
            from __init__ import __version__, __app_name__, __description__
        except ImportError:
            __version__ = "3.0.3"
            __app_name__ = "StampZ_III"
            __description__ = "Image analysis and color analysis tool"
        
        messagebox.showinfo(
            f"About {__app_name__}",
            f"{__app_name__} v{__version__}\\n\\n"
            f"{__description__}\\n\\n"
            "Features:\\n"
            "• Image cropping with polygon selection\\n"
            "• Color analysis and measurement\\n"
            "• Black Ink Extractor for cancellation isolation\\n"
            "• Compare mode for color averaging\\n"
            "• Export to ODS, XLSX, and CSV formats\\n"
            "• Color library management\\n"
            "• Spectral analysis tools\\n\\n"
            "Built for precision philatelic analysis."
        )

    def open_preferences(self):
        """Open the preferences dialog."""
        try:
            from gui.preferences_dialog import show_preferences_dialog
            show_preferences_dialog(parent=self.root)
        except ImportError as e:
            messagebox.showerror(
                "Missing Component",
                f"Preferences dialog not available:\\n\\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to open preferences:\\n\\n{str(e)}"
            )

    def check_dependencies(self):
        """Check optional dependencies and show guidance if needed."""
        try:
            from utils.dependency_checker import DependencyChecker
            checker = DependencyChecker()
            
            # Only show dialog if important dependencies are missing
            if checker.should_show_dependency_dialog():
                self._show_dependency_dialog(checker)
                
        except Exception as e:
            logger.warning(f"Error checking dependencies: {e}")

    def _show_dependency_dialog(self, checker):
        """Show dependency status dialog to user."""
        try:
            from tkinter import Toplevel, Text, Scrollbar, Button, Frame, Label
            
            status = checker.get_dependency_status_summary()
            
            # Create dialog
            dialog = Toplevel(self.root)
            dialog.title("Optional Dependencies")
            dialog.geometry("700x500")
            dialog.resizable(True, True)
            
            # Center dialog
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Position dialog
            dialog.update_idletasks()
            x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
            y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
            dialog.geometry(f"+{x}+{y}")
            
            # Title
            title_label = Label(dialog, 
                               text=f"Optional Dependencies ({status['available_count']}/{status['total_dependencies']} available)",
                               font=("Arial", 14, "bold"))
            title_label.pack(pady=10)
            
            # Main content frame
            content_frame = Frame(dialog)
            content_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Status text area
            text_frame = Frame(content_frame)
            text_frame.pack(fill="both", expand=True)
            
            text_area = Text(text_frame, wrap="word", font=("Courier", 10))
            scrollbar = Scrollbar(text_frame, orient="vertical", command=text_area.yview)
            text_area.configure(yscrollcommand=scrollbar.set)
            
            text_area.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Insert dependency report
            report = checker.format_dependency_report()
            text_area.insert("1.0", report)
            text_area.configure(state="disabled")
            
            # Button frame
            button_frame = Frame(dialog)
            button_frame.pack(fill="x", padx=20, pady=10)
            
            def copy_install_commands():
                """Copy installation commands to clipboard."""
                try:
                    commands = "\\n".join([dep.installation_command for dep in checker.get_missing_dependencies()])
                    dialog.clipboard_clear()
                    dialog.clipboard_append(commands)
                    messagebox.showinfo("Copied", "Installation commands copied to clipboard!")
                except Exception as e:
                    messagebox.showerror("Copy Error", f"Failed to copy commands: {e}")
            
            def save_install_script():
                """Save installation script to file."""
                try:
                    from tkinter import filedialog
                    
                    script_path = filedialog.asksaveasfilename(
                        title="Save Installation Script",
                        defaultextension=".sh",
                        filetypes=[
                            ('Shell Script', '*.sh'),
                            ('Text files', '*.txt'),
                            ('All files', '*.*')
                        ],
                        initialfile="install_stampz_deps.sh"
                    )
                    
                    if script_path:
                        script_content = checker.get_installation_script()
                        with open(script_path, 'w') as f:
                            f.write(script_content)
                        messagebox.showinfo("Script Saved", 
                                           f"Installation script saved to:\\n{script_path}\\n\\n"
                                           f"Make it executable with:\\nchmod +x {script_path}")
                        
                except Exception as e:
                    messagebox.showerror("Save Error", f"Failed to save script: {e}")
            
            # Add buttons
            if checker.get_missing_dependencies():
                Button(button_frame, text="Copy Install Commands", 
                      command=copy_install_commands).pack(side="left", padx=5)
                Button(button_frame, text="Save Install Script", 
                      command=save_install_script).pack(side="left", padx=5)
            
            Button(button_frame, text="Continue", 
                  command=dialog.destroy).pack(side="right", padx=5)
            
            # Initial focus
            text_area.focus_set()
            
        except Exception as e:
            logger.error(f"Error showing dependency dialog: {e}")
            # Fallback to console output
            print(checker.format_dependency_report())
