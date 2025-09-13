import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import logging
from utils.rigid_plot3d_templates import create_rigid_plot3d_templates, RigidPlot3DTemplate

class TemplateSelector:
    def __init__(self, parent=None):
        self.file_path = None
        self.root = None
        self.parent = parent
        self.create_and_run_dialog()
        
    def create_and_run_dialog(self):
        """Create and run the file selection dialog with proper error handling."""
        try:
            # Create the main window - check if we have a parent (embedded mode)
            if self.parent:
                # Embedded mode - create as Toplevel window
                self.root = tk.Toplevel(self.parent)
                self.root.transient(self.parent)
                self.root.grab_set()  # Modal to parent only, not entire app
            else:
                # Standalone mode - create root window
                self.root = tk.Tk()
                
            self.root.title("Plot_3D File Selector")
            self.root.geometry("400x300")
            self.root.lift()
            self.root.attributes("-topmost", True)
            
            # Add heading
            heading = tk.Label(self.root, text="Plot_3D Template Selector", font=("Arial", 14, "bold"))
            heading.pack(pady=10)
            
            # Open existing .ods file button (most common use case)
            open_button = tk.Button(
                self.root, 
                text="📊 Open Existing .ods File",
                command=self.select_custom_file,
                width=25,
                height=2,
                bg="lightgreen",
                fg="black",
                font=("Arial", 11, "bold")
            )
            open_button.pack(pady=8)
            
            # Add description for open file
            open_desc = tk.Label(
                self.root,
                text="(For StampZ exports and other .ods data files)",
                font=("Arial", 9),
                fg="#666666"
            )
            open_desc.pack(pady=(0, 15))
            
            # Use existing template button
            existing_button = tk.Button(
                self.root, 
                text="📋 Use Built-in Template",
                command=self.select_existing_template,
                width=25,
                height=2,
                bg="lightblue",
                fg="black",
                font=("Arial", 11, "bold")
            )
            existing_button.pack(pady=8)
            
            # Add description for templates
            template_desc = tk.Label(
                self.root,
                text="(Start with pre-made Plot_3D templates)",
                font=("Arial", 9),
                fg="#666666"
            )
            template_desc.pack(pady=(0, 15))
            
            # Create rigid template button
            rigid_button = tk.Button(
                self.root, 
                text="➕ Create New Template",
                command=self.create_rigid_template,
                width=25,
                height=2,
                bg="lightyellow",
                fg="black",
                font=("Arial", 11, "bold")
            )
            rigid_button.pack(pady=8)
            
            # Add description for create template
            create_desc = tk.Label(
                self.root,
                text="(Generate blank template for manual data entry)",
                font=("Arial", 9),
                fg="#666666"
            )
            create_desc.pack(pady=(0, 10))
            
            # Center dialog if we have a parent
            if self.parent:
                self.root.update_idletasks()
                x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.root.winfo_width() // 2)
                y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.root.winfo_height() // 2)
                self.root.geometry(f"+{x}+{y}")
                
                # Wait for window instead of starting mainloop in embedded mode
                self.parent.wait_window(self.root)
            else:
                # Start the main loop only in standalone mode
                self.root.mainloop()
        except Exception as e:
            logging.error(f"Error creating template selector window: {str(e)}")
            if self.root:
                try:
                    self.root.destroy()
                except:
                    pass
            raise
    
    def select_custom_file(self):
        logging.debug("Select File")
        # Create a new temporary tkinter root for the file dialog
        file_root = tk.Tk()
        file_root.withdraw()
        
        file_types = [
            ('OpenDocument Spreadsheet', '*.ods'),
            ('All files', '*.*')
        ]
        
        selected_file = filedialog.askopenfilename(filetypes=file_types)
        file_root.destroy()
        
        if selected_file:
            # Convert to absolute path
            self.file_path = os.path.abspath(selected_file)
            logging.info(f"Selected file (absolute path): {self.file_path}")
            self.root.destroy()
        else:
            logging.warning("No file selected")
    
    def create_rigid_template(self):
        """Create a new rigid Plot_3D template."""
        logging.debug("Create Rigid Template")
        
        # Ask user for template name
        template_name = simpledialog.askstring(
            "Template Name",
            "Enter a name for the rigid template:",
            initialvalue="Plot3D_Rigid"
        )
        
        if not template_name:
            return
            
        # Get save location
        file_root = tk.Tk()
        file_root.withdraw()
        
        file_types = [
            ('OpenDocument Spreadsheet', '*.ods'),
            ('All files', '*.*')
        ]
        
        selected_file = filedialog.asksaveasfilename(
            title="Save Rigid Template",
            filetypes=file_types,
            defaultextension=".ods",
            initialfile=f"{template_name}_Template"
        )
        file_root.destroy()
        
        if selected_file:
            try:
                # Create rigid template
                rigid_creator = RigidPlot3DTemplate()
                success = rigid_creator.create_rigid_template(selected_file, template_name)
                
                if success:
                    self.file_path = os.path.abspath(selected_file)
                    logging.info(f"Created rigid template: {self.file_path}")
                    messagebox.showinfo(
                        "Template Created",
                        f"Rigid Plot_3D template created successfully!\n\n"
                        f"File: {os.path.basename(selected_file)}\n\n"
                        f"Features:\n"
                        f"• Protected column structure\n"
                        f"• Data validation dropdowns\n"
                        f"• Format compliance for K-means & ΔE\n"
                        f"• 'Refresh Data' compatible\n\n"
                        f"Ready for Plot_3D analysis!"
                    )
                    self.root.destroy()
                else:
                    messagebox.showerror(
                        "Creation Failed",
                        "Failed to create rigid template. Please try again."
                    )
            except Exception as e:
                logging.error(f"Error creating rigid template: {e}")
                messagebox.showerror(
                    "Error",
                    f"Error creating rigid template:\n\n{str(e)}"
                )
        else:
            logging.warning("No save location selected")
    
    def select_existing_template(self):
        """Select from existing rigid templates in the templates directory."""
        logging.debug("Select Existing Template")
        
        # Check templates directory
        templates_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "templates", "plot3d")
        
        if not os.path.exists(templates_dir):
            # Create templates directory and rigid templates
            try:
                os.makedirs(templates_dir, exist_ok=True)
                results = create_rigid_plot3d_templates()
                logging.info(f"Created template directory and templates: {results}")
            except Exception as e:
                logging.error(f"Error creating templates: {e}")
        
        # Look for existing rigid templates
        rigid_templates = []
        if os.path.exists(templates_dir):
            for file in os.listdir(templates_dir):
                if file.endswith('.ods') and 'Rigid' in file:
                    rigid_templates.append(os.path.join(templates_dir, file))
        
        # If no rigid templates found, offer to create them
        if not rigid_templates:
            create_new = messagebox.askyesno(
                "No Rigid Templates",
                "No rigid templates found in the templates directory.\n\n"
                "Would you like to create the standard rigid templates now?"
            )
            
            if create_new:
                try:
                    results = create_rigid_plot3d_templates()
                    if results:
                        messagebox.showinfo(
                            "Templates Created",
                            f"Created rigid templates:\n\n" + "\n".join(results)
                        )
                        # Refresh the list
                        for file in os.listdir(templates_dir):
                            if file.endswith('.ods') and 'Rigid' in file:
                                rigid_templates.append(os.path.join(templates_dir, file))
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create templates: {e}")
                    return
            else:
                return
        
        # Let user select from available templates
        if rigid_templates:
            file_root = tk.Tk()
            file_root.withdraw()
            
            selected_file = filedialog.askopenfilename(
                title="Select Rigid Template",
                initialdir=templates_dir,
                filetypes=[
                    ('OpenDocument Spreadsheet', '*.ods'),
                    ('All files', '*.*')
                ]
            )
            file_root.destroy()
            
            if selected_file:
                self.file_path = os.path.abspath(selected_file)
                logging.info(f"Selected existing rigid template: {self.file_path}")
                self.root.destroy()

