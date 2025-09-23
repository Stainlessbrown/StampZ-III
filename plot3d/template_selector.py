import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import os
import logging
from utils.rigid_plot3d_templates import create_rigid_plot3d_templates, RigidPlot3DTemplate
from utils.color_data_bridge import ColorDataBridge

class TemplateSelector:
    def __init__(self, parent=None):
        self.file_path = None
        self.root = None
        self.parent = parent
        self.database_mode = False  # Flag to indicate database selection vs file selection
        self.selected_database = None  # Store selected database name
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
            heading = tk.Label(self.root, text="Plot_3D Data Source Selector", font=("Arial", 14, "bold"))
            heading.pack(pady=10)
            
            # Check for available databases and add database option if available
            try:
                bridge = ColorDataBridge()
                sample_sets = bridge.get_available_sample_sets()
                if sample_sets:
                    # Load from internal database button (NEW - highest priority)
                    database_button = tk.Button(
                        self.root, 
                        text="🗄️ Load Internal Database",
                        command=self.select_database,
                        width=25,
                        height=2,
                        bg="lightcyan",
                        fg="black",
                        font=("Arial", 11, "bold")
                    )
                    database_button.pack(pady=8)
                    
                    # Add description for database
                    db_desc = tk.Label(
                        self.root,
                        text=f"(Access {len(sample_sets)} internal databases shared with Ternary)",
                        font=("Arial", 9),
                        fg="#006666"
                    )
                    db_desc.pack(pady=(0, 15))
            except Exception as e:
                logging.warning(f"Could not access internal databases: {e}")
            
            # Open existing .ods file button
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
                
                # Ensure proper focus and modal behavior
                self.root.focus_set()
                self.root.focus_force()
                
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
    
    def select_database(self):
        """Select from available internal databases."""
        logging.debug("Select Database")
        print("DEBUG: Starting database selection...")
        
        try:
            print("DEBUG: Creating ColorDataBridge...")
            bridge = ColorDataBridge()
            print("DEBUG: Getting available sample sets...")
            sample_sets = bridge.get_available_sample_sets()
            print(f"DEBUG: Found {len(sample_sets)} sample sets: {sample_sets}")
            
            if not sample_sets:
                print("DEBUG: No sample sets found, showing info dialog")
                messagebox.showinfo("No Databases", "No internal databases found.")
                return
            
            # Create database selection dialog first, then hide main dialog
            print("DEBUG: Creating database dialog...")
            try:
                dialog = tk.Toplevel(self.root)
                print("DEBUG: Dialog toplevel created")
                dialog.title("Select Database")
                print("DEBUG: Dialog title set")
                
                # Position dialog offset from main window to avoid overlap
                main_x = self.root.winfo_x()
                main_y = self.root.winfo_y()
                main_width = self.root.winfo_width()
                
                # Place dialog to the right of main window with some padding
                dialog_x = main_x + main_width + 20
                dialog_y = main_y
                
                dialog.geometry(f"400x500+{dialog_x}+{dialog_y}")
                print(f"DEBUG: Dialog positioned at {dialog_x},{dialog_y} (offset from main window)")
                dialog.transient(self.root)
                print("DEBUG: Dialog transient set")
                dialog.grab_set()
                print("DEBUG: Dialog grab set")
                
                # Position dialog and focus it
                dialog.lift()
                dialog.focus_set()
                dialog.focus_force()
                print("DEBUG: Dialog focused")
                
                ttk.Label(dialog, text="Select internal database:", font=('Arial', 12, 'bold')).pack(pady=10)
                
                # Create listbox for databases
                listbox_frame = tk.Frame(dialog)
                listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
                
                listbox = tk.Listbox(listbox_frame, height=12, font=('Arial', 10))
                scrollbar = tk.Scrollbar(listbox_frame, orient="vertical")
                listbox.configure(yscrollcommand=scrollbar.set)
                scrollbar.configure(command=listbox.yview)
                
                # Add databases to listbox
                for sample_set in sample_sets:
                    listbox.insert(tk.END, sample_set)
                    
                listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Info label
                info_label = tk.Label(dialog, 
                                     text="These databases are shared with Ternary Plot.\n" +
                                          "Changes made in Plot_3D can be saved back to database.",
                                     font=('Arial', 9), fg='#666666', justify=tk.CENTER)
                info_label.pack(pady=10)
                
                def on_select():
                    selection = listbox.curselection()
                    if selection:
                        selected_db = sample_sets[selection[0]]
                        self.database_mode = True
                        self.selected_database = selected_db
                        self.file_path = None  # Clear file path since we're using database
                        logging.info(f"Selected database: {selected_db}")
                        dialog.attributes('-topmost', False)  # Remove topmost
                        dialog.destroy()
                        
                        # Database selected, destroy main window
                        self.root.destroy()
                    else:
                        messagebox.showwarning("No Selection", "Please select a database first.")
                
                def on_cancel():
                    print("DEBUG: Database selection cancelled")
                    dialog.destroy()
                    self.root.lift()  # Bring main window back to focus
                
                # Handle dialog close button (X)
                def on_dialog_close():
                    print("DEBUG: Database dialog closed")
                    dialog.destroy()
                    self.root.lift()  # Bring main window back to focus
                
                dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
                
                # Buttons
                button_frame = tk.Frame(dialog)
                button_frame.pack(pady=10)
                
                tk.Button(button_frame, text="Load Database", command=on_select, 
                         bg="lightcyan", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
                tk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
                
                # Dialog is positioned separately, just ensure it has focus
                print("DEBUG: Database selection dialog ready")
                dialog.lift()
                dialog.focus_force()
                
            except Exception as dialog_error:
                # If dialog creation fails, show error
                logging.error(f"Error creating database dialog: {dialog_error}")
                print(f"DEBUG: Dialog creation error: {dialog_error}")
                messagebox.showerror("Dialog Error", f"Failed to create database selection dialog: {dialog_error}")
            
        except Exception as e:
            logging.error(f"Error selecting database: {e}")
            print(f"DEBUG: Database selection error: {e}")
            messagebox.showerror("Error", f"Failed to access databases: {e}")

