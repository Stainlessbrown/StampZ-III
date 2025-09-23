import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from typing import Set, List, Dict, Optional, Callable

class GroupDisplayManager:
    """
    Manages the display of selected groups of data points based on row ranges or clusters.
    Provides a minimalist UI for controlling which points are visible in the plot.
    """
    def __init__(self, master, frame, df, on_visibility_change):
        """
        Initialize the GroupDisplayManager.
        
        Args:
            master: The Tkinter root or parent window
            frame: The frame where controls should be placed
            df: The DataFrame containing the data points
            on_visibility_change: Callback function when visibility changes
        """
        self.master = master
        self.frame = frame
        self.df = df
        self.on_visibility_change = on_visibility_change
        
        # State tracking
        self.visible_indices: Set[int] = set()  # Set of visible DataFrame indices
        self.selection_mode = tk.StringVar(value='cluster')  # Only cluster mode now
        self.visibility_enabled = tk.BooleanVar(value=False)
        self.cluster_selection = tk.StringVar(value='')  # For cluster selection dropdown
        
        # Row mapping (from spreadsheet rows to DataFrame indices)
        self.row_mapping = {}  # Will be populated in update_references
        self.index_to_row = {}
        
        # Create controls
        self.create_controls()
        
        # Initialize row mapping
        self.update_row_mapping()
        
        # Set initial UI state
        self._initialize_ui_state()
        
    def create_controls(self):
        """Create the group display control panel"""
        # Main frame with bold title and prominent border
        controls_frame = ttk.LabelFrame(self.frame, text="Cluster Display")
        controls_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Force minimum size
        controls_frame.grid_propagate(False)
        controls_frame.configure(height=150, width=350)  # Set minimum dimensions
        
        # Cluster selection (primary method)
        self.cluster_frame = ttk.Frame(controls_frame)
        self.cluster_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Label(self.cluster_frame, text="Select Cluster:").grid(row=0, column=0, sticky='w', padx=5)
        self.cluster_dropdown = ttk.Combobox(self.cluster_frame, textvariable=self.cluster_selection, state='readonly')
        self.cluster_dropdown.grid(row=0, column=1, sticky='ew', padx=5)
        self.cluster_dropdown.bind('<<ComboboxSelected>>', lambda e: self._update_selection())
        
        self.cluster_frame.grid_columnconfigure(1, weight=1)
        
        # Note: Row Range functionality removed as no longer needed
        # Cluster selection is now the primary and only selection method
        
        # Visibility controls in their own frame
        control_frame = ttk.Frame(controls_frame)
        control_frame.grid(row=3, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Checkbutton(
            control_frame,
            text="Show Selection Only",
            variable=self.visibility_enabled,
            command=self._on_visibility_toggle
        ).grid(row=0, column=0, sticky='w', padx=5)
        
        ttk.Button(
            control_frame,
            text="Clear",
            command=self.clear_selection
        ).grid(row=0, column=1, sticky='w', padx=5)
        
        # Configure grid weights
        controls_frame.grid_columnconfigure(0, weight=1)
        
    def _initialize_ui_state(self):
        """Initialize the UI state based on available data"""
        try:
            # Always use cluster mode now
            self.selection_mode.set('cluster')
            
            # Check if we have cluster data available
            if 'Cluster' in self.df.columns and not self.df['Cluster'].isna().all():
                # Clusters available - update dropdown
                self._update_cluster_dropdown()
                print(f"DEBUG: Cluster Display initialized with {len(self.cluster_dropdown['values'])} clusters")
            else:
                # No clusters available - show message
                print("DEBUG: No cluster data available - user should run K-means first")
        except Exception as e:
            print(f"Error initializing Group Display UI state: {e}")

    def update_row_mapping(self):
        """Create a mapping between spreadsheet row numbers and DataFrame indices"""
        try:
            # Get the dataframe length for validation
            df_length = len(self.df)
            
            # Create both mappings:
            # 1. From spreadsheet row to DataFrame index
            # 2. From DataFrame index to spreadsheet row
            self.row_mapping = {}
            self.index_to_row = {}
            
            # CRITICAL FIX: Check for _original_sheet_row column from internal worksheet
            if '_original_sheet_row' in self.df.columns:
                print("Group Display: Using '_original_sheet_row' column for precise row mapping (REALTIME WORKSHEET)")
                for idx, row in self.df.iterrows():
                    orig_sheet_row = int(row['_original_sheet_row'])
                    # Display row = sheet row + 1 (sheet rows are 0-based, display rows are 1-based)
                    display_row = orig_sheet_row + 1
                    self.row_mapping[display_row] = idx
                    self.index_to_row[idx] = display_row
                    
                    # DEBUG: Show the mapping for first few rows
                    if idx < 5:
                        print(f"Group Display DEBUG: Display row {display_row} → DataFrame index {idx} (sheet row {orig_sheet_row})")
                        
            elif 'original_row' in self.df.columns:
                print("Group Display: Using 'original_row' column for precise row mapping (FILE-BASED)")
                for idx, row in self.df.iterrows():
                    orig_row = int(row['original_row'])
                    # Adjust for header row (+1) and zero-indexing (+1) = +2
                    spreadsheet_row = orig_row + 2
                    self.row_mapping[spreadsheet_row] = idx
                    self.index_to_row[idx] = spreadsheet_row
            else:
                # Fall back to default sequential mapping
                print("Group Display: Using default sequential row mapping (LEGACY - may be incorrect)")
                # Row 2 in spreadsheet = index 0 in DataFrame (accounting for header row)
                self.row_mapping = {i+2: i for i in range(df_length)}
                self.index_to_row = {i: i+2 for i in range(df_length)}
            
            print(f"Group Display: Created row mapping for {df_length} data points")
            
        except Exception as e:
            print(f"Error creating row mapping: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_mode_change(self):
        """Handle selection mode change (cluster mode only now)"""
        # Always use cluster mode
        self.selection_mode.set('cluster')
        
        # Check if clusters exist
        if 'Cluster' not in self.df.columns or self.df['Cluster'].isna().all():
            messagebox.showwarning(
                "No Clusters Available",
                "Please run K-means clustering first to create clusters."
            )
            return
        
        # Update cluster dropdown
        self._update_cluster_dropdown()
        
        # Clear current selection when changing modes
        self.visible_indices.clear()
        self.on_visibility_change()
    
    def _update_cluster_dropdown(self):
        """Update the cluster dropdown with available clusters"""
        if 'Cluster' not in self.df.columns:
            return
        
        # Get unique clusters
        clusters = sorted(self.df['Cluster'].dropna().unique())
        cluster_values = [str(c) for c in clusters]
        
        # Update dropdown
        self.cluster_dropdown['values'] = cluster_values
        if cluster_values and not self.cluster_selection.get():
            self.cluster_selection.set(cluster_values[0])
    
    def _on_visibility_toggle(self):
        """Handle visibility checkbox toggle"""
        if self.visibility_enabled.get():
            self._update_selection()
        else:
            # When toggled off, clear selection but keep the inputs
            self.visible_indices.clear()
            self.on_visibility_change()
    
    # Row range parsing method removed - no longer needed
    
    def _update_selection(self):
        """Update the current selection based on cluster selection"""
        if not self.visibility_enabled.get():
            return
            
        self.visible_indices.clear()
        
        # Cluster-based selection (only mode now)
        selected_cluster = self.cluster_selection.get()
        if 'Cluster' in self.df.columns and selected_cluster:
            try:
                # Convert string back to appropriate type (int, float, etc.)
                cluster_value = type(self.df['Cluster'].dropna().iloc[0])(selected_cluster)
                # Find rows with the selected cluster
                cluster_mask = self.df['Cluster'] == cluster_value
                selected_indices = self.df[cluster_mask].index.tolist()
                self.visible_indices.update(selected_indices)
                print(f"DEBUG: Selected cluster {cluster_value}, found {len(selected_indices)} points")
            except (ValueError, IndexError) as e:
                print(f"Error converting cluster value '{selected_cluster}': {e}")
        
        # Trigger update in main application
        self.on_visibility_change()
    
    def clear_selection(self):
        """Clear the current selection"""
        self.visible_indices.clear()
        self.cluster_selection.set('')  # Clear cluster selection
        self.visibility_enabled.set(False)
        self.on_visibility_change()
    
    def update_references(self, df):
        """Update the DataFrame reference when data changes"""
        self.df = df
        
        # Update row mapping
        self.update_row_mapping()
        
        # Update cluster dropdown (always in cluster mode now)
        self._update_cluster_dropdown()
        
        # Reapply current selection with updated DataFrame
        if self.visibility_enabled.get():
            self._update_selection()
    
    def get_visible_mask(self) -> pd.Series:
        """
        Return a boolean mask for visible rows.
        When disabled or no selection, all points are visible.
        When enabled, only selected points are visible.
        """
        try:
            enabled = self.visibility_enabled.get()
            num_selected = len(self.visible_indices)
            
            print(f"\n🔍 GROUP DISPLAY MASK DEBUG:")
            print(f"  Visibility enabled: {enabled}")
            print(f"  Selected indices count: {num_selected}")
            print(f"  DataFrame shape: {self.df.shape if self.df is not None else 'None'}")
            print(f"  DataFrame index: {list(self.df.index) if self.df is not None and len(self.df) < 10 else f'[0...{len(self.df)-1}]' if self.df is not None else 'None'}")
            print(f"  Selected indices: {sorted(list(self.visible_indices)) if self.visible_indices else 'None'}")
            
            # Safety check: ensure we have a valid DataFrame
            if self.df is None or len(self.df) == 0:
                print(f"  → ERROR: No valid DataFrame, returning empty mask")
                return pd.Series(dtype=bool)  # Return empty Series
            
            if not enabled:
                # When disabled, show all points
                print(f"  → DISABLED: Showing all {len(self.df)} points")
                return pd.Series(True, index=self.df.index)
            
            # When enabled, only show selected points
            mask = pd.Series(False, index=self.df.index)  # Start with all False
            if self.visible_indices:
                # CRITICAL FIX: Filter selected indices to only include valid ones
                valid_indices = [idx for idx in self.visible_indices if idx in self.df.index]
                invalid_indices = [idx for idx in self.visible_indices if idx not in self.df.index]
                
                if invalid_indices:
                    print(f"  → WARNING: Found {len(invalid_indices)} invalid indices: {invalid_indices}")
                    print(f"  → Removing invalid indices from selection")
                    # Update the visible_indices to remove invalid ones
                    self.visible_indices = set(valid_indices)
                
                if valid_indices:
                    mask.loc[valid_indices] = True  # Set True for valid selected indices
                    visible_count = mask.sum()
                    print(f"  → ENABLED: Showing {visible_count}/{len(self.df)} points (only selected)")
                else:
                    print(f"  → ENABLED: No valid selection after filtering, showing 0/{len(self.df)} points")
            else:
                print(f"  → ENABLED: No selection, showing 0/{len(self.df)} points")
            
            # Final safety check: ensure mask index matches DataFrame index
            if not mask.index.equals(self.df.index):
                print(f"  → ERROR: Mask index doesn't match DataFrame index, creating new aligned mask")
                # Create a new properly aligned mask
                aligned_mask = pd.Series(False, index=self.df.index)
                if enabled and valid_indices:
                    aligned_mask.loc[valid_indices] = True
                elif not enabled:
                    aligned_mask.loc[:] = True
                return aligned_mask
            
            return mask
            
        except Exception as e:
            print(f"  → CRITICAL ERROR in get_visible_mask: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback: return all-True mask to show all points
            if self.df is not None and len(self.df) > 0:
                print(f"  → FALLBACK: Returning all-True mask for {len(self.df)} points")
                return pd.Series(True, index=self.df.index)
            else:
                print(f"  → FALLBACK: Returning empty mask")
                return pd.Series(dtype=bool)

