import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import csv
import random
import os
from datetime import datetime
from itertools import combinations
import time

class SpacecraftBoltStudyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spacecraft Bolt Study Generator v1.0")
        self.root.geometry("1000x800")
        
        # Configuration variables
        self.study_type = tk.StringVar(value="S3-Limited")
        self.parameter_range = tk.StringVar(value="Sparse")
        self.cbush_selection = tk.StringVar(value="All")
        self.custom_cbushes = tk.StringVar(value="2,5,10")
        self.random_designs = tk.IntVar(value=5000)
        self.random_type = tk.StringVar(value="Pure Random")
        self.output_directory = tk.StringVar(value=os.getcwd())
        
        # CBUSH position mapping (user-editable)
        self.bottom_cbushes = tk.StringVar(value="2,3,4")
        self.middle_cbushes = tk.StringVar(value="5,6,7")
        self.top_cbushes = tk.StringVar(value="8,9,10")
        
        # Generation control
        self.cancel_generation = False
        self.generation_thread = None
        
        # Stiffness value mappings
        self.stiffness_ranges = {
            "Sparse": [1, 4, 7, 10],
            "Medium": [1, 3, 5, 7, 9, 11],
            "Full": list(range(1, 12)),
            "Custom": [1, 4, 7, 10]  # Default, user can modify
        }
        
        self.create_widgets()
        self.update_preview()
        
    def create_widgets(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Main Configuration Tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Study Configuration")
        
        # CBUSH Layout Tab
        cbush_frame = ttk.Frame(notebook)
        notebook.add(cbush_frame, text="CBUSH Layout")
        
        # Advanced Options Tab
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced Options")
        
        self.create_main_tab(main_frame)
        self.create_cbush_tab(cbush_frame)
        self.create_advanced_tab(advanced_frame)
        
    def create_main_tab(self, parent):
        # Study Type Selection
        study_group = ttk.LabelFrame(parent, text="Study Type Selection", padding="10")
        study_group.pack(fill="x", padx=5, pady=5)
        
        study_options = [
            ("S1: Single CBUSH (Limited: 576 designs)", "S1-Limited"),
            ("S1: Single CBUSH (Complete: 11,979 designs - ~14 days)", "S1-Complete"),
            ("S2: Adjacent Pairs (Limited: 128 designs)", "S2-Limited"),
            ("S2: Adjacent Pairs (Complete: 968 designs - ~1 day)", "S2-Complete"),
            ("S3: Multi-Bolt (Limited: 192 designs)", "S3-Limited"),
            ("S3: Multi-Bolt (Complete: 111,804 designs - ~44 days!)", "S3-Complete"),
            ("Random: Training Data (Recommended for ML)", "Random"),
            ("Custom: User-defined combinations", "Custom")
        ]
        
        for i, (text, value) in enumerate(study_options):
            ttk.Radiobutton(study_group, text=text, variable=self.study_type, 
                           value=value, command=self.update_preview).grid(row=i//2, column=i%2, sticky="w", padx=5, pady=2)
        
        # Parameter Range Selection
        range_group = ttk.LabelFrame(parent, text="Parameter Range Selection", padding="10")
        range_group.pack(fill="x", padx=5, pady=5)
        
        range_options = [
            ("Sparse [1,4,7,10] - Quick validation", "Sparse"),
            ("Medium [1,3,5,7,9,11] - Balanced coverage", "Medium"),
            ("Full [1,2,3,...,10,11] - Comprehensive analysis", "Full"),
            ("Custom - User defined", "Custom")
        ]
        
        for i, (text, value) in enumerate(range_options):
            ttk.Radiobutton(range_group, text=text, variable=self.parameter_range, 
                           value=value, command=self.update_preview).grid(row=i, column=0, sticky="w", padx=5, pady=2)
        
        # Custom range entry
        custom_frame = ttk.Frame(range_group)
        custom_frame.grid(row=4, column=0, sticky="w", padx=20, pady=5)
        ttk.Label(custom_frame, text="Custom range:").pack(side="left")
        self.custom_range_entry = ttk.Entry(custom_frame, width=20)
        self.custom_range_entry.pack(side="left", padx=5)
        self.custom_range_entry.insert(0, "1,3,5,7,9,11")
        
        # CBUSH Selection
        cbush_group = ttk.LabelFrame(parent, text="CBUSH Selection", padding="10")
        cbush_group.pack(fill="x", padx=5, pady=5)
        
        cbush_options = [
            ("All CBUSHes (2-10)", "All"),
            ("Bottom CBUSHes", "Bottom"),
            ("Middle CBUSHes", "Middle"),
            ("Top CBUSHes", "Top"),
            ("Bottom + Top", "BottomTop"),
            ("Custom Selection", "Custom")
        ]
        
        for i, (text, value) in enumerate(cbush_options):
            ttk.Radiobutton(cbush_group, text=text, variable=self.cbush_selection, 
                           value=value, command=self.update_preview).grid(row=i//3, column=i%3, sticky="w", padx=5, pady=2)
        
        # Custom CBUSH entry
        custom_cbush_frame = ttk.Frame(cbush_group)
        custom_cbush_frame.grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        ttk.Label(custom_cbush_frame, text="Custom CBUSHes:").pack(side="left")
        ttk.Entry(custom_cbush_frame, textvariable=self.custom_cbushes, width=15).pack(side="left", padx=5)
        
        # Random Data Configuration (shown when Random is selected)
        self.random_group = ttk.LabelFrame(parent, text="Random Data Configuration", padding="10")
        
        ttk.Label(self.random_group, text="Number of designs:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(self.random_group, textvariable=self.random_designs, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(self.random_group, text="Random type:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        random_combo = ttk.Combobox(self.random_group, textvariable=self.random_type, 
                                   values=["Pure Random", "Balanced Random", "Failure-Focused"], 
                                   state="readonly", width=15)
        random_combo.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        # Preview and Warnings
        self.preview_group = ttk.LabelFrame(parent, text="Generation Preview", padding="10")
        self.preview_group.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.preview_text = tk.Text(self.preview_group, height=8, width=80, state="disabled")
        scrollbar = ttk.Scrollbar(self.preview_group, orient="vertical", command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=scrollbar.set)
        self.preview_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Control buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        ttk.Button(button_frame, text="Select Output Directory", 
                  command=self.select_output_directory).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Generate Studies", 
                  command=self.start_generation).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Cancel Generation", 
                  command=self.cancel_generation_func).pack(side="left", padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side="right", fill="x", expand=True, padx=5)
        
        self.status_label = ttk.Label(button_frame, text="Ready")
        self.status_label.pack(side="right", padx=5)
        
    def create_cbush_tab(self, parent):
        # Interactive CBUSH layout definition
        layout_frame = ttk.LabelFrame(parent, text="Define CBUSH Spatial Layout", padding="10")
        layout_frame.pack(fill="x", padx=5, pady=5)
        
        # Visual representation
        ttk.Label(layout_frame, text="Cantilever Beam Layout (Base → Tip):", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Beam visualization
        beam_frame = ttk.Frame(layout_frame)
        beam_frame.pack(pady=10)
        
        # Draw beam elements
        beam_canvas = tk.Canvas(beam_frame, width=800, height=100, bg="white")
        beam_canvas.pack()
        
        # Draw beam structure
        beam_canvas.create_rectangle(50, 40, 750, 60, fill="lightgray", outline="black", width=2)
        beam_canvas.create_text(25, 50, text="Fixed\nBase", font=("Arial", 8))
        beam_canvas.create_text(775, 50, text="Free\nTip", font=("Arial", 8))
        
        # Draw CBUSH positions
        cbush_positions = [100, 170, 240, 310, 380, 450, 520, 590, 660, 730]
        cbush_labels = ["CB1\n(Drive)", "CB2", "CB3", "CB4", "CB5", "CB6", "CB7", "CB8", "CB9", "CB10"]
        
        for i, (pos, label) in enumerate(zip(cbush_positions, cbush_labels)):
            color = "red" if i == 0 else "lightblue"
            beam_canvas.create_oval(pos-15, 25, pos+15, 75, fill=color, outline="black")
            beam_canvas.create_text(pos, 90, text=label, font=("Arial", 8))
        
        # Category definitions
        category_frame = ttk.LabelFrame(parent, text="CBUSH Category Definitions", padding="10")
        category_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(category_frame, text="Bottom CBUSHes (Near Base):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(category_frame, textvariable=self.bottom_cbushes, width=20).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(category_frame, text="Middle CBUSHes (Center):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(category_frame, textvariable=self.middle_cbushes, width=20).grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        ttk.Label(category_frame, text="Top CBUSHes (Near Tip):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(category_frame, textvariable=self.top_cbushes, width=20).grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        # Quick presets
        preset_frame = ttk.LabelFrame(parent, text="Quick Presets", padding="10")
        preset_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(preset_frame, text="Reset to Default", command=self.reset_cbush_layout).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Every Other (2,4,6,8,10)", command=self.set_every_other).pack(side="left", padx=5)
        ttk.Button(preset_frame, text="Random 3", command=self.set_random_3).pack(side="left", padx=5)
        
    def create_advanced_tab(self, parent):
        # File naming options
        naming_frame = ttk.LabelFrame(parent, text="File Naming Options", padding="10")
        naming_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(naming_frame, text="Output directory:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(naming_frame, textvariable=self.output_directory, width=50).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        ttk.Button(naming_frame, text="Browse", command=self.select_output_directory).grid(row=0, column=2, padx=5, pady=2)
        
        # Validation options
        validation_frame = ttk.LabelFrame(parent, text="Validation Options", padding="10")
        validation_frame.pack(fill="x", padx=5, pady=5)
        
        self.validate_duplicates = tk.BooleanVar(value=True)
        self.export_summary = tk.BooleanVar(value=True)
        self.create_validation_subset = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(validation_frame, text="Check for duplicate designs", 
                       variable=self.validate_duplicates).pack(anchor="w", padx=5, pady=2)
        ttk.Checkbutton(validation_frame, text="Export generation summary", 
                       variable=self.export_summary).pack(anchor="w", padx=5, pady=2)
        ttk.Checkbutton(validation_frame, text="Create 10% validation subset", 
                       variable=self.create_validation_subset).pack(anchor="w", padx=5, pady=2)
        
        # Performance options
        perf_frame = ttk.LabelFrame(parent, text="Performance Options", padding="10")
        perf_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(perf_frame, text="Batch size for large studies:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.batch_size = tk.IntVar(value=1000)
        ttk.Entry(perf_frame, textvariable=self.batch_size, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Stiffness value reference
        ref_frame = ttk.LabelFrame(parent, text="Stiffness Value Reference", padding="10")
        ref_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ref_text = tk.Text(ref_frame, height=15, width=80, state="disabled")
        ref_scrollbar = ttk.Scrollbar(ref_frame, orient="vertical", command=ref_text.yview)
        ref_text.configure(yscrollcommand=ref_scrollbar.set)
        ref_text.pack(side="left", fill="both", expand=True)
        ref_scrollbar.pack(side="right", fill="y")
        
        # Populate reference table
        ref_content = """Complete Stiffness Encoding System:

Encoded Value | Actual Stiffness | Scientific Notation | Description
-------------|------------------|-------------------|-------------
1            | 10,000           | 1e4               | Extremely loose bolt (most loose)
2            | 100,000          | 1e5               | Very loose bolt
3            | 1,000,000        | 1e6               | Loose bolt
4            | 10,000,000       | 1e7               | Medium-loose bolt
5            | 100,000,000      | 1e8               | Driving CBUSH (reference)
6            | 1,000,000,000    | 1e9               | Medium bolt
7            | 10,000,000,000   | 1e10              | Medium-tight bolt
8            | 100,000,000,000  | 1e11              | Tight bolt
9            | 1,000,000,000,000| 1e12              | Very tight bolt (baseline)
10           | 10,000,000,000,000| 1e13             | Extremely tight bolt
11           | 100,000,000,000,000| 1e14            | Ultra tight bolt (most tight)

Pattern: encoded_value + 3 = exponent (e.g., 1 → 1e4, 4 → 1e7)

HEEDS Runtime Estimates (Based on Real Server Performance):
- 1,000 designs = ~10 hours
- 5,000 designs = ~2 days (verified)
- 10,000 designs = ~4 days  
- 50,000 designs = ~20 days
- 111,804 designs (S3-Complete Full) = ~44 days

Usage Notes:
- Sparse [1,4,7,10]: Quick validation studies
- Medium [1,3,5,7,9,11]: Balanced coverage (~1/6 of Full runtime)
- Full [1-11]: Comprehensive analysis (computationally expensive!)
- CBUSH 1: Always fixed at value 5 (1e8) - driving reference
- Non-variable CBUSHes: Always fixed at value 9 (1e12) - baseline tight

Recommendations:
- Start with Random data (5,000 designs) for ML training
- Use Medium range for validation before Full studies
- S3-Complete Medium = 18,144 designs = ~7 days (manageable)
- S3-Complete Full = 111,804 designs = ~44 days (plan accordingly!)"""
        
        ref_text.config(state="normal")
        ref_text.insert("1.0", ref_content)
        ref_text.config(state="disabled")
        
    def update_preview(self):
        """Update the preview based on current selections"""
        try:
            study_type = self.study_type.get()
            param_range = self.parameter_range.get()
            cbush_sel = self.cbush_selection.get()
            
            # Calculate design count and other metrics
            design_count, runtime_est, file_size, warnings = self.calculate_study_metrics(
                study_type, param_range, cbush_sel)
            
            # Update preview text
            preview_content = f"""Study Configuration:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Study Type: {study_type}
Parameter Range: {param_range} {self.stiffness_ranges.get(param_range, [])}
CBUSH Selection: {cbush_sel}
CBUSHes to vary: {self.get_selected_cbushes()}

Design Count: {design_count:,} designs
Estimated HEEDS Runtime: {runtime_est}
(Based on real data: 5,000 designs = 2 days on server)
Estimated File Size: {file_size}
Required RAM: {self.estimate_ram(design_count)}

{warnings}

Output File: {self.generate_filename(study_type, param_range, cbush_sel, design_count)}
Output Directory: {self.output_directory.get()}
"""
            
            self.preview_text.config(state="normal")
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert("1.0", preview_content)
            self.preview_text.config(state="disabled")
            
            # Show/hide random configuration
            if study_type == "Random":
                self.random_group.pack(fill="x", padx=5, pady=5, before=self.preview_group)
            else:
                self.random_group.pack_forget()
                
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def calculate_study_metrics(self, study_type, param_range, cbush_sel):
        """Calculate design count, runtime, file size, and warnings"""
        try:
            param_values = len(self.stiffness_ranges[param_range])
            cbushes = self.get_selected_cbushes()
            num_cbushes = len(cbushes)
            
            warnings = []
            
            if study_type == "S1-Limited":
                design_count = 9 * (4 ** 3)  # 9 sets × 4³
            elif study_type == "S1-Complete":
                design_count = 9 * (param_values ** 3)
            elif study_type == "S2-Limited":
                design_count = 8 * (4 ** 2)  # 8 pairs × 4²
            elif study_type == "S2-Complete":
                design_count = 8 * (param_values ** 2)
            elif study_type == "S3-Limited":
                design_count = 3 * (param_values ** 3)  # 3 scenarios
            elif study_type == "S3-Complete":
                # All possible 3-CBUSH combinations from CBUSHes 2-10
                from math import comb
                design_count = comb(9, 3) * (param_values ** 3)  # 84 scenarios
            elif study_type == "Random":
                design_count = self.random_designs.get()
            elif study_type == "Custom":
                if num_cbushes == 1:
                    design_count = param_values
                elif num_cbushes == 2:
                    design_count = param_values ** 2
                elif num_cbushes == 3:
                    design_count = param_values ** 3
                else:
                    design_count = param_values ** min(num_cbushes, 3)
            else:
                design_count = 0
            
            # Runtime estimation (34.6 seconds per design based on real HEEDS data)
            runtime_seconds = design_count * 34.6
            if runtime_seconds < 3600:
                runtime_est = f"{runtime_seconds/60:.1f} minutes"
            elif runtime_seconds < 86400:
                runtime_est = f"{runtime_seconds/3600:.1f} hours"
            else:
                runtime_est = f"{runtime_seconds/86400:.1f} days"
            
            # File size estimation (15KB per design)
            file_size_kb = design_count * 0.015
            if file_size_kb < 1024:
                file_size = f"{file_size_kb:.1f} KB"
            else:
                file_size = f"{file_size_kb/1024:.1f} MB"
            
            # Generate warnings (updated thresholds based on real HEEDS performance)
            if design_count > 1000:
                warnings.append("⚠️  MEDIUM STUDY - Will take several hours")
            if design_count > 5000:
                warnings.append("⚠️  LARGE STUDY - Will take 1-2 days")
            if design_count > 20000:
                warnings.append("🚫 EXTREMELY LARGE STUDY - Will take weeks to complete!")
            if runtime_seconds > 86400:  # 1 day
                warnings.append(f"⚠️  LONG RUNTIME - ~{runtime_seconds/86400:.1f} days estimated")
            if param_range == "Full" and study_type in ["S1-Complete", "S3-Complete"]:
                warnings.append("💡 CRITICAL: Full S3-Complete = ~44 days! Consider Medium range first")
            
            # Special warning for S3-Complete Full
            if study_type == "S3-Complete" and param_range == "Full":
                s3_full_days = (111804 * 34.6) / 86400
                warnings.append(f"🚨 S3-Complete Full = {s3_full_days:.0f} days runtime!")
                warnings.append("🎯 Recommended: Try S3-Complete Medium first (18,144 designs = ~7 days)")
                
            warnings_text = "\n".join(warnings) if warnings else "✅ Configuration looks good!"
            
            return design_count, runtime_est, file_size, warnings_text
            
        except Exception as e:
            return 0, "Unknown", "Unknown", f"Error calculating metrics: {e}"
    
    def get_selected_cbushes(self):
        """Get list of selected CBUSHes based on selection mode"""
        selection = self.cbush_selection.get()
        
        if selection == "All":
            return list(range(2, 11))
        elif selection == "Bottom":
            return [int(x.strip()) for x in self.bottom_cbushes.get().split(",") if x.strip()]
        elif selection == "Middle":
            return [int(x.strip()) for x in self.middle_cbushes.get().split(",") if x.strip()]
        elif selection == "Top":
            return [int(x.strip()) for x in self.top_cbushes.get().split(",") if x.strip()]
        elif selection == "BottomTop":
            bottom = [int(x.strip()) for x in self.bottom_cbushes.get().split(",") if x.strip()]
            top = [int(x.strip()) for x in self.top_cbushes.get().split(",") if x.strip()]
            return sorted(bottom + top)
        elif selection == "Custom":
            return [int(x.strip()) for x in self.custom_cbushes.get().split(",") if x.strip()]
        else:
            return []
    
    def estimate_ram(self, design_count):
        """Estimate RAM requirements"""
        ram_mb = design_count * 0.1  # 0.1 MB per design
        if ram_mb < 1024:
            return f"{ram_mb:.0f} MB"
        else:
            return f"{ram_mb/1024:.1f} GB"
    
    def generate_filename(self, study_type, param_range, cbush_sel, design_count):
        """Generate automatic filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean study type for filename
        study_clean = study_type.replace("-", "_")
        
        # Clean parameter range
        range_clean = param_range
        
        # Clean CBUSH selection
        if cbush_sel == "Custom":
            cbush_clean = f"CB{''.join(str(x) for x in self.get_selected_cbushes())}"
        else:
            cbush_clean = cbush_sel
        
        return f"{study_clean}_{range_clean}_{cbush_clean}_{design_count}_{timestamp}.csv"
    
    def select_output_directory(self):
        """Select output directory"""
        directory = filedialog.askdirectory(initialdir=self.output_directory.get())
        if directory:
            self.output_directory.set(directory)
            self.update_preview()
    
    def reset_cbush_layout(self):
        """Reset CBUSH layout to default"""
        self.bottom_cbushes.set("2,3,4")
        self.middle_cbushes.set("5,6,7")
        self.top_cbushes.set("8,9,10")
        self.update_preview()
    
    def set_every_other(self):
        """Set every other CBUSH"""
        self.custom_cbushes.set("2,4,6,8,10")
        self.cbush_selection.set("Custom")
        self.update_preview()
    
    def set_random_3(self):
        """Set 3 random CBUSHes"""
        cbushes = random.sample(range(2, 11), 3)
        self.custom_cbushes.set(",".join(map(str, sorted(cbushes))))
        self.cbush_selection.set("Custom")
        self.update_preview()
    
    def start_generation(self):
        """Start the generation process in a separate thread"""
        if self.generation_thread and self.generation_thread.is_alive():
            messagebox.showwarning("Generation in Progress", "A generation is already running. Please wait or cancel it first.")
            return
        
        # Validate configuration
        if not self.validate_configuration():
            return
        
        # Confirm large studies
        design_count, _, _, _ = self.calculate_study_metrics(
            self.study_type.get(), self.parameter_range.get(), self.cbush_selection.get())
        
        if design_count > 1000:  # Lowered threshold due to longer runtimes
            runtime_days = (design_count * 34.6) / 86400
            result = messagebox.askyesno(
                "Large Study Warning",
                f"This study will generate {design_count:,} designs.\n"
                f"Estimated HEEDS runtime: {runtime_days:.1f} days\n"
                f"(Based on real server performance: 5,000 designs = 2 days)\n\n"
                f"Are you sure you want to proceed?"
            )
            if not result:
                return
        
        # Reset cancellation flag
        self.cancel_generation = False
        
        # Start generation thread
        self.generation_thread = threading.Thread(target=self.generate_designs)
        self.generation_thread.daemon = True
        self.generation_thread.start()
    
    def cancel_generation_func(self):
        """Cancel the generation process"""
        self.cancel_generation = True
        self.status_label.config(text="Cancelling...")
    
    def validate_configuration(self):
        """Validate the current configuration"""
        try:
            # Check CBUSH selection
            cbushes = self.get_selected_cbushes()
            if not cbushes:
                messagebox.showerror("Configuration Error", "No CBUSHes selected!")
                return False
            
            if any(cbush < 2 or cbush > 10 for cbush in cbushes):
                messagebox.showerror("Configuration Error", "CBUSH numbers must be between 2 and 10!")
                return False
            
            # Check for S2/S3 requirements
            study_type = self.study_type.get()
            if study_type.startswith("S2") and len(cbushes) < 2:
                messagebox.showerror("Configuration Error", "S2 studies require at least 2 CBUSHes!")
                return False
            
            if study_type.startswith("S3") and len(cbushes) < 3:
                messagebox.showerror("Configuration Error", "S3 studies require at least 3 CBUSHes!")
                return False
            
            # Check output directory
            if not os.path.exists(self.output_directory.get()):
                messagebox.showerror("Configuration Error", "Output directory does not exist!")
                return False
            
            return True
            
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Validation failed: {e}")
            return False
    
    def generate_designs(self):
        """Main generation function (runs in separate thread)"""
        try:
            self.root.after(0, lambda: self.status_label.config(text="Generating..."))
            self.root.after(0, lambda: self.progress_var.set(0))
            
            study_type = self.study_type.get()
            param_range = self.parameter_range.get()
            
            # Get stiffness values
            if param_range == "Custom":
                try:
                    stiffness_values = [int(x.strip()) for x in self.custom_range_entry.get().split(",")]
                except:
                    stiffness_values = [1, 4, 7, 10]
            else:
                stiffness_values = self.stiffness_ranges[param_range]
            
            # Generate designs based on study type
            if study_type.startswith("S1"):
                designs = self.generate_s1_designs(stiffness_values, study_type == "S1-Complete")
            elif study_type.startswith("S2"):
                designs = self.generate_s2_designs(stiffness_values, study_type == "S2-Complete")
            elif study_type.startswith("S3"):
                designs = self.generate_s3_designs(stiffness_values, study_type == "S3-Complete")
            elif study_type == "Random":
                designs = self.generate_random_designs(stiffness_values)
            elif study_type == "Custom":
                designs = self.generate_custom_designs(stiffness_values)
            else:
                raise ValueError(f"Unknown study type: {study_type}")
            
            if not self.cancel_generation and designs:
                # Generate filename and save
                design_count = len(designs)
                filename = self.generate_filename(study_type, param_range, self.cbush_selection.get(), design_count)
                filepath = os.path.join(self.output_directory.get(), filename)
                
                self.root.after(0, lambda: self.status_label.config(text="Saving file..."))
                self.save_designs_to_csv(designs, filepath)
                
                # Generate summary if requested
                if self.export_summary.get():
                    self.generate_summary(designs, filepath)
                
                self.root.after(0, lambda: self.status_label.config(text=f"Complete! Generated {len(designs)} designs"))
                self.root.after(0, lambda: self.progress_var.set(100))
                
                messagebox.showinfo("Generation Complete", 
                                  f"Successfully generated {len(designs):,} designs!\n\nSaved to: {filepath}")
            elif self.cancel_generation:
                self.root.after(0, lambda: self.status_label.config(text="Cancelled"))
                self.root.after(0, lambda: self.progress_var.set(0))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text="Error"))
            self.root.after(0, lambda: self.progress_var.set(0))
            messagebox.showerror("Generation Error", f"An error occurred during generation:\n\n{str(e)}")
    
    def generate_s1_designs(self, stiffness_values, complete_mode):
        """Generate S1 single-CBUSH designs"""
        designs = []
        
        if complete_mode:
            cbush_range = range(2, 11)  # All CBUSHes 2-10
        else:
            cbush_range = self.get_selected_cbushes()
        
        total_combinations = len(cbush_range) * (len(stiffness_values) ** 3)
        current_combination = 0
        
        for cbush_num in cbush_range:
            if self.cancel_generation:
                break
                
            set_name = f"S1{cbush_num-1}"
            design_counter = 1
            
            for val1 in stiffness_values:
                for val2 in stiffness_values:
                    for val3 in stiffness_values:
                        if self.cancel_generation:
                            break
                            
                        design = self.create_base_design()
                        design['Name'] = f"{set_name}_{design_counter:03d}"
                        
                        # Set variable CBUSH (K4=K5=K6 constraint for sparse/limited mode)
                        if not complete_mode:
                            design[f'K4_{cbush_num}'] = val1
                            design[f'K5_{cbush_num}'] = val1
                            design[f'K6_{cbush_num}'] = val1
                        else:
                            design[f'K4_{cbush_num}'] = val1
                            design[f'K5_{cbush_num}'] = val2
                            design[f'K6_{cbush_num}'] = val3
                        
                        designs.append(design)
                        design_counter += 1
                        current_combination += 1
                        
                        # Update progress
                        if current_combination % 100 == 0:
                            progress = (current_combination / total_combinations) * 100
                            self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        return designs
    
    def generate_s2_designs(self, stiffness_values, complete_mode):
        """Generate S2 adjacent pairs designs"""
        designs = []
        
        if complete_mode:
            # All adjacent pairs
            adjacent_pairs = [(i, i+1) for i in range(2, 10)]
        else:
            # Based on selection
            cbushes = self.get_selected_cbushes()
            adjacent_pairs = [(cbushes[i], cbushes[i+1]) for i in range(len(cbushes)-1)]
        
        total_combinations = len(adjacent_pairs) * (len(stiffness_values) ** 2)
        current_combination = 0
        
        for pair_idx, (cbush1, cbush2) in enumerate(adjacent_pairs):
            if self.cancel_generation:
                break
                
            study_name = f"S2{chr(65+pair_idx)}"  # S2A, S2B, etc.
            design_counter = 1
            
            for val1 in stiffness_values:
                for val2 in stiffness_values:
                    if self.cancel_generation:
                        break
                        
                    design = self.create_base_design()
                    design['Name'] = f"{study_name}_{design_counter:03d}"
                    
                    # Set variable CBUSHes with K4=K5=K6 constraint
                    for axis in ['K4', 'K5', 'K6']:
                        design[f'{axis}_{cbush1}'] = val1
                        design[f'{axis}_{cbush2}'] = val2
                    
                    designs.append(design)
                    design_counter += 1
                    current_combination += 1
                    
                    # Update progress
                    if current_combination % 50 == 0:
                        progress = (current_combination / total_combinations) * 100
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        return designs
    
    def generate_s3_designs(self, stiffness_values, complete_mode):
        """Generate S3 multi-bolt designs"""
        designs = []
        
        if complete_mode:
            # All possible 3-CBUSH combinations from CBUSHes 2-10
            available_cbushes = list(range(2, 11))
            cbush_combinations = list(combinations(available_cbushes, 3))
        else:
            # Limited scenarios (current S3A, S3B, S3C)
            cbush_combinations = [(2,5,8), (3,6,9), (4,7,10)]
        
        total_combinations = len(cbush_combinations) * (len(stiffness_values) ** 3)
        current_combination = 0
        
        for scenario_idx, cbush_trio in enumerate(cbush_combinations):
            if self.cancel_generation:
                break
                
            if complete_mode:
                scenario_name = f"S3_{scenario_idx+1:03d}"
            else:
                scenario_name = f"S3{chr(65+scenario_idx)}"  # S3A, S3B, S3C
            
            design_counter = 1
            
            for val1 in stiffness_values:
                for val2 in stiffness_values:
                    for val3 in stiffness_values:
                        if self.cancel_generation:
                            break
                            
                        design = self.create_base_design()
                        if complete_mode:
                            design['Name'] = f"{scenario_name}_{design_counter:05d}"
                        else:
                            design['Name'] = f"{scenario_name}_{design_counter:03d}"
                        
                        # Set variable CBUSHes with K4=K5=K6 constraint
                        cbush1, cbush2, cbush3 = cbush_trio
                        for axis in ['K4', 'K5', 'K6']:
                            design[f'{axis}_{cbush1}'] = val1
                            design[f'{axis}_{cbush2}'] = val2
                            design[f'{axis}_{cbush3}'] = val3
                        
                        designs.append(design)
                        design_counter += 1
                        current_combination += 1
                        
                        # Update progress
                        if current_combination % 100 == 0:
                            progress = (current_combination / total_combinations) * 100
                            self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        return designs
    
    def generate_random_designs(self, stiffness_values):
        """Generate random training data"""
        designs = []
        num_designs = self.random_designs.get()
        random_type = self.random_type.get()
        cbush_selection = self.get_selected_cbushes()
        
        # Define categories for balanced random
        loose = [1, 2, 3, 4]
        medium = [5, 6, 7, 8]
        tight = [9, 10, 11]
        
        for i in range(num_designs):
            if self.cancel_generation:
                break
                
            design = self.create_base_design()
            design['Name'] = f"RAND_{i+1:05d}"
            
            for cbush in cbush_selection:
                if random_type == "Pure Random":
                    random_stiffness = random.choice(stiffness_values)
                elif random_type == "Balanced Random":
                    category = random.choice([loose, medium, tight])
                    available = [x for x in category if x in stiffness_values]
                    random_stiffness = random.choice(available) if available else random.choice(stiffness_values)
                elif random_type == "Failure-Focused":
                    # Higher probability of loose bolts
                    categories = [loose, medium, tight]
                    weights = [0.5, 0.3, 0.2]
                    category = random.choices(categories, weights=weights)[0]
                    available = [x for x in category if x in stiffness_values]
                    random_stiffness = random.choice(available) if available else random.choice(stiffness_values)
                else:
                    random_stiffness = random.choice(stiffness_values)
                
                # Apply K4=K5=K6 constraint
                for axis in ['K4', 'K5', 'K6']:
                    design[f'{axis}_{cbush}'] = random_stiffness
            
            designs.append(design)
            
            # Update progress
            if (i + 1) % 100 == 0:
                progress = ((i + 1) / num_designs) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        return designs
    
    def generate_custom_designs(self, stiffness_values):
        """Generate custom combination designs"""
        designs = []
        cbushes = self.get_selected_cbushes()
        
        if len(cbushes) == 1:
            total_combinations = len(stiffness_values)
            current_combination = 0
            
            for val in stiffness_values:
                if self.cancel_generation:
                    break
                    
                design = self.create_base_design()
                design['Name'] = f"CUSTOM_{current_combination+1:03d}"
                
                for axis in ['K4', 'K5', 'K6']:
                    design[f'{axis}_{cbushes[0]}'] = val
                
                designs.append(design)
                current_combination += 1
                
                progress = (current_combination / total_combinations) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        elif len(cbushes) == 2:
            total_combinations = len(stiffness_values) ** 2
            current_combination = 0
            
            for val1 in stiffness_values:
                for val2 in stiffness_values:
                    if self.cancel_generation:
                        break
                        
                    design = self.create_base_design()
                    design['Name'] = f"CUSTOM_{current_combination+1:03d}"
                    
                    for axis in ['K4', 'K5', 'K6']:
                        design[f'{axis}_{cbushes[0]}'] = val1
                        design[f'{axis}_{cbushes[1]}'] = val2
                    
                    designs.append(design)
                    current_combination += 1
                    
                    if current_combination % 10 == 0:
                        progress = (current_combination / total_combinations) * 100
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        elif len(cbushes) >= 3:
            # Use first 3 CBUSHes for 3D combination
            total_combinations = len(stiffness_values) ** 3
            current_combination = 0
            
            for val1 in stiffness_values:
                for val2 in stiffness_values:
                    for val3 in stiffness_values:
                        if self.cancel_generation:
                            break
                            
                        design = self.create_base_design()
                        design['Name'] = f"CUSTOM_{current_combination+1:03d}"
                        
                        for axis in ['K4', 'K5', 'K6']:
                            design[f'{axis}_{cbushes[0]}'] = val1
                            design[f'{axis}_{cbushes[1]}'] = val2
                            design[f'{axis}_{cbushes[2]}'] = val3
                        
                        designs.append(design)
                        current_combination += 1
                        
                        if current_combination % 50 == 0:
                            progress = (current_combination / total_combinations) * 100
                            self.root.after(0, lambda p=progress: self.progress_var.set(p))
        
        return designs
    
    def create_base_design(self):
        """Create base design template"""
        design = {
            'Map': 'FALSE',
            'K4_1': 5, 'K5_1': 5, 'K6_1': 5,  # Always driving
            'Show Resp.': 'FALSE',
            'Comment': ''
        }
        
        # Initialize all K values to baseline (9)
        for i in range(2, 11):
            for axis in ['K4', 'K5', 'K6']:
                design[f'{axis}_{i}'] = 9
        
        return design
    
    def save_designs_to_csv(self, designs, filepath):
        """Save designs to CSV file"""
        if not designs:
            return
        
        # Column order as specified in documentation
        columns = ['Name', 'Map', 'K4_1', 'K5_1', 'K6_1', 'K4_2', 'K5_2', 'K6_2', 
                  'K4_3', 'K5_3', 'K6_3', 'K4_4', 'K5_4', 'K6_4', 'K4_5', 'K5_5', 'K6_5',
                  'K4_6', 'K5_6', 'K6_6', 'K4_7', 'K5_7', 'K6_7', 'K4_8', 'K5_8', 'K6_8',
                  'K4_9', 'K5_9', 'K6_9', 'K4_10', 'K5_10', 'K6_10', 'Show Resp.', 'Comment']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            batch_size = self.batch_size.get()
            for i in range(0, len(designs), batch_size):
                if self.cancel_generation:
                    break
                    
                batch = designs[i:i+batch_size]
                writer.writerows(batch)
                
                # Small delay to prevent UI freezing
                time.sleep(0.001)
    
    def generate_summary(self, designs, filepath):
        """Generate summary file"""
        summary_path = filepath.replace('.csv', '_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write(f"Spacecraft Bolt Study Generation Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Study Type: {self.study_type.get()}\n")
            f.write(f"Parameter Range: {self.parameter_range.get()}\n")
            f.write(f"CBUSH Selection: {self.cbush_selection.get()}\n")
            f.write(f"Total Designs: {len(designs):,}\n")
            f.write(f"Output File: {os.path.basename(filepath)}\n\n")
            
            # Configuration details
            f.write(f"Configuration Details:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"CBUSHes Varied: {self.get_selected_cbushes()}\n")
            
            if self.parameter_range.get() == "Custom":
                f.write(f"Custom Range: {self.custom_range_entry.get()}\n")
            else:
                f.write(f"Stiffness Values: {self.stiffness_ranges[self.parameter_range.get()]}\n")
            
            if self.study_type.get() == "Random":
                f.write(f"Random Type: {self.random_type.get()}\n")
                f.write(f"Number of Designs: {self.random_designs.get()}\n")
            
            # Sample designs
            f.write(f"\nFirst 5 Designs:\n")
            f.write(f"-" * 15 + "\n")
            for i, design in enumerate(designs[:5]):
                f.write(f"{design['Name']}: ")
                varied_cbushes = []
                for cbush in range(2, 11):
                    if design[f'K4_{cbush}'] != 9:  # Not baseline
                        varied_cbushes.append(f"CB{cbush}={design[f'K4_{cbush}']}")
                f.write(", ".join(varied_cbushes) + "\n")

def main():
    root = tk.Tk()
    app = SpacecraftBoltStudyGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
