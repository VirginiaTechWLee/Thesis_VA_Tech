import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import threading
import os
from pathlib import Path
import time
import warnings
import gc
import psutil
from collections import defaultdict
warnings.filterwarnings('ignore')

# Import the updated modules (these should be in the same directory)
try:
    from feature_matrix_builder import BoltAnomalyFeatureExtractor
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    print("⚠️  feature_matrix_builder not found - some features will be disabled")
    FEATURE_EXTRACTOR_AVAILABLE = False

try:
    from multi_cbush_ml_pipeline import MultiCBUSHMLPipeline
    ML_PIPELINE_AVAILABLE = True
except ImportError:
    print("⚠️  multi_cbush_ml_pipeline not found - ML features will be disabled")
    ML_PIPELINE_AVAILABLE = False

try:
    from comprehensive_scoring_system import ComprehensiveMultiLabelScoring
    SCORING_SYSTEM_AVAILABLE = True
except ImportError:
    print("⚠️  comprehensive_scoring_system not found - scoring features will be disabled")
    SCORING_SYSTEM_AVAILABLE = False

class FlexibleCBUSHProcessor:
    """
    Enhanced processor for massive CBUSH bolt anomaly detection datasets.
    Supports flexible directory structures, auto-detection, failed design handling,
    and massive datasets (11K-111K designs).
    """
    
    def __init__(self):
        """Initialize the flexible processor."""
        # Main configuration
        self.heeds_directory = None
        self.processed_data = {}
        self.processing_stats = {}
        
        # Processing control
        self.is_paused = False
        self.should_stop = False
        self.current_operation = None
        
        # Memory management
        self.max_memory_gb = 16.0
        self.batch_size = 500
        self.chunk_processing = True
        
        # Error handling
        self.skip_failed_designs = True
        self.log_failed_designs = True
        self.max_failure_rate = 0.1  # Stop if >10% designs fail
        
    def set_heeds_directory(self, directory_path):
        """Set the main HEEDS output directory."""
        self.heeds_directory = directory_path
        print(f"✅ HEEDS directory set: {directory_path}")
    
    def validate_heeds_directory(self):
        """Validate HEEDS directory structure."""
        if not self.heeds_directory:
            raise ValueError("HEEDS directory not set")
        
        if not os.path.exists(self.heeds_directory):
            raise ValueError(f"HEEDS directory does not exist: {self.heeds_directory}")
        
        post_0_dir = os.path.join(self.heeds_directory, "POST_0")
        if not os.path.exists(post_0_dir):
            raise ValueError(f"POST_0 directory not found in {self.heeds_directory}")
        
        return True
    
    def scan_available_designs(self, post_0_directory=None):
        """
        Scan POST_0 directory for available designs and detect patterns.
        
        Args:
            post_0_directory: Path to POST_0 directory (optional, uses self.heeds_directory if not provided)
            
        Returns:
            dict: Available designs information
        """
        if post_0_directory is None:
            if not self.heeds_directory:
                raise ValueError("No HEEDS directory set")
            post_0_directory = os.path.join(self.heeds_directory, "POST_0")
        
        print(f"🔍 Scanning {post_0_directory} for available designs...")
        
        available_designs = {}
        failed_designs = {}
        design_dirs = []
        
        # Find all Design directories
        for item in os.listdir(post_0_directory):
            if item.startswith("Design") and os.path.isdir(os.path.join(post_0_directory, item)):
                try:
                    design_num = int(item.replace("Design", ""))
                    design_path = os.path.join(post_0_directory, item)
                    design_dirs.append((design_num, design_path))
                except ValueError:
                    continue
        
        design_dirs.sort()  # Sort by design number
        
        print(f"📊 Found {len(design_dirs)} Design directories")
        
        # Analyze each design directory
        for design_num, design_path in design_dirs:
            analysis_dir = os.path.join(design_path, "Analysis_1")
            
            if os.path.exists(analysis_dir):
                accel_file = os.path.join(analysis_dir, "acceleration_results.csv")
                disp_file = os.path.join(analysis_dir, "displacement_results.csv")
                
                if os.path.exists(accel_file) and os.path.exists(disp_file):
                    # Design is available
                    available_designs[design_num] = {
                        'design_path': design_path,
                        'analysis_path': analysis_dir,
                        'accel_file': accel_file,
                        'disp_file': disp_file,
                        'file_size_mb': self._get_total_file_size([accel_file, disp_file])
                    }
                else:
                    # Design failed - missing CSV files
                    missing_files = []
                    if not os.path.exists(accel_file):
                        missing_files.append("acceleration_results.csv")
                    if not os.path.exists(disp_file):
                        missing_files.append("displacement_results.csv")
                    
                    failed_designs[design_num] = f"Missing files: {', '.join(missing_files)}"
            else:
                # Design failed - missing Analysis_1 directory
                failed_designs[design_num] = "Missing Analysis_1 directory"
        
        total_designs = len(available_designs) + len(failed_designs)
        success_rate = len(available_designs) / total_designs if total_designs > 0 else 0
        
        print(f"✅ Scan complete: {len(available_designs)} available, {len(failed_designs)} failed ({success_rate:.1%} success)")
        
        return {
            'available_designs': available_designs,
            'failed_designs': failed_designs,
            'success_rate': success_rate,
            'total_scanned': total_designs
        }
    
    def _get_total_file_size(self, file_paths):
        """Get total size of files in MB."""
        total_size = 0
        for path in file_paths:
            if os.path.exists(path):
                total_size += os.path.getsize(path)
        return total_size / (1024 * 1024)
    
    def auto_detect_study_type(self, available_designs, study_name_hint=None):
        """
        Auto-detect study type based on available designs and patterns.
        
        Args:
            available_designs: Dictionary of available designs
            study_name_hint: Optional hint from CSV filename or user input
            
        Returns:
            dict: Study type information
        """
        design_numbers = sorted(available_designs.keys())
        total_designs = len(design_numbers)
        
        if total_designs == 0:
            return {'type': 'empty', 'description': 'No designs available'}
        
        min_design = min(design_numbers)
        max_design = max(design_numbers)
        design_range = max_design - min_design + 1
        
        # Pattern detection logic
        study_info = {
            'design_numbers': design_numbers,
            'total_designs': total_designs,
            'design_range': (min_design, max_design),
            'density': total_designs / design_range if design_range > 0 else 0,
            'name': f'Auto_Study_{min_design}_{max_design}'
        }
        
        # Study type classification
        if total_designs == 64:
            study_info.update({
                'type': 'single_cbush_complete',
                'description': 'Single-CBUSH Complete Study (64 designs)',
                'likely_cbush_count': 1,
                'expected_format': 'S1-Complete'
            })
        elif total_designs == 16:
            study_info.update({
                'type': 'adjacent_pairs',
                'description': 'Adjacent Pairs Study (16 designs)',
                'likely_cbush_count': 2,
                'expected_format': 'S2-Limited'
            })
        elif total_designs in [576, 11979]:
            study_info.update({
                'type': 'single_cbush_batch',
                'description': f'Single-CBUSH Batch Study ({total_designs} designs)',
                'likely_cbush_count': 1,
                'expected_format': 'S1-Complete' if total_designs > 1000 else 'S1-Limited'
            })
        elif total_designs in [128, 968]:
            study_info.update({
                'type': 'adjacent_pairs_batch',
                'description': f'Adjacent Pairs Batch Study ({total_designs} designs)',
                'likely_cbush_count': 2,
                'expected_format': 'S2-Complete' if total_designs > 500 else 'S2-Limited'
            })
        elif total_designs in [192, 111804]:
            study_info.update({
                'type': 'multi_bolt_batch',
                'description': f'Multi-Bolt Batch Study ({total_designs} designs)',
                'likely_cbush_count': 3,
                'expected_format': 'S3-Complete' if total_designs > 10000 else 'S3-Limited'
            })
        elif 1000 <= total_designs <= 20000:
            study_info.update({
                'type': 'random_study',
                'description': f'Random Study ({total_designs} designs)',
                'likely_cbush_count': 'variable',
                'expected_format': 'Random'
            })
        elif total_designs > 20000:
            study_info.update({
                'type': 'massive_study',
                'description': f'Massive Study ({total_designs} designs)',
                'likely_cbush_count': 'variable',
                'expected_format': 'S3-Complete or Large Random'
            })
        else:
            study_info.update({
                'type': 'custom_study',
                'description': f'Custom Study ({total_designs} designs)',
                'likely_cbush_count': 'unknown',
                'expected_format': 'Custom'
            })
        
        return study_info
    
    def process_study_batch(self, study_info, progress_callback=None):
        """
        Process a study batch with the detected designs.
        
        Args:
            study_info: Study information from auto-detection
            progress_callback: Optional progress callback function
            
        Returns:
            pandas.DataFrame: Processed study data
        """
        if not FEATURE_EXTRACTOR_AVAILABLE:
            raise ImportError("Feature extractor not available - cannot process studies")
        
        design_numbers = study_info.get('design_numbers', [])
        if not design_numbers:
            print("❌ No design numbers available for processing")
            return None
        
        print(f"\n🔄 Processing study: {study_info.get('description', 'Unknown Study')}")
        print(f"   Designs to process: {len(design_numbers):,}")
        
        # Initialize extractor
        extractor = BoltAnomalyFeatureExtractor(self.heeds_directory)
        
        # Process in batches
        all_features = []
        batch_size = min(self.batch_size, len(design_numbers))
        
        for i in range(0, len(design_numbers), batch_size):
            batch_end = min(i + batch_size, len(design_numbers))
            batch_designs = design_numbers[i:batch_end]
            
            if progress_callback:
                progress_callback(i, len(design_numbers), f"Processing batch {i//batch_size + 1}")
            
            print(f"   Processing batch {i//batch_size + 1}: designs {batch_designs[0]}-{batch_designs[-1]}")
            
            # Process batch
            batch_features = extractor.process_design_batch_enhanced(
                batch_designs, study_info, progress_callback
            )
            
            if batch_features:
                all_features.extend(batch_features)
        
        if all_features:
            # Convert to DataFrame
            study_df = pd.DataFrame(all_features)
            
            # Store processed data
            study_name = study_info.get('name', f"Study_{len(self.processed_data) + 1}")
            self.store_processed_study(study_name, all_features)
            
            print(f"✅ Study processing complete: {len(study_df):,} designs processed")
            return study_df
        else:
            print("❌ No features extracted from study")
            return None
    
    def store_processed_study(self, study_name, study_data):
        """Store processed study data with memory management."""
        if study_data:
            # Convert to DataFrame if not already
            if isinstance(study_data, list):
                study_df = pd.DataFrame(study_data)
            else:
                study_df = study_data
            
            # Store with metadata
            self.processed_data[study_name] = {
                'data': study_df,
                'design_count': len(study_df),
                'feature_count': len([col for col in study_df.columns if col.startswith(('ACCE_', 'DISP_'))]),
                'memory_mb': study_df.memory_usage(deep=True).sum() / 1024 / 1024,
                'processing_time': time.time(),
                'quality_score': self._calculate_quality_score(study_df)
            }
            
            print(f"💾 Stored study {study_name}: {len(study_df)} designs, {self.processed_data[study_name]['memory_mb']:.1f} MB")
            
            # Memory management
            self._manage_memory()
    
    def _calculate_quality_score(self, study_df):
        """Calculate a quality score for the processed study."""
        try:
            # Check for missing values
            missing_ratio = study_df.isnull().sum().sum() / (study_df.shape[0] * study_df.shape[1])
            
            # Check for feature variance (avoid constant features)
            feature_cols = [col for col in study_df.columns if col.startswith(('ACCE_', 'DISP_'))]
            if feature_cols:
                feature_variance = study_df[feature_cols].var().mean()
                variance_score = min(1.0, feature_variance / 1000)  # Normalize
            else:
                variance_score = 0.0
            
            # Check for label consistency
            cbush_cols = [col for col in study_df.columns if col.startswith('cbush_') and col.endswith('_loose')]
            if cbush_cols:
                label_consistency = 1 - (study_df[cbush_cols].sum(axis=1) == 0).mean()  # Avoid all-zero labels
            else:
                label_consistency = 0.0
            
            # Combined quality score
            quality_score = (1 - missing_ratio) * 0.4 + variance_score * 0.3 + label_consistency * 0.3
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception:
            return 0.5  # Default score if calculation fails
    
    def _manage_memory(self):
        """Manage memory usage by cleaning up data if necessary."""
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_gb * 1024:
                print(f"⚠️  High memory usage ({memory_mb:.0f} MB), cleaning up...")
                
                # Force garbage collection
                gc.collect()
                
                # If still high, could implement more aggressive cleanup
                memory_mb_after = process.memory_info().rss / 1024 / 1024
                print(f"   Memory after cleanup: {memory_mb_after:.0f} MB")
                
        except Exception:
            pass  # Memory management is optional
    
    def create_master_matrix(self, strategy="smart_combined", max_designs=50000):
        """Create master matrix from processed studies."""
        if not self.processed_data:
            print("❌ No processed data available")
            return None
        
        all_dataframes = []
        
        for study_name, study_info in self.processed_data.items():
            all_dataframes.append(study_info['data'])
            print(f"   Including {study_name}: {len(study_info['data']):,} designs")
        
        if all_dataframes:
            master_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"✅ Master matrix created: {len(master_df):,} designs")
            return master_df
        else:
            return None


class MassiveDatasetBoltDetectionGUI:
    """
    Enhanced GUI for massive dataset multi-CBUSH bolt anomaly detection.
    Supports 11K-111K+ designs with single HEEDS directory input.
    """
    
    def __init__(self, root):
        """Initialize the massive dataset GUI."""
        self.root = root
        self.root.title("Multi-CBUSH Bolt Detection System v3.0 - Massive Dataset Edition")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize backend components
        self.processor = FlexibleCBUSHProcessor()
        if ML_PIPELINE_AVAILABLE:
            self.ml_pipeline = MultiCBUSHMLPipeline()
        else:
            self.ml_pipeline = None
        if SCORING_SYSTEM_AVAILABLE:
            self.scoring_system = ComprehensiveMultiLabelScoring()
        else:
            self.scoring_system = None
        
        self.current_data = None
        self.scan_results = None
        self.study_info = None
        
        # Threading
        self.processing_thread = None
        self.scanning_thread = None
        
        # Create GUI
        self.create_widgets()
        
        # Initial status
        self.log_message("🚀 Massive Dataset Bolt Detection System v3.0 Ready!")
        self.log_message("📊 Supports: 11K-111K+ designs | Single HEEDS Directory | Auto-Detection")
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_setup_tab()
        self.create_scan_tab()
        self.create_processing_tab()
        self.create_matrix_tab()
        if ML_PIPELINE_AVAILABLE:
            self.create_ml_tab()
        if SCORING_SYSTEM_AVAILABLE:
            self.create_scoring_tab()
        self.create_results_tab()
    
    def create_setup_tab(self):
        """Create enhanced setup tab for massive datasets."""
        setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(setup_frame, text="🔧 Setup")
        
        # Title
        title_label = tk.Label(setup_frame, text="Massive Dataset Bolt Detection System v3.0", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(setup_frame, text="11K-111K+ Designs | Auto-Detection | Memory Optimized", 
                                 font=('Arial', 14), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=5)
        
        # HEEDS directory configuration
        config_frame = ttk.LabelFrame(setup_frame, text="HEEDS Output Directory Configuration")
        config_frame.pack(fill='x', padx=20, pady=20)
        
        # Single directory input (NEW - replaces 3 separate folders)
        dir_frame = tk.Frame(config_frame)
        dir_frame.pack(fill='x', padx=10, pady=15)
        
        tk.Label(dir_frame, text="HEEDS Output Directory (containing POST_0 folder):", 
                font=('Arial', 12, 'bold')).pack(anchor='w')
        
        path_input_frame = tk.Frame(dir_frame)
        path_input_frame.pack(fill='x', pady=10)
        
        self.heeds_path_var = tk.StringVar()
        path_entry = tk.Entry(path_input_frame, textvariable=self.heeds_path_var, 
                             width=100, font=('Arial', 10))
        path_entry.pack(side='left', fill='x', expand=True)
        
        tk.Button(path_input_frame, text="📁 Browse", 
                 command=self.browse_heeds_directory,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(side='right', padx=(10,0))
        
        # Directory validation
        validate_frame = tk.Frame(config_frame)
        validate_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(validate_frame, text="✅ Validate Directory", 
                 command=self.validate_directory,
                 bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=20).pack(side='left')
        
        self.validation_status = tk.Label(validate_frame, text="No directory selected", 
                                         font=('Arial', 11), fg='#7f8c8d')
        self.validation_status.pack(side='left', padx=20)
        
        # System status
        self.status_frame = ttk.LabelFrame(setup_frame, text="System Status & Log")
        self.status_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.status_text = scrolledtext.ScrolledText(self.status_frame, height=25, width=120,
                                                    font=('Consolas', 9))
        self.status_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_scan_tab(self):
        """Create scan and detection tab."""
        scan_frame = ttk.Frame(self.notebook)
        self.notebook.add(scan_frame, text="🔍 Scan & Detect")
        
        # Scan controls
        scan_control_frame = ttk.LabelFrame(scan_frame, text="Design Scanning & Auto-Detection")
        scan_control_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(scan_control_frame, text="Scan HEEDS directory for available designs and auto-detect study patterns:", 
                font=('Arial', 12)).pack(anchor='w', padx=10, pady=10)
        
        scan_button_frame = tk.Frame(scan_control_frame)
        scan_button_frame.pack(fill='x', padx=10, pady=15)
        
        tk.Button(scan_button_frame, text="🔍 Scan Available Designs", 
                 command=self.scan_designs,
                 bg='#3498db', fg='white', font=('Arial', 14, 'bold'),
                 height=2, width=25).pack(side='left')
        
        # Scan progress
        self.scan_progress_frame = tk.Frame(scan_control_frame)
        self.scan_progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.scan_progress_var = tk.DoubleVar()
        self.scan_progress_bar = ttk.Progressbar(self.scan_progress_frame, variable=self.scan_progress_var, 
                                               maximum=100, length=500, mode='indeterminate')
        self.scan_progress_bar.pack(pady=5)
        
        self.scan_status_label = tk.Label(self.scan_progress_frame, text="Ready to scan", 
                                         font=('Arial', 10), fg='#7f8c8d')
        self.scan_status_label.pack()
        
        # Scan results display
        results_frame = ttk.LabelFrame(scan_frame, text="Scan Results & Study Detection")
        results_frame.pack(fill='both', expand=True, padx=20, pady=15)
        
        # Results tree
        columns = ('Property', 'Value', 'Description')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == 'Property':
                self.results_tree.column(col, width=200)
            elif col == 'Value':
                self.results_tree.column(col, width=150)
            else:
                self.results_tree.column(col, width=400)
        
        # Add scrollbars
        tree_frame = tk.Frame(results_frame)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar_v = ttk.Scrollbar(tree_frame, orient='vertical', command=self.results_tree.yview)
        scrollbar_h = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar_v.pack(side='right', fill='y')
        scrollbar_h.pack(side='bottom', fill='x')
    
    def create_processing_tab(self):
        """Create data processing tab."""
        processing_frame = ttk.Frame(self.notebook)
        self.notebook.add(processing_frame, text="⚙️ Process Data")
        
        # Processing controls
        process_control_frame = ttk.LabelFrame(processing_frame, text="Study Processing")
        process_control_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(process_control_frame, text="Process detected study with feature extraction:", 
                font=('Arial', 12)).pack(anchor='w', padx=10, pady=10)
        
        process_button_frame = tk.Frame(process_control_frame)
        process_button_frame.pack(fill='x', padx=10, pady=15)
        
        tk.Button(process_button_frame, text="⚙️ Process Study", 
                 command=self.process_study,
                 bg='#27ae60', fg='white', font=('Arial', 14, 'bold'),
                 height=2, width=20).pack(side='left')
        
        tk.Button(process_button_frame, text="⏸️ Pause", 
                 command=self.pause_processing,
                 bg='#f39c12', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=10).pack(side='left', padx=10)
        
        tk.Button(process_button_frame, text="⏹️ Stop", 
                 command=self.stop_processing,
                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=10).pack(side='left', padx=5)
        
        # Processing progress
        progress_frame = tk.Frame(process_control_frame)
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(progress_frame, text="Processing Progress:", font=('Arial', 11, 'bold')).pack(anchor='w')
        
        self.process_progress_var = tk.DoubleVar()
        self.process_progress_bar = ttk.Progressbar(progress_frame, variable=self.process_progress_var, 
                                                  maximum=100, length=500)
        self.process_progress_bar.pack(pady=5)
        
        self.process_status_label = tk.Label(progress_frame, text="Ready to process", 
                                           font=('Arial', 10), fg='#7f8c8d')
        self.process_status_label.pack()
        
        # Processing stats
        stats_frame = ttk.LabelFrame(processing_frame, text="Processing Statistics")
        stats_frame.pack(fill='both', expand=True, padx=20, pady=15)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=20, width=120,
                                                   font=('Consolas', 9))
        self.stats_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_matrix_tab(self):
        """Create matrix creation tab."""
        matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(matrix_frame, text="📊 Create Matrix")
        
        # Matrix creation controls
        matrix_control_frame = ttk.LabelFrame(matrix_frame, text="Master Matrix Creation")
        matrix_control_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(matrix_control_frame, text="Create master feature matrix from processed data:", 
                font=('Arial', 12)).pack(anchor='w', padx=10, pady=10)
        
        # Matrix options
        options_frame = tk.Frame(matrix_control_frame)
        options_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(options_frame, text="Matrix Strategy:", font=('Arial', 11, 'bold')).pack(anchor='w')
        
        self.matrix_strategy = tk.StringVar(value="smart_combined")
        strategies = [
            ("Smart Combined (Recommended)", "smart_combined"),
            ("Complete Combined", "complete_combined"),
            ("Size Limited", "size_limited")
        ]
        
        for text, value in strategies:
            tk.Radiobutton(options_frame, text=text, variable=self.matrix_strategy, value=value,
                          font=('Arial', 10)).pack(anchor='w', padx=20, pady=2)
        
        # Create matrix button
        create_button_frame = tk.Frame(matrix_control_frame)
        create_button_frame.pack(fill='x', padx=10, pady=15)
        
        tk.Button(create_button_frame, text="📊 Create Master Matrix", 
                 command=self.create_matrix,
                 bg='#9b59b6', fg='white', font=('Arial', 14, 'bold'),
                 height=2, width=25).pack(side='left')
        
        tk.Button(create_button_frame, text="💾 Save Matrix", 
                 command=self.save_matrix,
                 bg='#16a085', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=15).pack(side='left', padx=10)
        
        # Matrix information
        self.matrix_info_text = scrolledtext.ScrolledText(matrix_frame, height=25, width=120,
                                                         font=('Consolas', 9))
        self.matrix_info_text.pack(fill='both', expand=True, padx=20, pady=15)
    
    def create_ml_tab(self):
        """Create machine learning tab."""
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="🤖 Machine Learning")
        
        # ML controls
        ml_control_frame = ttk.LabelFrame(ml_frame, text="Model Training (Massive Dataset Optimized)")
        ml_control_frame.pack(fill='x', padx=20, pady=15)
        
        # Strategy selection
        tk.Label(ml_control_frame, text="Select ML Strategy:", font=('Arial', 12, 'bold')).pack(anchor='w', padx=10, pady=10)
        
        self.ml_strategy = tk.StringVar(value="random_forest")
        strategies = [
            ("Random Forest (Massive Dataset)", "random_forest"),
            ("Neural Network (Batch Training)", "neural_network"),
            ("Incremental Ensemble", "ensemble")
        ]
        
        for text, value in strategies:
            tk.Radiobutton(ml_control_frame, text=text, variable=self.ml_strategy, value=value,
                          font=('Arial', 11)).pack(anchor='w', padx=20, pady=3)
        
        # Train button
        train_button_frame = tk.Frame(ml_control_frame)
        train_button_frame.pack(fill='x', padx=10, pady=15)
        
        tk.Button(train_button_frame, text="🤖 Train Model", 
                 command=self.train_model,
                 bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'),
                 height=2, width=20).pack()
        
        # Training progress
        training_progress_frame = tk.Frame(ml_control_frame)
        training_progress_frame.pack(fill='x', padx=10, pady=10)
        
        self.training_progress_var = tk.DoubleVar()
        self.training_progress_bar = ttk.Progressbar(training_progress_frame, variable=self.training_progress_var, 
                                                   maximum=100, length=500, mode='indeterminate')
        self.training_progress_bar.pack(pady=5)
        
        self.training_status_label = tk.Label(training_progress_frame, text="No model trained", 
                                            font=('Arial', 10), fg='#7f8c8d')
        self.training_status_label.pack()
        
        # Results display
        self.ml_results_text = scrolledtext.ScrolledText(ml_frame, height=20, width=120,
                                                        font=('Consolas', 9))
        self.ml_results_text.pack(fill='both', expand=True, padx=20, pady=15)
    
    def create_scoring_tab(self):
        """Create comprehensive scoring tab."""
        scoring_frame = ttk.Frame(self.notebook)
        self.notebook.add(scoring_frame, text="📈 Scoring")
        
        # Scoring controls
        scoring_control_frame = ttk.LabelFrame(scoring_frame, text="Comprehensive Model Scoring")
        scoring_control_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Button(scoring_control_frame, text="📈 Calculate Comprehensive Scores", 
                 command=self.calculate_scores,
                 bg='#3498db', fg='white', font=('Arial', 14, 'bold'),
                 height=2, width=30).pack(pady=15)
        
        # Scoring results
        self.scoring_results_text = scrolledtext.ScrolledText(scoring_frame, height=25, width=120,
                                                             font=('Consolas', 9))
        self.scoring_results_text.pack(fill='both', expand=True, padx=20, pady=15)
    
    def create_results_tab(self):
        """Create results and visualization tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📋 Results")
        
        # Export controls
        export_frame = ttk.LabelFrame(results_frame, text="Export Results")
        export_frame.pack(fill='x', padx=20, pady=15)
        
        export_buttons = [
            ("💾 Export Matrix", self.export_matrix, '#16a085'),
            ("📊 Export Results", self.export_results, '#9b59b6'),
            ("🤖 Export Models", self.export_models, '#e74c3c')
        ]
        
        button_frame = tk.Frame(export_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        for text, command, color in export_buttons:
            tk.Button(button_frame, text=text, command=command, bg=color, fg='white',
                     font=('Arial', 12, 'bold')).pack(side='left', padx=10, pady=5)
        
        # Results summary
        self.results_text = scrolledtext.ScrolledText(results_frame, height=25, width=120,
                                                     font=('Consolas', 9))
        self.results_text.pack(fill='both', expand=True, padx=20, pady=15)
    
    # Event handlers
    def browse_heeds_directory(self):
        """Browse for HEEDS output directory."""
        directory = filedialog.askdirectory(title="Select HEEDS Output Directory (containing POST_0)")
        if directory:
            self.heeds_path_var.set(directory)
            self.log_message(f"📁 Selected HEEDS directory: {directory}")
    
    def validate_directory(self):
        """Validate the selected HEEDS directory."""
        directory = self.heeds_path_var.get().strip()
        if not directory:
            messagebox.showerror("Error", "Please select a HEEDS directory first")
            return
        
        try:
            self.processor.set_heeds_directory(directory)
            self.processor.validate_heeds_directory()
            
            self.validation_status.config(text="✅ Directory validated", fg='#27ae60')
            self.log_message(f"✅ HEEDS directory validated: {directory}")
            messagebox.showinfo("Success", "HEEDS directory validated successfully!")
            
        except Exception as e:
            self.validation_status.config(text="❌ Validation failed", fg='#e74c3c')
            self.log_message(f"❌ Directory validation failed: {str(e)}")
            messagebox.showerror("Validation Error", str(e))
    
    def scan_designs(self):
        """Scan for available designs in background thread."""
        if not self.processor.heeds_directory:
            messagebox.showerror("Error", "Please set and validate HEEDS directory first")
            return
        
        # Start scanning in background
        self.scan_progress_bar.configure(mode='indeterminate')
        self.scan_progress_bar.start()
        self.scan_status_label.config(text="Scanning designs...")
        
        self.scanning_thread = threading.Thread(target=self._scan_designs_thread)
        self.scanning_thread.daemon = True
        self.scanning_thread.start()
    
    def _scan_designs_thread(self):
        """Scan designs in background thread."""
        try:
            self.log_message("🔍 Starting design scan...")
            
            # Scan available designs
            scan_results = self.processor.scan_available_designs()
            
            # Auto-detect study type
            available_designs = scan_results['available_designs']
            study_info = self.processor.auto_detect_study_type(available_designs)
            
            # Update UI on main thread
            self.root.after(0, self._scan_complete, scan_results, study_info)
            
        except Exception as e:
            error_msg = f"Scan error: {str(e)}"
            self.root.after(0, self._scan_failed, error_msg)
    
    def _scan_complete(self, scan_results, study_info):
        """Handle scan completion on main thread."""
        self.scan_progress_bar.stop()
        self.scan_status_label.config(text="Scan completed")
        
        self.scan_results = scan_results
        self.study_info = study_info
        
        # Update results tree
        self.results_tree.delete(*self.results_tree.get_children())
        
        # Add scan results
        self.results_tree.insert('', 'end', values=('Total Designs Found', f"{scan_results['total_scanned']:,}", 'Total Design directories found'))
        self.results_tree.insert('', 'end', values=('Available Designs', f"{len(scan_results['available_designs']):,}", 'Designs with complete data files'))
        self.results_tree.insert('', 'end', values=('Failed Designs', f"{len(scan_results['failed_designs']):,}", 'Designs with missing/corrupted files'))
        self.results_tree.insert('', 'end', values=('Success Rate', f"{scan_results['success_rate']:.1%}", 'Percentage of successful designs'))
        
        # Add study detection results
        self.results_tree.insert('', 'end', values=('', '', ''))  # Separator
        self.results_tree.insert('', 'end', values=('Study Type', study_info.get('type', 'unknown'), 'Auto-detected study classification'))
        self.results_tree.insert('', 'end', values=('Study Description', study_info.get('description', 'Unknown'), 'Detailed study description'))
        self.results_tree.insert('', 'end', values=('Expected Format', study_info.get('expected_format', 'Unknown'), 'Expected study format'))
        self.results_tree.insert('', 'end', values=('Design Range', f"{study_info['design_range'][0]}-{study_info['design_range'][1]}", 'Range of design numbers'))
        self.results_tree.insert('', 'end', values=('CBUSH Count', str(study_info.get('likely_cbush_count', 'Unknown')), 'Expected number of varied CBUSHes'))
        
        self.log_message(f"✅ Scan completed: {len(scan_results['available_designs']):,} designs available")
        self.log_message(f"📊 Study detected: {study_info.get('description', 'Unknown Study')}")
        
        messagebox.showinfo("Scan Complete", 
                           f"Scan completed successfully!\n\n"
                           f"Available: {len(scan_results['available_designs']):,} designs\n"
                           f"Failed: {len(scan_results['failed_designs']):,} designs\n"
                           f"Study Type: {study_info.get('description', 'Unknown')}")
    
    def _scan_failed(self, error_msg):
        """Handle scan failure on main thread."""
        self.scan_progress_bar.stop()
        self.scan_status_label.config(text="Scan failed")
        self.log_message(f"❌ {error_msg}")
        messagebox.showerror("Scan Failed", error_msg)
    
    def process_study(self):
        """Process the detected study."""
        if not self.study_info:
            messagebox.showerror("Error", "Please scan for designs first")
            return
        
        if not FEATURE_EXTRACTOR_AVAILABLE:
            messagebox.showerror("Error", "Feature extractor not available")
            return
        
        # Start processing in background
        self.process_progress_bar.configure(mode='indeterminate')
        self.process_progress_bar.start()
        self.process_status_label.config(text="Processing study...")
        
        self.processing_thread = threading.Thread(target=self._process_study_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_study_thread(self):
        """Process study in background thread."""
        try:
            def progress_callback(current, total, message):
                progress = (current / total) * 100 if total > 0 else 0
                self.root.after(0, lambda: self.process_progress_var.set(progress))
                self.root.after(0, lambda: self.process_status_label.config(text=f"{message} ({current}/{total})"))
            
            # Process the study
            result = self.processor.process_study_batch(self.study_info, progress_callback)
            
            # Update UI on main thread
            self.root.after(0, self._process_complete, result)
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.root.after(0, self._process_failed, error_msg)
    
    def _process_complete(self, result):
        """Handle processing completion on main thread."""
        self.process_progress_bar.stop()
        self.process_progress_bar.configure(mode='determinate')
        self.process_progress_var.set(100)
        self.process_status_label.config(text="Processing completed")
        
        if result is not None:
            # Update stats display
            self.stats_text.delete('1.0', tk.END)
            
            stats_text = f"""STUDY PROCESSING COMPLETE
{'='*50}

Study Information:
  Name: {self.study_info.get('name', 'Unknown')}
  Type: {self.study_info.get('description', 'Unknown')}
  Total Designs: {len(result):,}

Processing Results:
  Features Extracted: {len([col for col in result.columns if col.startswith(('ACCE_', 'DISP_'))])}
  Memory Usage: {result.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
  
Feature Categories:
"""
            
            # Add feature breakdown
            feature_counts = {}
            for col in result.columns:
                if col.startswith('ACCE_T1_PSD'):
                    feature_counts['Modal (PSD)'] = feature_counts.get('Modal (PSD)', 0) + 1
                elif col.startswith('ACCE_T1_Area'):
                    feature_counts['Energy (Area)'] = feature_counts.get('Energy (Area)', 0) + 1
                elif col.startswith('DISP_'):
                    feature_counts['Displacement'] = feature_counts.get('Displacement', 0) + 1
            
            for category, count in feature_counts.items():
                stats_text += f"  {category}: {count} features\n"
            
            stats_text += f"\n✅ Ready for matrix creation and ML training!"
            
            self.stats_text.insert('1.0', stats_text)
            
            self.log_message(f"✅ Study processing completed: {len(result):,} designs processed")
            messagebox.showinfo("Processing Complete", 
                               f"Study processing completed successfully!\n\n"
                               f"Processed: {len(result):,} designs\n"
                               f"Features: {len([col for col in result.columns if col.startswith(('ACCE_', 'DISP_'))])}\n"
                               f"Ready for matrix creation!")
        else:
            self.log_message("❌ Study processing failed")
            messagebox.showerror("Processing Failed", "Study processing failed")
    
    def _process_failed(self, error_msg):
        """Handle processing failure on main thread."""
        self.process_progress_bar.stop()
        self.process_status_label.config(text="Processing failed")
        self.log_message(f"❌ {error_msg}")
        messagebox.showerror("Processing Failed", error_msg)
    
    def pause_processing(self):
        """Pause processing."""
        self.processor.pause_processing()
        self.log_message("⏸️ Processing paused")
    
    def stop_processing(self):
        """Stop processing."""
        self.processor.stop_processing()
        self.log_message("⏹️ Processing stopped")
    
    def create_matrix(self):
        """Create master matrix."""
        if not self.processor.processed_data:
            messagebox.showerror("Error", "No processed data available. Process a study first.")
            return
        
        try:
            strategy = self.matrix_strategy.get()
            self.log_message(f"🔨 Creating master matrix using {strategy} strategy...")
            
            self.current_data = self.processor.create_master_matrix(strategy=strategy)
            
            if self.current_data is not None:
                # Display matrix information
                matrix_info = f"""MASTER MATRIX CREATED
{'='*50}

Matrix Information:
  Strategy: {strategy}
  Total Designs: {len(self.current_data):,}
  Total Features: {len(self.current_data.columns):,}
  Memory Usage: {self.current_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB

Study Composition:
"""
                if 'study_type' in self.current_data.columns:
                    study_counts = self.current_data['study_type'].value_counts()
                    for study_type, count in study_counts.items():
                        percentage = (count / len(self.current_data)) * 100
                        matrix_info += f"  {study_type}: {count:,} designs ({percentage:.1f}%)\n"
                
                matrix_info += f"\nCBUSH Involvement:\n"
                for cbush_num in range(2, 11):
                    col_name = f'cbush_{cbush_num}_loose'
                    if col_name in self.current_data.columns:
                        count = self.current_data[col_name].sum()
                        percentage = (count / len(self.current_data)) * 100
                        matrix_info += f"  CBUSH {cbush_num}: {count:,} designs ({percentage:.1f}%)\n"
                
                matrix_info += f"\n✅ Matrix ready for ML training!"
                
                self.matrix_info_text.delete('1.0', tk.END)
                self.matrix_info_text.insert('1.0', matrix_info)
                
                self.log_message(f"✅ Master matrix created: {len(self.current_data):,} designs")
                messagebox.showinfo("Success", f"Master matrix created successfully!\n{len(self.current_data):,} designs")
            else:
                messagebox.showerror("Error", "Failed to create master matrix")
                
        except Exception as e:
            error_msg = f"Matrix creation failed: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            messagebox.showerror("Error", error_msg)
    
    def save_matrix(self):
        """Save master matrix to CSV."""
        if self.current_data is None:
            messagebox.showerror("Error", "No matrix available. Create matrix first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Master Matrix",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialname=f"master_matrix_{len(self.current_data)}_designs.csv"
        )
        
        if filename:
            try:
                self.current_data.to_csv(filename, index=False)
                self.log_message(f"💾 Matrix saved: {filename}")
                messagebox.showinfo("Success", f"Matrix saved successfully!\n{filename}")
            except Exception as e:
                error_msg = f"Failed to save matrix: {str(e)}"
                self.log_message(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
    
    def train_model(self):
        """Train ML model."""
        if not ML_PIPELINE_AVAILABLE:
            messagebox.showerror("Error", "ML pipeline not available")
            return
        
        if self.current_data is None:
            messagebox.showerror("Error", "No matrix available. Create master matrix first.")
            return
        
        # Start training in background
        self.training_progress_bar.configure(mode='indeterminate')
        self.training_progress_bar.start()
        self.training_status_label.config(text="Training model...")
        
        strategy = self.ml_strategy.get()
        
        threading.Thread(target=self._train_model_thread, args=(strategy,), daemon=True).start()
    
    def _train_model_thread(self, strategy):
        """Train model in background thread."""
        try:
            # Load data
            self.ml_pipeline.load_data(self.current_data)
            self.ml_pipeline.prepare_targets()
            
            # Train model
            if strategy == "random_forest":
                model, results = self.ml_pipeline.train_random_forest_massive()
            elif strategy == "neural_network":
                model, results = self.ml_pipeline.train_neural_network_massive()
            elif strategy == "ensemble":
                model, results = self.ml_pipeline.train_incremental_ensemble()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Update UI
            self.root.after(0, self._training_complete, strategy, results)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.root.after(0, self._training_failed, error_msg)
    
    def _training_complete(self, strategy, results):
        """Handle training completion."""
        self.training_progress_bar.stop()
        
        if self.ml_pipeline.classification_mode == 'multilabel':
            metric = results.get('jaccard_score', 0)
            metric_name = "Jaccard Score"
        else:
            metric = results.get('accuracy', 0)
            metric_name = "Accuracy"
        
        self.training_status_label.config(text=f"✅ {strategy.title()} - {metric_name}: {metric:.3f}")
        
        # Display results
        results_text = f"""MODEL TRAINING COMPLETE
{'='*50}

Model: {strategy.title()}
Classification Mode: {self.ml_pipeline.classification_mode}
Training Samples: {results.get('training_samples', 0):,}
Test Samples: {results.get('test_samples', 0):,}
Training Time: {results.get('training_time', 0):.1f} seconds

Performance Metrics:
"""
        
        if self.ml_pipeline.classification_mode == 'multilabel':
            results_text += f"""  Jaccard Score: {results.get('jaccard_score', 0):.3f}
  Hamming Loss: {results.get('hamming_loss', 0):.3f}
  F1-Score (Macro): {results.get('f1_score', 0):.3f}
  Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.3f}
"""
        else:
            results_text += f"  Accuracy: {results.get('accuracy', 0):.3f}\n"
        
        results_text += f"\n✅ Model ready for scoring and deployment!"
        
        self.ml_results_text.delete('1.0', tk.END)
        self.ml_results_text.insert('1.0', results_text)
        
        self.log_message(f"✅ {strategy.title()} training completed - {metric_name}: {metric:.3f}")
        messagebox.showinfo("Training Complete", f"Model training completed!\n{metric_name}: {metric:.1%}")
    
    def _training_failed(self, error_msg):
        """Handle training failure."""
        self.training_progress_bar.stop()
        self.training_status_label.config(text="❌ Training failed")
        self.log_message(f"❌ {error_msg}")
        messagebox.showerror("Training Failed", error_msg)
    
    def calculate_scores(self):
        """Calculate comprehensive scores."""
        if not SCORING_SYSTEM_AVAILABLE:
            messagebox.showerror("Error", "Scoring system not available")
            return
        
        if not self.ml_pipeline or not self.ml_pipeline.models:
            messagebox.showerror("Error", "No trained models available")
            return
        
        self.log_message("📈 Calculating comprehensive scores...")
        
        # This would implement comprehensive scoring
        self.scoring_results_text.delete('1.0', tk.END)
        self.scoring_results_text.insert('1.0', "Comprehensive scoring functionality available but not implemented in this demo.\nScoring system is ready for integration.")
    
    def export_matrix(self):
        """Export matrix."""
        if self.current_data is None:
            messagebox.showerror("Error", "No matrix available")
            return
        
        self.save_matrix()
    
    def export_results(self):
        """Export results."""
        messagebox.showinfo("Export", "Results export functionality ready for implementation")
    
    def export_models(self):
        """Export models."""
        messagebox.showinfo("Export", "Model export functionality ready for implementation")
    
    def log_message(self, message):
        """Log message to status display."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, log_entry)
        self.status_text.see(tk.END)
        self.root.update_idletasks()


def main():
    """Main entry point for the massive dataset GUI."""
    root = tk.Tk()
    app = MassiveDatasetBoltDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
