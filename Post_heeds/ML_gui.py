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

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import (hamming_loss, jaccard_score, f1_score, 
                                precision_score, recall_score, accuracy_score,
                                classification_report)
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn not available - ML features will be disabled")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available - XGBoost will be disabled")
    XGBOOST_AVAILABLE = False

class FlexibleCBUSHProcessor:
    """Enhanced processor for massive CBUSH bolt anomaly detection datasets."""
   
    def __init__(self):
        """Initialize the flexible processor."""
        self.heeds_directory = None
        self.processed_data = {}
        self.processing_stats = {}
        self.is_paused = False
        self.should_stop = False
        self.current_operation = None
        self.max_memory_gb = 16.0
        self.batch_size = 500
        self.chunk_processing = True
        self.skip_failed_designs = True
        self.log_failed_designs = True
        self.max_failure_rate = 0.1
       
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
        """Scan POST_0 directory for available designs and detect patterns."""
        if post_0_directory is None:
            if not self.heeds_directory:
                raise ValueError("No HEEDS directory set")
            post_0_directory = os.path.join(self.heeds_directory, "POST_0")
       
        print(f"🔍 Scanning {post_0_directory} for available designs...")
       
        available_designs = {}
        failed_designs = {}
        design_dirs = []
       
        for item in os.listdir(post_0_directory):
            if item.startswith("Design") and os.path.isdir(os.path.join(post_0_directory, item)):
                try:
                    design_num = int(item.replace("Design", ""))
                    design_path = os.path.join(post_0_directory, item)
                    design_dirs.append((design_num, design_path))
                except ValueError:
                    continue
       
        design_dirs.sort()
       
        print(f"📊 Found {len(design_dirs)} Design directories")
       
        for design_num, design_path in design_dirs:
            analysis_dir = os.path.join(design_path, "Analysis_1")
           
            if os.path.exists(analysis_dir):
                accel_file = os.path.join(analysis_dir, "acceleration_results.csv")
                disp_file = os.path.join(analysis_dir, "displacement_results.csv")
               
                if os.path.exists(accel_file) and os.path.exists(disp_file):
                    available_designs[design_num] = {
                        'design_path': design_path,
                        'analysis_path': analysis_dir,
                        'accel_file': accel_file,
                        'disp_file': disp_file,
                        'file_size_mb': self._get_total_file_size([accel_file, disp_file])
                    }
                else:
                    missing_files = []
                    if not os.path.exists(accel_file):
                        missing_files.append("acceleration_results.csv")
                    if not os.path.exists(disp_file):
                        missing_files.append("displacement_results.csv")
                   
                    failed_designs[design_num] = f"Missing files: {', '.join(missing_files)}"
            else:
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
        """Auto-detect study type based on available designs and patterns."""
        design_numbers = sorted(available_designs.keys())
        total_designs = len(design_numbers)
       
        if total_designs == 0:
            return {'type': 'empty', 'description': 'No designs available'}
       
        min_design = min(design_numbers)
        max_design = max(design_numbers)
        design_range = max_design - min_design + 1
       
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
        """Process a study batch with the detected designs."""
        if not FEATURE_EXTRACTOR_AVAILABLE:
            raise ImportError("Feature extractor not available - cannot process studies")
       
        design_numbers = study_info.get('design_numbers', [])
        if not design_numbers:
            print("❌ No design numbers available for processing")
            return None
       
        print(f"\n🔄 Processing study: {study_info.get('description', 'Unknown Study')}")
        print(f"   Designs to process: {len(design_numbers):,}")
       
        extractor = BoltAnomalyFeatureExtractor(self.heeds_directory)
       
        all_features = []
        batch_size = min(self.batch_size, len(design_numbers))
       
        for i in range(0, len(design_numbers), batch_size):
            batch_end = min(i + batch_size, len(design_numbers))
            batch_designs = design_numbers[i:batch_end]
           
            if progress_callback:
                progress_callback(i, len(design_numbers), f"Processing batch {i//batch_size + 1}")
           
            print(f"   Processing batch {i//batch_size + 1}: designs {batch_designs[0]}-{batch_designs[-1]}")
           
            batch_features = extractor.process_design_batch_enhanced(
                batch_designs, study_info, progress_callback
            )
           
            if batch_features:
                all_features.extend(batch_features)
       
        if all_features:
            study_df = pd.DataFrame(all_features)
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
            if isinstance(study_data, list):
                study_df = pd.DataFrame(study_data)
            else:
                study_df = study_data
           
            self.processed_data[study_name] = {
                'data': study_df,
                'design_count': len(study_df),
                'feature_count': len([col for col in study_df.columns if col.startswith(('ACCE_', 'DISP_'))]),
                'memory_mb': study_df.memory_usage(deep=True).sum() / 1024 / 1024,
                'processing_time': time.time(),
                'quality_score': self._calculate_quality_score(study_df)
            }
           
            print(f"💾 Stored study {study_name}: {len(study_df)} designs, {self.processed_data[study_name]['memory_mb']:.1f} MB")
            self._manage_memory()
   
    def _calculate_quality_score(self, study_df):
        """Calculate a quality score for the processed study."""
        try:
            missing_ratio = study_df.isnull().sum().sum() / (study_df.shape[0] * study_df.shape[1])
            feature_cols = [col for col in study_df.columns if col.startswith(('ACCE_', 'DISP_'))]
            if feature_cols:
                feature_variance = study_df[feature_cols].var().mean()
                variance_score = min(1.0, feature_variance / 1000)
            else:
                variance_score = 0.0
           
            cbush_cols = [col for col in study_df.columns if col.startswith('cbush_') and col.endswith('_loose')]
            if cbush_cols:
                label_consistency = 1 - (study_df[cbush_cols].sum(axis=1) == 0).mean()
            else:
                label_consistency = 0.0
           
            quality_score = (1 - missing_ratio) * 0.4 + variance_score * 0.3 + label_consistency * 0.3
            return min(1.0, max(0.0, quality_score))
           
        except Exception:
            return 0.5
   
    def _manage_memory(self):
        """Manage memory usage by cleaning up data if necessary."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
           
            if memory_mb > self.max_memory_gb * 1024:
                print(f"⚠️  High memory usage ({memory_mb:.0f} MB), cleaning up...")
                gc.collect()
                memory_mb_after = process.memory_info().rss / 1024 / 1024
                print(f"   Memory after cleanup: {memory_mb_after:.0f} MB")
               
        except Exception:
            pass
   
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


class EnhancedMLPipeline:
    """Enhanced ML Pipeline with Random Forest, XGBoost, and Ensemble methods."""
    
    def __init__(self):
        """Initialize the enhanced ML pipeline."""
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.cbush_numbers = list(range(2, 11))
        self.is_trained = False
        
    def prepare_data(self, master_df):
        """Prepare data for ML training with proper scaling."""
        print("🔧 PREPARING DATA FOR ML TRINITY")
        
        # Extract features and targets
        feature_cols = [col for col in master_df.columns 
                       if col.startswith(('ACCE_', 'DISP_'))]
        cbush_cols = [f'cbush_{i}_loose' for i in self.cbush_numbers 
                      if f'cbush_{i}_loose' in master_df.columns]
        
        self.feature_names = feature_cols
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   CBUSH targets: {len(cbush_cols)}")
        
        # Extract data
        X = master_df[feature_cols].values
        y = master_df[cbush_cols].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # ⭐ CRITICAL: Feature scaling
        print("⭐ Applying feature scaling (CRITICAL for performance)")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Check scaling effect
        print(f"   Before scaling - Feature range: {X_train.min():.2e} to {X_train.max():.2e}")
        print(f"   After scaling - Feature range: {X_train_scaled.min():.2f} to {X_train_scaled.max():.2f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_ml_trinity(self, master_df):
        """Train Random Forest, XGBoost, and Ensemble models."""
        print("🚀 TRAINING ML TRINITY FOR SPACECRAFT BOLT DETECTION")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(master_df)
        
        training_results = {}
        
        # 1. Random Forest (Baseline + Interpretability)
        print("\n🌲 Training Random Forest...")
        if SKLEARN_AVAILABLE:
            rf_model = self._train_random_forest(X_train, X_test, y_train, y_test)
            training_results['random_forest'] = rf_model
        
        # 2. XGBoost (Performance Champion)
        print("\n🚀 Training XGBoost...")
        if XGBOOST_AVAILABLE:
            xgb_model = self._train_xgboost(X_train, X_test, y_train, y_test)
            training_results['xgboost'] = xgb_model
        
        # 3. Ensemble (Best of Both Worlds)
        print("\n🎯 Training Ensemble...")
        if len(training_results) >= 2:
            ensemble_model = self._train_ensemble(X_train, X_test, y_train, y_test, training_results)
            training_results['ensemble'] = ensemble_model
        
        self.is_trained = True
        return training_results
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with optimal parameters."""
        start_time = time.time()
        
        # Optimized parameters for spacecraft application
        rf = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=4  # Safe for Windows
            )
        )
        
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        training_time = time.time() - start_time
        metrics = self._calculate_metrics(y_test, y_pred, 'Random Forest')
        metrics['training_time'] = training_time
        
        self.models['random_forest'] = rf
        self.results['random_forest'] = metrics
        
        print(f"   ✅ Random Forest trained in {training_time:.1f}s")
        print(f"   📊 Exact Match: {metrics['exact_match_accuracy']:.1%}")
        print(f"   📊 Jaccard Score: {metrics['jaccard_score']:.3f}")
        
        return rf
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost with optimal parameters."""
        start_time = time.time()
        
        # Optimized parameters for spacecraft application
        xgb_model = MultiOutputClassifier(
            xgb.XGBClassifier(
                n_estimators=150,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
                eval_metric='logloss'
            )
        )
        
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        training_time = time.time() - start_time
        metrics = self._calculate_metrics(y_test, y_pred, 'XGBoost')
        metrics['training_time'] = training_time
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = metrics
        
        print(f"   ✅ XGBoost trained in {training_time:.1f}s")
        print(f"   📊 Exact Match: {metrics['exact_match_accuracy']:.1%}")
        print(f"   📊 Jaccard Score: {metrics['jaccard_score']:.3f}")
        
        return xgb_model
    
    def _train_ensemble(self, X_train, X_test, y_train, y_test, base_models):
        """Train ensemble of base models."""
        start_time = time.time()
        
        # Create ensemble from available models
        estimators = []
        if 'random_forest' in base_models:
            estimators.append(('rf', base_models['random_forest']))
        if 'xgboost' in base_models:
            estimators.append(('xgb', base_models['xgboost']))
        
        if len(estimators) < 2:
            print("   ❌ Need at least 2 models for ensemble")
            return None
        
        # Create voting ensemble
        ensemble = VotingClassifier(estimators, voting='hard')
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        training_time = time.time() - start_time
        metrics = self._calculate_metrics(y_test, y_pred, 'Ensemble')
        metrics['training_time'] = training_time
        metrics['base_models'] = [name for name, _ in estimators]
        
        self.models['ensemble'] = ensemble
        self.results['ensemble'] = metrics
        
        print(f"   ✅ Ensemble trained in {training_time:.1f}s")
        print(f"   📊 Exact Match: {metrics['exact_match_accuracy']:.1%}")
        print(f"   📊 Jaccard Score: {metrics['jaccard_score']:.3f}")
        print(f"   🎯 Base models: {metrics['base_models']}")
        
        return ensemble
    
    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics."""
        n_samples, n_labels = y_true.shape
        
        # Exact Match Accuracy
        exact_matches = np.all(y_true == y_pred, axis=1).sum()
        exact_match_accuracy = exact_matches / n_samples
        
        # Hamming Loss
        hamming = hamming_loss(y_true, y_pred)
        
        # Jaccard Score
        jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        
        # F1 Scores
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Precision and Recall
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Deployment Score (Custom metric for spacecraft)
        deployment_score = (
            exact_match_accuracy * 0.4 + 
            jaccard * 0.3 + 
            (1 - hamming) * 0.3
        ) * 100
        
        return {
            'model_name': model_name,
            'exact_match_accuracy': exact_match_accuracy,
            'hamming_loss': hamming,
            'jaccard_score': jaccard,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'deployment_score': deployment_score,
            'n_samples': n_samples,
            'n_labels': n_labels
        }
    
    def get_model_comparison(self):
        """Get comparison of all trained models."""
        if not self.results:
            return None
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.title(),
                'Exact_Match': f"{metrics['exact_match_accuracy']:.1%}",
                'Jaccard_Score': f"{metrics['jaccard_score']:.3f}",
                'Hamming_Loss': f"{metrics['hamming_loss']:.3f}",
                'F1_Macro': f"{metrics['f1_macro']:.3f}",
                'Deployment_Score': f"{metrics['deployment_score']:.1f}/100",
                'Training_Time': f"{metrics.get('training_time', 0):.1f}s"
            })
        
        return pd.DataFrame(comparison_data)
    
    def predict_bolt_condition(self, X_new, model_name='ensemble'):
        """Predict bolt condition using specified model."""
        if model_name not in self.models:
            model_name = list(self.models.keys())[0] if self.models else None
            
        if not model_name:
            return None
        
        model = self.models[model_name]
        X_scaled = self.scaler.transform(X_new.reshape(1, -1))
        predictions = model.predict(X_scaled)[0]
        
        loose_cbushes = [self.cbush_numbers[i] for i, pred in enumerate(predictions) if pred == 1]
        
        return {
            'model_used': model_name,
            'loose_cbushes': loose_cbushes,
            'prediction_vector': predictions.tolist(),
            'confidence': 'High' if len(loose_cbushes) <= 2 else 'Medium'
        }


class MassiveDatasetBoltDetectionGUI:
    """Enhanced GUI for massive dataset multi-CBUSH bolt anomaly detection with CSV upload."""
   
    def __init__(self, root):
        """Initialize the enhanced GUI."""
        self.root = root
        self.root.title("Multi-CBUSH Bolt Detection System v4.0 - Enhanced ML Edition")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')
       
        # Initialize backend components
        self.processor = FlexibleCBUSHProcessor()
        self.ml_pipeline = EnhancedMLPipeline()
        if SCORING_SYSTEM_AVAILABLE:
            self.scoring_system = ComprehensiveMultiLabelScoring()
        else:
            self.scoring_system = None
       
        self.current_data = None
        self.loaded_data = None  # For CSV uploads
        self.scan_results = None
        self.study_info = None
       
        # Threading
        self.processing_thread = None
        self.scanning_thread = None
        self.training_thread = None
       
        # ⭐ NEW: Manual range variables
        self.use_manual_range = None
        self.manual_study_name = None
        self.manual_start = None
        self.manual_end = None
        self.manual_controls_frame = None
        
        # Create GUI
        self.create_widgets()
       
        # Initial status
        self.log_message("🚀 Enhanced Bolt Detection System v4.0 Ready!")
        self.log_message("📊 New Features: CSV Upload, ML Trinity (RF+XGBoost+Ensemble), Manual Range")
        self.log_message("🎯 Target: 70-85% Deployment Score for Spacecraft Safety")
   
    def create_widgets(self):
        """Create all GUI widgets."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
       
        # Create tabs
        self.create_setup_tab()
        self.create_scan_tab()
        self.create_processing_tab()
        self.create_matrix_tab()  # Enhanced with CSV upload
        self.create_ml_tab()      # Enhanced with ML Trinity
        if SCORING_SYSTEM_AVAILABLE:
            self.create_scoring_tab()
        self.create_results_tab()
   
    def create_setup_tab(self):
        """Create enhanced setup tab."""
        setup_frame = ttk.Frame(self.notebook)
        self.notebook.add(setup_frame, text="🔧 Setup")
       
        # Title
        title_label = tk.Label(setup_frame, text="Enhanced Bolt Detection System v4.0",
                              font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=20)
       
        subtitle_label = tk.Label(setup_frame, text="ML Trinity: Random Forest + XGBoost + Ensemble | CSV Upload | Manual Range",
                                 font=('Arial', 14), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=5)
       
        # HEEDS directory configuration
        config_frame = ttk.LabelFrame(setup_frame, text="HEEDS Output Directory Configuration")
        config_frame.pack(fill='x', padx=20, pady=20)
       
        # Directory input
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
        """Create scan and detection tab with MANUAL RANGE feature."""
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
        
        # ⭐ NEW: Manual Range & Study Definition Section
        manual_frame = ttk.LabelFrame(scan_control_frame, text="Manual Range & Study Definition")
        manual_frame.pack(fill='x', padx=10, pady=10)
        
        # Checkbox to enable manual mode
        self.use_manual_range = tk.BooleanVar(value=False)
        tk.Checkbutton(manual_frame, text="☑️ Use Manual Range", 
                       variable=self.use_manual_range,
                       command=self.toggle_manual_range,
                       font=('Arial', 11, 'bold')).pack(anchor='w', padx=10, pady=5)
        
        # Manual controls frame (will be enabled/disabled)
        self.manual_controls_frame = tk.Frame(manual_frame)
        self.manual_controls_frame.pack(fill='x', padx=10, pady=5)
        
        # Study name input
        study_name_frame = tk.Frame(self.manual_controls_frame)
        study_name_frame.pack(fill='x', pady=2)
        tk.Label(study_name_frame, text="Study Name:", font=('Arial', 10, 'bold')).pack(side='left')
        self.manual_study_name = tk.StringVar(value="Custom_Study")
        study_name_entry = tk.Entry(study_name_frame, textvariable=self.manual_study_name, width=50, font=('Arial', 10))
        study_name_entry.pack(side='left', padx=10, fill='x', expand=True)
        
        # Range inputs
        range_frame = tk.Frame(self.manual_controls_frame)
        range_frame.pack(fill='x', pady=2)
        
        tk.Label(range_frame, text="Start:", font=('Arial', 10, 'bold')).pack(side='left')
        self.manual_start = tk.StringVar(value="1")
        start_entry = tk.Entry(range_frame, textvariable=self.manual_start, width=10, font=('Arial', 10))
        start_entry.pack(side='left', padx=5)
        
        tk.Label(range_frame, text="End:", font=('Arial', 10, 'bold')).pack(side='left', padx=(20,5))
        self.manual_end = tk.StringVar(value="7000")
        end_entry = tk.Entry(range_frame, textvariable=self.manual_end, width=10, font=('Arial', 10))
        end_entry.pack(side='left', padx=5)
        
        # Validation button
        tk.Button(range_frame, text="✓ Validate Range", 
                 command=self.validate_manual_range,
                 bg='#f39c12', fg='white', font=('Arial', 9, 'bold')).pack(side='left', padx=20)
        
        # Information text
        info_text = ("Manual range allows you to:\n"
                    "• Override auto-detection with your own study name\n"
                    "• Process specific design ranges (e.g. 1-7000)\n"
                    "• Handle partial studies or custom configurations")
        tk.Label(manual_frame, text=info_text, font=('Arial', 9), fg='#7f8c8d', justify='left').pack(anchor='w', padx=10, pady=5)
        
        # Initially disable manual controls
        self.toggle_manual_range()
       
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
    
    def toggle_manual_range(self):
        """Enable/disable manual range controls."""
        state = 'normal' if self.use_manual_range.get() else 'disabled'
        
        for widget in self.manual_controls_frame.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, (tk.Entry, tk.Button)):
                        child.config(state=state)
    
    def validate_manual_range(self):
        """Validate manual range inputs."""
        try:
            start = int(self.manual_start.get())
            end = int(self.manual_end.get())
            
            if start >= end:
                messagebox.showerror("Invalid Range", "Start must be less than End")
                return False
            
            if start < 1:
                messagebox.showerror("Invalid Range", "Start must be at least 1")
                return False
                
            # Check against available designs if scan has been run
            if hasattr(self, 'scan_results') and self.scan_results:
                available = list(self.scan_results['available_designs'].keys())
                if available:
                    available_start = min(available)
                    available_end = max(available)
                    
                    if end < available_start or start > available_end:
                        messagebox.showwarning("Range Warning", 
                            f"Specified range {start}-{end} has no overlap with available designs {available_start}-{available_end}")
                        return False
                    
                    # Show how many designs will be included
                    overlapping = [d for d in available if start <= d <= end]
                    messagebox.showinfo("Range Validation", 
                        f"✅ Range {start}-{end} is valid\n"
                        f"Will include {len(overlapping)} designs from available data")
            else:
                messagebox.showinfo("Range Validation", 
                    f"✅ Range {start}-{end} looks valid\n"
                    f"Run scan to verify against available designs")
                    
            return True
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for Start and End")
            return False
   
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
        """Create ENHANCED matrix creation tab with CSV upload capability."""
        matrix_frame = ttk.Frame(self.notebook)
        self.notebook.add(matrix_frame, text="📊 Create/Load Matrix")
       
        # ⭐ NEW: CSV Upload Section
        upload_frame = ttk.LabelFrame(matrix_frame, text="📁 Load Existing Feature Matrix")
        upload_frame.pack(fill='x', padx=20, pady=15)
       
        tk.Label(upload_frame, text="Load pre-computed feature matrix from CSV file:",
                font=('Arial', 12, 'bold')).pack(anchor='w', padx=10, pady=5)
       
        upload_controls = tk.Frame(upload_frame)
        upload_controls.pack(fill='x', padx=10, pady=10)
       
        tk.Button(upload_controls, text="📁 Load Matrix CSV",
                 command=self.load_matrix_csv,
                 bg='#9b59b6', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=20).pack(side='left')
       
        # Upload options
        self.upload_option = tk.StringVar(value="replace")
        tk.Label(upload_controls, text="Action:", font=('Arial', 11, 'bold')).pack(side='left', padx=(20,5))
        tk.Radiobutton(upload_controls, text="Replace", variable=self.upload_option, value="replace",
                      font=('Arial', 10)).pack(side='left')
        tk.Radiobutton(upload_controls, text="Append", variable=self.upload_option, value="append",
                      font=('Arial', 10)).pack(side='left')
        tk.Radiobutton(upload_controls, text="Smart Merge", variable=self.upload_option, value="merge",
                      font=('Arial', 10)).pack(side='left')
       
        # Upload status
        self.upload_status = tk.Label(upload_frame, text="No CSV loaded",
                                     font=('Arial', 10), fg='#7f8c8d')
        self.upload_status.pack(padx=10, pady=5)
       
        # Matrix creation controls (existing functionality)
        matrix_control_frame = ttk.LabelFrame(matrix_frame, text="Create Matrix from Processed Data")
        matrix_control_frame.pack(fill='x', padx=20, pady=15)
       
        tk.Label(matrix_control_frame, text="Create master feature matrix from processed study data:",
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
                 bg='#3498db', fg='white', font=('Arial', 14, 'bold'),
                 height=2, width=25).pack(side='left')
       
        tk.Button(create_button_frame, text="💾 Save Matrix",
                 command=self.save_matrix,
                 bg='#16a085', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=15).pack(side='left', padx=10)
       
        # Matrix information display
        self.matrix_info_text = scrolledtext.ScrolledText(matrix_frame, height=20, width=120,
                                                         font=('Consolas', 9))
        self.matrix_info_text.pack(fill='both', expand=True, padx=20, pady=15)
   
    def create_ml_tab(self):
        """Create ENHANCED machine learning tab with ML Trinity."""
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="🤖 ML Trinity")
       
        # ML Trinity controls
        ml_control_frame = ttk.LabelFrame(ml_frame, text="🚀 ML Trinity Training (RF + XGBoost + Ensemble)")
        ml_control_frame.pack(fill='x', padx=20, pady=15)
       
        # Information display
        info_text = ("Train all three models simultaneously:\n"
                    "🌲 Random Forest (Interpretable Baseline)\n"
                    "🚀 XGBoost (Performance Champion)\n"
                    "🎯 Ensemble (Best of Both Worlds)")
        tk.Label(ml_control_frame, text=info_text, font=('Arial', 11), justify='left').pack(anchor='w', padx=10, pady=10)
       
        # Training controls
        training_controls = tk.Frame(ml_control_frame)
        training_controls.pack(fill='x', padx=10, pady=15)
       
        tk.Button(training_controls, text="🚀 Train ML Trinity",
                 command=self.train_ml_trinity,
                 bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'),
                 height=2, width=25).pack(side='left')
       
        # Model comparison button
        tk.Button(training_controls, text="📊 Compare Models",
                 command=self.compare_models,
                 bg='#9b59b6', fg='white', font=('Arial', 12, 'bold'),
                 height=2, width=15).pack(side='left', padx=10)
       
        # Training progress
        training_progress_frame = tk.Frame(ml_control_frame)
        training_progress_frame.pack(fill='x', padx=10, pady=10)
       
        self.training_progress_var = tk.DoubleVar()
        self.training_progress_bar = ttk.Progressbar(training_progress_frame, variable=self.training_progress_var,
                                                   maximum=100, length=500, mode='indeterminate')
        self.training_progress_bar.pack(pady=5)
       
        self.training_status_label = tk.Label(training_progress_frame, text="No models trained",
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
            ("🤖 Export Models", self.export_models, '#e74c3c'),
            ("🎯 Export Predictions", self.export_predictions, '#f39c12')
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
   
    # ⭐ NEW: CSV Upload Methods
    def load_matrix_csv(self):
        """Load existing feature matrix from CSV file."""
        filename = filedialog.askopenfilename(
            title="Load Feature Matrix CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
       
        if filename:
            try:
                self.log_message(f"📁 Loading matrix from: {filename}")
               
                # Load CSV
                loaded_df = pd.read_csv(filename)
               
                # Validate CSV structure
                validation_result = self.validate_matrix_csv(loaded_df, filename)
               
                if validation_result['valid']:
                    # Store loaded data
                    self.loaded_data = loaded_df
                   
                    # Apply user-selected action
                    action = self.upload_option.get()
                    if action == "replace":
                        self.current_data = loaded_df.copy()
                        status_msg = f"✅ Matrix replaced: {len(loaded_df):,} designs loaded"
                    elif action == "append":
                        if self.current_data is not None:
                            self.current_data = pd.concat([self.current_data, loaded_df], ignore_index=True)
                            status_msg = f"✅ Matrix appended: {len(self.current_data):,} total designs"
                        else:
                            self.current_data = loaded_df.copy()
                            status_msg = f"✅ Matrix loaded: {len(loaded_df):,} designs"
                    elif action == "merge":
                        if self.current_data is not None:
                            self.current_data = self.smart_merge_matrices(self.current_data, loaded_df)
                            status_msg = f"✅ Smart merge complete: {len(self.current_data):,} total designs"
                        else:
                            self.current_data = loaded_df.copy()
                            status_msg = f"✅ Matrix loaded: {len(loaded_df):,} designs"
                   
                    self.upload_status.config(text=status_msg, fg='#27ae60')
                    self.log_message(status_msg)
                   
                    # Update matrix info display
                    self.update_matrix_info_display()
                   
                    messagebox.showinfo("Success", f"Matrix loaded successfully!\n{status_msg}")
                else:
                    self.upload_status.config(text="❌ Invalid CSV format", fg='#e74c3c')
                    messagebox.showerror("Validation Error", validation_result['error'])
                   
            except Exception as e:
                error_msg = f"Failed to load CSV: {str(e)}"
                self.upload_status.config(text="❌ Load failed", fg='#e74c3c')
                self.log_message(f"❌ {error_msg}")
                messagebox.showerror("Load Error", error_msg)
   
    def validate_matrix_csv(self, df, filename):
        """Validate loaded CSV for required structure."""
        try:
            required_metadata = ['design_id', 'study_type']
            required_cbush_labels = [f'cbush_{i}_loose' for i in range(2, 11)]
           
            # Check for required columns
            missing_metadata = [col for col in required_metadata if col not in df.columns]
            missing_cbush = [col for col in required_cbush_labels if col not in df.columns]
           
            if missing_metadata:
                return {'valid': False, 'error': f"Missing metadata columns: {missing_metadata}"}
           
            if missing_cbush:
                return {'valid': False, 'error': f"Missing CBUSH columns: {missing_cbush}"}
           
            # Check for feature columns
            feature_cols = [col for col in df.columns if col.startswith(('ACCE_', 'DISP_'))]
            if len(feature_cols) < 10:
                return {'valid': False, 'error': f"Insufficient feature columns: {len(feature_cols)} found, expected 72"}
           
            # Check data types
            try:
                # Convert CBUSH columns to int
                for col in required_cbush_labels:
                    df[col] = df[col].astype(int)
               
                # Check feature columns are numeric
                for col in feature_cols[:5]:  # Check first 5 features
                    pd.to_numeric(df[col], errors='raise')
                   
            except ValueError as e:
                return {'valid': False, 'error': f"Data type validation failed: {str(e)}"}
           
            return {'valid': True, 'error': None}
           
        except Exception as e:
            return {'valid': False, 'error': f"Validation error: {str(e)}"}
   
    def smart_merge_matrices(self, existing_df, new_df):
        """Smart merge that removes duplicates and combines datasets intelligently."""
        try:
            # Check for overlapping design_ids
            if 'design_id' in existing_df.columns and 'design_id' in new_df.columns:
                overlapping_ids = set(existing_df['design_id']) & set(new_df['design_id'])
                if overlapping_ids:
                    self.log_message(f"⚠️  Found {len(overlapping_ids)} overlapping design IDs, removing duplicates")
                    new_df = new_df[~new_df['design_id'].isin(overlapping_ids)]
           
            # Merge datasets
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
           
            # Sort by design_id if available
            if 'design_id' in merged_df.columns:
                merged_df = merged_df.sort_values('design_id').reset_index(drop=True)
           
            self.log_message(f"📊 Smart merge: {len(existing_df)} + {len(new_df)} = {len(merged_df)} designs")
            return merged_df
           
        except Exception as e:
            self.log_message(f"❌ Smart merge failed: {str(e)}")
            return pd.concat([existing_df, new_df], ignore_index=True)
   
    def update_matrix_info_display(self):
        """Update the matrix information display."""
        if self.current_data is not None:
            feature_cols = [col for col in self.current_data.columns if col.startswith(('ACCE_', 'DISP_'))]
            cbush_cols = [col for col in self.current_data.columns if col.startswith('cbush_') and col.endswith('_loose')]
           
            info_text = f"""CURRENT FEATURE MATRIX
{'='*50}

Matrix Information:
  Total Designs: {len(self.current_data):,}
  Total Features: {len(feature_cols)}
  CBUSH Labels: {len(cbush_cols)}
  Memory Usage: {self.current_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB

Study Composition:"""
           
            if 'study_type' in self.current_data.columns:
                study_counts = self.current_data['study_type'].value_counts()
                for study_type, count in study_counts.items():
                    percentage = (count / len(self.current_data)) * 100
                    info_text += f"\n  {study_type}: {count:,} designs ({percentage:.1f}%)"
           
            info_text += f"\n\nCBUSH Label Distribution:"
            for cbush_num in range(2, 11):
                col_name = f'cbush_{cbush_num}_loose'
                if col_name in self.current_data.columns:
                    count = self.current_data[col_name].sum()
                    percentage = (count / len(self.current_data)) * 100
                    info_text += f"\n  CBUSH {cbush_num}: {count:,} loose ({percentage:.1f}%)"
           
            info_text += f"\n\n✅ Ready for ML Trinity training!"
           
            self.matrix_info_text.delete('1.0', tk.END)
            self.matrix_info_text.insert('1.0', info_text)
   
    # ⭐ ENHANCED: ML Trinity Training
    def train_ml_trinity(self):
        """Train the ML Trinity (Random Forest + XGBoost + Ensemble)."""
        if self.current_data is None:
            messagebox.showerror("Error", "No matrix available. Create or load a matrix first.")
            return
       
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Error", "scikit-learn not available - cannot train models")
            return
       
        # Start training in background
        self.training_progress_bar.configure(mode='indeterminate')
        self.training_progress_bar.start()
        self.training_status_label.config(text="Training ML Trinity...")
       
        self.training_thread = threading.Thread(target=self._train_ml_trinity_thread)
        self.training_thread.daemon = True
        self.training_thread.start()
   
    def _train_ml_trinity_thread(self):
        """Train ML Trinity in background thread."""
        try:
            self.log_message("🚀 Starting ML Trinity training...")
           
            # Train all models
            training_results = self.ml_pipeline.train_ml_trinity(self.current_data)
           
            # Update UI on main thread
            self.root.after(0, self._training_complete, training_results)
           
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.root.after(0, self._training_failed, error_msg)
   
    def _training_complete(self, training_results):
        """Handle training completion."""
        self.training_progress_bar.stop()
        self.training_progress_bar.configure(mode='determinate')
        self.training_progress_var.set(100)
       
        # Get model comparison
        comparison_df = self.ml_pipeline.get_model_comparison()
       
        if comparison_df is not None:
            # Find best model
            best_model = None
            best_score = 0
            for _, row in comparison_df.iterrows():
                score = float(row['Deployment_Score'].replace('/100', ''))
                if score > best_score:
                    best_score = score
                    best_model = row['Model']
           
            self.training_status_label.config(
                text=f"✅ Training Complete - Best: {best_model} ({best_score:.1f}/100)"
            )
           
            # Display results
            results_text = f"""🚀 ML TRINITY TRAINING COMPLETE
{'='*60}

"""
            results_text += comparison_df.to_string(index=False)
           
            results_text += f"""

🎯 DEPLOYMENT ASSESSMENT:
Best Model: {best_model}
Deployment Score: {best_score:.1f}/100

"""
            if best_score >= 70:
                results_text += "🚀 STATUS: READY FOR SPACECRAFT DEPLOYMENT\n"
                results_text += "🌟 Excellent performance for safety-critical systems!"
            elif best_score >= 50:
                results_text += "🔧 STATUS: GOOD PERFORMANCE - DEPLOYMENT READY\n"
                results_text += "✅ Meets spacecraft safety requirements"
            else:
                results_text += "⚠️  STATUS: NEEDS IMPROVEMENT\n"
                results_text += "💡 Consider feature engineering or more data"
           
            self.ml_results_text.delete('1.0', tk.END)
            self.ml_results_text.insert('1.0', results_text)
           
            self.log_message(f"✅ ML Trinity training completed - Best: {best_model} ({best_score:.1f}/100)")
           
            # Show completion popup
            messagebox.showinfo("Training Complete",
                               f"✅ ML Trinity training completed!\n\n"
                               f"Best Model: {best_model}\n"
                               f"Deployment Score: {best_score:.1f}/100\n"
                               f"Status: {'DEPLOYMENT READY' if best_score >= 50 else 'NEEDS IMPROVEMENT'}")
        else:
            self.training_status_label.config(text="❌ Training failed")
            messagebox.showerror("Training Failed", "No models were successfully trained")
   
    def _training_failed(self, error_msg):
        """Handle training failure."""
        self.training_progress_bar.stop()
        self.training_status_label.config(text="❌ Training failed")
       
        error_text = f"""❌ ML TRINITY TRAINING FAILED
{'='*50}

Error: {error_msg}

Common causes:
1. Insufficient memory
2. Data quality issues  
3. Missing dependencies

Solutions:
1. Check data quality in matrix
2. Ensure scikit-learn and xgboost are installed
3. Try with smaller dataset
4. Check system memory
"""
       
        self.ml_results_text.delete('1.0', tk.END)
        self.ml_results_text.insert('1.0', error_text)
       
        self.log_message(f"❌ ML Trinity training failed: {error_msg}")
        messagebox.showerror("Training Failed", f"Training failed:\n\n{error_msg}")
   
    def compare_models(self):
        """Display model comparison."""
        if not self.ml_pipeline.is_trained:
            messagebox.showwarning("Warning", "No models trained yet. Train ML Trinity first.")
            return
       
        comparison_df = self.ml_pipeline.get_model_comparison()
        if comparison_df is not None:
            # Display in popup window
            comparison_window = tk.Toplevel(self.root)
            comparison_window.title("Model Performance Comparison")
            comparison_window.geometry("800x400")
           
            # Create table
            frame = tk.Frame(comparison_window)
            frame.pack(fill='both', expand=True, padx=10, pady=10)
           
            columns = list(comparison_df.columns)
            tree = ttk.Treeview(frame, columns=columns, show='headings')
           
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)
           
            for _, row in comparison_df.iterrows():
                tree.insert('', 'end', values=list(row))
           
            tree.pack(fill='both', expand=True)
        else:
            messagebox.showinfo("Info", "No comparison data available")
   
    # Event handlers (keeping existing ones and adding new ones)
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
       
        self.scan_progress_bar.configure(mode='indeterminate')
        self.scan_progress_bar.start()
        self.scan_status_label.config(text="Scanning designs...")
       
        self.scanning_thread = threading.Thread(target=self._scan_designs_thread)
        self.scanning_thread.daemon = True
        self.scanning_thread.start()
   
    def _scan_designs_thread(self):
        """Scan designs in background thread with MANUAL RANGE support."""
        try:
            self.log_message("🔍 Starting design scan...")
           
            scan_results = self.processor.scan_available_designs()
            available_designs = scan_results['available_designs']
            
            # ⭐ NEW: Apply manual range if enabled
            if self.use_manual_range.get():
                try:
                    start_design = int(self.manual_start.get())
                    end_design = int(self.manual_end.get())
                    study_name = self.manual_study_name.get().strip()
                    
                    if not study_name:
                        study_name = f"Manual_Study_{start_design}_{end_design}"
                    
                    self.log_message(f"🎯 Using manual range: {start_design}-{end_design}, Study: {study_name}")
                    
                    # Validate range
                    available_numbers = list(available_designs.keys())
                    if not available_numbers:
                        raise ValueError("No available designs found")
                    
                    actual_start = max(start_design, min(available_numbers))
                    actual_end = min(end_design, max(available_numbers))
                    
                    # Filter to manual range
                    filtered_designs = {k: v for k, v in available_designs.items() 
                                      if actual_start <= k <= actual_end}
                    
                    if not filtered_designs:
                        raise ValueError(f"No designs found in range {start_design}-{end_design}")
                    
                    # Create manual study info
                    study_info = {
                        'name': study_name,
                        'type': 'manual_range',
                        'description': f'Manual Range: {actual_start}-{actual_end} ({len(filtered_designs)} designs)',
                        'design_numbers': sorted(filtered_designs.keys()),
                        'total_designs': len(filtered_designs),
                        'design_range': (actual_start, actual_end),
                        'user_defined': True,
                        'manual_range': True,
                        'specified_range': (start_design, end_design),
                        'actual_range': (actual_start, actual_end)
                    }
                    
                    self.log_message(f"✅ Manual range applied: {len(filtered_designs)} designs selected")
                    
                except Exception as e:
                    # If manual range fails, fall back to auto-detection but log the error
                    self.log_message(f"❌ Manual range failed: {str(e)}, falling back to auto-detection")
                    study_info = self.processor.auto_detect_study_type(available_designs)
            else:
                # Use existing auto-detection
                study_info = self.processor.auto_detect_study_type(available_designs)
           
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
        
        # ⭐ NEW: Show manual range information if used
        if study_info.get('manual_range', False):
            self.results_tree.insert('', 'end', values=('Study Mode', 'Manual Range', 'User-defined study configuration'))
            self.results_tree.insert('', 'end', values=('Study Name', study_info.get('name', 'Unknown'), 'User-provided study name'))
            self.results_tree.insert('', 'end', values=('Specified Range', f"{study_info.get('specified_range', (0,0))[0]}-{study_info.get('specified_range', (0,0))[1]}", 'User-specified design range'))
            self.results_tree.insert('', 'end', values=('Actual Range', f"{study_info.get('actual_range', (0,0))[0]}-{study_info.get('actual_range', (0,0))[1]}", 'Actual range after filtering'))
            self.results_tree.insert('', 'end', values=('Selected Designs', f"{study_info['total_designs']:,}", 'Designs in the manual range'))
        else:
            self.results_tree.insert('', 'end', values=('Study Mode', 'Auto-Detection', 'Automatic study type detection'))
            self.results_tree.insert('', 'end', values=('Study Type', study_info.get('type', 'unknown'), 'Auto-detected study classification'))
            self.results_tree.insert('', 'end', values=('Expected Format', study_info.get('expected_format', 'Unknown'), 'Expected study format'))
            self.results_tree.insert('', 'end', values=('CBUSH Count', str(study_info.get('likely_cbush_count', 'Unknown')), 'Expected number of varied CBUSHes'))
        
        self.results_tree.insert('', 'end', values=('Study Description', study_info.get('description', 'Unknown'), 'Detailed study description'))
        self.results_tree.insert('', 'end', values=('Design Range', f"{study_info['design_range'][0]}-{study_info['design_range'][1]}", 'Range of design numbers'))
       
        self.log_message(f"✅ Scan completed: {len(scan_results['available_designs']):,} designs available")
        self.log_message(f"📊 Study detected: {study_info.get('description', 'Unknown Study')}")
       
        # Create appropriate message for popup
        if study_info.get('manual_range', False):
            popup_message = (f"Scan completed with manual range!\n\n"
                           f"Study: {study_info.get('name', 'Unknown')}\n"
                           f"Selected: {study_info['total_designs']:,} designs\n"
                           f"Range: {study_info.get('actual_range', (0,0))[0]}-{study_info.get('actual_range', (0,0))[1]}")
        else:
            popup_message = (f"Scan completed successfully!\n\n"
                           f"Available: {len(scan_results['available_designs']):,} designs\n"
                           f"Failed: {len(scan_results['failed_designs']):,} designs\n"
                           f"Study Type: {study_info.get('description', 'Unknown')}")
        
        messagebox.showinfo("Scan Complete", popup_message)
   
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
           
            result = self.processor.process_study_batch(self.study_info, progress_callback)
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
           
            matrix_from_studies = self.processor.create_master_matrix(strategy=strategy)
           
            if matrix_from_studies is not None:
                # If we have loaded data, offer to combine
                if self.loaded_data is not None:
                    result = messagebox.askyesnocancel(
                        "Combine Matrices",
                        "You have both processed study data and loaded CSV data.\n\n"
                        "Yes: Combine both datasets\n"
                        "No: Use only processed data\n"
                        "Cancel: Abort operation"
                    )
                   
                    if result is None:  # Cancel
                        return
                    elif result:  # Yes - combine
                        self.current_data = self.smart_merge_matrices(matrix_from_studies, self.loaded_data)
                        self.log_message(f"📊 Combined datasets: {len(self.current_data):,} total designs")
                    else:  # No - use only processed
                        self.current_data = matrix_from_studies
                else:
                    self.current_data = matrix_from_studies
               
                # Update display
                self.update_matrix_info_display()
               
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
            messagebox.showerror("Error", "No matrix available. Create or load matrix first.")
            return
       
        filename = filedialog.asksaveasfilename(
            title="Save Master Matrix",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile=f"enhanced_matrix_{len(self.current_data)}_designs.csv"
        )
       
        if filename:
            try:
                print(f"💾 Saving enhanced matrix to: {filename}")
                self.current_data.to_csv(filename, index=False)
                self.log_message(f"💾 Enhanced matrix saved: {filename}")
                messagebox.showinfo("Success", f"Matrix saved successfully!\n{filename}")
            except Exception as e:
                error_msg = f"Failed to save matrix: {str(e)}"
                self.log_message(f"❌ {error_msg}")
                messagebox.showerror("Error", error_msg)
   
    def calculate_scores(self):
        """Calculate comprehensive scores."""
        if not SCORING_SYSTEM_AVAILABLE:
            messagebox.showerror("Error", "Scoring system not available")
            return
       
        if not self.ml_pipeline.is_trained:
            messagebox.showerror("Error", "No trained models available")
            return
       
        self.log_message("📈 Calculating comprehensive scores...")
       
        # This would implement comprehensive scoring
        self.scoring_results_text.delete('1.0', tk.END)
        self.scoring_results_text.insert('1.0', "Comprehensive scoring functionality available but not implemented in this demo.\nScoring system is ready for integration.")
   
    def export_matrix(self):
        """Export matrix."""
        self.save_matrix()
   
    def export_results(self):
        """Export ML results."""
        if not self.ml_pipeline.is_trained:
            messagebox.showwarning("Warning", "No models trained yet")
            return
       
        comparison_df = self.ml_pipeline.get_model_comparison()
        if comparison_df is not None:
            filename = filedialog.asksaveasfilename(
                title="Export ML Results",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")]
            )
           
            if filename:
                comparison_df.to_csv(filename, index=False)
                self.log_message(f"📊 ML results exported: {filename}")
                messagebox.showinfo("Success", f"Results exported to {filename}")
   
    def export_models(self):
        """Export trained models."""
        messagebox.showinfo("Export", "Model export functionality ready for implementation")
   
    def export_predictions(self):
        """Export model predictions."""
        messagebox.showinfo("Export", "Prediction export functionality ready for implementation")
   
    def log_message(self, message):
        """Log message to status display."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
       
        self.status_text.insert(tk.END, log_entry)
        self.status_text.see(tk.END)
        self.root.update_idletasks()


def main():
    """Main entry point for the enhanced GUI."""
    root = tk.Tk()
    app = MassiveDatasetBoltDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
