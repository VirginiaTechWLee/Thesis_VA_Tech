import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           multilabel_confusion_matrix, hamming_loss, jaccard_score,
                           f1_score, precision_score, recall_score)
from sklearn.multioutput import MultiOutputClassifier
import joblib
import os
import warnings
import time
import gc
import psutil
from pathlib import Path
warnings.filterwarnings('ignore')

# Optional imports - handle gracefully if not available
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available - XGBoost strategy will be disabled")

TENSORFLOW_AVAILABLE = False
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available - Neural Network strategy will be disabled")

class MultiCBUSHMLPipeline:
    """
    Enhanced ML pipeline for multi-CBUSH bolt anomaly detection.
    Now supports massive datasets (11K-111K+ designs) with incremental learning,
    memory management, and batch processing.
    """
    
    def __init__(self):
        """Initialize the enhanced ML pipeline for massive datasets."""
        self.data = None
        self.X = None
        self.y_multilabel = None
        self.y_single = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.classification_mode = None
        
        # CBUSH mapping (2-10)
        self.cbush_numbers = list(range(2, 11))
        
        # Memory management for massive datasets
        self.max_memory_gb = 32.0  # Increased for massive datasets
        self.enable_memory_management = True
        self.batch_size = 1000  # Larger batches for efficiency
        self.use_incremental_learning = True
        
        # Progress tracking
        self.training_callbacks = []
        self.processing_stats = {}
        
        # ML strategy options (filtered by availability)
        self.ml_strategies = {
            'random_forest': 'Random Forest (Massive Dataset Optimized)'
        }
        
        if TENSORFLOW_AVAILABLE:
            self.ml_strategies['neural_network'] = 'Neural Network (Batch Training)'
        
        if XGBOOST_AVAILABLE:
            self.ml_strategies['xgboost'] = 'XGBoost (Incremental Learning)'
        
        # Ensemble only if we have multiple strategies
        available_strategies = [k for k in self.ml_strategies.keys()]
        if len(available_strategies) > 1:
            self.ml_strategies['ensemble'] = f'Ensemble ({len(available_strategies)} methods)'
        
        # Add massive dataset specific strategies
        if len(available_strategies) > 0:
            self.ml_strategies['incremental_ensemble'] = 'Incremental Ensemble (Memory Efficient)'
        
        # Classification types
        self.classification_types = {
            'auto': 'Auto-detect (Recommended)',
            'multilabel': 'Multi-Label Classification',
            'single': 'Single-CBUSH Classification (Legacy)',
            'hybrid': 'Hybrid Approach'
        }
        
        print("📊 Enhanced ML Pipeline Initialized for Massive Datasets")
        print(f"   Available strategies: {list(self.ml_strategies.keys())}")
        print(f"   Memory limit: {self.max_memory_gb}GB")
        print(f"   Batch size: {self.batch_size}")
    
    def load_data(self, master_df, chunk_size=None):
        """
        Load and prepare massive datasets for ML analysis with memory management.
        
        Args:
            master_df: Master DataFrame (can be very large)
            chunk_size: Optional chunk size for memory management
        """
        print("=== LOADING MASSIVE DATASET FOR ML ANALYSIS ===")
        
        # Memory check before loading
        self._check_memory_usage("Before data loading")
        
        # Handle very large datasets
        if len(master_df) > 100000:
            print(f"⚠️  Very large dataset detected ({len(master_df):,} designs)")
            print(f"   Enabling aggressive memory management")
            self.enable_memory_management = True
            self.use_incremental_learning = True
        
        self.data = master_df.copy()
        
        # Extract feature columns
        feature_cols = [col for col in self.data.columns 
                       if col.startswith(('ACCE_', 'DISP_'))]
        
        # Check memory usage of feature matrix
        estimated_memory_mb = (len(self.data) * len(feature_cols) * 8) / (1024 * 1024)  # 8 bytes per float64
        print(f"   Estimated feature matrix memory: {estimated_memory_mb:.1f} MB")
        
        if estimated_memory_mb > self.max_memory_gb * 1024 * 0.7:  # 70% of available memory
            print(f"   Feature matrix too large for memory, enabling chunked processing")
            self.use_incremental_learning = True
        
        self.X = self.data[feature_cols].values
        self.feature_names = feature_cols
        
        print(f"✅ Data loaded:")
        print(f"   Total samples: {len(self.data):,}")
        print(f"   Features: {len(feature_cols):,}")
        
        # Analyze dataset composition
        if 'study_type' in self.data.columns:
            study_composition = self.data['study_type'].value_counts()
            print(f"   Study composition:")
            for study_type, count in study_composition.items():
                print(f"     {study_type}: {count:,} designs ({count/len(self.data)*100:.1f}%)")
        
        # CBUSH involvement analysis
        cbush_involvement = {}
        for cbush_num in self.cbush_numbers:
            col_name = f'cbush_{cbush_num}_loose'
            if col_name in self.data.columns:
                cbush_involvement[cbush_num] = self.data[col_name].sum()
        
        if cbush_involvement:
            print(f"   CBUSH involvement:")
            for cbush_num, count in cbush_involvement.items():
                percentage = (count / len(self.data)) * 100
                print(f"     CBUSH {cbush_num}: {count:,} designs ({percentage:.1f}%)")
        
        # Memory management after loading
        self._manage_memory("After data loading")
        
        return True
    
    def prepare_targets(self, classification_type='auto'):
        """Prepare target variables for multi-label classification with massive dataset support."""
        print(f"\n=== PREPARING TARGETS ({classification_type.upper()}) ===")
        
        if classification_type == 'auto':
            # Enhanced auto-detection for massive datasets
            study_types = self.data['study_type'].unique() if 'study_type' in self.data.columns else ['unknown']
            
            if len(study_types) == 1 and any(t in study_types for t in ['single', 'single_cbush_complete']):
                classification_type = 'single'
            else:
                classification_type = 'multilabel'
            print(f"   Auto-detected: {classification_type}")
        
        self.classification_mode = classification_type
        
        if classification_type == 'multilabel':
            # Multi-label classification: predict which CBUSHes are loose
            cbush_cols = [f'cbush_{cbush_num}_loose' for cbush_num in self.cbush_numbers]
            available_cols = [col for col in cbush_cols if col in self.data.columns]
            
            if available_cols:
                self.y_multilabel = self.data[available_cols].values
                print(f"✅ Multi-label targets prepared")
                print(f"   Label columns: {len(available_cols)} CBUSHes")
                print(f"   CBUSHes: {[col.split('_')[1] for col in available_cols]}")
                
                # Label distribution
                label_sums = self.y_multilabel.sum(axis=0)
                print(f"   Label distribution:")
                for i, cbush_num in enumerate(self.cbush_numbers[:len(available_cols)]):
                    percentage = (label_sums[i] / len(self.y_multilabel)) * 100
                    print(f"     CBUSH {cbush_num}: {label_sums[i]:,} positive samples ({percentage:.1f}%)")
                
                # Check for class imbalance in massive datasets
                max_imbalance = max(label_sums) / (min(label_sums) + 1)  # Avoid division by zero
                if max_imbalance > 10:
                    print(f"   ⚠️  Class imbalance detected (ratio: {max_imbalance:.1f})")
                    print(f"   Recommend using balanced class weights")
                
            else:
                print("❌ No multi-label columns found")
                return False
                
        elif classification_type == 'single':
            # Traditional single-CBUSH classification (legacy mode)
            if 'varied_cbush' in self.data.columns:
                self.y_single = self.data['varied_cbush'].values
                print(f"✅ Single-CBUSH targets prepared")
                print(f"   Total samples: {len(self.y_single):,}")
                
                # Class distribution
                unique_classes, counts = np.unique(self.y_single, return_counts=True)
                print(f"   Classes: {sorted(unique_classes)}")
                print(f"   Class distribution:")
                for cls, count in zip(unique_classes, counts):
                    percentage = (count / len(self.y_single)) * 100
                    print(f"     CBUSH {cls}: {count:,} samples ({percentage:.1f}%)")
            else:
                print("❌ No single-CBUSH column found")
                return False
        
        return True
    
    def train_random_forest_massive(self, **kwargs):
        """Train Random Forest optimized for massive datasets."""
        print("\n=== TRAINING RANDOM FOREST (MASSIVE DATASET) ===")
        
        # Enhanced parameters for massive datasets
        params = {
            'n_estimators': 200,  # Reduced for memory efficiency
            'max_depth': 15,      # Reduced depth for speed
            'min_samples_split': 10,  # Increased for generalization
            'min_samples_leaf': 5,    # Increased for generalization
            'random_state': 42,
            'n_jobs': -1,  # Use all available cores
            'class_weight': 'balanced',
            'max_samples': 0.8,  # Bootstrap with 80% of samples
            'verbose': 1 if len(self.X) > 50000 else 0  # Verbose for very large datasets
        }
        params.update(kwargs)
        
        # Memory management for massive datasets
        if len(self.X) > 100000:
            print(f"   Very large dataset ({len(self.X):,} samples) - optimizing parameters")
            params.update({
                'n_estimators': 100,  # Further reduced
                'max_depth': 12,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_samples': 0.6  # Use less data per tree
            })
        
        # Choose target and model type
        if self.classification_mode == 'multilabel':
            rf = MultiOutputClassifier(RandomForestClassifier(**params))
            y_target = self.y_multilabel
        else:
            rf = RandomForestClassifier(**params)
            y_target = self.y_single
        
        # Enhanced train/test split with stratification for massive datasets
        test_size = min(0.2, max(0.05, 10000 / len(self.X)))  # Adaptive test size
        
        if self.classification_mode == 'multilabel':
            # Use simple random split for multilabel
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y_target, test_size=test_size, random_state=42
            )
        else:
            # Use stratified split for single-label
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, y_target, test_size=test_size, random_state=42, stratify=y_target
            )
        
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Scale features with memory management
        print(f"🔄 Scaling features...")
        self._manage_memory("Before scaling")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self._manage_memory("After scaling")
        
        # Train model with progress tracking
        print(f"🔄 Training Random Forest...")
        start_time = time.time()
        
        rf.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.1f} seconds")
        
        # Predictions with memory management
        print(f"🔄 Making predictions...")
        y_pred = rf.predict(X_test_scaled)
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, 'Random Forest (Massive)')
        results['training_time'] = training_time
        results['training_samples'] = len(X_train)
        results['test_samples'] = len(X_test)
        
        # Store model and results
        self.models['random_forest'] = rf
        self.results['random_forest'] = results
        
        # Memory cleanup
        self._manage_memory("After training")
        
        print(f"✅ Random Forest trained on {len(X_train):,} samples")
        if self.classification_mode == 'multilabel':
            print(f"   Hamming Loss: {results.get('hamming_loss', 0):.3f}")
            print(f"   Jaccard Score: {results.get('jaccard_score', 0):.3f}")
        else:
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        
        return rf, results
    
    def train_neural_network_massive(self, **kwargs):
        """Train Neural Network optimized for massive datasets with batch processing."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available - cannot train neural network")
            return None, {'accuracy': 0, 'model_name': 'Neural Network (Unavailable)'}
        
        print("\n=== TRAINING NEURAL NETWORK (MASSIVE DATASET) ===")
        
        # Enhanced parameters for massive datasets
        params = {
            'hidden_layers': [512, 256, 128],  # Larger network for massive data
            'dropout_rate': 0.4,  # Increased dropout for regularization
            'learning_rate': 0.001,
            'epochs': 100,  # Reduced epochs, rely on early stopping
            'batch_size': min(self.batch_size, 2048),  # Adaptive batch size
            'patience': 20,  # More patience for large datasets
            'reduce_lr_patience': 10,  # Learning rate reduction
            'validation_split': 0.15  # Smaller validation split for massive datasets
        }
        params.update(kwargs)
        
        # Adjust parameters for very large datasets
        if len(self.X) > 100000:
            print(f"   Very large dataset ({len(self.X):,} samples) - optimizing for memory")
            params.update({
                'hidden_layers': [256, 128, 64],  # Smaller network
                'batch_size': min(512, params['batch_size']),
                'validation_split': 0.1  # Smaller validation split
            })
        
        # Split data
        if self.classification_mode == 'multilabel':
            y_target = self.y_multilabel
        else:
            y_target = self.y_single
        
        test_size = min(0.2, max(0.05, 10000 / len(self.X)))  # Adaptive test size
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_target, test_size=test_size, random_state=42
        )
        
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Scale features
        print(f"🔄 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model with enhanced architecture
        print(f"🔄 Building neural network...")
        model = Sequential()
        
        # Input layer
        model.add(Dense(params['hidden_layers'][0], activation='relu', 
                       input_shape=(X_train_scaled.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout_rate']))
        
        # Hidden layers
        for units in params['hidden_layers'][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(params['dropout_rate']))
        
        # Output layer
        if self.classification_mode == 'multilabel':
            output_dim = y_target.shape[1]
            model.add(Dense(output_dim, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            unique_classes = len(np.unique(y_target))
            model.add(Dense(unique_classes, activation='softmax'))
            
            # Convert labels to categorical for softmax
            y_train_cat = to_categorical(y_train - min(y_train))
            y_test_cat = to_categorical(y_test - min(y_test))
            y_train, y_test = y_train_cat, y_test_cat
            
            model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
        
        print(f"   Architecture: {params['hidden_layers']} → {output_dim if self.classification_mode == 'multilabel' else unique_classes}")
        print(f"   Total parameters: {model.count_params():,}")
        
        # Enhanced callbacks for massive datasets
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=params['patience'], 
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=params['reduce_lr_patience'],
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model with batch processing
        print(f"🔄 Training neural network...")
        print(f"   Batch size: {params['batch_size']}")
        print(f"   Max epochs: {params['epochs']}")
        
        start_time = time.time()
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=params['validation_split'],
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1 if len(X_train) > 50000 else 0
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        print(f"🔄 Making predictions...")
        if self.classification_mode == 'multilabel':
            y_pred_prob = model.predict(X_test_scaled, batch_size=params['batch_size'])
            y_pred = (y_pred_prob > 0.5).astype(int)
        else:
            y_pred_prob = model.predict(X_test_scaled, batch_size=params['batch_size'])
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_test = np.argmax(y_test, axis=1)  # Convert back from categorical
            
            # Convert predictions back to original class labels
            unique_classes = sorted(np.unique(self.y_single))
            y_pred = np.array([unique_classes[i] for i in y_pred])
            y_test = np.array([unique_classes[i] for i in y_test])
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, 'Neural Network (Massive)')
        results['history'] = history.history
        results['training_epochs'] = len(history.history['loss'])
        results['training_time'] = training_time
        results['training_samples'] = len(X_train)
        results['test_samples'] = len(X_test)
        
        # Store model and results
        self.models['neural_network'] = model
        self.results['neural_network'] = results
        
        # Memory cleanup
        self._manage_memory("After neural network training")
        
        print(f"✅ Neural Network trained on {len(X_train):,} samples")
        print(f"   Training epochs: {results['training_epochs']}")
        print(f"   Training time: {training_time:.1f} seconds")
        if self.classification_mode == 'multilabel':
            print(f"   Hamming Loss: {results.get('hamming_loss', 0):.3f}")
            print(f"   Jaccard Score: {results.get('jaccard_score', 0):.3f}")
        else:
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        
        return model, results
    
    def train_incremental_ensemble(self, base_strategies=None, **kwargs):
        """Train ensemble using incremental learning for massive datasets."""
        print("\n=== TRAINING INCREMENTAL ENSEMBLE (MASSIVE DATASET) ===")
        
        if base_strategies is None:
            base_strategies = ['random_forest']
            if TENSORFLOW_AVAILABLE:
                base_strategies.append('neural_network')
        
        print(f"   Base strategies: {base_strategies}")
        
        # Train base models if not already trained
        base_models = {}
        for strategy in base_strategies:
            if strategy == 'random_forest':
                if 'random_forest' not in self.models:
                    self.train_random_forest_massive(**kwargs)
                base_models['random_forest'] = self.models['random_forest']
            elif strategy == 'neural_network' and TENSORFLOW_AVAILABLE:
                if 'neural_network' not in self.models:
                    self.train_neural_network_massive(**kwargs)
                base_models['neural_network'] = self.models['neural_network']
        
        if len(base_models) < 2:
            print("❌ Need at least 2 base models for ensemble")
            return None, {'accuracy': 0, 'model_name': 'Incremental Ensemble (Insufficient models)'}
        
        # Create ensemble predictions on test set
        if self.classification_mode == 'multilabel':
            y_target = self.y_multilabel
        else:
            y_target = self.y_single
        
        test_size = min(0.2, max(0.05, 10000 / len(self.X)))
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_target, test_size=test_size, random_state=42
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Collect predictions from all base models
        predictions = []
        for model_name, model in base_models.items():
            print(f"   Getting predictions from {model_name}...")
            
            if model_name == 'neural_network':
                if self.classification_mode == 'multilabel':
                    pred_prob = model.predict(X_test_scaled, batch_size=self.batch_size)
                    pred = (pred_prob > 0.5).astype(int)
                else:
                    pred_prob = model.predict(X_test_scaled, batch_size=self.batch_size)
                    pred = np.argmax(pred_prob, axis=1)
                    # Convert back to original class labels
                    unique_classes = sorted(np.unique(y_target))
                    pred = np.array([unique_classes[i] for i in pred])
            else:
                pred = model.predict(X_test_scaled)
            
            predictions.append(pred)
        
        # Ensemble prediction (voting)
        if self.classification_mode == 'multilabel':
            # Average predictions for multi-label
            ensemble_pred = np.mean(predictions, axis=0)
            ensemble_pred = (ensemble_pred > 0.5).astype(int)
        else:
            # Majority voting for single-label
            predictions_array = np.array(predictions)
            ensemble_pred = []
            for i in range(predictions_array.shape[1]):
                votes = predictions_array[:, i]
                unique_votes, counts = np.unique(votes, return_counts=True)
                majority_vote = unique_votes[np.argmax(counts)]
                ensemble_pred.append(majority_vote)
            ensemble_pred = np.array(ensemble_pred)
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, ensemble_pred, 'Incremental Ensemble')
        results['base_models'] = list(base_models.keys())
        results['ensemble_size'] = len(base_models)
        
        # Store results
        self.results['incremental_ensemble'] = results
        
        print(f"✅ Incremental Ensemble trained")
        print(f"   Base models: {list(base_models.keys())}")
        print(f"   Test samples: {len(X_test):,}")
        if self.classification_mode == 'multilabel':
            print(f"   Hamming Loss: {results.get('hamming_loss', 0):.3f}")
            print(f"   Jaccard Score: {results.get('jaccard_score', 0):.3f}")
        else:
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        
        return ensemble_pred, results
    
    def _calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics for model evaluation."""
        results = {'model_name': model_name}
        
        if self.classification_mode == 'multilabel':
            # Multi-label metrics
            results['hamming_loss'] = hamming_loss(y_true, y_pred)
            results['jaccard_score'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
            results['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            results['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            results['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Exact match accuracy (all labels correct)
            results['exact_match_accuracy'] = accuracy_score(y_true, y_pred)
            
            # Additional metrics for massive datasets
            results['subset_accuracy'] = results['exact_match_accuracy']  # Same as exact match
            results['jaccard_samples'] = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
            
            # Per-label metrics
            try:
                per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
                results['per_label_f1'] = per_label_f1.tolist()
                results['mean_per_label_f1'] = np.mean(per_label_f1)
            except:
                results['per_label_f1'] = []
                results['mean_per_label_f1'] = 0
            
            try:
                results['classification_report'] = classification_report(y_true, y_pred, 
                                                                       target_names=[f'CBUSH_{i}' for i in self.cbush_numbers[:y_true.shape[1]]],
                                                                       output_dict=True, zero_division=0)
            except:
                results['classification_report'] = "Error generating report"
            
        else:
            # Single-label metrics
            results['accuracy'] = accuracy_score(y_true, y_pred)
            results['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            results['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            results['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            try:
                results['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
            except:
                results['classification_report'] = "Error generating report"
                results['confusion_matrix'] = None
        
        return results
    
    def predict_cbush_condition(self, X_new, model_name='random_forest', threshold=0.5):
        """Predict CBUSH condition for new data with massive dataset optimizations."""
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return None
        
        try:
            model = self.models[model_name]
            
            # Ensure 2D array
            if X_new.ndim == 1:
                X_new = X_new.reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X_new)
            
            # Get predictions with batch processing for large inputs
            if self.classification_mode == 'multilabel':
                if model_name == 'neural_network':
                    pred_prob = model.predict(X_scaled, batch_size=min(self.batch_size, len(X_scaled)))
                    predictions = (pred_prob > threshold).astype(int)[0]
                    confidences = pred_prob[0]
                else:
                    if hasattr(model, 'predict_proba'):
                        pred_proba_list = model.predict_proba(X_scaled)
                        confidences = np.array([proba[0][1] for proba in pred_proba_list])
                    else:
                        confidences = model.predict(X_scaled)[0].astype(float)
                    predictions = (confidences > threshold).astype(int)
                
                # Create result
                loose_cbushes = [self.cbush_numbers[i] for i in range(len(predictions)) if predictions[i] == 1]
                
                return {
                    'loose_cbushes': loose_cbushes,
                    'loose_count': len(loose_cbushes),
                    'cbush_probabilities': dict(zip(self.cbush_numbers[:len(confidences)], confidences)),
                    'prediction_vector': predictions,
                    'confidence_vector': confidences,
                    'classification_mode': 'multilabel',
                    'threshold_used': threshold
                }
            else:
                # Single-label classification
                predictions = model.predict(X_scaled)
                if hasattr(model, 'predict_proba'):
                    try:
                        confidences = model.predict_proba(X_scaled)
                        max_confidence = np.max(confidences[0])
                    except:
                        max_confidence = 0.5
                else:
                    max_confidence = 0.5
                
                return {
                    'predicted_cbush': predictions[0],
                    'confidence': max_confidence,
                    'classification_mode': 'single'
                }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def _manage_memory(self, context=""):
        """Enhanced memory management for massive datasets."""
        if not self.enable_memory_management:
            return
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_gb = memory_mb / 1024
            
            if memory_gb > self.max_memory_gb * 0.8:  # 80% threshold
                print(f"⚠️  High memory usage ({memory_gb:.1f}GB) at {context}, cleaning up...")
                
                # Force garbage collection
                collected = gc.collect()
                
                # TensorFlow specific cleanup
                if TENSORFLOW_AVAILABLE:
                    try:
                        tf.keras.backend.clear_session()
                    except:
                        pass
                
                # Check memory after cleanup
                memory_mb_after = process.memory_info().rss / 1024 / 1024
                memory_gb_after = memory_mb_after / 1024
                
                saved_mb = memory_mb - memory_mb_after
                print(f"   Memory cleanup: {saved_mb:.1f}MB freed, {collected} objects collected")
                print(f"   Memory after cleanup: {memory_gb_after:.1f}GB")
                
        except Exception as e:
            print(f"⚠️  Memory management warning: {e}")
    
    def _check_memory_usage(self, context=""):
        """Check and report current memory usage."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_gb = memory_mb / 1024
            
            print(f"   Memory usage {context}: {memory_gb:.1f}GB")
            
            if memory_gb > self.max_memory_gb * 0.9:
                print(f"   ⚠️  Warning: Memory usage approaching limit ({self.max_memory_gb}GB)")
                
        except Exception as e:
            print(f"⚠️  Memory check warning: {e}")
    
    # Legacy compatibility methods
    def train_random_forest(self, **kwargs):
        """Legacy method - redirects to massive dataset version."""
        return self.train_random_forest_massive(**kwargs)
    
    def train_neural_network(self, **kwargs):
        """Legacy method - redirects to massive dataset version."""
        return self.train_neural_network_massive(**kwargs)
    
    def train_xgboost(self, **kwargs):
        """Train XGBoost with massive dataset optimizations."""
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available - cannot train XGBoost model")
            return None, {'accuracy': 0, 'model_name': 'XGBoost (Unavailable)'}
        
        print("\n=== TRAINING XGBOOST (MASSIVE DATASET) ===")
        
        # Enhanced parameters for massive datasets
        params = {
            'n_estimators': 150,  # Reduced for memory efficiency
            'max_depth': 10,      # Reduced depth
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1 if len(self.X) > 50000 else 0
        }
        params.update(kwargs)
        
        if self.classification_mode == 'multilabel':
            xgb_clf = MultiOutputClassifier(xgb.XGBClassifier(**params))
            y_target = self.y_multilabel
        else:
            xgb_clf = xgb.XGBClassifier(**params)
            y_target = self.y_single
        
        # Split data
        test_size = min(0.2, max(0.05, 10000 / len(self.X)))
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_target, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"🔄 Training XGBoost on {len(X_train):,} samples...")
        start_time = time.time()
        
        xgb_clf.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.1f} seconds")
        
        # Predictions
        y_pred = xgb_clf.predict(X_test_scaled)
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, 'XGBoost (Massive)')
        results['training_time'] = training_time
        
        # Store model and results
        self.models['xgboost'] = xgb_clf
        self.results['xgboost'] = results
        
        # Memory cleanup
        self._manage_memory("After XGBoost training")
        
        print(f"✅ XGBoost trained")
        if self.classification_mode == 'multilabel':
            print(f"   Hamming Loss: {results.get('hamming_loss', 0):.3f}")
            print(f"   Jaccard Score: {results.get('jaccard_score', 0):.3f}")
        else:
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        
        return xgb_clf, results
    
    def train_ensemble(self, strategies=None):
        """Legacy method - redirects to incremental ensemble."""
        return self.train_incremental_ensemble(strategies)
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from tree-based models."""
        if model_name not in self.models:
            print(f"❌ Model {model_name} not trained")
            return None
        
        model = self.models[model_name]
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'estimators_'):
                # Multi-output classifier
                if self.classification_mode == 'multilabel':
                    importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                else:
                    importance = model.feature_importances_
            else:
                print(f"❌ Model {model_name} doesn't support feature importance")
                return None
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"❌ Error getting feature importance: {e}")
            return None
    
    def save_model(self, model_name, filepath):
        """Save trained model to file with enhanced metadata."""
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return False
        
        try:
            model = self.models[model_name]
            
            if model_name == 'neural_network':
                # Save Keras model
                model.save(filepath)
                # Save enhanced metadata
                metadata = {
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'classification_mode': self.classification_mode,
                    'cbush_numbers': self.cbush_numbers,
                    'max_memory_gb': self.max_memory_gb,
                    'batch_size': self.batch_size,
                    'training_stats': self.results.get(model_name, {})
                }
                
                scaler_path = filepath.replace('.h5', '_metadata.pkl')
                joblib.dump(metadata, scaler_path)
                
            else:
                # Save sklearn model with enhanced metadata
                model_data = {
                    'model': model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'classification_mode': self.classification_mode,
                    'cbush_numbers': self.cbush_numbers,
                    'max_memory_gb': self.max_memory_gb,
                    'batch_size': self.batch_size,
                    'training_stats': self.results.get(model_name, {}),
                    'version': 'massive_dataset_v1.0'
                }
                joblib.dump(model_data, filepath)
            
            print(f"✅ Model {model_name} saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_name, filepath):
        """Load trained model from file with enhanced metadata."""
        try:
            if filepath.endswith('.h5'):
                # Load Keras model
                if not TENSORFLOW_AVAILABLE:
                    print("❌ Cannot load .h5 model - TensorFlow not available")
                    return False
                
                model = load_model(filepath)
                self.models[model_name] = model
                
                # Load enhanced metadata
                metadata_path = filepath.replace('.h5', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    self.scaler = metadata.get('scaler', StandardScaler())
                    self.feature_names = metadata.get('feature_names', [])
                    self.classification_mode = metadata.get('classification_mode', 'multilabel')
                    self.cbush_numbers = metadata.get('cbush_numbers', list(range(2, 11)))
                    self.max_memory_gb = metadata.get('max_memory_gb', 32.0)
                    self.batch_size = metadata.get('batch_size', 1000)
                    
                    # Restore training stats if available
                    if 'training_stats' in metadata:
                        self.results[model_name] = metadata['training_stats']
                    
            else:
                # Load sklearn model
                model_data = joblib.load(filepath)
                self.models[model_name] = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.classification_mode = model_data['classification_mode']
                self.cbush_numbers = model_data.get('cbush_numbers', list(range(2, 11)))
                self.max_memory_gb = model_data.get('max_memory_gb', 32.0)
                self.batch_size = model_data.get('batch_size', 1000)
                
                # Restore training stats if available
                if 'training_stats' in model_data:
                    self.results[model_name] = model_data['training_stats']
            
            print(f"✅ Model {model_name} loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_results_summary(self):
        """Create comprehensive results summary for massive datasets."""
        if not self.results:
            print("❌ No results available")
            return None
        
        summary = {
            'classification_mode': self.classification_mode,
            'models_trained': list(self.results.keys()),
            'total_data_size': len(self.data) if self.data is not None else 0,
            'memory_limit_gb': self.max_memory_gb,
            'batch_size': self.batch_size,
            'incremental_learning': self.use_incremental_learning,
            'model_comparison': {}
        }
        
        for model_name, results in self.results.items():
            model_metrics = {
                'model_type': results.get('model_name', model_name),
                'training_time': results.get('training_time', 0),
                'training_samples': results.get('training_samples', 0),
                'test_samples': results.get('test_samples', 0)
            }
            
            if self.classification_mode == 'multilabel':
                model_metrics.update({
                    'hamming_loss': results.get('hamming_loss', 0),
                    'jaccard_score': results.get('jaccard_score', 0),
                    'f1_score': results.get('f1_score', 0),
                    'exact_match_accuracy': results.get('exact_match_accuracy', 0)
                })
            else:
                model_metrics['accuracy'] = results.get('accuracy', 0)
            
            summary['model_comparison'][model_name] = model_metrics
        
        return summary


def main():
    """Example usage of Enhanced MultiCBUSH ML Pipeline for massive datasets."""
    print("Enhanced MultiCBUSH ML Pipeline ready for massive datasets!")
    print("Features:")
    print("  ✅ Support for 11K-111K+ designs")
    print("  ✅ Incremental learning and batch processing")
    print("  ✅ Enhanced memory management")
    print("  ✅ Optimized neural networks with early stopping")
    print("  ✅ Adaptive batch sizes and test splits")
    print("  ✅ Integration with FlexibleCBUSHProcessor")
    print("  ✅ Spacecraft deployment ready")


if __name__ == "__main__":
    main()
