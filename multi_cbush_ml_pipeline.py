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
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available - Neural Network strategy will be disabled")

class MultiCBUSHMLPipeline:
    """
    Enhanced ML pipeline for multi-CBUSH bolt anomaly detection.
    Supports the full combined dataset (896 designs) with multi-label classification.
    """
    
    def __init__(self):
        """Initialize the enhanced ML pipeline."""
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
        
        # ML strategy options (filtered by availability)
        self.ml_strategies = {
            'random_forest': 'Random Forest'
        }
        
        if TENSORFLOW_AVAILABLE:
            self.ml_strategies['neural_network'] = 'Neural Network'
        
        if XGBOOST_AVAILABLE:
            self.ml_strategies['xgboost'] = 'XGBoost'
        
        # Ensemble only if we have multiple strategies
        available_strategies = [k for k in self.ml_strategies.keys()]
        if len(available_strategies) > 1:
            self.ml_strategies['ensemble'] = f'Ensemble ({len(available_strategies)} methods)'
        
        # Classification types
        self.classification_types = {
            'auto': 'Auto-detect (Recommended)',
            'multilabel': 'Multi-Label Classification',
            'single': 'Single-CBUSH Classification (Legacy)',
            'hybrid': 'Hybrid Approach'
        }
        
        print("📊 Enhanced ML Pipeline Initialized")
        print(f"   Available strategies: {list(self.ml_strategies.keys())}")
    
    def load_data(self, master_df):
        """Load and prepare the combined dataset for ML analysis."""
        print("=== LOADING COMBINED DATASET FOR ML ANALYSIS ===")
        
        self.data = master_df.copy()
        
        # Extract feature columns
        feature_cols = [col for col in self.data.columns 
                       if col.startswith(('ACCE_', 'DISP_'))]
        
        self.X = self.data[feature_cols].values
        self.feature_names = feature_cols
        
        print(f"✅ Data loaded:")
        print(f"   Total samples: {len(self.data)}")
        print(f"   Features: {len(feature_cols)}")
        
        # Analyze dataset composition
        study_composition = self.data['study_type'].value_counts()
        print(f"   Study composition:")
        for study_type, count in study_composition.items():
            print(f"     {study_type}: {count} designs")
        
        # CBUSH involvement analysis
        cbush_involvement = {}
        for cbush_num in self.cbush_numbers:
            col_name = f'cbush_{cbush_num}_loose'
            if col_name in self.data.columns:
                cbush_involvement[cbush_num] = self.data[col_name].sum()
        
        print(f"   CBUSH involvement:")
        for cbush_num, count in cbush_involvement.items():
            print(f"     CBUSH {cbush_num}: {count} designs ({count/len(self.data)*100:.1f}%)")
        
        return True
    
    def prepare_targets(self, classification_type='auto'):
        """Prepare target variables for multi-label classification."""
        print(f"\n=== PREPARING TARGETS ({classification_type.upper()}) ===")
        
        if classification_type == 'auto':
            # Auto-detect: Use multi-label for combined dataset
            study_types = self.data['study_type'].unique()
            if len(study_types) == 1 and 'single' in study_types:
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
                    print(f"     CBUSH {cbush_num}: {label_sums[i]} positive samples")
                
            else:
                print("❌ No multi-label columns found")
                return False
                
        elif classification_type == 'single':
            # Traditional single-CBUSH classification (legacy mode)
            if 'varied_cbush' in self.data.columns:
                # Filter only single-CBUSH studies
                single_data = self.data[self.data['study_type'] == 'single']
                if len(single_data) > 0:
                    self.y_single = single_data['varied_cbush'].values
                    self.X = single_data[self.feature_names].values
                    print(f"✅ Single-CBUSH targets prepared")
                    print(f"   Filtered to {len(single_data)} single-CBUSH designs")
                    print(f"   Classes: {sorted(np.unique(self.y_single))}")
                else:
                    print("❌ No single-CBUSH data found")
                    return False
            else:
                print("❌ No single-CBUSH column found")
                return False
        
        return True
    
    def train_random_forest(self, **kwargs):
        """Train Random Forest for multi-label classification."""
        print("\n=== TRAINING RANDOM FOREST ===")
        
        # Default parameters optimized for multi-label
        params = {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': 1,  # Windows-safe
            'class_weight': 'balanced'
        }
        params.update(kwargs)
        
        # Choose target and model type
        if self.classification_mode == 'multilabel':
            # Multi-output classifier for multi-label
            rf = MultiOutputClassifier(RandomForestClassifier(**params))
            y_target = self.y_multilabel
        else:
            rf = RandomForestClassifier(**params)
            y_target = self.y_single
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"🔄 Training on {len(X_train)} samples...")
        
        # Train model
        rf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test_scaled)
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, 'Random Forest')
        
        # Store model and results
        self.models['random_forest'] = rf
        self.results['random_forest'] = results
        
        print(f"✅ Random Forest trained")
        if self.classification_mode == 'multilabel':
            print(f"   Hamming Loss: {results.get('hamming_loss', 0):.3f}")
            print(f"   Jaccard Score: {results.get('jaccard_score', 0):.3f}")
        else:
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        
        return rf, results
    
    def train_neural_network(self, **kwargs):
        """Train Neural Network for multi-label classification."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available - cannot train neural network")
            return None, {'accuracy': 0, 'model_name': 'Neural Network (Unavailable)'}
        
        print("\n=== TRAINING NEURAL NETWORK ===")
        
        # Default parameters optimized for multi-label
        params = {
            'hidden_layers': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 150,
            'batch_size': 32,
            'patience': 15
        }
        params.update(kwargs)
        
        # Split data
        if self.classification_mode == 'multilabel':
            y_target = self.y_multilabel
        else:
            y_target = self.y_single
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
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
            # Multi-label output with sigmoid
            output_dim = y_target.shape[1]
            model.add(Dense(output_dim, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
        else:
            # Single-label output with softmax
            output_dim = len(np.unique(y_target))
            model.add(Dense(output_dim, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        
        print(f"🔄 Training neural network...")
        print(f"   Architecture: {params['hidden_layers']} → {output_dim}")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=params['patience'], 
                                     restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predictions
        if self.classification_mode == 'multilabel':
            y_pred_prob = model.predict(X_test_scaled)
            y_pred = (y_pred_prob > 0.5).astype(int)
        else:
            y_pred_prob = model.predict(X_test_scaled)
            y_pred = np.argmax(y_pred_prob, axis=1)
            # Convert back to original class labels
            unique_classes = sorted(np.unique(y_target))
            y_pred = np.array([unique_classes[i] for i in y_pred])
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, 'Neural Network')
        results['history'] = history.history
        results['training_epochs'] = len(history.history['loss'])
        
        # Store model and results
        self.models['neural_network'] = model
        self.results['neural_network'] = results
        
        print(f"✅ Neural Network trained")
        print(f"   Training epochs: {results['training_epochs']}")
        if self.classification_mode == 'multilabel':
            print(f"   Hamming Loss: {results.get('hamming_loss', 0):.3f}")
            print(f"   Jaccard Score: {results.get('jaccard_score', 0):.3f}")
        else:
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        
        return model, results
    
    def train_xgboost(self, **kwargs):
        """Train XGBoost for multi-label classification."""
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available - cannot train XGBoost model")
            return None, {'accuracy': 0, 'model_name': 'XGBoost (Unavailable)'}
        
        print("\n=== TRAINING XGBOOST ===")
        
        # Default parameters
        params = {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': 1
        }
        params.update(kwargs)
        
        if self.classification_mode == 'multilabel':
            # Multi-output classifier for multi-label
            xgb_clf = MultiOutputClassifier(xgb.XGBClassifier(**params))
            y_target = self.y_multilabel
        else:
            xgb_clf = xgb.XGBClassifier(**params)
            y_target = self.y_single
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"🔄 Training XGBoost...")
        
        # Train model
        xgb_clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = xgb_clf.predict(X_test_scaled)
        
        # Calculate metrics
        results = self._calculate_metrics(y_test, y_pred, 'XGBoost')
        
        # Store model and results
        self.models['xgboost'] = xgb_clf
        self.results['xgboost'] = results
        
        print(f"✅ XGBoost trained")
        if self.classification_mode == 'multilabel':
            print(f"   Hamming Loss: {results.get('hamming_loss', 0):.3f}")
            print(f"   Jaccard Score: {results.get('jaccard_score', 0):.3f}")
        else:
            print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        
        return xgb_clf, results
    
    def train_ensemble(self, strategies=None):
        """Train ensemble of multiple strategies."""
        print("\n=== TRAINING ENSEMBLE ===")
        
        # Use only available strategies
        if strategies is None:
            available_base_strategies = []
            if 'random_forest' in self.ml_strategies:
                available_base_strategies.append('random_forest')
            if 'neural_network' in self.ml_strategies:
                available_base_strategies.append('neural_network')
            if 'xgboost' in self.ml_strategies:
                available_base_strategies.append('xgboost')
            strategies = available_base_strategies
        
        # Filter out unavailable strategies
        valid_strategies = []
        for strategy in strategies:
            if strategy == 'neural_network' and not TENSORFLOW_AVAILABLE:
                print(f"⚠️  Skipping {strategy} - TensorFlow not available")
                continue
            elif strategy == 'xgboost' and not XGBOOST_AVAILABLE:
                print(f"⚠️  Skipping {strategy} - XGBoost not available")
                continue
            else:
                valid_strategies.append(strategy)
        
        if len(valid_strategies) < 2:
            print("❌ Need at least 2 available strategies for ensemble")
            return None, {'accuracy': 0, 'model_name': 'Ensemble (Insufficient models)'}
        
        strategies = valid_strategies
        print(f"📊 Using strategies: {strategies}")
        
        # Ensure all base models are trained
        for strategy in strategies:
            if strategy not in self.models:
                if strategy == 'random_forest':
                    self.train_random_forest()
                elif strategy == 'neural_network':
                    self.train_neural_network()
                elif strategy == 'xgboost':
                    self.train_xgboost()
        
        # Split data (use same split as individual models)
        if self.classification_mode == 'multilabel':
            y_target = self.y_multilabel
        else:
            y_target = self.y_single
            
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_target, test_size=0.2, random_state=42
        )
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions from all models
        predictions = []
        for strategy in strategies:
            if strategy in self.models:
                model = self.models[strategy]
                if strategy == 'neural_network':
                    if self.classification_mode == 'multilabel':
                        pred_prob = model.predict(X_test_scaled)
                        pred = (pred_prob > 0.5).astype(int)
                    else:
                        pred_prob = model.predict(X_test_scaled)
                        pred = np.argmax(pred_prob, axis=1)
                        # Convert back to original class labels
                        unique_classes = sorted(np.unique(y_target))
                        pred = np.array([unique_classes[i] for i in pred])
                else:
                    pred = model.predict(X_test_scaled)
                
                predictions.append(pred)
        
        if len(predictions) == 0:
            print("❌ No models available for ensemble")
            return None, None
        
        # Ensemble prediction
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
        results = self._calculate_metrics(y_test, ensemble_pred, 'Ensemble')
        results['strategies_used'] = strategies
        
        # Store results
        self.results['ensemble'] = results
        
        print(f"✅ Ensemble trained")
        print(f"   Strategies: {strategies}")
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
            
            # Per-label metrics
            try:
                results['classification_report'] = classification_report(y_true, y_pred, 
                                                                       target_names=[f'CBUSH_{i}' for i in self.cbush_numbers[:y_true.shape[1]]],
                                                                       output_dict=True, zero_division=0)
            except:
                results['classification_report'] = "Error generating report"
            
        else:
            # Single-label metrics
            results['accuracy'] = accuracy_score(y_true, y_pred)
            try:
                results['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
            except:
                results['classification_report'] = "Error generating report"
                results['confusion_matrix'] = None
        
        return results
    
    def predict_cbush_condition(self, X_new, model_name='random_forest', threshold=0.5):
        """Predict CBUSH condition for new data."""
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
            
            # Get predictions
            if self.classification_mode == 'multilabel':
                if model_name == 'neural_network':
                    pred_prob = model.predict(X_scaled)
                    predictions = (pred_prob > threshold).astype(int)[0]
                    confidences = pred_prob[0]
                else:
                    if hasattr(model, 'predict_proba'):
                        # Multi-output classifier with predict_proba
                        pred_proba_list = model.predict_proba(X_scaled)
                        confidences = np.array([proba[0][1] for proba in pred_proba_list])  # Probability of class 1
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
                    'classification_mode': 'multilabel'
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
                    # Average importance across all output estimators
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
        """Save trained model to file."""
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found")
            return False
        
        try:
            model = self.models[model_name]
            
            if model_name == 'neural_network':
                # Save Keras model
                model.save(filepath)
                # Also save scaler and metadata
                scaler_path = filepath.replace('.h5', '_scaler.pkl')
                metadata_path = filepath.replace('.h5', '_metadata.pkl')
                
                joblib.dump(self.scaler, scaler_path)
                metadata = {
                    'feature_names': self.feature_names,
                    'classification_mode': self.classification_mode,
                    'cbush_numbers': self.cbush_numbers
                }
                joblib.dump(metadata, metadata_path)
                
            else:
                # Save sklearn model with all metadata
                model_data = {
                    'model': model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'classification_mode': self.classification_mode,
                    'cbush_numbers': self.cbush_numbers
                }
                joblib.dump(model_data, filepath)
            
            print(f"✅ Model {model_name} saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, model_name, filepath):
        """Load trained model from file."""
        try:
            if filepath.endswith('.h5'):
                # Load Keras model
                if not TENSORFLOW_AVAILABLE:
                    print("❌ Cannot load .h5 model - TensorFlow not available")
                    return False
                
                model = load_model(filepath)
                self.models[model_name] = model
                
                # Load scaler and metadata
                scaler_path = filepath.replace('.h5', '_scaler.pkl')
                metadata_path = filepath.replace('.h5', '_metadata.pkl')
                
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    self.feature_names = metadata.get('feature_names', [])
                    self.classification_mode = metadata.get('classification_mode', 'multilabel')
                    self.cbush_numbers = metadata.get('cbush_numbers', list(range(2, 11)))
                    
            else:
                # Load sklearn model
                model_data = joblib.load(filepath)
                self.models[model_name] = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.classification_mode = model_data['classification_mode']
                self.cbush_numbers = model_data.get('cbush_numbers', list(range(2, 11)))
            
            print(f"✅ Model {model_name} loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_results_summary(self):
        """Create comprehensive results summary."""
        if not self.results:
            print("❌ No results available")
            return None
        
        summary = {
            'classification_mode': self.classification_mode,
            'models_trained': list(self.results.keys()),
            'model_comparison': {}
        }
        
        for model_name, results in self.results.items():
            model_metrics = {'model_type': results.get('model_name', model_name)}
            
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
    """Example usage of EnhancedMultiCBUSHMLPipeline."""
    print("Enhanced MultiCBUSH ML Pipeline ready for GUI integration!")
    print("Features:")
    print("  ✅ Multi-label classification for combined dataset")
    print("  ✅ Support for 896-design matrix (single + adjacent + multi-bolt)")
    print("  ✅ Advanced neural networks with TensorFlow")
    print("  ✅ Ensemble methods")
    print("  ✅ Spacecraft deployment ready")


if __name__ == "__main__":
    main()