import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    # Multi-label metrics
    hamming_loss, jaccard_score, accuracy_score,
    precision_score, recall_score, f1_score,
    multilabel_confusion_matrix, classification_report,
    # Probability-based metrics
    roc_auc_score, average_precision_score,
    # Ranking metrics
    coverage_error, label_ranking_average_precision_score,
    label_ranking_loss,
    # Curves
    precision_recall_curve, roc_curve
)
import warnings
import time
import gc
import psutil
from pathlib import Path
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
warnings.filterwarnings('ignore')

class ComprehensiveMultiLabelScoring:
    """
    Enhanced comprehensive scoring system for massive multi-label CBUSH classification.
    Supports datasets with 11K-111K+ designs with memory-efficient computation,
    batch processing, and distributed metrics calculation.
    """
    
    def __init__(self, cbush_numbers=None, max_memory_gb=32.0, enable_parallel=True):
        """
        Initialize the enhanced scoring system for massive datasets.
        
        Args:
            cbush_numbers: List of CBUSH numbers (default: 2-10)
            max_memory_gb: Maximum memory limit in GB
            enable_parallel: Enable parallel processing for large datasets
        """
        self.cbush_numbers = cbush_numbers or list(range(2, 11))
        self.cbush_names = [f'CBUSH_{i}' for i in self.cbush_numbers]
        
        # Memory management for massive datasets
        self.max_memory_gb = max_memory_gb
        self.enable_memory_management = True
        self.batch_size = 10000  # Batch size for metrics computation
        
        # Parallel processing
        self.enable_parallel = enable_parallel
        self.max_workers = min(mp.cpu_count(), 8)  # Limit workers to prevent memory issues
        
        # Caching for expensive computations
        self.metrics_cache = {}
        self.enable_caching = True
        
        print(f"🔍 Enhanced Scoring System Initialized for Massive Datasets")
        print(f"   Memory limit: {self.max_memory_gb}GB")
        print(f"   Batch size: {self.batch_size:,}")
        print(f"   Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}")
        print(f"   Max workers: {self.max_workers}")
        
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """
        Calculate all multi-label metrics for massive datasets with batch processing.
        
        Args:
            y_true: True labels (can be very large)
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            model_name: Name of the model
            
        Returns:
            dict: Comprehensive metrics
        """
        print(f"\n🔍 CALCULATING COMPREHENSIVE METRICS FOR {model_name.upper()} (MASSIVE DATASET)")
        print(f"   Dataset size: {len(y_true):,} samples")
        
        # Memory check
        self._check_memory_usage("Before metrics calculation")
        
        # Cache key for expensive computations
        cache_key = f"{model_name}_{len(y_true)}_{hash(y_true.tobytes()) if hasattr(y_true, 'tobytes') else 'no_hash'}"
        
        if self.enable_caching and cache_key in self.metrics_cache:
            print(f"   📋 Using cached results for {model_name}")
            return self.metrics_cache[cache_key]
        
        metrics = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'n_labels': y_true.shape[1] if y_true.ndim > 1 else len(self.cbush_numbers),
            'computation_time': time.time()
        }
        
        # === SAMPLE-BASED METRICS (BATCH PROCESSING) ===
        print("📊 Sample-based metrics (batch processing)...")
        
        if len(y_true) > self.batch_size:
            # Use batch processing for large datasets
            sample_metrics = self._calculate_sample_metrics_batched(y_true, y_pred)
        else:
            # Direct calculation for smaller datasets
            sample_metrics = self._calculate_sample_metrics_direct(y_true, y_pred)
        
        metrics.update(sample_metrics)
        
        # === LABEL-BASED METRICS (PARALLEL PROCESSING) ===
        print("🏷️  Label-based metrics (parallel processing)...")
        
        if self.enable_parallel and len(y_true) > 50000:
            label_metrics = self._calculate_label_metrics_parallel(y_true, y_pred)
        else:
            label_metrics = self._calculate_label_metrics_direct(y_true, y_pred)
        
        metrics.update(label_metrics)
        
        # === PROBABILITY-BASED METRICS ===
        if y_prob is not None:
            print("📈 Probability-based metrics...")
            
            try:
                if len(y_true) > 100000:
                    # Use sampling for very large datasets
                    prob_metrics = self._calculate_probability_metrics_sampled(y_true, y_prob)
                else:
                    prob_metrics = self._calculate_probability_metrics_direct(y_true, y_prob)
                
                metrics.update(prob_metrics)
                
            except Exception as e:
                print(f"⚠️  Warning: Some probability-based metrics failed: {e}")
                metrics.update({
                    'roc_auc_macro': 0.5,
                    'average_precision_macro': 0.5,
                    'coverage_error': float('inf'),
                    'label_ranking_avg_precision': 0.5,
                    'label_ranking_loss': 1.0
                })
        
        # === CONFUSION MATRICES (MEMORY EFFICIENT) ===
        print("📋 Confusion matrices (memory efficient)...")
        try:
            if len(y_true) > 50000:
                cm_metrics = self._calculate_confusion_matrices_efficient(y_true, y_pred)
            else:
                cm_metrics = self._calculate_confusion_matrices_direct(y_true, y_pred)
            
            metrics.update(cm_metrics)
            
        except Exception as e:
            print(f"⚠️  Warning: Confusion matrix calculation failed: {e}")
        
        # === PERFORMANCE QUALITY ASSESSMENT ===
        print("🎯 Performance quality assessment...")
        metrics.update(self._assess_performance_quality(metrics))
        
        # Calculate total computation time
        metrics['computation_time'] = time.time() - metrics['computation_time']
        
        # Cache results for expensive computations
        if self.enable_caching:
            self.metrics_cache[cache_key] = metrics
        
        # Memory cleanup
        self._manage_memory("After metrics calculation")
        
        print(f"✅ Comprehensive metrics calculated for {model_name}")
        print(f"   Computation time: {metrics['computation_time']:.1f} seconds")
        
        return metrics
    
    def _calculate_sample_metrics_batched(self, y_true, y_pred):
        """Calculate sample-based metrics using batch processing."""
        print(f"   Processing {len(y_true):,} samples in batches of {self.batch_size:,}")
        
        exact_matches = 0
        jaccard_scores = []
        hamming_losses = []
        
        for i in range(0, len(y_true), self.batch_size):
            batch_end = min(i + self.batch_size, len(y_true))
            y_true_batch = y_true[i:batch_end]
            y_pred_batch = y_pred[i:batch_end]
            
            # Exact match (subset accuracy)
            exact_matches += np.sum(np.all(y_true_batch == y_pred_batch, axis=1))
            
            # Jaccard score per sample
            for j in range(len(y_true_batch)):
                intersection = np.sum(y_true_batch[j] & y_pred_batch[j])
                union = np.sum(y_true_batch[j] | y_pred_batch[j])
                jaccard_scores.append(intersection / union if union > 0 else 0)
            
            # Hamming loss per sample
            hamming_losses.extend(np.mean(y_true_batch != y_pred_batch, axis=1))
        
        return {
            'exact_match_ratio': exact_matches / len(y_true),
            'subset_accuracy': exact_matches / len(y_true),
            'jaccard_score': np.mean(jaccard_scores),
            'hamming_loss': np.mean(hamming_losses)
        }
    
    def _calculate_sample_metrics_direct(self, y_true, y_pred):
        """Calculate sample-based metrics directly for smaller datasets."""
        return {
            'exact_match_ratio': accuracy_score(y_true, y_pred),
            'subset_accuracy': accuracy_score(y_true, y_pred),
            'jaccard_score': jaccard_score(y_true, y_pred, average='samples', zero_division=0),
            'hamming_loss': hamming_loss(y_true, y_pred)
        }
    
    def _calculate_label_metrics_parallel(self, y_true, y_pred):
        """Calculate label-based metrics using parallel processing."""
        print(f"   Using {self.max_workers} workers for parallel computation")
        
        # Prepare data for parallel processing
        label_data = []
        for i in range(y_true.shape[1]):
            label_data.append({
                'label_index': i,
                'y_true_label': y_true[:, i],
                'y_pred_label': y_pred[:, i],
                'cbush_name': self.cbush_names[i] if i < len(self.cbush_names) else f'Label_{i}'
            })
        
        # Parallel computation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._calculate_single_label_metrics, data): data for data in label_data}
            
            per_label_results = {}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    per_label_results[result['cbush_name']] = result
        
        # Aggregate results
        if per_label_results:
            precisions = [r['precision'] for r in per_label_results.values()]
            recalls = [r['recall'] for r in per_label_results.values()]
            f1_scores = [r['f1'] for r in per_label_results.values()]
            
            return {
                'precision_macro': np.mean(precisions),
                'recall_macro': np.mean(recalls),
                'f1_macro': np.mean(f1_scores),
                'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
                'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
                'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'per_label_metrics': per_label_results
            }
        else:
            return self._calculate_label_metrics_direct(y_true, y_pred)
    
    def _calculate_label_metrics_direct(self, y_true, y_pred):
        """Calculate label-based metrics directly."""
        # Standard sklearn metrics
        metrics = {}
        
        for avg_type in ['micro', 'macro', 'weighted']:
            metrics[f'precision_{avg_type}'] = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
            metrics[f'recall_{avg_type}'] = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
            metrics[f'f1_{avg_type}'] = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
        
        # Per-label metrics
        per_label_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_label_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        per_label_metrics = {}
        for i, cbush_name in enumerate(self.cbush_names[:len(per_label_precision)]):
            per_label_metrics[cbush_name] = {
                'precision': per_label_precision[i],
                'recall': per_label_recall[i],
                'f1': per_label_f1[i]
            }
        
        metrics['per_label_metrics'] = per_label_metrics
        
        return metrics
    
    def _calculate_single_label_metrics(self, data):
        """Calculate metrics for a single label (used in parallel processing)."""
        try:
            y_true_label = data['y_true_label']
            y_pred_label = data['y_pred_label']
            cbush_name = data['cbush_name']
            
            # Avoid calculations if no positive samples
            if np.sum(y_true_label) == 0 and np.sum(y_pred_label) == 0:
                return {
                    'cbush_name': cbush_name,
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0,
                    'support': 0
                }
            
            precision = precision_score(y_true_label, y_pred_label, zero_division=0)
            recall = recall_score(y_true_label, y_pred_label, zero_division=0)
            f1 = f1_score(y_true_label, y_pred_label, zero_division=0)
            support = np.sum(y_true_label)
            
            return {
                'cbush_name': cbush_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            }
            
        except Exception as e:
            print(f"⚠️  Error calculating metrics for {data['cbush_name']}: {e}")
            return None
    
    def _calculate_probability_metrics_sampled(self, y_true, y_prob, sample_size=50000):
        """Calculate probability-based metrics using sampling for very large datasets."""
        print(f"   Using sampling ({sample_size:,} samples) for probability metrics")
        
        # Random sampling
        if len(y_true) > sample_size:
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true[indices]
            y_prob_sample = y_prob[indices]
        else:
            y_true_sample = y_true
            y_prob_sample = y_prob
        
        return self._calculate_probability_metrics_direct(y_true_sample, y_prob_sample)
    
    def _calculate_probability_metrics_direct(self, y_true, y_prob):
        """Calculate probability-based metrics directly."""
        try:
            roc_aucs = []
            avg_precisions = []
            
            for i in range(y_true.shape[1]):
                # ROC AUC
                try:
                    if len(np.unique(y_true[:, i])) > 1:  # Need both classes
                        roc_auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                        roc_aucs.append(roc_auc)
                    else:
                        roc_aucs.append(0.5)  # Random performance
                except:
                    roc_aucs.append(0.5)
                
                # Average Precision
                try:
                    avg_prec = average_precision_score(y_true[:, i], y_prob[:, i])
                    avg_precisions.append(avg_prec)
                except:
                    avg_precisions.append(0.0)
            
            # Update per-label metrics with probability-based scores
            if hasattr(self, 'per_label_metrics'):
                for i, cbush_name in enumerate(self.cbush_names[:len(roc_aucs)]):
                    if cbush_name in self.per_label_metrics:
                        self.per_label_metrics[cbush_name]['roc_auc'] = roc_aucs[i]
                        self.per_label_metrics[cbush_name]['avg_precision'] = avg_precisions[i]
            
            # Ranking metrics (computationally expensive, use sampling if needed)
            try:
                if len(y_true) > 20000:
                    # Sample for ranking metrics
                    sample_indices = np.random.choice(len(y_true), 20000, replace=False)
                    y_true_rank = y_true[sample_indices]
                    y_prob_rank = y_prob[sample_indices]
                else:
                    y_true_rank = y_true
                    y_prob_rank = y_prob
                
                coverage_err = coverage_error(y_true_rank, y_prob_rank)
                label_ranking_ap = label_ranking_average_precision_score(y_true_rank, y_prob_rank)
                label_ranking_loss_val = label_ranking_loss(y_true_rank, y_prob_rank)
                
            except Exception as e:
                print(f"⚠️  Warning: Ranking metrics failed: {e}")
                coverage_err = float('inf')
                label_ranking_ap = 0.0
                label_ranking_loss_val = 1.0
            
            return {
                'roc_auc_macro': np.mean(roc_aucs),
                'average_precision_macro': np.mean(avg_precisions),
                'coverage_error': coverage_err,
                'label_ranking_avg_precision': label_ranking_ap,
                'label_ranking_loss': label_ranking_loss_val,
                'per_label_roc_aucs': roc_aucs,
                'per_label_avg_precisions': avg_precisions
            }
            
        except Exception as e:
            print(f"⚠️  Error in probability metrics calculation: {e}")
            return {
                'roc_auc_macro': 0.5,
                'average_precision_macro': 0.5,
                'coverage_error': float('inf'),
                'label_ranking_avg_precision': 0.5,
                'label_ranking_loss': 1.0
            }
    
    def _calculate_confusion_matrices_efficient(self, y_true, y_pred):
        """Calculate confusion matrices efficiently for large datasets."""
        print(f"   Computing confusion matrices for {len(y_true):,} samples")
        
        try:
            # Use sklearn's multilabel_confusion_matrix but with memory management
            cm_matrices = multilabel_confusion_matrix(y_true, y_pred)
            
            confusion_matrices = {}
            for i, cbush_name in enumerate(self.cbush_names[:len(cm_matrices)]):
                cm = cm_matrices[i]
                tn, fp, fn, tp = cm.ravel()
                
                confusion_matrices[cbush_name] = {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp),
                    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                }
            
            return {'confusion_matrices': confusion_matrices}
            
        except Exception as e:
            print(f"⚠️  Error in confusion matrix calculation: {e}")
            return {'confusion_matrices': {}}
    
    def _calculate_confusion_matrices_direct(self, y_true, y_pred):
        """Calculate confusion matrices directly for smaller datasets."""
        return self._calculate_confusion_matrices_efficient(y_true, y_pred)
    
    def _assess_performance_quality(self, metrics):
        """Assess overall performance quality for massive datasets."""
        exact_match = metrics.get('exact_match_ratio', 0)
        jaccard = metrics.get('jaccard_score', 0)
        hamming = metrics.get('hamming_loss', 0)
        f1_macro = metrics.get('f1_macro', 0)
        
        # Quality thresholds adjusted for massive datasets
        quality_assessment = {
            'deployment_ready': False,
            'quality_score': 0.0,
            'quality_grade': 'F',
            'recommendations': []
        }
        
        # Calculate overall quality score
        quality_components = []
        
        # Exact match component (0-25 points)
        if exact_match >= 0.8:
            quality_components.append(25)
        elif exact_match >= 0.6:
            quality_components.append(20)
        elif exact_match >= 0.4:
            quality_components.append(15)
        elif exact_match >= 0.2:
            quality_components.append(10)
        else:
            quality_components.append(5)
        
        # Jaccard component (0-25 points)
        if jaccard >= 0.7:
            quality_components.append(25)
        elif jaccard >= 0.5:
            quality_components.append(20)
        elif jaccard >= 0.3:
            quality_components.append(15)
        else:
            quality_components.append(10)
        
        # Hamming loss component (0-25 points)
        if hamming <= 0.1:
            quality_components.append(25)
        elif hamming <= 0.2:
            quality_components.append(20)
        elif hamming <= 0.3:
            quality_components.append(15)
        else:
            quality_components.append(10)
        
        # F1 macro component (0-25 points)
        if f1_macro >= 0.8:
            quality_components.append(25)
        elif f1_macro >= 0.6:
            quality_components.append(20)
        elif f1_macro >= 0.4:
            quality_components.append(15)
        else:
            quality_components.append(10)
        
        quality_assessment['quality_score'] = sum(quality_components) / 100.0
        
        # Assign grade
        if quality_assessment['quality_score'] >= 0.9:
            quality_assessment['quality_grade'] = 'A'
        elif quality_assessment['quality_score'] >= 0.8:
            quality_assessment['quality_grade'] = 'B'
        elif quality_assessment['quality_score'] >= 0.7:
            quality_assessment['quality_grade'] = 'C'
        elif quality_assessment['quality_score'] >= 0.6:
            quality_assessment['quality_grade'] = 'D'
        else:
            quality_assessment['quality_grade'] = 'F'
        
        # Deployment readiness (stricter for spacecraft)
        spacecraft_ready = (
            exact_match >= 0.6 and
            jaccard >= 0.5 and
            f1_macro >= 0.6 and
            hamming <= 0.25
        )
        
        quality_assessment['deployment_ready'] = spacecraft_ready
        
        # Recommendations
        if exact_match < 0.6:
            quality_assessment['recommendations'].append("Improve exact match accuracy - consider ensemble methods")
        if jaccard < 0.5:
            quality_assessment['recommendations'].append("Improve label overlap - check class balance")
        if hamming > 0.25:
            quality_assessment['recommendations'].append("Reduce Hamming loss - fine-tune decision thresholds")
        if f1_macro < 0.6:
            quality_assessment['recommendations'].append("Improve F1 scores - consider cost-sensitive learning")
        
        if spacecraft_ready:
            quality_assessment['recommendations'].append("✅ Ready for spacecraft deployment")
        else:
            quality_assessment['recommendations'].append("⚠️  Not ready for spacecraft deployment - improve metrics")
        
        return quality_assessment
    
    def compare_models(self, model_results):
        """Compare multiple models across all metrics with enhanced visualization."""
        print(f"\n📊 COMPARING {len(model_results)} MODELS (MASSIVE DATASET)")
        
        if not model_results:
            print("❌ No model results to compare")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, metrics in model_results.items():
            row = {
                'Model': model_name.title(),
                'Samples': f"{metrics.get('n_samples', 0):,}",
                'Computation_Time': f"{metrics.get('computation_time', 0):.1f}s",
                'Exact_Match': f"{metrics.get('exact_match_ratio', 0):.3f}",
                'Jaccard_Score': f"{metrics.get('jaccard_score', 0):.3f}",
                'Hamming_Loss': f"{metrics.get('hamming_loss', 0):.3f}",
                'F1_Macro': f"{metrics.get('f1_macro', 0):.3f}",
                'F1_Micro': f"{metrics.get('f1_micro', 0):.3f}",
                'Precision_Macro': f"{metrics.get('precision_macro', 0):.3f}",
                'Recall_Macro': f"{metrics.get('recall_macro', 0):.3f}",
                'Quality_Score': f"{metrics.get('quality_score', 0):.3f}",
                'Quality_Grade': metrics.get('quality_grade', 'N/A'),
                'Deployment_Ready': '✅' if metrics.get('deployment_ready', False) else '❌'
            }
            
            # Add probability-based metrics if available
            if 'roc_auc_macro' in metrics:
                row['ROC_AUC_Macro'] = f"{metrics['roc_auc_macro']:.3f}"
                row['Avg_Precision_Macro'] = f"{metrics['average_precision_macro']:.3f}"
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by quality score
        try:
            comparison_df['Quality_Score_Numeric'] = comparison_df['Quality_Score'].astype(float)
            comparison_df = comparison_df.sort_values('Quality_Score_Numeric', ascending=False)
            comparison_df = comparison_df.drop('Quality_Score_Numeric', axis=1)
        except:
            pass
        
        print(f"✅ Model comparison completed")
        return comparison_df
    
    def create_detailed_report(self, metrics, model_name):
        """Create detailed text report for a single model with massive dataset insights."""
        report_lines = [
            f"{'='*80}",
            f"DETAILED PERFORMANCE REPORT: {model_name.upper()} (MASSIVE DATASET)",
            f"{'='*80}",
            f"",
            f"📊 DATASET INFORMATION:",
            f"   Total Samples: {metrics.get('n_samples', 0):,}",
            f"   Total Labels: {metrics.get('n_labels', 0)}",
            f"   Computation Time: {metrics.get('computation_time', 0):.1f} seconds",
            f"   Processing Rate: {metrics.get('n_samples', 0) / max(metrics.get('computation_time', 1), 1):,.0f} samples/second",
            f"",
            f"📊 SAMPLE-BASED METRICS:",
            f"   Exact Match Ratio: {metrics.get('exact_match_ratio', 0):.3f} ({metrics.get('exact_match_ratio', 0)*100:.1f}%)",
            f"   Jaccard Score (Samples): {metrics.get('jaccard_score', 0):.3f}",
            f"   Hamming Loss: {metrics.get('hamming_loss', 0):.3f}",
            f"",
            f"🏷️  LABEL-BASED METRICS:",
            f"   Precision (Macro): {metrics.get('precision_macro', 0):.3f}",
            f"   Precision (Micro): {metrics.get('precision_micro', 0):.3f}",
            f"   Precision (Weighted): {metrics.get('precision_weighted', 0):.3f}",
            f"   Recall (Macro): {metrics.get('recall_macro', 0):.3f}",
            f"   Recall (Micro): {metrics.get('recall_micro', 0):.3f}",
            f"   Recall (Weighted): {metrics.get('recall_weighted', 0):.3f}",
            f"   F1-Score (Macro): {metrics.get('f1_macro', 0):.3f}",
            f"   F1-Score (Micro): {metrics.get('f1_micro', 0):.3f}",
            f"   F1-Score (Weighted): {metrics.get('f1_weighted', 0):.3f}",
        ]
        
        # Add probability-based metrics if available
        if 'roc_auc_macro' in metrics:
            report_lines.extend([
                f"",
                f"📈 PROBABILITY-BASED METRICS:",
                f"   ROC AUC (Macro): {metrics['roc_auc_macro']:.3f}",
                f"   Average Precision (Macro): {metrics['average_precision_macro']:.3f}",
                f"   Coverage Error: {metrics.get('coverage_error', 0):.2f}",
                f"   Label Ranking Avg Precision: {metrics.get('label_ranking_avg_precision', 0):.3f}",
                f"   Label Ranking Loss: {metrics.get('label_ranking_loss', 0):.3f}",
            ])
        
        # Per-label breakdown
        if 'per_label_metrics' in metrics:
            report_lines.extend([
                f"",
                f"📋 PER-CBUSH BREAKDOWN:",
                f"   {'CBUSH':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}"
            ])
            
            if 'roc_auc_macro' in metrics:
                report_lines[-1] += f" {'ROC-AUC':<8} {'Avg-Prec':<8}"
            
            for cbush_name, cbush_metrics in metrics['per_label_metrics'].items():
                precision = cbush_metrics.get('precision', 0)
                recall = cbush_metrics.get('recall', 0)
                f1 = cbush_metrics.get('f1', 0)
                
                line = f"   {cbush_name:<8} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f}"
                
                if 'roc_auc' in cbush_metrics:
                    roc_auc = cbush_metrics.get('roc_auc', 0)
                    avg_prec = cbush_metrics.get('avg_precision', 0)
                    line += f" {roc_auc:<8.3f} {avg_prec:<8.3f}"
                
                report_lines.append(line)
        
        # Performance interpretation
        report_lines.extend([
            f"",
            f"🎯 PERFORMANCE INTERPRETATION:",
            self._interpret_performance_massive(metrics),
            f"",
            f"🚀 SPACECRAFT DEPLOYMENT ASSESSMENT:",
            self._deployment_readiness_massive(metrics),
            f"",
            f"💡 RECOMMENDATIONS FOR MASSIVE DATASETS:",
            self._generate_massive_dataset_recommendations(metrics)
        ])
        
        return "\n".join(report_lines)
    
    def _interpret_performance_massive(self, metrics):
        """Interpret model performance for massive datasets."""
        exact_match = metrics.get('exact_match_ratio', 0)
        jaccard = metrics.get('jaccard_score', 0)
        hamming = metrics.get('hamming_loss', 0)
        f1_macro = metrics.get('f1_macro', 0)
        n_samples = metrics.get('n_samples', 0)
        
        interpretations = []
        
        # Exact Match interpretation (adjusted for massive datasets)
        if exact_match >= 0.7:
            interpretations.append(f"   ✅ Excellent: {exact_match:.1%} predictions are completely correct")
        elif exact_match >= 0.5:
            interpretations.append(f"   ✅ Good: {exact_match:.1%} predictions are completely correct")
        elif exact_match >= 0.3:
            interpretations.append(f"   ⚠️  Fair: {exact_match:.1%} predictions are completely correct")
        else:
            interpretations.append(f"   ❌ Poor: Only {exact_match:.1%} predictions are completely correct")
        
        # Jaccard interpretation
        if jaccard >= 0.6:
            interpretations.append("   ✅ Excellent label overlap with true conditions")
        elif jaccard >= 0.4:
            interpretations.append("   ✅ Good label overlap with true conditions")
        elif jaccard >= 0.2:
            interpretations.append("   ⚠️  Fair label overlap with true conditions")
        else:
            interpretations.append("   ❌ Poor label overlap - mostly wrong predictions")
        
        # Performance scaling assessment
        if n_samples > 100000:
            interpretations.append(f"   📊 Massive dataset performance: {n_samples:,} samples processed")
            comp_time = metrics.get('computation_time', 0)
            if comp_time > 0:
                rate = n_samples / comp_time
                interpretations.append(f"   ⚡ Processing rate: {rate:,.0f} samples/second")
        
        return "\n".join(interpretations)
    
    def _deployment_readiness_massive(self, metrics):
        """Assess spacecraft deployment readiness for massive datasets."""
        exact_match = metrics.get('exact_match_ratio', 0)
        jaccard = metrics.get('jaccard_score', 0)
        f1_macro = metrics.get('f1_macro', 0)
        hamming = metrics.get('hamming_loss', 0)
        quality_score = metrics.get('quality_score', 0)
        
        # Enhanced deployment criteria for massive datasets
        deployment_criteria = {
            'exact_match': exact_match >= 0.5,      # 50%+ completely correct
            'jaccard': jaccard >= 0.4,              # 40%+ label overlap
            'f1_macro': f1_macro >= 0.5,            # 50%+ F1 score
            'hamming': hamming <= 0.3,              # <30% label errors
            'quality': quality_score >= 0.6         # 60%+ quality score
        }
        
        passed_criteria = sum(deployment_criteria.values())
        total_criteria = len(deployment_criteria)
        
        readiness_lines = []
        
        if passed_criteria == total_criteria:
            readiness_lines.append("   ✅ READY for spacecraft deployment - all criteria met")
        elif passed_criteria >= total_criteria * 0.8:
            readiness_lines.append("   ⚠️  CONDITIONALLY READY - most criteria met")
        else:
            readiness_lines.append("   ❌ NOT READY - significant improvements needed")
        
        # Detailed criteria breakdown
        readiness_lines.append(f"   Criteria passed: {passed_criteria}/{total_criteria}")
        
        for criterion, passed in deployment_criteria.items():
            status = "✅" if passed else "❌"
            readiness_lines.append(f"     {status} {criterion.replace('_', ' ').title()}")
        
        return "\n".join(readiness_lines)
    
    def _generate_massive_dataset_recommendations(self, metrics):
        """Generate recommendations specific to massive datasets."""
        recommendations = []
        
        # Performance recommendations
        exact_match = metrics.get('exact_match_ratio', 0)
        jaccard = metrics.get('jaccard_score', 0)
        f1_macro = metrics.get('f1_macro', 0)
        n_samples = metrics.get('n_samples', 0)
        
        if exact_match < 0.5:
            recommendations.append("   • Use ensemble methods to improve exact match accuracy")
            recommendations.append("   • Consider threshold optimization for better precision")
        
        if jaccard < 0.4:
            recommendations.append("   • Implement class balancing for better label overlap")
            recommendations.append("   • Use cost-sensitive learning for rare CBUSHes")
        
        if f1_macro < 0.5:
            recommendations.append("   • Increase model complexity for better F1 scores")
            recommendations.append("   • Use focal loss for difficult examples")
        
        # Massive dataset specific recommendations
        if n_samples > 100000:
            recommendations.append("   • Consider incremental learning for even larger datasets")
            recommendations.append("   • Use distributed training for faster processing")
            recommendations.append("   • Implement online learning for real-time updates")
        
        # Memory and performance recommendations
        comp_time = metrics.get('computation_time', 0)
        if comp_time > 300:  # 5 minutes
            recommendations.append("   • Use parallel processing to reduce computation time")
            recommendations.append("   • Consider model approximation techniques")
        
        if not recommendations:
            recommendations.append("   • Model performance is satisfactory for massive datasets")
            recommendations.append("   • Consider A/B testing for production deployment")
        
        return "\n".join(recommendations)
    
    def create_performance_visualization(self, model_results, save_plots=True, output_dir=None):
        """Create comprehensive performance visualization plots for massive datasets."""
        print(f"\n📊 CREATING PERFORMANCE VISUALIZATIONS (MASSIVE DATASET)")
        
        if not model_results:
            print("❌ No model results to visualize")
            return []
        
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        plots_created = []
        
        try:
            # Set style for better visualization
            plt.style.use('seaborn-v0_8')
            
            # Plot 1: Enhanced Overall metrics comparison
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Massive Dataset Model Performance Comparison', fontsize=16, fontweight='bold')
            
            models = list(model_results.keys())
            
            # Exact Match comparison
            exact_matches = [model_results[m].get('exact_match_ratio', 0) for m in models]
            axes[0, 0].bar(models, exact_matches, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Exact Match Ratio', fontweight='bold')
            axes[0, 0].set_ylabel('Ratio')
            axes[0, 0].set_ylim(0, 1)
            for i, v in enumerate(exact_matches):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Jaccard Score comparison
            jaccard_scores = [model_results[m].get('jaccard_score', 0) for m in models]
            axes[0, 1].bar(models, jaccard_scores, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Jaccard Score', fontweight='bold')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_ylim(0, 1)
            for i, v in enumerate(jaccard_scores):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Hamming Loss comparison (lower is better)
            hamming_losses = [model_results[m].get('hamming_loss', 0) for m in models]
            axes[0, 2].bar(models, hamming_losses, color='lightcoral', alpha=0.7)
            axes[0, 2].set_title('Hamming Loss (Lower is Better)', fontweight='bold')
            axes[0, 2].set_ylabel('Loss')
            for i, v in enumerate(hamming_losses):
                axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # F1 Score comparison
            f1_scores = [model_results[m].get('f1_macro', 0) for m in models]
            axes[1, 0].bar(models, f1_scores, color='gold', alpha=0.7)
            axes[1, 0].set_title('F1 Score (Macro)', fontweight='bold')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_ylim(0, 1)
            for i, v in enumerate(f1_scores):
                axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Quality Score comparison
            quality_scores = [model_results[m].get('quality_score', 0) for m in models]
            axes[1, 1].bar(models, quality_scores, color='mediumpurple', alpha=0.7)
            axes[1, 1].set_title('Quality Score', fontweight='bold')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)
            for i, v in enumerate(quality_scores):
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Computation Time comparison
            comp_times = [model_results[m].get('computation_time', 0) for m in models]
            axes[1, 2].bar(models, comp_times, color='orange', alpha=0.7)
            axes[1, 2].set_title('Computation Time (seconds)', fontweight='bold')
            axes[1, 2].set_ylabel('Time (s)')
            for i, v in enumerate(comp_times):
                axes[1, 2].text(i, v + 0.01, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels for better readability
            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                filename = output_dir / 'massive_dataset_performance_comparison.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plots_created.append(str(filename))
                print(f"✅ Saved: {filename}")
            
            plt.close()
            
            # Plot 2: Per-CBUSH performance heatmap (if available)
            if any('per_label_metrics' in results for results in model_results.values()):
                heatmap_file = self._create_per_cbush_heatmap_massive(model_results, save_plots, output_dir)
                if heatmap_file:
                    plots_created.append(heatmap_file)
            
            # Plot 3: Deployment readiness radar chart
            radar_file = self._create_deployment_readiness_radar(model_results, save_plots, output_dir)
            if radar_file:
                plots_created.append(radar_file)
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
        
        return plots_created
    
    def _create_per_cbush_heatmap_massive(self, model_results, save_plots, output_dir):
        """Create per-CBUSH performance heatmap for massive datasets."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            fig.suptitle('Per-CBUSH Performance Analysis (Massive Dataset)', fontsize=16, fontweight='bold')
            
            metrics_to_plot = ['precision', 'recall', 'f1']
            metric_titles = ['Precision', 'Recall', 'F1-Score']
            
            for idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
                data = []
                model_names = []
                cbush_names = []
                
                for model_name, results in model_results.items():
                    if 'per_label_metrics' in results:
                        model_names.append(model_name.title())
                        model_data = []
                        
                        for cbush_name in self.cbush_names:
                            if cbush_name in results['per_label_metrics']:
                                value = results['per_label_metrics'][cbush_name].get(metric, 0)
                                model_data.append(value)
                            else:
                                model_data.append(0)
                        
                        data.append(model_data)
                        if not cbush_names:
                            cbush_names = self.cbush_names[:len(model_data)]
                
                if data:
                    data_array = np.array(data)
                    
                    im = axes[idx].imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                    axes[idx].set_title(f'Per-CBUSH {title}', fontweight='bold')
                    axes[idx].set_xticks(range(len(cbush_names)))
                    axes[idx].set_xticklabels([name.replace('CBUSH_', 'CB') for name in cbush_names], rotation=45)
                    axes[idx].set_yticks(range(len(model_names)))
                    axes[idx].set_yticklabels(model_names)
                    
                    # Add text annotations
                    for i in range(len(model_names)):
                        for j in range(len(cbush_names)):
                            text = axes[idx].text(j, i, f'{data_array[i, j]:.2f}',
                                                ha="center", va="center", color="black", fontsize=10, fontweight='bold')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=axes[idx])
            
            plt.tight_layout()
            
            if save_plots:
                filename = output_dir / 'per_cbush_performance_heatmap_massive.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✅ Saved: {filename}")
                plt.close()
                return str(filename)
            
        except Exception as e:
            print(f"❌ Error creating per-CBUSH heatmap: {e}")
        
        return None
    
    def _create_deployment_readiness_radar(self, model_results, save_plots, output_dir):
        """Create deployment readiness radar chart."""
        try:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            fig.suptitle('Spacecraft Deployment Readiness Assessment', fontsize=16, fontweight='bold')
            
            # Define readiness criteria
            criteria = ['Exact Match', 'Jaccard Score', 'F1 Macro', 'Low Hamming Loss', 'Quality Score']
            
            # Number of criteria
            N = len(criteria)
            
            # Angle for each criterion
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Colors for different models
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (model_name, results) in enumerate(model_results.items()):
                # Calculate readiness scores (0-1)
                readiness_scores = [
                    results.get('exact_match_ratio', 0),
                    results.get('jaccard_score', 0),
                    results.get('f1_macro', 0),
                    1 - results.get('hamming_loss', 1),  # Invert hamming loss
                    results.get('quality_score', 0)
                ]
                
                # Complete the circle
                readiness_scores += readiness_scores[:1]
                
                # Plot
                ax.plot(angles, readiness_scores, 'o-', linewidth=2, 
                       label=model_name.title(), color=colors[i % len(colors)])
                ax.fill(angles, readiness_scores, alpha=0.25, color=colors[i % len(colors)])
            
            # Add criteria labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(criteria)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            # Add grid
            ax.grid(True)
            
            if save_plots:
                filename = output_dir / 'deployment_readiness_radar.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✅ Saved: {filename}")
                plt.close()
                return str(filename)
            
        except Exception as e:
            print(f"❌ Error creating radar chart: {e}")
        
        return None
    
    def find_optimal_thresholds(self, y_true, y_prob, model_name="Model"):
        """Find optimal prediction thresholds for each CBUSH in massive datasets."""
        print(f"\n🎯 FINDING OPTIMAL THRESHOLDS FOR {model_name.upper()} (MASSIVE DATASET)")
        
        # Use sampling for very large datasets
        if len(y_true) > 100000:
            print(f"   Using sampling for threshold optimization ({len(y_true):,} samples)")
            sample_size = 50000
            indices = np.random.choice(len(y_true), sample_size, replace=False)
            y_true_sample = y_true[indices]
            y_prob_sample = y_prob[indices]
        else:
            y_true_sample = y_true
            y_prob_sample = y_prob
        
        optimal_thresholds = {}
        threshold_results = {}
        
        # Use parallel processing for multiple CBUSHes
        if self.enable_parallel:
            print(f"   Using parallel processing with {self.max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                for i, cbush_name in enumerate(self.cbush_names[:y_true_sample.shape[1]]):
                    future = executor.submit(
                        self._find_optimal_threshold_single_label,
                        y_true_sample[:, i],
                        y_prob_sample[:, i],
                        cbush_name
                    )
                    futures[future] = cbush_name
                
                for future in as_completed(futures):
                    cbush_name = futures[future]
                    try:
                        result = future.result()
                        optimal_thresholds[cbush_name] = result['optimal']
                        threshold_results[cbush_name] = result['details']
                    except Exception as e:
                        print(f"⚠️  Error finding threshold for {cbush_name}: {e}")
        else:
            # Sequential processing
            for i, cbush_name in enumerate(self.cbush_names[:y_true_sample.shape[1]]):
                result = self._find_optimal_threshold_single_label(
                    y_true_sample[:, i],
                    y_prob_sample[:, i],
                    cbush_name
                )
                optimal_thresholds[cbush_name] = result['optimal']
                threshold_results[cbush_name] = result['details']
        
        return optimal_thresholds, threshold_results
    
    def _find_optimal_threshold_single_label(self, y_true_label, y_prob_label, cbush_name):
        """Find optimal threshold for a single label."""
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        threshold_details = {
            'thresholds': [],
            'f1_scores': [],
            'precisions': [],
            'recalls': []
        }
        
        for threshold in thresholds:
            y_pred_threshold = (y_prob_label > threshold).astype(int)
            
            if len(np.unique(y_pred_threshold)) > 1:
                f1 = f1_score(y_true_label, y_pred_threshold, zero_division=0)
                precision = precision_score(y_true_label, y_pred_threshold, zero_division=0)
                recall = recall_score(y_true_label, y_pred_threshold, zero_division=0)
            else:
                f1 = precision = recall = 0
            
            threshold_details['thresholds'].append(threshold)
            threshold_details['f1_scores'].append(f1)
            threshold_details['precisions'].append(precision)
            threshold_details['recalls'].append(recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return {
            'optimal': {
                'threshold': best_threshold,
                'f1_score': best_f1
            },
            'details': threshold_details
        }
    
    def _manage_memory(self, context=""):
        """Enhanced memory management for massive datasets."""
        if not self.enable_memory_management:
            return
        
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_gb = memory_mb / 1024
            
            if memory_gb > self.max_memory_gb * 0.8:
                print(f"⚠️  High memory usage ({memory_gb:.1f}GB) at {context}, cleaning up...")
                
                # Clear cache if memory is high
                if len(self.metrics_cache) > 10:
                    self.metrics_cache.clear()
                    print(f"   Cleared metrics cache")
                
                # Force garbage collection
                collected = gc.collect()
                
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
    
    def save_comprehensive_report(self, model_results, output_file):
        """Save comprehensive report for all models to file."""
        try:
            with open(output_file, 'w') as f:
                f.write("COMPREHENSIVE MULTI-LABEL SCORING REPORT (MASSIVE DATASET)\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"System: Enhanced Scoring System v2.0 for Massive Datasets\n\n")
                
                # Overall summary
                f.write("OVERALL SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Models Evaluated: {len(model_results)}\n")
                
                total_samples = sum(results.get('n_samples', 0) for results in model_results.values())
                f.write(f"Total Samples Processed: {total_samples:,}\n")
                
                total_computation_time = sum(results.get('computation_time', 0) for results in model_results.values())
                f.write(f"Total Computation Time: {total_computation_time:.1f} seconds\n\n")
                
                # Model comparison
                comparison_df = self.compare_models(model_results)
                f.write("MODEL COMPARISON\n")
                f.write("-" * 40 + "\n")
                f.write(comparison_df.to_string(index=False))
                f.write("\n\n")
                
                # Detailed reports for each model
                for model_name, results in model_results.items():
                    detailed_report = self.create_detailed_report(results, model_name)
                    f.write(detailed_report)
                    f.write("\n\n")
                
                # Deployment recommendations
                f.write("DEPLOYMENT RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                
                deployment_ready_models = [
                    model_name for model_name, results in model_results.items()
                    if results.get('deployment_ready', False)
                ]
                
                if deployment_ready_models:
                    f.write(f"✅ Models ready for deployment: {', '.join(deployment_ready_models)}\n")
                else:
                    f.write("⚠️  No models currently meet deployment criteria\n")
                
                f.write("\nFor spacecraft deployment, prioritize models with:\n")
                f.write("• Exact Match Ratio ≥ 0.5\n")
                f.write("• Jaccard Score ≥ 0.4\n")
                f.write("• F1 Macro Score ≥ 0.5\n")
                f.write("• Hamming Loss ≤ 0.3\n")
                f.write("• Quality Score ≥ 0.6\n")
            
            print(f"✅ Comprehensive report saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return False


def main():
    """Example usage of Enhanced Comprehensive Multi-Label Scoring for massive datasets."""
    print("Enhanced Comprehensive Multi-Label Scoring ready for massive datasets!")
    print("Features:")
    print("  ✅ Support for 11K-111K+ designs")
    print("  ✅ Batch processing for memory efficiency")
    print("  ✅ Parallel computation for speed")
    print("  ✅ Memory management and caching")
    print("  ✅ Enhanced visualizations")
    print("  ✅ Deployment readiness assessment")
    print("  ✅ Comprehensive reporting")
    print("  ✅ Spacecraft deployment criteria")


if __name__ == "__main__":
    main()
