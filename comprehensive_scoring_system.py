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
warnings.filterwarnings('ignore')

class ComprehensiveMultiLabelScoring:
    """
    Comprehensive scoring system for multi-label CBUSH classification.
    Provides detailed metrics for Random Forest, Neural Network, and XGBoost.
    """
    
    def __init__(self, cbush_numbers=None):
        """Initialize the scoring system."""
        self.cbush_numbers = cbush_numbers or list(range(2, 11))
        self.cbush_names = [f'CBUSH_{i}' for i in self.cbush_numbers]
        
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """Calculate all multi-label metrics for a model."""
        print(f"\n🔍 CALCULATING COMPREHENSIVE METRICS FOR {model_name.upper()}")
        
        metrics = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'n_labels': y_true.shape[1] if y_true.ndim > 1 else len(self.cbush_numbers)
        }
        
        # === SAMPLE-BASED METRICS ===
        print("📊 Sample-based metrics...")
        
        # Exact match (subset accuracy)
        metrics['exact_match_ratio'] = accuracy_score(y_true, y_pred)
        metrics['subset_accuracy'] = metrics['exact_match_ratio']  # Same thing
        
        # Hamming loss
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # Jaccard score (IoU)
        metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
        metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        
        # === LABEL-BASED METRICS ===
        print("🏷️  Label-based metrics...")
        
        # Precision, Recall, F1 with different averaging
        for avg_type in ['micro', 'macro', 'weighted']:
            metrics[f'precision_{avg_type}'] = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
            metrics[f'recall_{avg_type}'] = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
            metrics[f'f1_{avg_type}'] = f1_score(y_true, y_pred, average=avg_type, zero_division=0)
        
        # Per-label metrics
        per_label_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_label_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_label_metrics'] = {}
        for i, cbush_name in enumerate(self.cbush_names[:len(per_label_precision)]):
            metrics['per_label_metrics'][cbush_name] = {
                'precision': per_label_precision[i],
                'recall': per_label_recall[i],
                'f1': per_label_f1[i]
            }
        
        # === PROBABILITY-BASED METRICS ===
        if y_prob is not None:
            print("📈 Probability-based metrics...")
            
            try:
                # ROC AUC per label
                roc_aucs = []
                avg_precisions = []
                
                for i in range(y_true.shape[1]):
                    # ROC AUC
                    try:
                        roc_auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                        roc_aucs.append(roc_auc)
                    except:
                        roc_aucs.append(0.5)  # Random performance
                    
                    # Average Precision
                    try:
                        avg_prec = average_precision_score(y_true[:, i], y_prob[:, i])
                        avg_precisions.append(avg_prec)
                    except:
                        avg_precisions.append(0.0)
                
                metrics['roc_auc_macro'] = np.mean(roc_aucs)
                metrics['average_precision_macro'] = np.mean(avg_precisions)
                
                # Per-label probability metrics
                for i, cbush_name in enumerate(self.cbush_names[:len(roc_aucs)]):
                    if cbush_name not in metrics['per_label_metrics']:
                        metrics['per_label_metrics'][cbush_name] = {}
                    metrics['per_label_metrics'][cbush_name]['roc_auc'] = roc_aucs[i]
                    metrics['per_label_metrics'][cbush_name]['avg_precision'] = avg_precisions[i]
                
                # === RANKING METRICS ===
                print("🎯 Ranking metrics...")
                
                # Coverage Error
                try:
                    metrics['coverage_error'] = coverage_error(y_true, y_prob)
                except:
                    metrics['coverage_error'] = float('inf')
                
                # Label Ranking Average Precision  
                try:
                    metrics['label_ranking_avg_precision'] = label_ranking_average_precision_score(y_true, y_prob)
                except:
                    metrics['label_ranking_avg_precision'] = 0.0
                
                # Label Ranking Loss
                try:
                    metrics['label_ranking_loss'] = label_ranking_loss(y_true, y_prob)
                except:
                    metrics['label_ranking_loss'] = 1.0
                    
            except Exception as e:
                print(f"⚠️  Warning: Some probability-based metrics failed: {e}")
        
        # === CONFUSION MATRICES ===
        print("📋 Confusion matrices...")
        try:
            # Multi-label confusion matrices (one per label)
            cm_matrices = multilabel_confusion_matrix(y_true, y_pred)
            metrics['confusion_matrices'] = {}
            
            for i, cbush_name in enumerate(self.cbush_names[:len(cm_matrices)]):
                cm = cm_matrices[i]
                tn, fp, fn, tp = cm.ravel()
                
                metrics['confusion_matrices'][cbush_name] = {
                    'true_negative': int(tn),
                    'false_positive': int(fp),
                    'false_negative': int(fn),
                    'true_positive': int(tp),
                    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0
                }
        except Exception as e:
            print(f"⚠️  Warning: Confusion matrix calculation failed: {e}")
        
        print(f"✅ Comprehensive metrics calculated for {model_name}")
        return metrics
    
    def compare_models(self, model_results):
        """Compare multiple models across all metrics."""
        print(f"\n📊 COMPARING {len(model_results)} MODELS")
        
        comparison_df = pd.DataFrame()
        
        for model_name, metrics in model_results.items():
            row = {
                'Model': model_name.title(),
                'Exact Match': f"{metrics.get('exact_match_ratio', 0):.3f}",
                'Jaccard (Samples)': f"{metrics.get('jaccard_score', 0):.3f}",
                'Jaccard (Macro)': f"{metrics.get('jaccard_macro', 0):.3f}",
                'Hamming Loss': f"{metrics.get('hamming_loss', 0):.3f}",
                'F1 (Macro)': f"{metrics.get('f1_macro', 0):.3f}",
                'F1 (Micro)': f"{metrics.get('f1_micro', 0):.3f}",
                'Precision (Macro)': f"{metrics.get('precision_macro', 0):.3f}",
                'Recall (Macro)': f"{metrics.get('recall_macro', 0):.3f}",
            }
            
            # Add probability-based metrics if available
            if 'roc_auc_macro' in metrics:
                row['ROC AUC (Macro)'] = f"{metrics['roc_auc_macro']:.3f}"
                row['Avg Precision (Macro)'] = f"{metrics['average_precision_macro']:.3f}"
                row['Coverage Error'] = f"{metrics.get('coverage_error', 0):.1f}"
                row['Label Ranking AP'] = f"{metrics.get('label_ranking_avg_precision', 0):.3f}"
            
            comparison_df = pd.concat([comparison_df, pd.DataFrame([row])], ignore_index=True)
        
        return comparison_df
    
    def create_detailed_report(self, metrics, model_name):
        """Create detailed text report for a single model."""
        report_lines = [
            f"{'='*60}",
            f"DETAILED PERFORMANCE REPORT: {model_name.upper()}",
            f"{'='*60}",
            f"",
            f"📊 SAMPLE-BASED METRICS:",
            f"   Exact Match Ratio: {metrics.get('exact_match_ratio', 0):.3f} ({metrics.get('exact_match_ratio', 0)*100:.1f}%)",
            f"   Jaccard Score (Samples): {metrics.get('jaccard_score', 0):.3f}",
            f"   Jaccard Score (Macro): {metrics.get('jaccard_macro', 0):.3f}",
            f"   Jaccard Score (Micro): {metrics.get('jaccard_micro', 0):.3f}",
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
                f"",
                f"🎯 RANKING METRICS:",
                f"   Coverage Error: {metrics.get('coverage_error', 0):.2f}",
                f"   Label Ranking Avg Precision: {metrics.get('label_ranking_avg_precision', 0):.3f}",
                f"   Label Ranking Loss: {metrics.get('label_ranking_loss', 0):.3f}",
            ])
        
        # Per-label breakdown
        if 'per_label_metrics' in metrics:
            report_lines.extend([
                f"",
                f"📋 PER-CBUSH BREAKDOWN:",
                f"   {'CBUSH':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'ROC-AUC':<8} {'Avg-Prec':<8}"
            ])
            
            for cbush_name, cbush_metrics in metrics['per_label_metrics'].items():
                precision = cbush_metrics.get('precision', 0)
                recall = cbush_metrics.get('recall', 0)
                f1 = cbush_metrics.get('f1', 0)
                roc_auc = cbush_metrics.get('roc_auc', 0)
                avg_prec = cbush_metrics.get('avg_precision', 0)
                
                report_lines.append(
                    f"   {cbush_name:<8} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f} {roc_auc:<8.3f} {avg_prec:<8.3f}"
                )
        
        # Performance interpretation
        report_lines.extend([
            f"",
            f"🎯 PERFORMANCE INTERPRETATION:",
            self._interpret_performance(metrics),
            f"",
            f"🚀 SPACECRAFT DEPLOYMENT READINESS:",
            self._deployment_readiness(metrics)
        ])
        
        return "\n".join(report_lines)
    
    def _interpret_performance(self, metrics):
        """Interpret model performance in plain language."""
        exact_match = metrics.get('exact_match_ratio', 0)
        jaccard = metrics.get('jaccard_score', 0)
        hamming = metrics.get('hamming_loss', 0)
        f1_macro = metrics.get('f1_macro', 0)
        
        interpretations = []
        
        # Exact Match interpretation
        if exact_match >= 0.8:
            interpretations.append("   ✅ Excellent: 80%+ predictions are completely correct")
        elif exact_match >= 0.6:
            interpretations.append("   ✅ Good: 60%+ predictions are completely correct")
        elif exact_match >= 0.4:
            interpretations.append("   ⚠️  Fair: 40%+ predictions are completely correct")
        else:
            interpretations.append("   ❌ Poor: <40% predictions are completely correct")
        
        # Jaccard interpretation
        if jaccard >= 0.7:
            interpretations.append("   ✅ Excellent label overlap with true conditions")
        elif jaccard >= 0.5:
            interpretations.append("   ✅ Good label overlap with true conditions")
        elif jaccard >= 0.3:
            interpretations.append("   ⚠️  Fair label overlap with true conditions")
        else:
            interpretations.append("   ❌ Poor label overlap - mostly wrong predictions")
        
        # Hamming Loss interpretation (assuming 9 labels)
        error_percentage = hamming * 100
        if hamming <= 0.1:
            interpretations.append(f"   ✅ Excellent: Only {error_percentage:.1f}% labels wrong on average")
        elif hamming <= 0.2:
            interpretations.append(f"   ✅ Good: {error_percentage:.1f}% labels wrong on average")
        elif hamming <= 0.3:
            interpretations.append(f"   ⚠️  Fair: {error_percentage:.1f}% labels wrong on average")
        else:
            interpretations.append(f"   ❌ Poor: {error_percentage:.1f}% labels wrong on average")
        
        return "\n".join(interpretations)
    
    def _deployment_readiness(self, metrics):
        """Assess spacecraft deployment readiness."""
        exact_match = metrics.get('exact_match_ratio', 0)
        jaccard = metrics.get('jaccard_score', 0)
        f1_macro = metrics.get('f1_macro', 0)
        
        # Spacecraft deployment criteria (stricter than general ML)
        spacecraft_ready = (
            exact_match >= 0.6 and  # 60%+ completely correct predictions
            jaccard >= 0.5 and     # 50%+ label overlap
            f1_macro >= 0.6         # 60%+ F1 score
        )
        
        if spacecraft_ready:
            return "   ✅ READY for spacecraft deployment - meets reliability criteria"
        else:
            readiness_issues = []
            if exact_match < 0.6:
                readiness_issues.append(f"exact match too low ({exact_match:.1%} < 60%)")
            if jaccard < 0.5:
                readiness_issues.append(f"label overlap too low ({jaccard:.1%} < 50%)")
            if f1_macro < 0.6:
                readiness_issues.append(f"F1 score too low ({f1_macro:.1%} < 60%)")
            
            return f"   ⚠️  NOT READY: {', '.join(readiness_issues)}"
    
    def create_performance_visualization(self, model_results, save_plots=True):
        """Create comprehensive performance visualization plots."""
        print(f"\n📊 CREATING PERFORMANCE VISUALIZATIONS")
        
        if not model_results:
            print("❌ No model results to visualize")
            return []
        
        plots_created = []
        
        # Plot 1: Overall metrics comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(model_results.keys())
        
        # Exact Match comparison
        exact_matches = [model_results[m].get('exact_match_ratio', 0) for m in models]
        ax1.bar(models, exact_matches, color='skyblue', alpha=0.7)
        ax1.set_title('Exact Match Ratio')
        ax1.set_ylabel('Ratio')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(exact_matches):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Jaccard Score comparison
        jaccard_scores = [model_results[m].get('jaccard_score', 0) for m in models]
        ax2.bar(models, jaccard_scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Jaccard Score (Samples)')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(jaccard_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Hamming Loss comparison (lower is better)
        hamming_losses = [model_results[m].get('hamming_loss', 0) for m in models]
        ax3.bar(models, hamming_losses, color='lightcoral', alpha=0.7)
        ax3.set_title('Hamming Loss (Lower is Better)')
        ax3.set_ylabel('Loss')
        for i, v in enumerate(hamming_losses):
            ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1 Score comparison
        f1_scores = [model_results[m].get('f1_macro', 0) for m in models]
        ax4.bar(models, f1_scores, color='gold', alpha=0.7)
        ax4.set_title('F1 Score (Macro)')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'model_performance_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plots_created.append(filename)
            print(f"✅ Saved: {filename}")
        
        plt.close()
        
        # Plot 2: Per-CBUSH performance heatmap
        if any('per_label_metrics' in results for results in model_results.values()):
            self._create_per_cbush_heatmap(model_results, save_plots, plots_created)
        
        return plots_created
    
    def _create_per_cbush_heatmap(self, model_results, save_plots, plots_created):
        """Create per-CBUSH performance heatmap."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
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
                axes[idx].set_title(f'Per-CBUSH {title}')
                axes[idx].set_xticks(range(len(cbush_names)))
                axes[idx].set_xticklabels([name.replace('CBUSH_', 'CB') for name in cbush_names], rotation=45)
                axes[idx].set_yticks(range(len(model_names)))
                axes[idx].set_yticklabels(model_names)
                
                # Add text annotations
                for i in range(len(model_names)):
                    for j in range(len(cbush_names)):
                        text = axes[idx].text(j, i, f'{data_array[i, j]:.2f}',
                                            ha="center", va="center", color="black", fontsize=8)
                
                # Add colorbar
                plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'per_cbush_performance_heatmap.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plots_created.append(filename)
            print(f"✅ Saved: {filename}")
        
        plt.close()
    
    def find_optimal_thresholds(self, y_true, y_prob, model_name="Model"):
        """Find optimal prediction thresholds for each CBUSH."""
        print(f"\n🎯 FINDING OPTIMAL THRESHOLDS FOR {model_name.upper()}")
        
        optimal_thresholds = {}
        threshold_results = {}
        
        for i, cbush_name in enumerate(self.cbush_names[:y_true.shape[1]]):
            y_true_label = y_true[:, i]
            y_prob_label = y_prob[:, i]
            
            # Try different thresholds
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_threshold = 0.5
            
            threshold_results[cbush_name] = {
                'thresholds': [],
                'f1_scores': [],
                'precisions': [],
                'recalls': []
            }
            
            for threshold in thresholds:
                y_pred_threshold = (y_prob_label > threshold).astype(int)
                
                if len(np.unique(y_pred_threshold)) > 1:  # Avoid division by zero
                    f1 = f1_score(y_true_label, y_pred_threshold, zero_division=0)
                    precision = precision_score(y_true_label, y_pred_threshold, zero_division=0)
                    recall = recall_score(y_true_label, y_pred_threshold, zero_division=0)
                else:
                    f1 = precision = recall = 0
                
                threshold_results[cbush_name]['thresholds'].append(threshold)
                threshold_results[cbush_name]['f1_scores'].append(f1)
                threshold_results[cbush_name]['precisions'].append(precision)
                threshold_results[cbush_name]['recalls'].append(recall)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds[cbush_name] = {
                'threshold': best_threshold,
                'f1_score': best_f1
            }
            
            print(f"   {cbush_name}: Optimal threshold = {best_threshold:.2f} (F1 = {best_f1:.3f})")
        
        return optimal_thresholds, threshold_results


def main():
    """Example usage of ComprehensiveMultiLabelScoring."""
    print("Comprehensive Multi-Label Scoring System ready!")
    print("Features:")
    print("  ✅ 15+ metrics for multi-label classification")
    print("  ✅ Model comparison across all metrics")
    print("  ✅ Per-CBUSH performance breakdown")
    print("  ✅ Spacecraft deployment readiness assessment")
    print("  ✅ Optimal threshold finding")
    print("  ✅ Comprehensive visualizations")


if __name__ == "__main__":
    main()