#!/usr/bin/env python3
"""
Feature Matrix Editor
====================

Complete tool for editing and enhancing HEEDS feature matrices with parameter data.
Handles failed case detection, individual study processing, and multi-study combination.

Features:
- Automatic POST_0 folder scanning for failed case detection
- Individual study processing with full traceability
- Multi-study combination with global ML IDs
- Comprehensive validation and logging

Usage:
    python feature_matrix_editor.py --help

Author: Enhanced Bolt Detection System
Version: 1.0
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')


class FeatureMatrixEditor:
    """
    Complete editor for HEEDS feature matrices with parameter integration and traceability.
    """
    
    def __init__(self, output_dir="edited_output", verbose=True):
        """Initialize the feature matrix editor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Tracking
        self.processing_log = []
        self.failed_operations = []
        self.processed_studies = {}
        
        # Define parameter columns (pre-HEEDS inputs)
        self.parameter_columns = [
            'K4_1', 'K5_1', 'K6_1', 'K4_2', 'K5_2', 'K6_2', 'K4_3', 'K5_3', 'K6_3',
            'K4_4', 'K5_4', 'K6_4', 'K4_5', 'K5_5', 'K6_5', 'K4_6', 'K5_6', 'K6_6',
            'K4_7', 'K5_7', 'K6_7', 'K4_8', 'K5_8', 'K6_8', 'K4_9', 'K5_9', 'K6_9',
            'K4_10', 'K5_10', 'K6_10'
        ]
        
        self.log("🚀 Feature Matrix Editor Initialized")
        self.log(f"📁 Output directory: {self.output_dir}")
    
    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.processing_log.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def scan_post0_folder(self, heeds_directory):
        """
        Scan POST_0 folder to identify successful HEEDS designs.
        
        Args:
            heeds_directory: Path to HEEDS output directory containing POST_0
            
        Returns:
            dict: Scan results with successful and failed designs
        """
        try:
            self.log("🔍 Scanning POST_0 folder for successful designs...")
            
            heeds_path = Path(heeds_directory)
            post0_path = heeds_path / "POST_0"
            
            if not post0_path.exists():
                raise ValueError(f"POST_0 folder not found in {heeds_directory}")
            
            successful_designs = []
            failed_designs = []
            
            # Scan for Design_XXXX folders
            for item in post0_path.iterdir():
                if item.is_dir() and item.name.startswith("Design"):
                    try:
                        # Extract design number
                        design_num = int(item.name.replace("Design", ""))
                        
                        # Check if Analysis_1 folder exists with required files
                        analysis_path = item / "Analysis_1"
                        if analysis_path.exists():
                            accel_file = analysis_path / "acceleration_results.csv"
                            disp_file = analysis_path / "displacement_results.csv"
                            
                            if accel_file.exists() and disp_file.exists():
                                successful_designs.append(design_num)
                            else:
                                failed_designs.append(design_num)
                        else:
                            failed_designs.append(design_num)
                            
                    except ValueError:
                        # Skip non-numeric Design folders
                        continue
            
            # Sort for consistent ordering
            successful_designs.sort()
            failed_designs.sort()
            
            scan_results = {
                'successful_designs': successful_designs,
                'failed_designs': failed_designs,
                'total_scanned': len(successful_designs) + len(failed_designs),
                'success_rate': len(successful_designs) / (len(successful_designs) + len(failed_designs)) if (len(successful_designs) + len(failed_designs)) > 0 else 0
            }
            
            self.log(f"   ✅ Successful designs: {len(successful_designs):,}")
            self.log(f"   ❌ Failed designs: {len(failed_designs):,}")
            self.log(f"   📊 Success rate: {scan_results['success_rate']:.1%}")
            
            if failed_designs:
                self.log(f"   🔍 Failed design examples: {failed_designs[:10]}")
            
            return scan_results
            
        except Exception as e:
            error_msg = f"❌ POST_0 scan failed: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None
    
    def load_parameter_file(self, param_file, study_name=None):
        """
        Load HEEDS parameter file and identify structure.
        
        Args:
            param_file: Path to parameter CSV file
            study_name: Optional study name override
            
        Returns:
            dict: Parameter file information
        """
        try:
            param_file = Path(param_file)
            self.log(f"📋 Loading parameter file: {param_file.name}")
            
            # Auto-detect delimiter
            with open(param_file, 'r') as f:
                first_line = f.readline()
                delimiter = '\t' if '\t' in first_line else ','
            
            # Load parameter file
            param_df = pd.read_csv(param_file, delimiter=delimiter)
            
            # Create design_id mapping (row number = design number)
            param_df['heeds_design_id'] = range(1, len(param_df) + 1)
            
            # Identify available parameter columns
            available_params = [col for col in self.parameter_columns if col in param_df.columns]
            missing_params = [col for col in self.parameter_columns if col not in param_df.columns]
            
            param_info = {
                'file': param_file,
                'study_name': study_name or param_file.stem,
                'data': param_df,
                'total_designs': len(param_df),
                'available_parameters': available_params,
                'missing_parameters': missing_params,
                'delimiter': delimiter
            }
            
            self.log(f"   ✅ Loaded {len(param_df):,} parameter sets")
            self.log(f"   📊 Available parameters: {len(available_params)}/{len(self.parameter_columns)}")
            
            if missing_params:
                self.log(f"   ⚠️  Missing parameters: {missing_params}")
            
            return param_info
            
        except Exception as e:
            error_msg = f"❌ Failed to load parameter file {param_file}: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None
    
    def load_feature_matrix(self, matrix_file):
        """
        Load extracted feature matrix (post-HEEDS data).
        
        Args:
            matrix_file: Path to feature matrix CSV
            
        Returns:
            dict: Feature matrix information
        """
        try:
            matrix_file = Path(matrix_file)
            self.log(f"📈 Loading feature matrix: {matrix_file.name}")
            
            # Load feature matrix
            feature_df = pd.read_csv(matrix_file)
            
            # Identify column types
            feature_cols = [col for col in feature_df.columns 
                           if col.startswith(('ACCE_', 'DISP_'))]
            cbush_cols = [col for col in feature_df.columns 
                         if col.startswith('cbush_') and col.endswith('_loose')]
            meta_cols = [col for col in feature_df.columns 
                        if col in ['design_id', 'study_name', 'study_type', 'extraction_method', 'study_description']]
            
            # Get design IDs
            if 'design_id' in feature_df.columns:
                design_ids = sorted(feature_df['design_id'].unique())
            else:
                raise ValueError("Feature matrix missing 'design_id' column")
            
            matrix_info = {
                'file': matrix_file,
                'data': feature_df,
                'total_designs': len(feature_df),
                'feature_columns': feature_cols,
                'cbush_columns': cbush_cols,
                'meta_columns': meta_cols,
                'design_ids': design_ids,
                'design_range': (min(design_ids), max(design_ids))
            }
            
            self.log(f"   ✅ Loaded {len(feature_df):,} feature vectors")
            self.log(f"   📊 Feature columns: {len(feature_cols)}")
            self.log(f"   🔧 CBUSH columns: {len(cbush_cols)}")
            self.log(f"   📏 Design range: {matrix_info['design_range'][0]}-{matrix_info['design_range'][1]}")
            
            return matrix_info
            
        except Exception as e:
            error_msg = f"❌ Failed to load feature matrix {matrix_file}: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None
    
    def filter_to_successful_designs(self, param_info, matrix_info, scan_results):
        """
        Filter both parameter and feature data to only successful designs.
        
        Args:
            param_info: Parameter file information
            matrix_info: Feature matrix information  
            scan_results: POST_0 scan results
            
        Returns:
            tuple: (filtered_param_df, filtered_feature_df, alignment_stats)
        """
        try:
            self.log("🔄 Filtering to successful designs only...")
            
            successful_designs = set(scan_results['successful_designs'])
            param_designs = set(param_info['data']['heeds_design_id'].values)
            feature_designs = set(matrix_info['design_ids'])
            
            # Find intersection of all three
            valid_designs = successful_designs & param_designs & feature_designs
            
            # Calculate alignment statistics
            alignment_stats = {
                'post0_successful': len(successful_designs),
                'param_available': len(param_designs),
                'feature_available': len(feature_designs),
                'final_valid': len(valid_designs),
                'post0_failed': len(scan_results['failed_designs']),
                'param_missing': len(param_designs - successful_designs),
                'feature_missing': len(feature_designs - successful_designs),
                'alignment_rate': len(valid_designs) / len(param_designs) if len(param_designs) > 0 else 0
            }
            
            # Filter parameter data
            param_df = param_info['data'].copy()
            filtered_param_df = param_df[param_df['heeds_design_id'].isin(valid_designs)].copy()
            
            # Filter feature data  
            feature_df = matrix_info['data'].copy()
            filtered_feature_df = feature_df[feature_df['design_id'].isin(valid_designs)].copy()
            
            # Sort both by design_id for consistent ordering
            filtered_param_df = filtered_param_df.sort_values('heeds_design_id').reset_index(drop=True)
            filtered_feature_df = filtered_feature_df.sort_values('design_id').reset_index(drop=True)
            
            self.log(f"   📊 POST_0 successful: {alignment_stats['post0_successful']:,}")
            self.log(f"   📊 Parameter available: {alignment_stats['param_available']:,}")
            self.log(f"   📊 Feature available: {alignment_stats['feature_available']:,}")
            self.log(f"   ✅ Final valid designs: {alignment_stats['final_valid']:,}")
            self.log(f"   📈 Alignment rate: {alignment_stats['alignment_rate']:.1%}")
            
            if alignment_stats['alignment_rate'] < 0.8:
                self.log(f"   ⚠️  LOW ALIGNMENT RATE: {alignment_stats['alignment_rate']:.1%}")
            
            return filtered_param_df, filtered_feature_df, alignment_stats
            
        except Exception as e:
            error_msg = f"❌ Failed to filter to successful designs: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None, None, None
    
    def merge_parameter_features(self, param_df, feature_df, study_name):
        """
        Merge filtered parameter and feature data with traceability.
        
        Args:
            param_df: Filtered parameter dataframe
            feature_df: Filtered feature dataframe
            study_name: Study name for identification
            
        Returns:
            pd.DataFrame: Merged enhanced dataset
        """
        try:
            self.log("🔗 Merging parameter and feature data...")
            
            # Merge on design IDs
            merged_df = pd.merge(
                feature_df,
                param_df,
                left_on='design_id',
                right_on='heeds_design_id',
                how='inner'
            )
            
            # Create ML ID (sequential within study)
            merged_df['ml_id'] = range(1, len(merged_df) + 1)
            
            # Ensure study name consistency
            merged_df['study_name'] = study_name
            
            # Reorder columns for clarity
            id_columns = ['ml_id', 'heeds_design_id', 'study_name']
            param_columns = [col for col in merged_df.columns if col in self.parameter_columns]
            cbush_columns = [col for col in merged_df.columns if col.startswith('cbush_') and col.endswith('_loose')]
            feature_columns = [col for col in merged_df.columns if col.startswith(('ACCE_', 'DISP_'))]
            meta_columns = [col for col in merged_df.columns if col in ['study_type', 'study_description', 'extraction_method']]
            
            # Final column order
            final_columns = (id_columns + param_columns + cbush_columns + 
                           feature_columns + meta_columns)
            
            # Only include columns that exist
            available_columns = [col for col in final_columns if col in merged_df.columns]
            merged_df = merged_df[available_columns]
            
            self.log(f"   ✅ Merged dataset: {len(merged_df):,} designs")
            self.log(f"   📊 Total columns: {len(merged_df.columns)}")
            self.log(f"   🔧 Parameters: {len(param_columns)}")
            self.log(f"   📈 Features: {len(feature_columns)}")
            self.log(f"   🎯 CBUSH labels: {len(cbush_columns)}")
            
            return merged_df
            
        except Exception as e:
            error_msg = f"❌ Failed to merge parameter and feature data: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None
    
    def process_single_study(self, param_file, matrix_file, heeds_directory, study_name=None, output_name=None):
        """
        Process a single study: parameter file + feature matrix + POST_0 validation.
        
        Args:
            param_file: Path to parameter file
            matrix_file: Path to feature matrix
            heeds_directory: Path to HEEDS directory with POST_0
            study_name: Optional study name
            output_name: Optional output filename
            
        Returns:
            dict: Processing results
        """
        self.log(f"\n🔄 PROCESSING SINGLE STUDY")
        self.log("=" * 50)
        
        try:
            # Step 1: Scan POST_0 for successful designs
            scan_results = self.scan_post0_folder(heeds_directory)
            if not scan_results:
                return None
            
            # Step 2: Load parameter file
            param_info = self.load_parameter_file(param_file, study_name)
            if not param_info:
                return None
            
            # Step 3: Load feature matrix
            matrix_info = self.load_feature_matrix(matrix_file)
            if not matrix_info:
                return None
            
            # Step 4: Filter to successful designs only
            filtered_param_df, filtered_feature_df, alignment_stats = self.filter_to_successful_designs(
                param_info, matrix_info, scan_results
            )
            if filtered_param_df is None:
                return None
            
            # Step 5: Merge parameter and feature data
            study_name = study_name or param_info['study_name']
            merged_df = self.merge_parameter_features(filtered_param_df, filtered_feature_df, study_name)
            if merged_df is None:
                return None
            
            # Step 6: Save enhanced study
            if output_name is None:
                output_name = f"enhanced_{study_name.lower().replace(' ', '_')}.csv"
            
            output_path = self.output_dir / output_name
            merged_df.to_csv(output_path, index=False)
            
            self.log(f"💾 Saved enhanced study: {output_path}")
            
            # Create processing summary
            processing_summary = {
                'study_name': study_name,
                'param_file': str(param_file),
                'matrix_file': str(matrix_file),
                'heeds_directory': str(heeds_directory),
                'output_file': str(output_path),
                'scan_results': scan_results,
                'alignment_stats': alignment_stats,
                'final_dataset_shape': merged_df.shape,
                'parameter_count': len([col for col in merged_df.columns if col in self.parameter_columns]),
                'feature_count': len([col for col in merged_df.columns if col.startswith(('ACCE_', 'DISP_'))]),
                'cbush_count': len([col for col in merged_df.columns if col.startswith('cbush_') and col.endswith('_loose')]),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Store for later combination
            self.processed_studies[study_name] = {
                'data': merged_df,
                'summary': processing_summary
            }
            
            return processing_summary
            
        except Exception as e:
            error_msg = f"❌ Single study processing failed: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None
    
    def combine_multiple_studies(self, study_files=None, output_name="final_enhanced_dataset.csv"):
        """
        Combine multiple processed studies into one dataset with global ML IDs.
        
        Args:
            study_files: List of enhanced study CSV files (if None, uses processed_studies)
            output_name: Output filename
            
        Returns:
            dict: Combination results
        """
        self.log(f"\n🔗 COMBINING MULTIPLE STUDIES")
        self.log("=" * 50)
        
        try:
            all_study_data = []
            study_composition = {}
            
            # Load studies (either from memory or files)
            if study_files:
                self.log("📁 Loading studies from files...")
                for study_file in study_files:
                    study_df = pd.read_csv(study_file)
                    study_name = study_df['study_name'].iloc[0] if 'study_name' in study_df.columns else Path(study_file).stem
                    all_study_data.append(study_df)
                    study_composition[study_name] = len(study_df)
                    self.log(f"   📊 {study_name}: {len(study_df):,} designs")
            else:
                self.log("💾 Using processed studies from memory...")
                for study_name, study_info in self.processed_studies.items():
                    study_df = study_info['data']
                    all_study_data.append(study_df)
                    study_composition[study_name] = len(study_df)
                    self.log(f"   📊 {study_name}: {len(study_df):,} designs")
            
            if not all_study_data:
                raise ValueError("No studies available for combination")
            
            # Combine all studies
            self.log("🔗 Combining datasets...")
            combined_df = pd.concat(all_study_data, ignore_index=True)
            
            # Create global ML IDs
            combined_df['ml_id'] = range(1, len(combined_df) + 1)
            
            # Check for design ID conflicts and handle them
            original_heeds_ids = combined_df['heeds_design_id'].tolist()
            study_names = combined_df['study_name'].tolist()
            
            # Create unique identifier combinations
            unique_combinations = set(zip(study_names, original_heeds_ids))
            
            combination_stats = {
                'total_studies': len(study_composition),
                'study_composition': study_composition,
                'total_combined_designs': len(combined_df),
                'unique_heeds_combinations': len(unique_combinations),
                'has_heeds_id_conflicts': len(unique_combinations) < len(combined_df),
                'global_ml_id_range': (1, len(combined_df))
            }
            
            self.log(f"   ✅ Combined {len(all_study_data)} studies")
            self.log(f"   📊 Total designs: {len(combined_df):,}")
            self.log(f"   🔢 Global ML ID range: 1-{len(combined_df)}")
            
            # Log study composition
            self.log("📈 Final dataset composition:")
            for study, count in study_composition.items():
                percentage = (count / len(combined_df)) * 100
                self.log(f"   {study}: {count:,} designs ({percentage:.1f}%)")
            
            # Save combined dataset
            output_path = self.output_dir / output_name
            combined_df.to_csv(output_path, index=False)
            
            self.log(f"💾 Saved combined dataset: {output_path}")
            
            # Create combination summary
            combination_summary = {
                'output_file': str(output_path),
                'combination_stats': combination_stats,
                'final_dataset_shape': combined_df.shape,
                'column_summary': {
                    'total_columns': len(combined_df.columns),
                    'parameter_columns': len([col for col in combined_df.columns if col in self.parameter_columns]),
                    'feature_columns': len([col for col in combined_df.columns if col.startswith(('ACCE_', 'DISP_'))]),
                    'cbush_columns': len([col for col in combined_df.columns if col.startswith('cbush_') and col.endswith('_loose')])
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return combination_summary
            
        except Exception as e:
            error_msg = f"❌ Study combination failed: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None
    
    def validate_enhanced_dataset(self, dataset_file):
        """
        Validate the final enhanced dataset for ML readiness.
        
        Args:
            dataset_file: Path to enhanced dataset CSV
            
        Returns:
            dict: Validation results
        """
        try:
            self.log(f"\n🔍 VALIDATING ENHANCED DATASET")
            self.log("=" * 40)
            
            # Load dataset
            df = pd.read_csv(dataset_file)
            
            # Basic structure validation
            required_columns = ['ml_id', 'heeds_design_id', 'study_name']
            missing_required = [col for col in required_columns if col not in df.columns]
            
            # Count different column types
            param_cols = [col for col in df.columns if col in self.parameter_columns]
            feature_cols = [col for col in df.columns if col.startswith(('ACCE_', 'DISP_'))]
            cbush_cols = [col for col in df.columns if col.startswith('cbush_') and col.endswith('_loose')]
            
            # Data quality checks
            validation_results = {
                'total_designs': len(df),
                'total_columns': len(df.columns),
                'missing_required_columns': missing_required,
                'parameter_columns': len(param_cols),
                'feature_columns': len(feature_cols),
                'cbush_label_columns': len(cbush_cols),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_ml_ids': df['ml_id'].duplicated().sum(),
                'ml_id_sequential': list(df['ml_id']) == list(range(1, len(df) + 1)),
                'study_composition': df['study_name'].value_counts().to_dict(),
                'heeds_id_range_by_study': {},
                'cbush_label_distribution': {},
                'data_quality_score': 0
            }
            
            # Analyze HEEDS ID ranges by study
            for study in df['study_name'].unique():
                study_data = df[df['study_name'] == study]
                heeds_ids = study_data['heeds_design_id'].values
                validation_results['heeds_id_range_by_study'][study] = {
                    'min': int(heeds_ids.min()),
                    'max': int(heeds_ids.max()),
                    'count': len(heeds_ids),
                    'has_gaps': len(set(heeds_ids)) != (heeds_ids.max() - heeds_ids.min() + 1)
                }
            
            # Analyze CBUSH label distribution
            for cbush_col in cbush_cols:
                loose_count = df[cbush_col].sum()
                validation_results['cbush_label_distribution'][cbush_col] = {
                    'loose_count': int(loose_count),
                    'tight_count': int(len(df) - loose_count),
                    'loose_percentage': float(loose_count / len(df) * 100)
                }
            
            # Calculate overall data quality score
            structure_score = 1.0 if not missing_required else 0.5
            completeness_score = 1.0 - (validation_results['missing_values'] / (len(df) * len(df.columns)))
            consistency_score = 1.0 if validation_results['ml_id_sequential'] and validation_results['duplicate_ml_ids'] == 0 else 0.5
            
            validation_results['data_quality_score'] = (structure_score + completeness_score + consistency_score) / 3
            
            # Log validation results
            self.log(f"📊 Dataset validation results:")
            self.log(f"   Total designs: {validation_results['total_designs']:,}")
            self.log(f"   Total columns: {validation_results['total_columns']}")
            self.log(f"   Parameter columns: {validation_results['parameter_columns']}")
            self.log(f"   Feature columns: {validation_results['feature_columns']}")
            self.log(f"   CBUSH label columns: {validation_results['cbush_label_columns']}")
            self.log(f"   Missing values: {validation_results['missing_values']:,}")
            self.log(f"   Data quality score: {validation_results['data_quality_score']:.1%}")
            
            if validation_results['data_quality_score'] >= 0.95:
                self.log("   ✅ EXCELLENT - Ready for ML training!")
            elif validation_results['data_quality_score'] >= 0.85:
                self.log("   👍 GOOD - Suitable for ML training")
            else:
                self.log("   ⚠️  Issues detected - Review before ML training")
            
            return validation_results
            
        except Exception as e:
            error_msg = f"❌ Dataset validation failed: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None
    
    def generate_comprehensive_report(self):
        """Generate comprehensive processing report."""
        report_path = self.output_dir / "processing_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("FEATURE MATRIX EDITOR PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Processing completed: {datetime.now()}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            
            if self.processed_studies:
                f.write(f"PROCESSED STUDIES ({len(self.processed_studies)}):\n")
                f.write("-" * 30 + "\n")
                for study_name, study_info in self.processed_studies.items():
                    summary = study_info['summary']
                    f.write(f"Study: {study_name}\n")
                    f.write(f"  Designs: {summary['final_dataset_shape'][0]:,}\n")
                    f.write(f"  Columns: {summary['final_dataset_shape'][1]}\n")
                    f.write(f"  Alignment: {summary['alignment_stats']['alignment_rate']:.1%}\n")
                    f.write(f"  Output: {summary['output_file']}\n\n")
            
            f.write("PROCESSING LOG:\n")
            f.write("-" * 20 + "\n")
            for log_entry in self.processing_log:
                f.write(log_entry + "\n")
            
            if self.failed_operations:
                f.write(f"\nFAILED OPERATIONS ({len(self.failed_operations)}):\n")
                f.write("-" * 30 + "\n")
                for failure in self.failed_operations:
                    f.write(failure + "\n")
        
        self.log(f"📋 Generated comprehensive report: {report_path}")
        return report_path


class FeatureMatrixEditorGUI:
    """
    Simple GUI interface for the Feature Matrix Editor.
    """
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Feature Matrix Editor v1.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize backend
        self.editor = None
        self.processing_thread = None
        
        # Variables for GUI inputs
        self.param_file_var = tk.StringVar()
        self.matrix_file_var = tk.StringVar()
        self.heeds_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value="edited_output")
        self.study_name_var = tk.StringVar()
        self.validate_var = tk.BooleanVar(value=True)
        
        # Create GUI
        self.create_widgets()
        
        # Data storage for previews
        self.param_data = None
        self.matrix_data = None
        self.combined_data = None
        self.loaded_full_matrix = None
        
        self.log_message("🚀 Feature Matrix Editor GUI Ready!")
        self.log_message("📋 Select your files and click 'Process Study' to begin")
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Title
        title_label = tk.Label(self.root, text="Feature Matrix Editor", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(self.root, text="HEEDS Parameter-Feature Integration with Traceability",
                                 font=('Arial', 12), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=5)
        
        # Create main container with horizontal layout
        main_container = tk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = tk.Frame(main_container)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Right panel for data previews
        right_panel = tk.Frame(main_container)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Create control widgets in left panel
        self.create_control_widgets(left_panel)
        
        # Create data preview tabs in right panel
        self.create_data_preview_tabs(right_panel)
    
    def create_control_widgets(self, parent):
        """Create the control widgets in the left panel."""
        # File selection frame
        file_frame = ttk.LabelFrame(parent, text="📁 File Selection")
        file_frame.pack(fill='x', pady=10)
        
        # Parameter file
        param_frame = tk.Frame(file_frame)
        param_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(param_frame, text="Parameter File (CSV/TSV):", font=('Arial', 10, 'bold')).pack(anchor='w')
        param_input_frame = tk.Frame(param_frame)
        param_input_frame.pack(fill='x', pady=2)
        tk.Entry(param_input_frame, textvariable=self.param_file_var, width=50, font=('Arial', 9)).pack(side='left', fill='x', expand=True)
        tk.Button(param_input_frame, text="Browse", command=self.browse_param_file, 
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold')).pack(side='right', padx=(5,0))
        
        # Feature matrix file
        matrix_frame = tk.Frame(file_frame)
        matrix_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(matrix_frame, text="Feature Matrix File (CSV):", font=('Arial', 10, 'bold')).pack(anchor='w')
        matrix_input_frame = tk.Frame(matrix_frame)
        matrix_input_frame.pack(fill='x', pady=2)
        tk.Entry(matrix_input_frame, textvariable=self.matrix_file_var, width=50, font=('Arial', 9)).pack(side='left', fill='x', expand=True)
        tk.Button(matrix_input_frame, text="Browse", command=self.browse_matrix_file,
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold')).pack(side='right', padx=(5,0))
        
        # HEEDS directory
        heeds_frame = tk.Frame(file_frame)
        heeds_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(heeds_frame, text="HEEDS Directory (containing POST_0):", font=('Arial', 10, 'bold')).pack(anchor='w')
        heeds_input_frame = tk.Frame(heeds_frame)
        heeds_input_frame.pack(fill='x', pady=2)
        tk.Entry(heeds_input_frame, textvariable=self.heeds_dir_var, width=50, font=('Arial', 9)).pack(side='left', fill='x', expand=True)
        tk.Button(heeds_input_frame, text="Browse", command=self.browse_heeds_dir,
                 bg='#3498db', fg='white', font=('Arial', 9, 'bold')).pack(side='right', padx=(5,0))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(parent, text="⚙️ Processing Settings")
        settings_frame.pack(fill='x', pady=10)
        
        # Study name
        study_frame = tk.Frame(settings_frame)
        study_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(study_frame, text="Study Name:", font=('Arial', 10, 'bold')).pack(anchor='w')
        tk.Entry(study_frame, textvariable=self.study_name_var, width=40, font=('Arial', 9)).pack(fill='x', pady=2)
        
        # Output directory
        output_frame = tk.Frame(settings_frame)
        output_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(output_frame, text="Output Directory:", font=('Arial', 10, 'bold')).pack(anchor='w')
        output_input_frame = tk.Frame(output_frame)
        output_input_frame.pack(fill='x', pady=2)
        tk.Entry(output_input_frame, textvariable=self.output_dir_var, width=30, font=('Arial', 9)).pack(side='left', fill='x', expand=True)
        tk.Button(output_input_frame, text="Browse", command=self.browse_output_dir,
                 bg='#95a5a6', fg='white', font=('Arial', 8, 'bold')).pack(side='right', padx=(5,0))
        
        # Options
        options_frame = tk.Frame(settings_frame)
        options_frame.pack(fill='x', padx=10, pady=5)
        tk.Checkbutton(options_frame, text="Validate final dataset", variable=self.validate_var,
                      font=('Arial', 10)).pack(anchor='w')
        
        # Processing controls
        control_frame = ttk.LabelFrame(parent, text="🚀 Processing Controls")
        control_frame.pack(fill='x', pady=10)
        
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text="🔄 Process Study", command=self.process_single_study,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), height=2, width=18).pack(fill='x', pady=2)
        
        tk.Button(button_frame, text="🔗 Combine Studies", command=self.combine_studies,
                 bg='#9b59b6', fg='white', font=('Arial', 11, 'bold'), height=2, width=18).pack(fill='x', pady=2)
        
        tk.Button(button_frame, text="📋 Clear Log", command=self.clear_log,
                 bg='#95a5a6', fg='white', font=('Arial', 10, 'bold'), height=1, width=18).pack(fill='x', pady=2)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(parent, text="📊 Progress")
        progress_frame.pack(fill='x', pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=300, mode='indeterminate')
        self.progress_bar.pack(pady=10)
        
        self.status_label = tk.Label(progress_frame, text="Ready", font=('Arial', 10), fg='#7f8c8d')
        self.status_label.pack()
        
        # Log frame
        log_frame = ttk.LabelFrame(parent, text="📜 Processing Log")
        log_frame.pack(fill='both', expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, width=60, 
                                                 font=('Consolas', 8))
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_data_preview_tabs(self, parent):
        """Create the data preview tabs in the right panel."""
        # Create notebook for data preview tabs
        self.preview_notebook = ttk.Notebook(parent)
        self.preview_notebook.pack(fill='both', expand=True)
        
        # Tab 1: Parameter File Preview
        self.param_tab = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.param_tab, text="📋 Parameter File")
        self.create_param_preview_tab(self.param_tab)
        
        # Tab 2: Feature Matrix Preview
        self.matrix_tab = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.matrix_tab, text="📈 Feature Matrix")
        self.create_matrix_preview_tab(self.matrix_tab)
        
        # Tab 3: Full Feature Matrix (Combined)
        self.combined_tab = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.combined_tab, text="🔗 Full Feature Matrix")
        self.create_combined_preview_tab(self.combined_tab)
        
        # Tab 4: Load Full Feature Matrix
        self.load_tab = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.load_tab, text="📁 Load Full Matrix")
        self.create_load_preview_tab(self.load_tab)
    
    def create_param_preview_tab(self, parent):
        """Create parameter file preview tab."""
        # Info frame
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.param_info_label = tk.Label(info_frame, text="No parameter file loaded", 
                                        font=('Arial', 11, 'bold'), fg='#7f8c8d')
        self.param_info_label.pack(anchor='w')
        
        # Table frame
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.param_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Scrollbars
        param_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.param_tree.yview)
        param_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.param_tree.xview)
        self.param_tree.configure(yscrollcommand=param_v_scroll.set, xscrollcommand=param_h_scroll.set)
        
        # Pack treeview and scrollbars
        self.param_tree.pack(side='left', fill='both', expand=True)
        param_v_scroll.pack(side='right', fill='y')
        param_h_scroll.pack(side='bottom', fill='x')
    
    def create_matrix_preview_tab(self, parent):
        """Create feature matrix preview tab."""
        # Info frame
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.matrix_info_label = tk.Label(info_frame, text="No feature matrix loaded", 
                                         font=('Arial', 11, 'bold'), fg='#7f8c8d')
        self.matrix_info_label.pack(anchor='w')
        
        # Table frame
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.matrix_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Scrollbars
        matrix_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.matrix_tree.yview)
        matrix_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.matrix_tree.xview)
        self.matrix_tree.configure(yscrollcommand=matrix_v_scroll.set, xscrollcommand=matrix_h_scroll.set)
        
        # Pack treeview and scrollbars
        self.matrix_tree.pack(side='left', fill='both', expand=True)
        matrix_v_scroll.pack(side='right', fill='y')
        matrix_h_scroll.pack(side='bottom', fill='x')
    
    def create_combined_preview_tab(self, parent):
        """Create combined data preview tab."""
        # Info frame
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.combined_info_label = tk.Label(info_frame, text="No combined data available - process a study first", 
                                           font=('Arial', 11, 'bold'), fg='#7f8c8d')
        self.combined_info_label.pack(anchor='w')
        
        # Table frame
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.combined_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Scrollbars
        combined_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.combined_tree.yview)
        combined_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.combined_tree.xview)
        self.combined_tree.configure(yscrollcommand=combined_v_scroll.set, xscrollcommand=combined_h_scroll.set)
        
        # Pack treeview and scrollbars
        self.combined_tree.pack(side='left', fill='both', expand=True)
        combined_v_scroll.pack(side='right', fill='y')
        combined_h_scroll.pack(side='bottom', fill='x')
    
    def create_load_preview_tab(self, parent):
        """Create load full matrix preview tab."""
        # Controls frame
        controls_frame = tk.Frame(parent)
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(controls_frame, text="Load existing Full Feature Matrix:", 
                font=('Arial', 12, 'bold')).pack(anchor='w')
        
        load_button_frame = tk.Frame(controls_frame)
        load_button_frame.pack(fill='x', pady=10)
        
        tk.Button(load_button_frame, text="📁 Browse Full Feature Matrix", 
                 command=self.browse_full_matrix,
                 bg='#e67e22', fg='white', font=('Arial', 11, 'bold'),
                 height=2, width=30).pack(side='left')
        
        # Info frame
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.load_info_label = tk.Label(info_frame, text="No full feature matrix loaded", 
                                       font=('Arial', 11, 'bold'), fg='#7f8c8d')
        self.load_info_label.pack(anchor='w')
        
        # Table frame
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.load_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Scrollbars
        load_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.load_tree.yview)
        load_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.load_tree.xview)
        self.load_tree.configure(yscrollcommand=load_v_scroll.set, xscrollcommand=load_h_scroll.set)
        
        # Pack treeview and scrollbars
        self.load_tree.pack(side='left', fill='both', expand=True)
        load_v_scroll.pack(side='right', fill='y')
        load_h_scroll.pack(side='bottom', fill='x')
    
    def populate_data_table(self, tree, data, info_label, max_rows=500):
        """Populate a treeview with data from a DataFrame."""
        # Clear existing data
        for item in tree.get_children():
            tree.delete(item)
        
        if data is None or data.empty:
            info_label.config(text="No data available", fg='#e74c3c')
            return
        
        # Limit rows for performance
        display_data = data.head(max_rows)
        
        # Configure columns
        columns = list(display_data.columns)
        tree['columns'] = columns
        
        # Set column headings and widths
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=50)
        
        # Insert data
        for index, row in display_data.iterrows():
            values = []
            for col in columns:
                value = row[col]
                # Handle different data types
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, (int, float)):
                    if isinstance(value, float):
                        values.append(f"{value:.3f}")
                    else:
                        values.append(str(value))
                else:
                    values.append(str(value))
            
            tree.insert('', 'end', values=values)
        
        # Update info label
        total_rows = len(data)
        showing_rows = len(display_data)
        missing_values = data.isnull().sum().sum()
        
        info_text = f"📊 Showing {showing_rows:,} of {total_rows:,} rows | {len(columns)} columns | {missing_values:,} missing values"
        if showing_rows < total_rows:
            info_text += f" | ⚠️ Limited to first {max_rows:,} rows for performance"
        
        info_label.config(text=info_text, fg='#27ae60')
    
    def load_param_file_preview(self, filename):
        """Load and display parameter file preview."""
        try:
            self.log_message(f"📋 Loading parameter file preview: {Path(filename).name}")
            
            # Auto-detect delimiter
            with open(filename, 'r') as f:
                first_line = f.readline()
                delimiter = '\t' if '\t' in first_line else ','
            
            # Load data
            self.param_data = pd.read_csv(filename, delimiter=delimiter)
            
            # Add heeds_design_id for reference
            self.param_data['heeds_design_id'] = range(1, len(self.param_data) + 1)
            
            # Populate table
            self.populate_data_table(self.param_tree, self.param_data, self.param_info_label)
            
            self.log_message(f"   ✅ Parameter preview loaded: {len(self.param_data):,} designs")
            
        except Exception as e:
            error_msg = f"Failed to load parameter preview: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            self.param_info_label.config(text=f"❌ Error: {error_msg}", fg='#e74c3c')
    
    def load_matrix_file_preview(self, filename):
        """Load and display feature matrix preview."""
        try:
            self.log_message(f"📊 Loading feature matrix preview: {Path(filename).name}")
            
            # Load data
            self.matrix_data = pd.read_csv(filename)
            
            # Populate table
            self.populate_data_table(self.matrix_tree, self.matrix_data, self.matrix_info_label)
            
            self.log_message(f"   ✅ Feature matrix preview loaded: {len(self.matrix_data):,} designs")
            
        except Exception as e:
            error_msg = f"Failed to load feature matrix preview: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            self.matrix_info_label.config(text=f"❌ Error: {error_msg}", fg='#e74c3c')
    
    def browse_full_matrix(self):
        """Browse for existing full feature matrix file."""
        filename = filedialog.askopenfilename(
            title="Select Full Feature Matrix CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            self.load_full_matrix_preview(filename)
            self.log_message(f"📁 Full feature matrix selected: {Path(filename).name}")
    
    def load_full_matrix_preview(self, filename):
        """Load and display full feature matrix preview."""
        try:
            self.log_message(f"📁 Loading full feature matrix preview: {Path(filename).name}")
            
            # Load data
            self.loaded_full_matrix = pd.read_csv(filename)
            
            # Populate table
            self.populate_data_table(self.load_tree, self.loaded_full_matrix, self.load_info_label)
            
            self.log_message(f"   ✅ Full feature matrix preview loaded: {len(self.loaded_full_matrix):,} designs")
            
        except Exception as e:
            error_msg = f"Failed to load full feature matrix preview: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            self.load_info_label.config(text=f"❌ Error: {error_msg}", fg='#e74c3c')
    
    def browse_param_file(self):
        """Browse for parameter file."""
        filename = filedialog.askopenfilename(
            title="Select Parameter File",
            filetypes=[("CSV Files", "*.csv"), ("TSV Files", "*.tsv"), ("All Files", "*.*")]
        )
        if filename:
            self.param_file_var.set(filename)
            self.log_message(f"📋 Parameter file selected: {Path(filename).name}")
            # Auto-load preview
            self.load_param_file_preview(filename)
    
    def browse_matrix_file(self):
        """Browse for feature matrix file."""
        filename = filedialog.askopenfilename(
            title="Select Feature Matrix File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            self.matrix_file_var.set(filename)
            self.log_message(f"📊 Feature matrix selected: {Path(filename).name}")
            # Auto-load preview
            self.load_matrix_file_preview(filename)
    
    def browse_heeds_dir(self):
        """Browse for HEEDS directory."""
        dirname = filedialog.askdirectory(title="Select HEEDS Directory (containing POST_0)")
        if dirname:
            self.heeds_dir_var.set(dirname)
            self.log_message(f"📁 HEEDS directory selected: {Path(dirname).name}")
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir_var.set(dirname)
            self.log_message(f"💾 Output directory set: {Path(dirname).name}")
    
    def validate_inputs(self):
        """Validate all required inputs."""
        if not self.param_file_var.get():
            messagebox.showerror("Missing Input", "Please select a parameter file")
            return False
        
        if not self.matrix_file_var.get():
            messagebox.showerror("Missing Input", "Please select a feature matrix file")
            return False
        
        if not self.heeds_dir_var.get():
            messagebox.showerror("Missing Input", "Please select a HEEDS directory")
            return False
        
        # Check if files exist
        if not Path(self.param_file_var.get()).exists():
            messagebox.showerror("File Not Found", f"Parameter file not found:\n{self.param_file_var.get()}")
            return False
        
        if not Path(self.matrix_file_var.get()).exists():
            messagebox.showerror("File Not Found", f"Feature matrix file not found:\n{self.matrix_file_var.get()}")
            return False
        
        if not Path(self.heeds_dir_var.get()).exists():
            messagebox.showerror("Directory Not Found", f"HEEDS directory not found:\n{self.heeds_dir_var.get()}")
            return False
        
        return True
    
    def log_message(self, message):
        """Log message to the text display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete('1.0', tk.END)
        self.log_message("📋 Log cleared")
    
    def update_status(self, message):
        """Update status label."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def start_progress(self):
        """Start progress bar animation."""
        self.progress_bar.configure(mode='indeterminate')
        self.progress_bar.start()
    
    def stop_progress(self):
        """Stop progress bar animation."""
        self.progress_bar.stop()
        self.progress_bar.configure(mode='determinate')
        self.progress_var.set(100)
    
    def process_single_study(self):
        """Process a single study in background thread."""
        if not self.validate_inputs():
            return
        
        # Start processing in background
        self.start_progress()
        self.update_status("Processing single study...")
        self.log_message("🚀 Starting single study processing...")
        
        self.processing_thread = threading.Thread(target=self._process_single_study_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_single_study_thread(self):
        """Process single study in background thread."""
        try:
            # Initialize editor with GUI log callback
            self.editor = FeatureMatrixEditor(
                output_dir=self.output_dir_var.get(),
                verbose=False  # We'll handle logging through GUI
            )
            
            # Override the editor's log method to send to GUI
            original_log = self.editor.log
            def gui_log(message):
                self.root.after(0, lambda: self.log_message(message))
            self.editor.log = gui_log
            
            # Process the study
            result = self.editor.process_single_study(
                param_file=self.param_file_var.get(),
                matrix_file=self.matrix_file_var.get(),
                heeds_directory=self.heeds_dir_var.get(),
                study_name=self.study_name_var.get() or None
            )
            
            # Update GUI on main thread
            self.root.after(0, self._process_single_complete, result)
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            self.root.after(0, self._process_failed, error_msg)
    
    def _process_single_complete(self, result):
        """Handle single study processing completion."""
        self.stop_progress()
        
        if result:
            self.update_status("✅ Single study processing complete!")
            
            # Load and display the combined data
            try:
                output_file = result['output_file']
                self.combined_data = pd.read_csv(output_file)
                
                # Add cbush_list and cbush_count columns
                self.combined_data = self.add_cbush_summary_columns(self.combined_data)
                
                # Save updated data
                self.combined_data.to_csv(output_file, index=False)
                
                # Display in preview tab
                self.populate_data_table(self.combined_tree, self.combined_data, self.combined_info_label)
                
                # Switch to the combined tab to show results
                self.preview_notebook.select(self.combined_tab)
                
                self.log_message(f"   📊 Combined data preview updated: {len(self.combined_data):,} designs")
                
            except Exception as e:
                self.log_message(f"⚠️ Could not load combined data preview: {str(e)}")
            
            # Validate if requested
            if self.validate_var.get():
                self.log_message("🔍 Validating dataset...")
                validation_result = self.editor.validate_enhanced_dataset(result['output_file'])
                
                if validation_result:
                    quality_score = validation_result['data_quality_score']
                    self.log_message(f"📊 Data quality score: {quality_score:.1%}")
                    
                    if quality_score >= 0.95:
                        status_msg = f"✅ EXCELLENT quality - ML ready! ({quality_score:.1%})"
                    elif quality_score >= 0.85:
                        status_msg = f"👍 GOOD quality - ML suitable ({quality_score:.1%})"
                    else:
                        status_msg = f"⚠️  Quality issues detected ({quality_score:.1%})"
                    
                    self.update_status(status_msg)
            
            # Generate report
            self.editor.generate_comprehensive_report()
            
            # Show success message
            output_file = result['output_file']
            messagebox.showinfo("Success!", 
                               f"✅ Study processed successfully!\n\n"
                               f"Output: {Path(output_file).name}\n"
                               f"Designs: {result['final_dataset_shape'][0]:,}\n"
                               f"Columns: {result['final_dataset_shape'][1]}\n"
                               f"Alignment: {result['alignment_stats']['alignment_rate']:.1%}")
        else:
            self.update_status("❌ Processing failed")
            messagebox.showerror("Processing Failed", "Study processing failed. Check the log for details.")
    
    def add_cbush_summary_columns(self, df):
        """Add cbush_list and cbush_count summary columns."""
        try:
            def get_loose_cbushes(row):
                loose_list = []
                for cbush_num in range(2, 11):
                    if row[f'cbush_{cbush_num}_loose'] == 1:
                        loose_list.append(cbush_num)
                
                # Format to match original data: "[2 3 4 5 6 7 8 9 10]"
                if loose_list:
                    return str(loose_list).replace(',', '')
                else:
                    return "[]"
            
            # Add cbush_list column
            df['cbush_list'] = df.apply(get_loose_cbushes, axis=1)
            
            # Add cbush_count column
            cbush_cols = [f'cbush_{i}_loose' for i in range(2, 11) if f'cbush_{i}_loose' in df.columns]
            df['cbush_count'] = df[cbush_cols].sum(axis=1)
            
            self.log_message(f"   ✅ Added cbush_list and cbush_count columns")
            
            return df
            
        except Exception as e:
            self.log_message(f"⚠️ Warning: Could not add cbush summary columns: {str(e)}")
            return df
    
    def _process_failed(self, error_msg):
        """Handle processing failure."""
        self.stop_progress()
        self.update_status("❌ Processing failed")
        self.log_message(f"❌ {error_msg}")
        messagebox.showerror("Processing Failed", f"An error occurred:\n\n{error_msg}")
    
    def combine_studies(self):
        """Load and combine multiple enhanced study files."""
        study_files = filedialog.askopenfilenames(
            title="Select Enhanced Study Files to Combine",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not study_files:
            return
        
        if len(study_files) < 2:
            messagebox.showwarning("Insufficient Files", "Please select at least 2 study files to combine")
            return
        
        # Start combination process
        self.start_progress()
        self.update_status("Combining studies...")
        self.log_message(f"🔗 Combining {len(study_files)} study files...")
        
        self.processing_thread = threading.Thread(target=self._combine_studies_thread, args=(study_files,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _combine_studies_thread(self, study_files):
        """Combine studies in background thread."""
        try:
            # Initialize editor if not already done
            if not self.editor:
                self.editor = FeatureMatrixEditor(
                    output_dir=self.output_dir_var.get(),
                    verbose=False
                )
                
                # Override logging
                def gui_log(message):
                    self.root.after(0, lambda: self.log_message(message))
                self.editor.log = gui_log
            
            # Combine studies
            combination_result = self.editor.combine_multiple_studies(study_files=study_files)
            
            # Update GUI on main thread
            self.root.after(0, self._combine_complete, combination_result)
            
        except Exception as e:
            error_msg = f"Combination error: {str(e)}"
            self.root.after(0, self._process_failed, error_msg)
    
    def _combine_complete(self, result):
        """Handle study combination completion."""
        self.stop_progress()
        
        if result:
            self.update_status("✅ Study combination complete!")
            
            # Load and display the combined data
            try:
                output_file = result['output_file']
                self.combined_data = pd.read_csv(output_file)
                
                # Add cbush_list and cbush_count columns if not present
                if 'cbush_list' not in self.combined_data.columns:
                    self.combined_data = self.add_cbush_summary_columns(self.combined_data)
                    self.combined_data.to_csv(output_file, index=False)
                
                # Display in preview tab
                self.populate_data_table(self.combined_tree, self.combined_data, self.combined_info_label)
                
                # Switch to the combined tab to show results
                self.preview_notebook.select(self.combined_tab)
                
                self.log_message(f"   📊 Combined data preview updated: {len(self.combined_data):,} designs")
                
            except Exception as e:
                self.log_message(f"⚠️ Could not load combined data preview: {str(e)}")
            
            # Validate if requested
            if self.validate_var.get():
                self.log_message("🔍 Validating combined dataset...")
                validation_result = self.editor.validate_enhanced_dataset(result['output_file'])
                
                if validation_result:
                    quality_score = validation_result['data_quality_score']
                    self.log_message(f"📊 Combined data quality: {quality_score:.1%}")
            
            # Show success message
            output_file = result['output_file']
            stats = result['combination_stats']
            
            messagebox.showinfo("Combination Success!", 
                               f"✅ Studies combined successfully!\n\n"
                               f"Output: {Path(output_file).name}\n"
                               f"Total designs: {stats['total_combined_designs']:,}\n"
                               f"Studies combined: {stats['total_studies']}\n"
                               f"Global ML IDs: {stats['global_ml_id_range'][0]}-{stats['global_ml_id_range'][1]}")
        else:
            self.update_status("❌ Combination failed")
            messagebox.showerror("Combination Failed", "Study combination failed. Check the log for details.")


def create_gui():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = FeatureMatrixEditorGUI(root)
    root.mainloop()


def main():
    """Main entry point - can run GUI or command line."""
    # Check if any command line arguments provided
    if len(sys.argv) > 1:
        # Run command line interface
        parser = argparse.ArgumentParser(description="Feature Matrix Editor for HEEDS Parameter-Feature Integration")
        
        # Required arguments
        parser.add_argument('--param', '-p', required=True,
                           help='Parameter file (CSV/TSV)')
        parser.add_argument('--matrix', '-m', required=True,
                           help='Feature matrix file (CSV)')
        parser.add_argument('--heeds', '-d', required=True,
                           help='HEEDS directory containing POST_0 folder')
        
        # Optional arguments
        parser.add_argument('--output', '-o', default='edited_output',
                           help='Output directory')
        parser.add_argument('--study-name', '-s', default=None,
                           help='Study name override')
        parser.add_argument('--combine-with', '-c', nargs='+', default=None,
                           help='Additional enhanced study files to combine with')
        parser.add_argument('--validate', '-v', action='store_true',
                           help='Validate final dataset')
        parser.add_argument('--quiet', '-q', action='store_true',
                           help='Quiet mode')
        
        args = parser.parse_args()
        
        # Initialize feature matrix editor
        editor = FeatureMatrixEditor(
            output_dir=args.output,
            verbose=not args.quiet
        )
        
        try:
            # Process primary study
            result = editor.process_single_study(
                param_file=args.param,
                matrix_file=args.matrix,
                heeds_directory=args.heeds,
                study_name=args.study_name
            )
            
            if not result:
                print("❌ Failed to process primary study")
                return 1
            
            # Combine with additional studies if specified
            if args.combine_with:
                # Load additional study files
                additional_files = []
                for study_file in args.combine_with:
                    if Path(study_file).exists():
                        additional_files.append(study_file)
                    else:
                        editor.log(f"⚠️  Study file not found: {study_file}")
                
                if additional_files:
                    # Add primary study output to combination
                    primary_output = result['output_file']
                    all_files = [primary_output] + additional_files
                    
                    combination_result = editor.combine_multiple_studies(
                        study_files=all_files
                    )
                    
                    if combination_result:
                        final_file = combination_result['output_file']
                    else:
                        print("❌ Failed to combine studies")
                        return 1
                else:
                    final_file = result['output_file']
            else:
                # Just process single study
                final_file = result['output_file']
            
            # Validate final dataset if requested
            if args.validate:
                validation_result = editor.validate_enhanced_dataset(final_file)
                if not validation_result:
                    print("❌ Dataset validation failed")
                    return 1
            
            # Generate comprehensive report
            editor.generate_comprehensive_report()
            
            # Final success message
            print(f"\n🎉 SUCCESS: Enhanced feature matrix created!")
            print(f"📁 Output directory: {editor.output_dir}")
            print(f"📊 Final dataset: {final_file}")
            
            if args.validate and validation_result:
                print(f"🎯 Data quality: {validation_result['data_quality_score']:.1%}")
                print(f"📈 Total designs: {validation_result['total_designs']:,}")
                print(f"📋 Total columns: {validation_result['total_columns']}")
            
            return 0
            
        except Exception as e:
            print(f"\n❌ FEATURE MATRIX EDITING FAILED: {str(e)}")
            return 1
    
    else:
        # No command line arguments - run GUI
        print("🚀 Starting Feature Matrix Editor GUI...")
        create_gui()
        return 0


if __name__ == "__main__":
    sys.exit(main())
