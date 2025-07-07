#!/usr/bin/env python3
"""
Feature Matrix Editor v7
========================

Complete tool for editing and enhancing HEEDS feature matrices with parameter data.
Handles failed case detection, individual study processing, and multi-study combination.

Features:
- Automatic POST_0 folder scanning for failed case detection
- Individual study processing with full traceability
- Multi-study combination with global ML IDs
- User-configurable CBUSH loose detection thresholds
- Interactive stiffness reference table
- Comprehensive validation and logging

Usage:
    python feature_matrix_editor_gui_v6.py --help

Author: Enhanced Bolt Detection System
Version: 7.0
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
        
        # HEEDS parameter to stiffness mapping
        self.stiffness_mapping = {
            1: 1e4,   # loose
            2: 1e5,   # very loose threshold
            3: 1e6,   # borderline
            4: 1e7,   # borderline 
            5: 1e8,   # standard (industry baseline)
            6: 1e9,   # tight
            7: 1e10,  # tight
            8: 1e11,  # tight
            9: 1e12,  # very tight
            10: 1e13, # approaching validation limit
            11: 1e14, # validated healthy (beam study confirmed)
        }
        
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
                # Log first 20 failed designs for debugging
                failed_sample = failed_designs[:20]
                if len(failed_designs) > 20:
                    self.log(f"   🚫 Omitted designs: {failed_sample} (showing first 20 of {len(failed_designs)})")
                else:
                    self.log(f"   🚫 Omitted designs: {failed_designs}")
            
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
            
            # Log detailed omission analysis
            self.log("🔍 Detailed alignment analysis:")
            
            # Designs missing from POST_0 success
            param_but_not_post0 = param_designs - successful_designs
            if param_but_not_post0:
                sample_param_missing = sorted(list(param_but_not_post0))[:10]
                if len(param_but_not_post0) > 10:
                    self.log(f"   ❌ POST_0 failed: {sample_param_missing} (showing first 10 of {len(param_but_not_post0)})")
                else:
                    self.log(f"   ❌ POST_0 failed: {sample_param_missing}")
            
            # Designs missing from feature matrix
            param_but_not_feature = param_designs - feature_designs
            if param_but_not_feature:
                sample_feature_missing = sorted(list(param_but_not_feature))[:10]
                if len(param_but_not_feature) > 10:
                    self.log(f"   📈 Feature extraction failed: {sample_feature_missing} (showing first 10 of {len(param_but_not_feature)})")
                else:
                    self.log(f"   📈 Feature extraction failed: {sample_feature_missing}")
            
            # Overall omitted designs (parameter designs that didn't make it to final)
            omitted_designs = param_designs - valid_designs
            if omitted_designs:
                sample_omitted = sorted(list(omitted_designs))[:20]
                if len(omitted_designs) > 20:
                    self.log(f"   🚫 Total omitted from final: {sample_omitted} (showing first 20 of {len(omitted_designs)})")
                else:
                    self.log(f"   🚫 Total omitted from final: {sorted(list(omitted_designs))}")
            
            if alignment_stats['alignment_rate'] < 0.8:
                self.log(f"   ⚠️  LOW ALIGNMENT RATE: {alignment_stats['alignment_rate']:.1%}")
                self.log(f"   💡 Consider checking: POST_0 simulation stability, feature extraction robustness")
            
            return filtered_param_df, filtered_feature_df, alignment_stats
            
        except Exception as e:
            error_msg = f"❌ Failed to filter to successful designs: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return None, None, None
    
    def merge_parameter_features(self, param_df, feature_df, study_name, user_threshold=1e8):
        """
        Merge filtered parameter and feature data with traceability.
        
        Args:
            param_df: Filtered parameter dataframe
            feature_df: Filtered feature dataframe
            study_name: Study name for identification
            user_threshold: User-defined threshold for loose detection
            
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
            
            # Apply parameter-based CBUSH loose detection (replace simulation-based)
            merged_df = self.add_parameter_based_cbush_detection(merged_df, user_threshold)
            
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
    
    def add_parameter_based_cbush_detection(self, df, user_threshold=1e8):
        """
        Add parameter-based CBUSH loose detection using user-defined threshold.
        
        IMPORTANT: This replaces the simulation-based cbush_X_loose columns which appear
        to have incorrect results (high stiffness CBUSHes showing as loose).
        
        TODO: Update the main GUI feature extraction process to use this logic instead
        of post-HEEDS simulation analysis. The simulation-based detection is showing
        physically incorrect results.
        
        Args:
            df: DataFrame with parameter columns (K4_X, K5_X, K6_X)
            user_threshold: User-defined threshold for loose detection (default 1e8)
            
        Returns:
            DataFrame with updated cbush_X_loose columns
        """
        try:
            self.log(f"🔧 Applying parameter-based CBUSH loose detection (threshold < {user_threshold:.0e})...")
            
            # Normalize column names to handle case sensitivity
            df_normalized = df.copy()
            column_mapping = {}
            for col in df.columns:
                normalized_col = col.upper()
                column_mapping[normalized_col] = col
            
            # Keep original simulation-based columns for comparison
            simulation_cols = [col for col in df.columns if col.startswith('cbush_') and col.endswith('_loose')]
            for col in simulation_cols:
                if col in df.columns:
                    df[f'{col}_simulation'] = df[col].copy()
            
            # Apply parameter-based logic
            loose_count_by_cbush = {}
            
            # Helper function to safely convert to numeric
            def safe_numeric_conversion(value, default=5):
                """Convert value to integer, handle various data types."""
                if pd.isna(value):
                    return default
                try:
                    # Handle string representations
                    if isinstance(value, str):
                        value = value.strip()
                        if value == '':
                            return default
                    return int(float(value))
                except (ValueError, TypeError):
                    self.log(f"   ⚠️  Could not convert '{value}' to integer, using default {default}")
                    return default
            
            # Helper function to get stiffness value
            def get_stiffness_value(param_value, param_name=""):
                """Get stiffness from parameter value with robust conversion."""
                numeric_value = safe_numeric_conversion(param_value)
                stiffness = self.stiffness_mapping.get(numeric_value, 1e8)
                return stiffness, numeric_value
            
            self.log("🔍 Detailed CBUSH analysis:")
            
            for cbush_num in range(2, 11):  # CBUSHes 2-10 only (omit CBUSH 1)
                # Try different case combinations for column names
                possible_k4_names = [f'K4_{cbush_num}', f'k4_{cbush_num}', f'K4{cbush_num}', f'k4{cbush_num}']
                possible_k5_names = [f'K5_{cbush_num}', f'k5_{cbush_num}', f'K5{cbush_num}', f'k5{cbush_num}']
                possible_k6_names = [f'K6_{cbush_num}', f'k6_{cbush_num}', f'K6{cbush_num}', f'k6{cbush_num}']
                
                # Find actual column names
                k4_col = None
                k5_col = None  
                k6_col = None
                
                for name in possible_k4_names:
                    if name in df.columns:
                        k4_col = name
                        break
                
                for name in possible_k5_names:
                    if name in df.columns:
                        k5_col = name
                        break
                        
                for name in possible_k6_names:
                    if name in df.columns:
                        k6_col = name
                        break
                
                if all([k4_col, k5_col, k6_col]):
                    # Process this CBUSH with robust logic
                    loose_results = []
                    
                    # Debug first row for CBUSH 2
                    if cbush_num == 2:
                        sample_row = df.iloc[0]
                        k4_raw = sample_row[k4_col]
                        k5_raw = sample_row[k5_col]
                        k6_raw = sample_row[k6_col]
                        
                        k4_stiffness, k4_numeric = get_stiffness_value(k4_raw, k4_col)
                        k5_stiffness, k5_numeric = get_stiffness_value(k5_raw, k5_col)
                        k6_stiffness, k6_numeric = get_stiffness_value(k6_raw, k6_col)
                        
                        self.log(f"   🔍 DEBUG CBUSH 2 (first row):")
                        self.log(f"      {k4_col} raw: '{k4_raw}' → numeric: {k4_numeric} → stiffness: {k4_stiffness:.0e}")
                        self.log(f"      {k5_col} raw: '{k5_raw}' → numeric: {k5_numeric} → stiffness: {k5_stiffness:.0e}")
                        self.log(f"      {k6_col} raw: '{k6_raw}' → numeric: {k6_numeric} → stiffness: {k6_stiffness:.0e}")
                        
                        min_stiffness = min(k4_stiffness, k5_stiffness, k6_stiffness)
                        is_loose = 1 if min_stiffness < user_threshold else 0
                        
                        self.log(f"      Min stiffness: {min_stiffness:.0e}")
                        self.log(f"      Threshold: {user_threshold:.0e}")
                        self.log(f"      Is loose: {is_loose} ({'LOOSE' if is_loose else 'TIGHT'})")
                    
                    # Apply to all rows
                    for idx, row in df.iterrows():
                        k4_stiffness, _ = get_stiffness_value(row[k4_col])
                        k5_stiffness, _ = get_stiffness_value(row[k5_col])
                        k6_stiffness, _ = get_stiffness_value(row[k6_col])
                        
                        min_stiffness = min(k4_stiffness, k5_stiffness, k6_stiffness)
                        is_loose = 1 if min_stiffness < user_threshold else 0
                        loose_results.append(is_loose)
                    
                    # Update dataframe
                    df[f'cbush_{cbush_num}_loose'] = loose_results
                    loose_count = sum(loose_results)
                    loose_count_by_cbush[cbush_num] = loose_count
                    
                else:
                    missing_cols = []
                    if not k4_col: missing_cols.append(f'K4_{cbush_num}')
                    if not k5_col: missing_cols.append(f'K5_{cbush_num}')
                    if not k6_col: missing_cols.append(f'K6_{cbush_num}')
                    
                    self.log(f"   ⚠️  Missing parameter columns for CBUSH {cbush_num}: {missing_cols}")
                    loose_count_by_cbush[cbush_num] = df.get(f'cbush_{cbush_num}_loose', pd.Series([0])).sum()
            
            # CBUSH 1 is omitted as requested (stays constant)
            if 'cbush_1_loose' in df.columns:
                loose_count_by_cbush[1] = df['cbush_1_loose'].sum()
            
            # Log the new distribution
            self.log("   📊 Parameter-based CBUSH loose distribution:")
            self.log(f"   🎯 Using threshold: {user_threshold:.0e} (below this = loose)")
            total_designs = len(df)
            for cbush_num in range(1, 11):
                if cbush_num in loose_count_by_cbush:
                    count = loose_count_by_cbush[cbush_num]
                    percentage = (count / total_designs * 100) if total_designs > 0 else 0
                    if cbush_num == 1:
                        self.log(f"      CBUSH {cbush_num}: {count:,} loose ({percentage:.1f}%) - CONSTANT")
                    else:
                        self.log(f"      CBUSH {cbush_num}: {count:,} loose ({percentage:.1f}%)")
            
            self.log("   ✅ Parameter-based CBUSH detection complete")
            
            # Add documentation columns
            df['cbush_detection_method'] = 'parameter_based_user_threshold'
            df[f'loose_threshold'] = f'< {user_threshold:.0e}'
            
            return df
            
        except Exception as e:
            error_msg = f"❌ Failed to apply parameter-based CBUSH detection: {str(e)}"
            self.log(error_msg)
            self.failed_operations.append(error_msg)
            return df
    
    def process_single_study(self, param_file, matrix_file, heeds_directory, study_name=None, output_name=None, user_threshold=1e8):
        """
        Process a single study: parameter file + feature matrix + POST_0 validation.
        
        Args:
            param_file: Path to parameter file
            matrix_file: Path to feature matrix
            heeds_directory: Path to HEEDS directory with POST_0
            study_name: Optional study name
            output_name: Optional output filename
            user_threshold: User-defined threshold for loose detection
            
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
            merged_df = self.merge_parameter_features(filtered_param_df, filtered_feature_df, study_name, user_threshold)
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
                'user_threshold': user_threshold,
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
                    f.write(f"  Threshold: {summary.get('user_threshold', 'N/A')}\n")
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
        self.root.title("Feature Matrix Editor v7.0")
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
        self.threshold_var = tk.StringVar(value="1e8")  # Default to industry standard
        self.custom_threshold_var = tk.StringVar()
        
        # Combine studies variables
        self.study1_file_var = tk.StringVar()
        self.study2_file_var = tk.StringVar()
        
        # Stiffness mapping for reference table
        self.stiffness_mapping = {
            1: 1e4, 2: 1e5, 3: 1e6, 4: 1e7, 5: 1e8, 6: 1e9, 
            7: 1e10, 8: 1e11, 9: 1e12, 10: 1e13, 11: 1e14
        }
        
        # Create GUI
        self.create_widgets()
        
        # Data storage for previews
        self.param_data = None
        self.matrix_data = None
        self.combined_data = None
        self.loaded_full_matrix = None
        
        self.log_message("🚀 Feature Matrix Editor v6.0 Ready!")
        self.log_message("📋 Select your files and click 'Process Study' to begin")
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Title
        title_label = tk.Label(self.root, text="Feature Matrix Editor v6.0", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(self.root, text="HEEDS Parameter-Feature Integration with Traceability",
                                 font=('Arial', 12), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=5)
        
        # Stiffness Reference Table
        self.create_stiffness_reference_table()
        
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
    
    def create_stiffness_reference_table(self):
        """Create collapsible stiffness reference table on main page."""
        # Stiffness Reference Table (collapsible)
        self.ref_frame = ttk.LabelFrame(self.root, text="📊 Stiffness Reference (Click to Set Threshold)")
        self.ref_frame.pack(fill='x', padx=10, pady=5)
        
        # Collapse/Expand button
        button_frame = tk.Frame(self.ref_frame)
        button_frame.pack(fill='x', padx=5, pady=2)
        
        self.table_expanded = tk.BooleanVar(value=True)
        self.expand_button = tk.Button(button_frame, text="▼ Hide Stiffness Table", command=self.toggle_table,
                                      font=('Arial', 8), bg='#ecf0f1', relief='flat')
        self.expand_button.pack(side='left')
        
        # Table container (collapsible)
        self.table_container = tk.Frame(self.ref_frame)
        self.table_container.pack(fill='x', padx=5, pady=5)
        
        # Create scrollable table with both horizontal and vertical scrollbars
        table_frame = tk.Frame(self.table_container)
        table_frame.pack(fill='both', expand=True)
        
        # Configure grid for proper scrollbar layout
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for scrolling
        canvas = tk.Canvas(table_frame, height=150, bg='white')
        canvas.grid(row=0, column=0, sticky='nsew')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        
        h_scrollbar = ttk.Scrollbar(table_frame, orient="horizontal", command=canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Bind scrolling events
        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        def configure_canvas_width(event):
            canvas_width = canvas.winfo_width()
            canvas.itemconfig(canvas_frame, width=canvas_width)
        
        scrollable_frame.bind("<Configure>", configure_scroll_region)
        canvas.bind("<Configure>", configure_canvas_width)
        
        # Bind mousewheel to canvas for smooth scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", on_mousewheel)  # Windows
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux
        
        # Categories for display
        categories = ["Loose", "Very Loose", "Borderline", "Borderline", "Industry Std", 
                      "Tight", "Very Tight", "Extremely Tight", "Ultra Tight", "Near Limit", "Validated"]
        
        # Create table headers
        tk.Label(scrollable_frame, text="ID", font=('Arial', 9, 'bold'), relief="solid", borderwidth=1, padx=5).grid(row=0, column=0, sticky='ew')
        tk.Label(scrollable_frame, text="Stiffness (N·m/rad)", font=('Arial', 9, 'bold'), relief="solid", borderwidth=1, padx=5).grid(row=0, column=1, sticky='ew')
        tk.Label(scrollable_frame, text="Category", font=('Arial', 9, 'bold'), relief="solid", borderwidth=1, padx=5).grid(row=0, column=2, sticky='ew')
        
        # Data rows - make clickable
        for i, (id_num, stiffness) in enumerate(self.stiffness_mapping.items()):
            # Create clickable row
            id_label = tk.Label(scrollable_frame, text=str(id_num), relief="solid", borderwidth=1, padx=5, cursor="hand2")
            stiff_label = tk.Label(scrollable_frame, text=f"{stiffness:.0e}", relief="solid", borderwidth=1, padx=5, cursor="hand2")
            cat_label = tk.Label(scrollable_frame, text=categories[i], relief="solid", borderwidth=1, padx=5, cursor="hand2")
            
            id_label.grid(row=i+1, column=0, sticky='ew')
            stiff_label.grid(row=i+1, column=1, sticky='ew')
            cat_label.grid(row=i+1, column=2, sticky='ew')
            
            # Bind click events to set threshold
            for label in [id_label, stiff_label, cat_label]:
                label.bind("<Button-1>", lambda e, val=stiffness: self.set_threshold_from_table(val))
                # Hover effect
                label.bind("<Enter>", lambda e, lbl=label: lbl.configure(bg='#e8e8e8'))
                label.bind("<Leave>", lambda e, lbl=label: lbl.configure(bg='SystemButtonFace'))
        
        # Update scroll region after all widgets are added
        scrollable_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    def toggle_table(self):
        """Toggle visibility of stiffness reference table."""
        if self.table_expanded.get():
            # Collapse table
            self.table_container.pack_forget()
            self.expand_button.config(text="▶ Show Stiffness Table")
            self.table_expanded.set(False)
        else:
            # Expand table
            self.table_container.pack(fill='x', padx=5, pady=5)
            self.expand_button.config(text="▼ Hide Stiffness Table")
            self.table_expanded.set(True)
    
    def set_threshold_from_table(self, stiffness_value):
        """Set threshold from clicking on reference table."""
        self.threshold_var.set(f"{stiffness_value:.0e}")
        self.log_message(f"🎯 Threshold set to {stiffness_value:.0e} from reference table")
    
    def on_threshold_change(self, event=None):
        """Handle threshold dropdown selection change."""
        if self.threshold_var.get() == "Custom":
            self.custom_entry.pack(fill='x', pady=2)
            self.custom_entry.focus()
        else:
            self.custom_entry.pack_forget()
    
    def validate_threshold(self, threshold_str):
        """Validate threshold is valid scientific notation."""
        try:
            value = float(threshold_str)
            if value <= 0:
                raise ValueError("Threshold must be positive")
            return value
        except:
            raise ValueError(f"Invalid threshold format: {threshold_str}")
    
    def get_user_threshold(self):
        """Get threshold from GUI, with validation."""
        try:
            if self.threshold_var.get() == "Custom":
                if not self.custom_threshold_var.get():
                    raise ValueError("Custom threshold cannot be empty")
                return self.validate_threshold(self.custom_threshold_var.get())
            else:
                return float(self.threshold_var.get())
        except Exception as e:
            # Default to industry standard if error
            self.log_message(f"⚠️  Threshold error: {str(e)}, using default 1e8")
            return 1e8
    
    def update_scaling_description(self, event=None):
        """Update scaling method description."""
        method = self.scaling_method_var.get()
        descriptions = {
            "StandardScaler": "Mean=0, Std=1 (best for most ML algorithms)",
            "MinMaxScaler": "Scale to 0-1 range (preserves relationships)", 
            "RobustScaler": "Median-based, robust to outliers",
            "MaxAbsScaler": "Scale by maximum absolute value",
            "Normalizer": "Scale individual samples to unit norm"
        }
        self.method_desc_label.config(text=descriptions.get(method, "Unknown scaling method"))
    
    def apply_scaling(self):
        """Apply scaling to selected features."""
        if self.combined_data is None:
            messagebox.showwarning("No Data", "Please process a study first to enable scaling")
            return
        
        try:
            # Import scaling modules
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
            import numpy as np
            
            # Get scaling method
            method_name = self.scaling_method_var.get()
            scalers = {
                "StandardScaler": StandardScaler(),
                "MinMaxScaler": MinMaxScaler(),
                "RobustScaler": RobustScaler(),
                "MaxAbsScaler": MaxAbsScaler(),
                "Normalizer": Normalizer()
            }
            
            scaler = scalers[method_name]
            
            # Identify columns to scale
            columns_to_scale = []
            
            if self.scale_params_var.get():
                param_cols = [col for col in self.combined_data.columns if col.startswith(('K4_', 'K5_', 'K6_'))]
                columns_to_scale.extend(param_cols)
            
            if self.scale_features_var.get():
                feature_cols = [col for col in self.combined_data.columns if col.startswith(('ACCE_', 'DISP_'))]
                columns_to_scale.extend(feature_cols)
            
            if self.scale_cbush_var.get():
                cbush_cols = [col for col in self.combined_data.columns if col.startswith('cbush_') and col.endswith('_loose')]
                columns_to_scale.extend(cbush_cols)
            
            if not columns_to_scale:
                messagebox.showwarning("No Columns", "Please select at least one column type to scale")
                return
            
            # Create scaled copy
            self.scaled_data = self.combined_data.copy()
            
            # Apply scaling
            scaled_values = scaler.fit_transform(self.combined_data[columns_to_scale])
            self.scaled_data[columns_to_scale] = scaled_values
            
            # Show before/after statistics
            self.show_scaling_comparison(columns_to_scale[:10])  # Show first 10 for brevity
            
            self.log_message(f"✅ Applied {method_name} to {len(columns_to_scale)} columns")
            
        except ImportError:
            messagebox.showerror("Missing Library", "sklearn not installed. Please install: pip install scikit-learn")
        except Exception as e:
            messagebox.showerror("Scaling Error", f"Failed to apply scaling: {str(e)}")
    
    def show_scaling_comparison(self, sample_columns):
        """Show before/after scaling statistics."""
        if self.combined_data is None or not hasattr(self, 'scaled_data'):
            return
        
        # Clear previous results
        self.before_scaling_text.delete('1.0', tk.END)
        self.after_scaling_text.delete('1.0', tk.END)
        
        # Before scaling statistics
        self.before_scaling_text.insert(tk.END, "BEFORE SCALING STATISTICS:\n")
        self.before_scaling_text.insert(tk.END, "=" * 40 + "\n\n")
        
        for col in sample_columns:
            if col in self.combined_data.columns:
                data = self.combined_data[col]
                stats = f"{col}:\n"
                stats += f"  Mean: {data.mean():.4f}\n"
                stats += f"  Std:  {data.std():.4f}\n"
                stats += f"  Min:  {data.min():.4f}\n"
                stats += f"  Max:  {data.max():.4f}\n\n"
                self.before_scaling_text.insert(tk.END, stats)
        
        # After scaling statistics
        self.after_scaling_text.insert(tk.END, "AFTER SCALING STATISTICS:\n")
        self.after_scaling_text.insert(tk.END, "=" * 40 + "\n\n")
        
        for col in sample_columns:
            if col in self.scaled_data.columns:
                data = self.scaled_data[col]
                stats = f"{col}:\n"
                stats += f"  Mean: {data.mean():.4f}\n"
                stats += f"  Std:  {data.std():.4f}\n"
                stats += f"  Min:  {data.min():.4f}\n"
                stats += f"  Max:  {data.max():.4f}\n\n"
                self.after_scaling_text.insert(tk.END, stats)
    
    def save_scaled_data(self):
        """Save scaled data to file."""
        if not hasattr(self, 'scaled_data'):
            messagebox.showwarning("No Scaled Data", "Please apply scaling first")
            return
        
        # Get save filename
        filename = filedialog.asksaveasfilename(
            title="Save Scaled Dataset",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                self.scaled_data.to_csv(filename, index=False)
                self.log_message(f"💾 Scaled data saved: {Path(filename).name}")
                messagebox.showinfo("Success", f"Scaled data saved successfully!\n\n{Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save scaled data: {str(e)}")
    
    def generate_correlation(self):
        """Generate correlation matrix for selected features."""
        if self.combined_data is None:
            messagebox.showwarning("No Data", "Please process a study first")
            return
        
        try:
            # Use scaled data if available, otherwise original
            data = self.scaled_data if hasattr(self, 'scaled_data') else self.combined_data
            
            # Get target variable
            target = self.correlation_target_var.get()
            if target not in data.columns:
                messagebox.showwarning("Invalid Target", f"Target '{target}' not found in data")
                return
            
            # Identify columns to include
            columns_to_analyze = [target]  # Always include target
            
            if self.corr_params_var.get():
                param_cols = [col for col in data.columns if col.startswith(('K4_', 'K5_', 'K6_'))]
                columns_to_analyze.extend(param_cols)
            
            if self.corr_features_var.get():
                feature_cols = [col for col in data.columns if col.startswith(('ACCE_', 'DISP_'))]
                columns_to_analyze.extend(feature_cols)
            
            if self.corr_cbush_var.get():
                cbush_cols = [col for col in data.columns if col.startswith('cbush_') and col.endswith('_loose')]
                columns_to_analyze.extend(cbush_cols)
            
            # Remove duplicates and ensure target is included
            columns_to_analyze = list(set(columns_to_analyze))
            
            if len(columns_to_analyze) < 2:
                messagebox.showwarning("Insufficient Columns", "Select at least one feature type for correlation analysis")
                return
            
            # Calculate correlations with target
            correlations = data[columns_to_analyze].corr()[target].drop(target).abs().sort_values(ascending=False)
            
            # Display results
            self.correlation_text.delete('1.0', tk.END)
            self.correlation_text.insert(tk.END, f"CORRELATION WITH {target}:\n")
            self.correlation_text.insert(tk.END, "=" * 50 + "\n\n")
            
            data_source = "SCALED DATA" if hasattr(self, 'scaled_data') else "RAW DATA"
            self.correlation_text.insert(tk.END, f"Source: {data_source}\n")
            self.correlation_text.insert(tk.END, f"Features analyzed: {len(columns_to_analyze)-1}\n\n")
            
            self.correlation_text.insert(tk.END, "TOP CORRELATIONS:\n")
            self.correlation_text.insert(tk.END, "-" * 30 + "\n")
            
            for feature, corr_value in correlations.head(20).items():
                self.correlation_text.insert(tk.END, f"{feature:<25} {corr_value:.4f}\n")
            
            self.log_message(f"🔥 Generated correlation matrix for {target}")
            
        except Exception as e:
            messagebox.showerror("Correlation Error", f"Failed to generate correlation matrix: {str(e)}")
    
    def generate_ai_suggestions(self):
        """Generate AI-powered feature suggestions."""
        if self.combined_data is None:
            messagebox.showwarning("No Data", "Please process a study first")
            return
        
        try:
            # Use scaled data if available, otherwise original
            data = self.scaled_data if hasattr(self, 'scaled_data') else self.combined_data
            
            # Get target variable
            target = self.correlation_target_var.get()
            if target not in data.columns:
                messagebox.showwarning("Invalid Target", f"Target '{target}' not found in data")
                return
            
            # Calculate correlations
            numeric_data = data.select_dtypes(include=[np.number])
            if target not in numeric_data.columns:
                messagebox.showwarning("Invalid Target", f"Target '{target}' is not numeric")
                return
            
            correlations = numeric_data.corr()[target].drop(target).abs().sort_values(ascending=False)
            
            # Generate AI insights
            self.ai_suggestions_text.delete('1.0', tk.END)
            self.ai_suggestions_text.insert(tk.END, f"🧠 AI INSIGHTS FOR {target}:\n")
            self.ai_suggestions_text.insert(tk.END, "=" * 50 + "\n\n")
            
            data_source = "SCALED" if hasattr(self, 'scaled_data') else "RAW"
            self.ai_suggestions_text.insert(tk.END, f"Analysis of: {data_source} DATA\n\n")
            
            # Top overall features
            top_feature = correlations.index[0]
            top_corr = correlations.iloc[0]
            
            self.ai_suggestions_text.insert(tk.END, "🎯 TOP RECOMMENDATION:\n")
            self.ai_suggestions_text.insert(tk.END, f"Feature: {top_feature}\n")
            self.ai_suggestions_text.insert(tk.END, f"Correlation: {top_corr:.4f}\n\n")
            
            # Categorized suggestions
            param_features = [f for f in correlations.index if f.startswith(('K4_', 'K5_', 'K6_'))]
            accel_features = [f for f in correlations.index if f.startswith('ACCE_')]
            disp_features = [f for f in correlations.index if f.startswith('DISP_')]
            
            if param_features:
                self.ai_suggestions_text.insert(tk.END, "🔧 TOP PARAMETERS:\n")
                for feat in param_features[:3]:
                    self.ai_suggestions_text.insert(tk.END, f"  {feat}: {correlations[feat]:.4f}\n")
                self.ai_suggestions_text.insert(tk.END, "\n")
            
            if accel_features:
                self.ai_suggestions_text.insert(tk.END, "📈 TOP ACCELERATIONS:\n")
                for feat in accel_features[:3]:
                    self.ai_suggestions_text.insert(tk.END, f"  {feat}: {correlations[feat]:.4f}\n")
                self.ai_suggestions_text.insert(tk.END, "\n")
            
            if disp_features:
                self.ai_suggestions_text.insert(tk.END, "📏 TOP DISPLACEMENTS:\n")
                for feat in disp_features[:3]:
                    self.ai_suggestions_text.insert(tk.END, f"  {feat}: {correlations[feat]:.4f}\n")
                self.ai_suggestions_text.insert(tk.END, "\n")
            
            # AI interpretation
            self.ai_suggestions_text.insert(tk.END, "💡 AI INTERPRETATION:\n")
            if top_feature.startswith(('K4_', 'K5_', 'K6_')):
                self.ai_suggestions_text.insert(tk.END, f"The stiffness parameter {top_feature} shows strongest correlation.\n")
                self.ai_suggestions_text.insert(tk.END, "This suggests the bolt's inherent stiffness directly impacts looseness.\n")
            elif top_feature.startswith('ACCE_'):
                self.ai_suggestions_text.insert(tk.END, f"Acceleration response {top_feature} is most predictive.\n")
                self.ai_suggestions_text.insert(tk.END, "Dynamic response patterns reveal structural behavior changes.\n")
            elif top_feature.startswith('DISP_'):
                self.ai_suggestions_text.insert(tk.END, f"Displacement response {top_feature} is most predictive.\n")
                self.ai_suggestions_text.insert(tk.END, "Structural deformation patterns indicate loose connections.\n")
            
            # Update target combo with available targets
            cbush_targets = [col for col in data.columns if col.startswith('cbush_') and col.endswith('_loose')]
            self.target_combo['values'] = cbush_targets
            
            self.log_message(f"🧠 Generated AI suggestions for {target}")
            
        except Exception as e:
            messagebox.showerror("AI Analysis Error", f"Failed to generate AI suggestions: {str(e)}")
    
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
        
        # CBUSH Loose Detection Threshold
        threshold_frame = tk.Frame(settings_frame)
        threshold_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(threshold_frame, text="CBUSH Loose Threshold:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        # Dropdown with presets
        threshold_combo = ttk.Combobox(threshold_frame, textvariable=self.threshold_var, 
                                      values=["1e5", "1e6", "1e7", "1e8", "1e9", "1e10", "1e11", "1e12", "1e13", "1e14", "Custom"],
                                      width=15, font=('Arial', 9))
        threshold_combo.pack(fill='x', pady=2)
        
        # Custom entry (hidden initially)
        self.custom_entry = tk.Entry(threshold_frame, textvariable=self.custom_threshold_var, font=('Arial', 9))
        self.custom_entry.pack(fill='x', pady=2)
        self.custom_entry.pack_forget()  # Hidden initially
        
        # Bind event to show/hide custom entry
        threshold_combo.bind('<<ComboboxSelected>>', self.on_threshold_change)
        
        # Add tooltip explanation
        tooltip_label = tk.Label(threshold_frame, text="Values below threshold are considered 'loose'", 
                               font=('Arial', 8), fg='#666666')
        tooltip_label.pack(anchor='w')
        
        # Processing controls
        control_frame = ttk.LabelFrame(parent, text="🚀 Processing Controls")
        control_frame.pack(fill='x', pady=10)
        
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text="🔄 Process Study", command=self.process_single_study,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), height=1, width=18).pack(fill='x', pady=2)
        
        tk.Button(button_frame, text="📋 Clear All", command=self.clear_all_inputs,
                 bg='#f39c12', fg='white', font=('Arial', 11, 'bold'), height=1, width=18).pack(fill='x', pady=2)
        
        # Combine Studies section
        combine_frame = ttk.LabelFrame(parent, text="🔗 Combine Studies")
        combine_frame.pack(fill='x', pady=10)
        
        # Study 1 selection
        study1_frame = tk.Frame(combine_frame)
        study1_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(study1_frame, text="Study File 1:", font=('Arial', 10, 'bold')).pack(anchor='w')
        study1_input_frame = tk.Frame(study1_frame)
        study1_input_frame.pack(fill='x', pady=2)
        tk.Entry(study1_input_frame, textvariable=self.study1_file_var, width=40, font=('Arial', 9)).pack(side='left', fill='x', expand=True)
        tk.Button(study1_input_frame, text="Browse", command=self.browse_study1_file,
                 bg='#9b59b6', fg='white', font=('Arial', 9, 'bold')).pack(side='right', padx=(5,0))
        
        # Study 2 selection
        study2_frame = tk.Frame(combine_frame)
        study2_frame.pack(fill='x', padx=10, pady=5)
        tk.Label(study2_frame, text="Study File 2:", font=('Arial', 10, 'bold')).pack(anchor='w')
        study2_input_frame = tk.Frame(study2_frame)
        study2_input_frame.pack(fill='x', pady=2)
        tk.Entry(study2_input_frame, textvariable=self.study2_file_var, width=40, font=('Arial', 9)).pack(side='left', fill='x', expand=True)
        tk.Button(study2_input_frame, text="Browse", command=self.browse_study2_file,
                 bg='#9b59b6', fg='white', font=('Arial', 9, 'bold')).pack(side='right', padx=(5,0))
        
        # Combine button
        combine_button_frame = tk.Frame(combine_frame)
        combine_button_frame.pack(fill='x', padx=10, pady=10)
        tk.Button(combine_button_frame, text="🔗 Combine Selected Studies", command=self.combine_two_studies,
                 bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'), height=1, width=25).pack()
        
        tk.Button(button_frame, text="📋 Clear Log", command=self.clear_log,
                 bg='#95a5a6', fg='white', font=('Arial', 9, 'bold'), height=1, width=18).pack(fill='x', pady=2)
        
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
        
        # Tab 4: Feature Scaling
        self.scaling_tab = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.scaling_tab, text="📏 Feature Scaling")
        self.create_scaling_tab(self.scaling_tab)
        
        # Tab 5: Correlation Matrix
        self.correlation_tab = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.correlation_tab, text="🔥 Correlation Matrix")
        self.create_correlation_tab(self.correlation_tab)
        
        # Tab 5: Heatmap Visualization
        self.heatmap_tab = ttk.Frame(self.preview_notebook)
        self.preview_notebook.add(self.heatmap_tab, text="🔥 Heatmap")
        self.create_heatmap_tab(self.heatmap_tab)
        
        # Tab 6: Load Full Feature Matrix
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
        
        # Table frame with proper scrollbar layout
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.param_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Create scrollbars
        param_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.param_tree.yview)
        param_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.param_tree.xview)
        
        # Configure treeview to use scrollbars
        self.param_tree.configure(yscrollcommand=param_v_scroll.set, xscrollcommand=param_h_scroll.set)
        
        # Grid layout for proper scrollbar positioning
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        self.param_tree.grid(row=0, column=0, sticky='nsew')
        param_v_scroll.grid(row=0, column=1, sticky='ns')
        param_h_scroll.grid(row=1, column=0, sticky='ew')
    
    def create_matrix_preview_tab(self, parent):
        """Create feature matrix preview tab."""
        # Info frame
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.matrix_info_label = tk.Label(info_frame, text="No feature matrix loaded", 
                                         font=('Arial', 11, 'bold'), fg='#7f8c8d')
        self.matrix_info_label.pack(anchor='w')
        
        # Table frame with proper scrollbar layout
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.matrix_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Create scrollbars
        matrix_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.matrix_tree.yview)
        matrix_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.matrix_tree.xview)
        
        # Configure treeview to use scrollbars
        self.matrix_tree.configure(yscrollcommand=matrix_v_scroll.set, xscrollcommand=matrix_h_scroll.set)
        
        # Grid layout for proper scrollbar positioning
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        self.matrix_tree.grid(row=0, column=0, sticky='nsew')
        matrix_v_scroll.grid(row=0, column=1, sticky='ns')
        matrix_h_scroll.grid(row=1, column=0, sticky='ew')
    
    def create_combined_preview_tab(self, parent):
        """Create combined data preview tab."""
        # Info frame
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.combined_info_label = tk.Label(info_frame, text="No combined data available - process a study first", 
                                           font=('Arial', 11, 'bold'), fg='#7f8c8d')
        self.combined_info_label.pack(anchor='w')
        
        # Table frame with proper scrollbar layout
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.combined_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Create scrollbars
        combined_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.combined_tree.yview)
        combined_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.combined_tree.xview)
        
        # Configure treeview to use scrollbars
        self.combined_tree.configure(yscrollcommand=combined_v_scroll.set, xscrollcommand=combined_h_scroll.set)
        
        # Grid layout for proper scrollbar positioning
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        self.combined_tree.grid(row=0, column=0, sticky='nsew')
        combined_v_scroll.grid(row=0, column=1, sticky='ns')
        combined_h_scroll.grid(row=1, column=0, sticky='ew')
    
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
        
        # Table frame with proper scrollbar layout
        table_frame = tk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview for data display
        self.load_tree = ttk.Treeview(table_frame, show='headings', height=20)
        
        # Create scrollbars
        load_v_scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.load_tree.yview)
        load_h_scroll = ttk.Scrollbar(table_frame, orient='horizontal', command=self.load_tree.xview)
        
        # Configure treeview to use scrollbars
        self.load_tree.configure(yscrollcommand=load_v_scroll.set, xscrollcommand=load_h_scroll.set)
        
        # Grid layout for proper scrollbar positioning
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        self.load_tree.grid(row=0, column=0, sticky='nsew')
        load_v_scroll.grid(row=0, column=1, sticky='ns')
        load_h_scroll.grid(row=1, column=0, sticky='ew')
    
    def create_scaling_tab(self, parent):
        """Create feature scaling tab."""
        # Controls frame
        controls_frame = ttk.LabelFrame(parent, text="📏 Feature Scaling Options")
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # Scaling method selection
        method_frame = tk.Frame(controls_frame)
        method_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(method_frame, text="Scaling Method:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.scaling_method_var = tk.StringVar(value="StandardScaler")
        scaling_methods = ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer"]
        method_combo = ttk.Combobox(method_frame, textvariable=self.scaling_method_var, 
                                   values=scaling_methods, width=15, font=('Arial', 9))
        method_combo.pack(fill='x', pady=2)
        
        # Method descriptions
        method_desc_frame = tk.Frame(controls_frame)
        method_desc_frame.pack(fill='x', padx=10, pady=5)
        
        self.method_desc_label = tk.Label(method_desc_frame, text="StandardScaler: Mean=0, Std=1 (best for ML)", 
                                         font=('Arial', 9), fg='#666666', wraplength=400)
        self.method_desc_label.pack(anchor='w')
        
        # Bind method change to update description
        method_combo.bind('<<ComboboxSelected>>', self.update_scaling_description)
        
        # Column selection frame
        selection_frame = ttk.LabelFrame(controls_frame, text="Select Columns to Scale")
        selection_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Column type checkboxes
        type_frame = tk.Frame(selection_frame)
        type_frame.pack(fill='x', padx=10, pady=5)
        
        self.scale_params_var = tk.BooleanVar(value=False)
        self.scale_features_var = tk.BooleanVar(value=True)
        self.scale_cbush_var = tk.BooleanVar(value=False)
        
        tk.Checkbutton(type_frame, text="Parameters (K4_X, K5_X, K6_X)", variable=self.scale_params_var,
                      font=('Arial', 10)).pack(anchor='w')
        tk.Checkbutton(type_frame, text="Features (ACCE_X, DISP_X)", variable=self.scale_features_var,
                      font=('Arial', 10)).pack(anchor='w')
        tk.Checkbutton(type_frame, text="CBUSH Labels (cbush_X_loose)", variable=self.scale_cbush_var,
                      font=('Arial', 10)).pack(anchor='w')
        
        # Action buttons
        button_frame = tk.Frame(controls_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text="📏 Scale Selected Features", command=self.apply_scaling,
                 bg='#e67e22', fg='white', font=('Arial', 11, 'bold'), height=1, width=25).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="💾 Save Scaled Data", command=self.save_scaled_data,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), height=1, width=20).pack(side='left', padx=5)
        
        # Scaling results frame
        results_frame = ttk.LabelFrame(parent, text="📊 Scaling Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Before/After comparison
        comparison_frame = tk.Frame(results_frame)
        comparison_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Before scaling (left side)
        before_frame = ttk.LabelFrame(comparison_frame, text="Before Scaling")
        before_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.before_scaling_text = scrolledtext.ScrolledText(before_frame, height=8, width=40, font=('Consolas', 8))
        self.before_scaling_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # After scaling (right side)
        after_frame = ttk.LabelFrame(comparison_frame, text="After Scaling")
        after_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        self.after_scaling_text = scrolledtext.ScrolledText(after_frame, height=8, width=40, font=('Consolas', 8))
        self.after_scaling_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize with default description
        self.update_scaling_description()
    
    def create_correlation_tab(self, parent):
        """Create correlation matrix tab."""
        # Controls frame
        controls_frame = ttk.LabelFrame(parent, text="🔥 Correlation Analysis")
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # Target selection
        target_frame = tk.Frame(controls_frame)
        target_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(target_frame, text="Target Variable:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.correlation_target_var = tk.StringVar(value="cbush_2_loose")
        self.target_combo = ttk.Combobox(target_frame, textvariable=self.correlation_target_var, 
                                        width=20, font=('Arial', 9))
        self.target_combo.pack(fill='x', pady=2)
        
        # Feature selection checkboxes
        feature_frame = tk.Frame(controls_frame)
        feature_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(feature_frame, text="Include Features:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.corr_params_var = tk.BooleanVar(value=True)
        self.corr_features_var = tk.BooleanVar(value=True)
        self.corr_cbush_var = tk.BooleanVar(value=True)
        
        tk.Checkbutton(feature_frame, text="Parameters (K4_X, K5_X, K6_X)", variable=self.corr_params_var,
                      font=('Arial', 10)).pack(anchor='w')
        tk.Checkbutton(feature_frame, text="Features (ACCE_X, DISP_X)", variable=self.corr_features_var,
                      font=('Arial', 10)).pack(anchor='w')
        tk.Checkbutton(feature_frame, text="CBUSH Labels (cbush_X_loose)", variable=self.corr_cbush_var,
                      font=('Arial', 10)).pack(anchor='w')
        
        # Analysis buttons
        button_frame = tk.Frame(controls_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text="🔥 Generate Correlation Matrix", command=self.generate_correlation,
                 bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'), height=1, width=25).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="🧠 AI Feature Suggestions", command=self.generate_ai_suggestions,
                 bg='#9b59b6', fg='white', font=('Arial', 11, 'bold'), height=1, width=20).pack(side='left', padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="📊 Correlation Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Split into correlation display and AI suggestions
        analysis_frame = tk.Frame(results_frame)
        analysis_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Correlation matrix (left side)
        corr_frame = ttk.LabelFrame(analysis_frame, text="Correlation Matrix")
        corr_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        self.correlation_text = scrolledtext.ScrolledText(corr_frame, height=12, width=50, font=('Consolas', 8))
        self.correlation_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # AI suggestions (right side)
        ai_frame = ttk.LabelFrame(analysis_frame, text="🧠 AI Feature Insights")
        ai_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        self.ai_suggestions_text = scrolledtext.ScrolledText(ai_frame, height=12, width=40, font=('Consolas', 8))
        self.ai_suggestions_text.pack(fill='both', expand=True, padx=5, pady=5)
    
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
            
            # Get user-defined threshold
            user_threshold = self.get_user_threshold()
            self.root.after(0, lambda: self.log_message(f"🎯 Using threshold: {user_threshold:.0e}"))
            
            # Process the study
            result = self.editor.process_single_study(
                param_file=self.param_file_var.get(),
                matrix_file=self.matrix_file_var.get(),
                heeds_directory=self.heeds_dir_var.get(),
                study_name=self.study_name_var.get() or None,
                user_threshold=user_threshold
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
            threshold_used = result.get('user_threshold', 'N/A')
            messagebox.showinfo("Success!", 
                               f"✅ Study processed successfully!\n\n"
                               f"Output: {Path(output_file).name}\n"
                               f"Designs: {result['final_dataset_shape'][0]:,}\n"
                               f"Columns: {result['final_dataset_shape'][1]}\n"
                               f"Alignment: {result['alignment_stats']['alignment_rate']:.1%}\n"
                               f"Threshold: {threshold_used}")
        else:
            self.update_status("❌ Processing failed")
            messagebox.showerror("Processing Failed", "Study processing failed. Check the log for details.")
    
    def add_cbush_summary_columns(self, df):
        """Add cbush_list and cbush_count summary columns."""
        try:
            def get_loose_cbushes(row):
                loose_list = []
                for cbush_num in range(2, 11):
                    if f'cbush_{cbush_num}_loose' in df.columns and row[f'cbush_{cbush_num}_loose'] == 1:
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
    
    def clear_all_inputs(self):
        """Clear all input fields and data."""
        # Clear file paths
        self.param_file_var.set("")
        self.matrix_file_var.set("")
        self.heeds_dir_var.set("")
        self.study_name_var.set("")
        self.study1_file_var.set("")
        self.study2_file_var.set("")
        
        # Clear data
        self.param_data = None
        self.matrix_data = None
        self.combined_data = None
        self.loaded_full_matrix = None
        if hasattr(self, 'scaled_data'):
            delattr(self, 'scaled_data')
        
        # Clear all preview tables
        for tree in [self.param_tree, self.matrix_tree, self.combined_tree, self.load_tree]:
            for item in tree.get_children():
                tree.delete(item)
        
        # Reset info labels
        self.param_info_label.config(text="No parameter file loaded", fg='#7f8c8d')
        self.matrix_info_label.config(text="No feature matrix loaded", fg='#7f8c8d')
        self.combined_info_label.config(text="No combined data available - process a study first", fg='#7f8c8d')
        self.load_info_label.config(text="No full feature matrix loaded", fg='#7f8c8d')
        
        # Clear scaling results
        if hasattr(self, 'before_scaling_text'):
            self.before_scaling_text.delete('1.0', tk.END)
            self.after_scaling_text.delete('1.0', tk.END)
        
        # Clear correlation results
        if hasattr(self, 'correlation_text'):
            self.correlation_text.delete('1.0', tk.END)
            self.ai_suggestions_text.delete('1.0', tk.END)
        
        # Reset status
        self.update_status("Ready")
        
        self.log_message("🧹 All inputs and data cleared")
    
    def browse_study1_file(self):
        """Browse for first study file."""
        filename = filedialog.askopenfilename(
            title="Select First Study File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            self.study1_file_var.set(filename)
            self.log_message(f"📁 Study 1 selected: {Path(filename).name}")
    
    def browse_study2_file(self):
        """Browse for second study file."""
        filename = filedialog.askopenfilename(
            title="Select Second Study File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            self.study2_file_var.set(filename)
            self.log_message(f"📁 Study 2 selected: {Path(filename).name}")
    
    def combine_two_studies(self):
        """Combine two selected study files."""
        study1_file = self.study1_file_var.get()
        study2_file = self.study2_file_var.get()
        
        if not study1_file:
            messagebox.showwarning("Missing File", "Please select Study File 1")
            return
        
        if not study2_file:
            messagebox.showwarning("Missing File", "Please select Study File 2")
            return
        
        if not Path(study1_file).exists():
            messagebox.showerror("File Not Found", f"Study File 1 not found:\n{study1_file}")
            return
        
        if not Path(study2_file).exists():
            messagebox.showerror("File Not Found", f"Study File 2 not found:\n{study2_file}")
            return
        
        # Start combination process
        self.start_progress()
        self.update_status("Combining studies...")
        self.log_message(f"🔗 Combining two study files...")
        
        study_files = [study1_file, study2_file]
        self.processing_thread = threading.Thread(target=self._combine_studies_thread, args=(study_files,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def create_heatmap_tab(self, parent):
        """Create heatmap visualization tab."""
        # Info frame
        info_frame = tk.Frame(parent)
        info_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(info_frame, text="🔥 Correlation Heatmap Visualization", 
                font=('Arial', 14, 'bold')).pack(anchor='w')
        tk.Label(info_frame, text="Generate heatmaps based on correlation matrix selections", 
                font=('Arial', 10), fg='#666666').pack(anchor='w')
        
        # Controls frame
        controls_frame = ttk.LabelFrame(parent, text="Heatmap Settings")
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # Settings
        settings_frame = tk.Frame(controls_frame)
        settings_frame.pack(fill='x', padx=10, pady=5)
        
        # Colormap selection
        colormap_frame = tk.Frame(settings_frame)
        colormap_frame.pack(fill='x', pady=5)
        tk.Label(colormap_frame, text="Color Scheme:", font=('Arial', 10, 'bold')).pack(side='left')
        self.heatmap_colormap_var = tk.StringVar(value="RdYlBu_r")
        colormap_combo = ttk.Combobox(colormap_frame, textvariable=self.heatmap_colormap_var,
                                     values=["RdYlBu_r", "coolwarm", "viridis", "plasma", "inferno", "magma"],
                                     width=15)
        colormap_combo.pack(side='left', padx=10)
        
        # Size settings
        size_frame = tk.Frame(settings_frame)
        size_frame.pack(fill='x', pady=5)
        tk.Label(size_frame, text="Figure Size:", font=('Arial', 10, 'bold')).pack(side='left')
        self.heatmap_width_var = tk.StringVar(value="12")
        self.heatmap_height_var = tk.StringVar(value="8")
        tk.Label(size_frame, text="Width:").pack(side='left', padx=(10,2))
        tk.Entry(size_frame, textvariable=self.heatmap_width_var, width=5).pack(side='left')
        tk.Label(size_frame, text="Height:").pack(side='left', padx=(10,2))
        tk.Entry(size_frame, textvariable=self.heatmap_height_var, width=5).pack(side='left')
        
        # Options
        options_frame = tk.Frame(settings_frame)
        options_frame.pack(fill='x', pady=5)
        self.heatmap_annot_var = tk.BooleanVar(value=True)
        self.heatmap_cbar_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Show values", variable=self.heatmap_annot_var).pack(side='left')
        tk.Checkbutton(options_frame, text="Show colorbar", variable=self.heatmap_cbar_var).pack(side='left', padx=20)
        
        # Generate button
        button_frame = tk.Frame(controls_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(button_frame, text="🔥 Generate Heatmap", command=self.generate_heatmap,
                 bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'), height=1, width=20).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="💾 Save Heatmap", command=self.save_heatmap,
                 bg='#27ae60', fg='white', font=('Arial', 11, 'bold'), height=1, width=15).pack(side='left', padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="📊 Heatmap Display")
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Placeholder for matplotlib figure
        self.heatmap_info_label = tk.Label(results_frame, 
                                          text="🔥 Generate a correlation heatmap using the settings above.\n\nNote: Requires matplotlib. Install with: pip install matplotlib seaborn", 
                                          font=('Arial', 12), fg='#666666', justify='center')
        self.heatmap_info_label.pack(expand=True)
        
        self.current_heatmap_data = None
    
    def generate_heatmap(self):
        """Generate correlation heatmap visualization."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Use scaled data if available, otherwise original
            data = self.scaled_data if hasattr(self, 'scaled_data') else self.combined_data
            
            if data is None:
                messagebox.showwarning("No Data", "Please process a study first")
                return
            
            # Get feature selection from correlation tab
            columns_to_analyze = []
            
            if hasattr(self, 'corr_params_var') and self.corr_params_var.get():
                param_cols = [col for col in data.columns if col.startswith(('K4_', 'K5_', 'K6_'))]
                columns_to_analyze.extend(param_cols)
            
            if hasattr(self, 'corr_features_var') and self.corr_features_var.get():
                feature_cols = [col for col in data.columns if col.startswith(('ACCE_', 'DISP_'))]
                columns_to_analyze.extend(feature_cols[:20])  # Limit for readability
            
            if hasattr(self, 'corr_cbush_var') and self.corr_cbush_var.get():
                cbush_cols = [col for col in data.columns if col.startswith('cbush_') and col.endswith('_loose')]
                columns_to_analyze.extend(cbush_cols)
            
            if not columns_to_analyze:
                messagebox.showwarning("No Features Selected", "Please select features in the Correlation Matrix tab first")
                return
            
            # Calculate correlation matrix
            corr_matrix = data[columns_to_analyze].corr()
            self.current_heatmap_data = corr_matrix
            
            # Clear previous plot
            for widget in self.heatmap_info_label.master.winfo_children():
                if isinstance(widget, tk.Canvas):
                    widget.destroy()
            
            # Create matplotlib figure
            fig_width = float(self.heatmap_width_var.get())
            fig_height = float(self.heatmap_height_var.get())
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Generate heatmap
            sns.heatmap(corr_matrix, 
                       annot=self.heatmap_annot_var.get(),
                       cmap=self.heatmap_colormap_var.get(),
                       cbar=self.heatmap_cbar_var.get(),
                       center=0,
                       square=True,
                       fmt='.2f' if self.heatmap_annot_var.get() else '',
                       ax=ax)
            
            plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.heatmap_info_label.master)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Hide the info label
            self.heatmap_info_label.pack_forget()
            
            self.log_message(f"🔥 Generated heatmap with {len(columns_to_analyze)} features")
            
        except ImportError:
            messagebox.showerror("Missing Libraries", 
                               "Please install required packages:\n\npip install matplotlib seaborn")
        except Exception as e:
            messagebox.showerror("Heatmap Error", f"Failed to generate heatmap: {str(e)}")
    
    def save_heatmap(self):
        """Save the current heatmap to file."""
        if self.current_heatmap_data is None:
            messagebox.showwarning("No Heatmap", "Please generate a heatmap first")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Heatmap",
                defaultextension=".png",
                filetypes=[("PNG Files", "*.png"), ("PDF Files", "*.pdf"), ("SVG Files", "*.svg"), ("All Files", "*.*")]
            )
            
            if filename:
                import matplotlib.pyplot as plt
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.log_message(f"💾 Heatmap saved: {Path(filename).name}")
                messagebox.showinfo("Success", f"Heatmap saved successfully!\n\n{Path(filename).name}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save heatmap: {str(e)}")


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
        parser.add_argument('--threshold', '-t', type=float, default=1e8,
                           help='CBUSH loose threshold (default: 1e8)')
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
                study_name=args.study_name,
                user_threshold=args.threshold
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
            print(f"🎯 Threshold used: {args.threshold}")
            
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
        print("🚀 Starting Feature Matrix Editor v6.0 GUI...")
        create_gui()
        return 0


if __name__ == "__main__":
    sys.exit(main())
