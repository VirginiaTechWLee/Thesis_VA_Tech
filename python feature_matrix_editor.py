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
from pathlib import Path
from datetime import datetime
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


def main():
    """Main entry point for the feature matrix editor."""
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


if __name__ == "__main__":
    exit(main())
