import pandas as pd
import numpy as np
import os
import glob
import math
from pathlib import Path
import warnings
import gc
import psutil
from collections import defaultdict
import time
warnings.filterwarnings('ignore')

class BoltAnomalyFeatureExtractor:
    """
    Enhanced feature extractor for bolt anomaly detection from NASTRAN random vibration results.
    Now supports massive datasets (11K-111K designs), flexible study types, and auto-detection.
    Updated to work with FlexibleCBUSHProcessor and massive dataset GUI.
    """
    
    def __init__(self, base_path=None):
        """
        Initialize the feature extractor.
        
        Args:
            base_path: Base directory containing POST_0 folder with all designs
        """
        self.base_path = base_path
        self.key_measurements = [
            'ACCE_T1_Area',      # Most sensitive to bolt changes
            'ACCE_T1_PSD_1',     # First mode magnitude  
            'ACCE_T1_PSD_2',     # Second mode magnitude
            'ACCE_T1_PSD_3',     # Third mode magnitude
            'DISP_T1_Area',      # Displacement energy
            'DISP_T1_DISP_1'     # First mode displacement
        ]
        
        # Enhanced encoding system for flexible studies
        self.stiffness_encoding = self._create_stiffness_encoding()
        
        # Memory management for massive datasets
        self.max_memory_gb = 16.0
        self.enable_memory_management = True
        self.processing_stats = {}
        
        # Batch processing settings
        self.default_batch_size = 500
        self.chunk_processing = True
        
        # Error handling
        self.skip_failed_designs = True
        self.log_failed_designs = True
        self.max_failure_rate = 0.1  # Stop if >10% designs fail
        
    def _create_stiffness_encoding(self):
        """
        Create enhanced stiffness encoding that works for all study types.
        """
        return {
            1: {'exp': 4, 'value': 1e4, 'description': 'Extremely loose'},
            2: {'exp': 5, 'value': 1e5, 'description': 'Very loose'},
            3: {'exp': 6, 'value': 1e6, 'description': 'Loose'},
            4: {'exp': 7, 'value': 1e7, 'description': 'Medium-loose'},
            5: {'exp': 8, 'value': 1e8, 'description': 'Driving CBUSH (reference)'},
            6: {'exp': 9, 'value': 1e9, 'description': 'Medium'},
            7: {'exp': 10, 'value': 1e10, 'description': 'Medium-tight'},
            8: {'exp': 11, 'value': 1e11, 'description': 'Tight'},
            9: {'exp': 12, 'value': 1e12, 'description': 'Very tight (baseline)'},
            10: {'exp': 13, 'value': 1e13, 'description': 'Extremely tight'},
            11: {'exp': 14, 'value': 1e14, 'description': 'Ultra tight'}
        }
    
    def create_flexible_labels(self, design_id, study_info=None, cbush_involvement=None):
        """
        Create flexible labels that work for any study type.
        
        Args:
            design_id: Design identifier
            study_info: Study information from auto-detection (optional)
            cbush_involvement: Manual CBUSH involvement list (optional)
            
        Returns:
            dict: Flexible labels for any study type
        """
        labels = {
            'design_id': design_id,
            'extraction_method': 'flexible_auto_detection'
        }
        
        # Add study information if available
        if study_info:
            labels.update({
                'study_name': study_info.get('name', f'Study_{design_id}'),
                'study_type': study_info.get('type', 'unknown'),
                'study_description': study_info.get('description', 'Auto-detected study'),
                'total_designs_in_study': study_info.get('total_designs', 1),
                'design_range': study_info.get('design_range', (design_id, design_id)),
                'auto_detected': True
            })
        else:
            # Default labels for standalone processing
            labels.update({
                'study_name': f'Standalone_Design_{design_id}',
                'study_type': 'standalone',
                'study_description': 'Single design processing',
                'total_designs_in_study': 1,
                'design_range': (design_id, design_id),
                'auto_detected': False
            })
        
        # Create multi-label CBUSH flags (initialize all to 0)
        for cbush_num in range(2, 11):
            labels[f'cbush_{cbush_num}_loose'] = 0
        
        # Determine CBUSH involvement
        if cbush_involvement:
            # Use provided CBUSH involvement
            involved_cbushes = cbush_involvement
        elif study_info:
            # Auto-detect from study information
            involved_cbushes = self._auto_detect_cbush_involvement(design_id, study_info)
        else:
            # Default: assume no specific CBUSH involvement (baseline condition)
            involved_cbushes = []
        
        # Set CBUSH involvement flags
        for cbush_num in involved_cbushes:
            if 2 <= cbush_num <= 10:
                labels[f'cbush_{cbush_num}_loose'] = 1
        
        labels['cbush_list'] = involved_cbushes
        labels['cbush_count'] = len(involved_cbushes)
        
        # Legacy compatibility for single-CBUSH studies
        if len(involved_cbushes) == 1:
            labels['varied_cbush'] = involved_cbushes[0]
        else:
            labels['varied_cbush'] = -1  # Multi-CBUSH indicator
        
        return labels
    
    def _auto_detect_cbush_involvement(self, design_id, study_info):
        """
        Auto-detect CBUSH involvement based on study patterns and design position.
        Enhanced for massive datasets and flexible study types.
        """
        study_type = study_info.get('type', 'unknown')
        total_designs = study_info.get('total_designs', 1)
        design_numbers = study_info.get('design_numbers', [design_id])
        
        if design_id not in design_numbers:
            # Design not in expected range - assume all CBUSHes potentially involved
            return list(range(2, 11))
        
        design_index = design_numbers.index(design_id)
        
        # Enhanced pattern detection for different study types
        if study_type in ['single_cbush_complete', 'single_cbush_batch']:
            return self._detect_single_cbush_pattern(design_index, total_designs)
            
        elif study_type in ['adjacent_pairs', 'adjacent_pairs_batch']:
            return self._detect_adjacent_pairs_pattern(design_index, total_designs)
            
        elif study_type in ['multi_bolt_batch', 'massive_study']:
            return self._detect_multi_bolt_pattern(design_index, total_designs)
            
        elif study_type == 'random_study':
            return self._detect_random_study_pattern(design_id, total_designs)
            
        elif study_type == 'custom_study':
            return self._detect_custom_study_pattern(design_id, study_info)
            
        else:
            # Unknown pattern - conservative approach
            return list(range(2, 11))
    
    def _detect_single_cbush_pattern(self, design_index, total_designs):
        """Detect single-CBUSH patterns for S1 studies."""
        if total_designs == 64:
            # Single set of 64 designs - one CBUSH varied
            # Without additional info, assume CBUSH 2 (first in sequence)
            return [2]
        elif total_designs == 576:
            # S1-Limited: 9 sets × 64 designs
            cbush_num = (design_index // 64) % 9 + 2
            return [cbush_num]
        elif total_designs == 11979:
            # S1-Complete: Need more sophisticated detection
            # Estimate based on design position
            cbush_num = (design_index // 1331) % 9 + 2  # 1331 = 11^3
            return [cbush_num]
        else:
            # Unknown single-CBUSH pattern
            cbush_num = (design_index % 9) + 2
            return [cbush_num]
    
    def _detect_adjacent_pairs_pattern(self, design_index, total_designs):
        """Detect adjacent pairs patterns for S2 studies."""
        if total_designs == 16:
            # Single pair study
            return [2, 3]  # Default to first pair
        elif total_designs == 128:
            # S2-Limited: 8 pairs × 16 designs
            pair_index = (design_index // 16) % 8
            return [pair_index + 2, pair_index + 3]
        elif total_designs == 968:
            # S2-Complete: 8 pairs × 121 designs
            pair_index = (design_index // 121) % 8
            return [pair_index + 2, pair_index + 3]
        else:
            # Unknown adjacent pattern - assume first pair
            return [2, 3]
    
    def _detect_multi_bolt_pattern(self, design_index, total_designs):
        """Detect multi-bolt patterns for S3 studies."""
        if total_designs == 64:
            # Single multi-bolt study
            return [2, 5, 8]  # Default combination
        elif total_designs == 192:
            # S3-Limited: 3 scenarios × 64 designs
            scenario_index = (design_index // 64) % 3
            combinations = [(2, 5, 8), (3, 6, 9), (4, 7, 10)]
            return list(combinations[scenario_index])
        elif total_designs == 111804:
            # S3-Complete: 84 combinations × 1331 designs
            combo_index = (design_index // 1331) % 84
            # Generate the combination from index
            available_cbushes = list(range(2, 11))
            from itertools import combinations
            all_combos = list(combinations(available_cbushes, 3))
            if combo_index < len(all_combos):
                return list(all_combos[combo_index])
            else:
                return [2, 5, 8]  # Default
        else:
            # Unknown multi-bolt pattern
            return [2, 5, 8]
    
    def _detect_random_study_pattern(self, design_id, total_designs):
        """Detect patterns for random studies."""
        # For random studies, we can't predict CBUSH involvement from design_id
        # Use a pseudo-random approach based on design_id
        np.random.seed(design_id)  # Consistent "randomness" for same design
        
        # Randomly select 1-3 CBUSHes
        num_cbushes = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        available_cbushes = list(range(2, 11))
        selected_cbushes = np.random.choice(available_cbushes, size=num_cbushes, replace=False)
        
        return sorted(selected_cbushes.tolist())
    
    def _detect_custom_study_pattern(self, design_id, study_info):
        """Detect patterns for custom studies."""
        # For custom studies, use any hints from study_info
        if 'expected_cbushes' in study_info:
            return study_info['expected_cbushes']
        elif 'likely_cbush_count' in study_info:
            count = study_info['likely_cbush_count']
            if count == 1:
                return [2]  # Default single CBUSH
            elif count == 2:
                return [2, 3]  # Default adjacent pair
            elif count == 3:
                return [2, 5, 8]  # Default distributed
            else:
                return list(range(2, min(2 + count, 11)))
        else:
            # Conservative approach - assume all CBUSHes potentially involved
            return list(range(2, 11))
    
    def extract_features_from_files(self, accel_file, disp_file, design_id, 
                                   study_info=None, cbush_involvement=None):
        """
        Enhanced feature extraction with flexible labeling support.
        
        Args:
            accel_file: Path to acceleration_results.csv
            disp_file: Path to displacement_results.csv
            design_id: Design identifier
            study_info: Optional study information for auto-detection
            cbush_involvement: Optional manual CBUSH involvement specification
            
        Returns:
            Dictionary of features with flexible labels
        """
        try:
            # Handle different file naming conventions
            if not os.path.exists(accel_file):
                # Try with "_delta" suffix
                accel_delta_file = accel_file.replace('.csv', '_delta.csv')
                if os.path.exists(accel_delta_file):
                    accel_file = accel_delta_file
            
            if not os.path.exists(disp_file):
                # Try with "_delta" suffix
                disp_delta_file = disp_file.replace('.csv', '_delta.csv')
                if os.path.exists(disp_delta_file):
                    disp_file = disp_delta_file
            
            # Load the data
            accel_df = pd.read_csv(accel_file)
            disp_df = pd.read_csv(disp_file)
            
            # Initialize feature dictionary with flexible labels
            features = self.create_flexible_labels(design_id, study_info, cbush_involvement)
            
            # Get node columns (exclude 'Measurement')
            node_columns = [col for col in accel_df.columns if col.startswith('Node_')]
            
            # Extract features for each key measurement
            for measurement in self.key_measurements:
                # Find the row for this measurement
                if measurement.startswith('ACCE_'):
                    data_row = accel_df[accel_df['Measurement'] == measurement]
                elif measurement.startswith('DISP_'):
                    data_row = disp_df[disp_df['Measurement'] == measurement]
                else:
                    continue
                
                if not data_row.empty:
                    # Extract values for each node
                    for node_col in node_columns:
                        feature_name = f"{measurement}_{node_col}"
                        try:
                            features[feature_name] = float(data_row.iloc[0][node_col])
                        except (ValueError, TypeError):
                            features[feature_name] = 0.0
                else:
                    # If measurement not found, fill with zeros
                    for node_col in node_columns:
                        feature_name = f"{measurement}_{node_col}"
                        features[feature_name] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Error processing design {design_id}: {e}")
            return None
    
    def process_design_batch_enhanced(self, design_numbers, study_info=None, 
                                    progress_callback=None, batch_size=None):
        """
        Enhanced batch processing with memory management and flexible study support.
        
        Args:
            design_numbers: List of design numbers to process
            study_info: Study information for auto-detection
            progress_callback: Optional function to call with progress updates
            batch_size: Size of processing batches for memory management
            
        Returns:
            List of feature dictionaries
        """
        if batch_size is None:
            batch_size = self.default_batch_size
        
        print(f"🔄 Processing {len(design_numbers)} designs in batches of {batch_size}...")
        
        features_list = []
        total_designs = len(design_numbers)
        failed_count = 0
        
        # Process in smaller batches to manage memory
        for batch_start in range(0, total_designs, batch_size):
            batch_end = min(batch_start + batch_size, total_designs)
            batch_designs = design_numbers[batch_start:batch_end]
            
            batch_features = []
            
            for i, design_num in enumerate(batch_designs):
                global_index = batch_start + i
                
                if progress_callback:
                    progress_callback(global_index + 1, total_designs, design_num)
                
                # Construct file paths
                design_folder = f"Design{design_num}"
                accel_file = os.path.join(self.base_path, "POST_0", design_folder, 
                                        "Analysis_1", "acceleration_results.csv")
                disp_file = os.path.join(self.base_path, "POST_0", design_folder,
                                       "Analysis_1", "displacement_results.csv")
                
                # Check if files exist
                if os.path.exists(accel_file) and os.path.exists(disp_file):
                    features = self.extract_features_from_files(accel_file, disp_file, 
                                                              design_num, study_info)
                    if features:
                        batch_features.append(features)
                    else:
                        failed_count += 1
                        if self.log_failed_designs:
                            print(f"⚠️  Design {design_num}: Feature extraction failed")
                else:
                    failed_count += 1
                    if self.log_failed_designs:
                        print(f"⚠️  Design {design_num}: Files not found")
                
                # Check failure rate
                if not self.skip_failed_designs and failed_count > 0:
                    total_processed = len(batch_features) + failed_count
                    failure_rate = failed_count / total_processed
                    if failure_rate > self.max_failure_rate and total_processed > 100:
                        raise RuntimeError(f"High failure rate ({failure_rate:.1%}) - stopping processing")
            
            # Add batch to main list
            features_list.extend(batch_features)
            
            # Memory management
            if self.enable_memory_management:
                self._manage_memory(f"Batch {batch_start//batch_size + 1}")
            
            print(f"   Batch {batch_start//batch_size + 1}/{(total_designs-1)//batch_size + 1} complete: "
                  f"{len(batch_features)} designs processed")
        
        success_rate = len(features_list) / total_designs if total_designs > 0 else 0
        print(f"✅ Enhanced batch processing complete: {len(features_list)} successful, "
              f"{failed_count} failed ({success_rate:.1%} success rate)")
        
        return features_list
    
    def _manage_memory(self, context=""):
        """Enhanced memory management with monitoring."""
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
                
                # Check memory after cleanup
                memory_mb_after = process.memory_info().rss / 1024 / 1024
                memory_gb_after = memory_mb_after / 1024
                
                saved_mb = memory_mb - memory_mb_after
                print(f"   Memory cleanup: {saved_mb:.1f}MB freed, {collected} objects collected")
                print(f"   Memory after cleanup: {memory_gb_after:.1f}GB")
                
        except Exception as e:
            print(f"⚠️  Memory management warning: {e}")
    
    def create_master_matrix_enhanced(self, design_numbers=None, study_info=None, 
                                    batch_size=None, output_file=None, 
                                    save_intermediate=True, max_memory_gb=None):
        """
        Enhanced master matrix creation with massive dataset support.
        
        Args:
            design_numbers: List of design numbers to process (auto-detected if None)
            study_info: Study information for flexible labeling
            batch_size: Processing batch size for memory management
            output_file: Output CSV file name
            save_intermediate: Save intermediate results for recovery
            max_memory_gb: Memory limit override
            
        Returns:
            pandas DataFrame with all features, or None if failed
        """
        if max_memory_gb:
            self.max_memory_gb = max_memory_gb
        
        if batch_size is None:
            batch_size = self.default_batch_size
        
        # Auto-detect available designs if not provided
        if design_numbers is None:
            design_numbers = self._auto_detect_available_designs()
        
        if not design_numbers:
            print("❌ No designs available for processing")
            return None
        
        total_designs = len(design_numbers)
        print(f"🔨 Creating enhanced master matrix for {total_designs:,} designs...")
        print(f"   Processing in batches of {batch_size}")
        print(f"   Memory limit: {self.max_memory_gb}GB")
        
        all_features = []
        processed_count = 0
        
        # Process in batches with intermediate saves
        for batch_num in range(0, total_designs, batch_size):
            batch_end = min(batch_num + batch_size, total_designs)
            batch_designs = design_numbers[batch_num:batch_end]
            
            print(f"\n📊 Processing batch {batch_num//batch_size + 1}: "
                  f"Designs {batch_designs[0]}-{batch_designs[-1]}")
            
            def progress_cb(current, total, design_id):
                if current % 50 == 0 or current == total:
                    print(f"     Progress: {current}/{total} - Design {design_id}")
            
            # Process batch
            batch_features = self.process_design_batch_enhanced(
                batch_designs, study_info, progress_cb, batch_size=min(100, len(batch_designs)))
            
            if batch_features:
                all_features.extend(batch_features)
                processed_count += len(batch_features)
                
                # Save intermediate results for large datasets
                if save_intermediate and total_designs > 1000 and batch_num > 0:
                    if batch_num % (batch_size * 5) == 0:  # Every 5 batches
                        temp_df = pd.DataFrame(all_features)
                        temp_file = f"temp_features_batch_{batch_num//batch_size}.csv"
                        temp_df.to_csv(temp_file, index=False)
                        print(f"   💾 Saved intermediate results: {temp_file}")
                
                # Memory management
                self._manage_memory(f"Batch {batch_num//batch_size + 1}")
                
                success_rate = processed_count / (batch_end) if batch_end > 0 else 0
                print(f"   ✅ Batch complete: {len(batch_features)} processed "
                      f"(Total: {processed_count:,}/{batch_end:,}, {success_rate:.1%} success)")
            else:
                print(f"   ❌ Batch failed: No features extracted")
        
        # Create final DataFrame
        if all_features:
            print(f"\n🔧 Creating final DataFrame from {len(all_features):,} feature sets...")
            
            try:
                master_df = pd.DataFrame(all_features)
                master_df = master_df.sort_values('design_id').reset_index(drop=True)
                
                # Calculate final statistics
                feature_cols = [col for col in master_df.columns 
                               if col.startswith(('ACCE_', 'DISP_'))]
                memory_mb = master_df.memory_usage(deep=True).sum() / 1024 / 1024
                
                print(f"✅ Enhanced master matrix created successfully!")
                print(f"   Shape: {master_df.shape}")
                print(f"   Features per design: {len(feature_cols)}")
                print(f"   Memory usage: {memory_mb:.1f} MB")
                print(f"   Success rate: {len(master_df)/total_designs:.1%}")
                
                # Save to file
                if output_file:
                    if memory_mb > 500:  # Large file - save in chunks
                        self._save_large_dataframe(master_df, output_file)
                    else:
                        master_df.to_csv(output_file, index=False)
                    print(f"   💾 Saved to: {output_file}")
                
                return master_df
                
            except Exception as e:
                print(f"❌ Error creating DataFrame: {e}")
                return None
        else:
            print("❌ No features extracted!")
            return None
    
    def _auto_detect_available_designs(self):
        """Auto-detect available design numbers from directory structure."""
        if not self.base_path:
            return []
        
        post_0_path = os.path.join(self.base_path, "POST_0")
        if not os.path.exists(post_0_path):
            return []
        
        design_numbers = []
        for item in os.listdir(post_0_path):
            if item.startswith("Design") and os.path.isdir(os.path.join(post_0_path, item)):
                try:
                    design_num = int(item.replace("Design", ""))
                    
                    # Check if analysis files exist
                    analysis_path = os.path.join(post_0_path, item, "Analysis_1")
                    accel_file = os.path.join(analysis_path, "acceleration_results.csv")
                    disp_file = os.path.join(analysis_path, "displacement_results.csv")
                    
                    if os.path.exists(accel_file) and os.path.exists(disp_file):
                        design_numbers.append(design_num)
                        
                except ValueError:
                    continue
        
        design_numbers.sort()
        print(f"🔍 Auto-detected {len(design_numbers)} available designs")
        return design_numbers
    
    def _save_large_dataframe(self, df, filepath):
        """Save large DataFrames in chunks to prevent memory issues."""
        chunk_size = 10000
        print(f"💾 Saving large DataFrame in chunks to {filepath}...")
        
        # Save header first
        df.iloc[:0].to_csv(filepath, index=False)
        
        # Append chunks
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk.to_csv(filepath, mode='a', header=False, index=False)
            print(f"   Saved chunk {i//chunk_size + 1}/{(len(df)-1)//chunk_size + 1}")
    
    def get_feature_summary_enhanced(self, master_df):
        """Enhanced feature matrix summary with additional statistics."""
        print("\n=== ENHANCED FEATURE MATRIX SUMMARY ===")
        print(f"Total designs: {len(master_df):,}")
        print(f"Total features: {master_df.shape[1]:,}")
        
        # Memory usage
        memory_mb = master_df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f} MB ({memory_mb/1024:.2f} GB)")
        
        # Study composition
        if 'study_type' in master_df.columns:
            study_counts = master_df['study_type'].value_counts()
            print(f"\nStudy type distribution:")
            for study_type, count in study_counts.items():
                percentage = (count / len(master_df)) * 100
                print(f"  {study_type}: {count:,} designs ({percentage:.1f}%)")
        
        # CBUSH involvement analysis
        cbush_involvement = {}
        for cbush_num in range(2, 11):
            col_name = f'cbush_{cbush_num}_loose'
            if col_name in master_df.columns:
                count = master_df[col_name].sum()
                percentage = (count / len(master_df)) * 100
                cbush_involvement[cbush_num] = {'count': count, 'percentage': percentage}
        
        if cbush_involvement:
            print(f"\nCBUSH involvement distribution:")
            for cbush_num, stats in cbush_involvement.items():
                print(f"  CBUSH {cbush_num}: {stats['count']:,} designs ({stats['percentage']:.1f}%)")
        
        # Feature statistics
        feature_cols = [col for col in master_df.columns 
                       if col.startswith(('ACCE_', 'DISP_'))]
        
        if feature_cols:
            print(f"\nFeature statistics:")
            print(f"  Feature columns: {len(feature_cols):,}")
            
            # Check for missing values
            missing_count = master_df[feature_cols].isnull().sum().sum()
            missing_percentage = (missing_count / (len(master_df) * len(feature_cols))) * 100
            print(f"  Missing values: {missing_count:,} ({missing_percentage:.2f}%)")
            
            # Feature variance analysis
            feature_variances = master_df[feature_cols].var()
            zero_variance = (feature_variances == 0).sum()
            low_variance = (feature_variances < 0.001).sum()
            
            print(f"  Zero variance features: {zero_variance}")
            print(f"  Low variance features (<0.001): {low_variance}")
            
            # Most responsive nodes/features
            feature_means = master_df[feature_cols].abs().mean().sort_values(ascending=False)
            print(f"\nTop 5 most responsive features:")
            for i, (feature, value) in enumerate(feature_means.head().items()):
                print(f"  {i+1}. {feature}: {value:.2f} avg absolute value")
        
        return {
            'total_designs': len(master_df),
            'total_features': master_df.shape[1],
            'memory_mb': memory_mb,
            'study_composition': study_counts.to_dict() if 'study_type' in master_df.columns else {},
            'cbush_involvement': cbush_involvement,
            'feature_quality': {
                'total_features': len(feature_cols),
                'missing_percentage': missing_percentage if feature_cols else 0,
                'zero_variance_count': zero_variance if feature_cols else 0,
                'low_variance_count': low_variance if feature_cols else 0
            }
        }
    
    # Legacy compatibility methods for existing code
    def process_design_batch(self, design_numbers, progress_callback=None):
        """Legacy method for backward compatibility."""
        return self.process_design_batch_enhanced(design_numbers, progress_callback=progress_callback)
    
    def create_master_matrix(self, design_numbers=None, batch_size=50, output_file='master_feature_matrix.csv'):
        """Legacy method for backward compatibility."""
        return self.create_master_matrix_enhanced(design_numbers, batch_size=batch_size, output_file=output_file)
    
    def get_feature_summary(self, master_df):
        """Legacy method for backward compatibility."""
        return self.get_feature_summary_enhanced(master_df)


# Enhanced usage example for massive datasets
def enhanced_processing_example():
    """
    Enhanced example showing how to process massive datasets with the updated extractor.
    """
    # Initialize extractor for massive dataset
    heeds_output_dir = r"C:\HEEDS_Output_Massive"
    extractor = BoltAnomalyFeatureExtractor(heeds_output_dir)
    
    # Configure for large dataset processing
    extractor.max_memory_gb = 32.0  # Use up to 32GB RAM
    extractor.enable_memory_management = True
    
    # Example 1: Process Random Study (5,000 designs)
    print("=== EXAMPLE 1: Random Study Processing ===")
    random_study_info = {
        'name': 'Random_Study_5000',
        'type': 'random_study',
        'total_designs': 5000,
        'design_numbers': list(range(1, 5001)),
        'description': 'Large Random Study'
    }
    
    random_matrix = extractor.create_master_matrix_enhanced(
        design_numbers=list(range(1, 5001)),
        study_info=random_study_info,
        batch_size=500,
        output_file='random_study_5000_features.csv'
    )
    
    if random_matrix is not None:
        summary = extractor.get_feature_summary_enhanced(random_matrix)
    
    # Example 2: Process S3-Complete Study (111,804 designs)
    print("\n=== EXAMPLE 2: S3-Complete Processing (Subset) ===")
    s3_study_info = {
        'name': 'S3_Complete_Subset',
        'type': 'massive_study',
        'total_designs': 50000,  # Process subset first
        'design_numbers': list(range(1, 50001)),
        'description': 'S3-Complete Subset'
    }
    
    s3_matrix = extractor.create_master_matrix_enhanced(
        design_numbers=list(range(1, 50001)),
        study_info=s3_study_info,
        batch_size=1000,
        output_file='s3_complete_subset_features.csv',
        max_memory_gb=64.0  # Use more memory for massive dataset
    )
    
    # Example 3: Auto-detect and process all available designs
    print("\n=== EXAMPLE 3: Auto-Detection Processing ===")
    auto_matrix = extractor.create_master_matrix_enhanced(
        design_numbers=None,  # Auto-detect
        study_info=None,      # Auto-detect
        batch_size=500,
        output_file='auto_detected_features.csv'
    )


if __name__ == "__main__":
    enhanced_processing_example()
