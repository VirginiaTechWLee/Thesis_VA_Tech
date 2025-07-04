import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import warnings
import threading
from collections import defaultdict
import gc
import psutil
warnings.filterwarnings('ignore')

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
    
    def scan_available_designs(self, post_0_directory):
        """
        Scan POST_0 directory for available designs and detect patterns.
        
        Args:
            post_0_directory: Path to POST_0 directory
            
        Returns:
            dict: Available designs information
        """
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
            'density': total_designs / design_range if design_range > 0 else 0
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
        
        # Add pattern analysis
        if len(design_numbers) > 1:
            gaps = []
            for i in range(len(design_numbers) - 1):
                gap = design_numbers[i + 1] - design_numbers[i]
                if gap > 1:
                    gaps.append(gap - 1)
            
            study_info['gaps'] = gaps
            study_info['has_gaps'] = len(gaps) > 0
            study_info['gap_pattern'] = 'regular' if len(set(gaps)) <= 2 else 'irregular' if gaps else 'none'
        
        return study_info
    
    def create_study_labels_flexible(self, design_id, study_info, auto_cbush_detection=True):
        """
        Create flexible study labels for different types of studies.
        
        Args:
            design_id: Design identifier
            study_info: Study information from auto-detection
            auto_cbush_detection: Whether to auto-detect CBUSH involvement
            
        Returns:
            dict: Study labels
        """
        labels = {
            'design_id': design_id,
            'study_name': study_info.get('name', f"Auto_Study_{study_info['design_range'][0]}_{study_info['design_range'][1]}"),
            'study_type': study_info.get('type', 'unknown'),
            'study_description': study_info.get('description', 'Unknown Study'),
            'auto_detected': True
        }
        
        # Create multi-label CBUSH flags (initialize all to 0)
        for cbush_num in range(2, 11):
            labels[f'cbush_{cbush_num}_loose'] = 0
        
        if auto_cbush_detection:
            # Auto-detect CBUSH involvement based on study type and design position
            involved_cbushes = self._auto_detect_cbush_involvement(design_id, study_info)
            
            for cbush_num in involved_cbushes:
                if 2 <= cbush_num <= 10:
                    labels[f'cbush_{cbush_num}_loose'] = 1
            
            labels['cbush_list'] = involved_cbushes
            labels['cbush_count'] = len(involved_cbushes)
        else:
            # Default: assume all CBUSHes potentially involved
            labels['cbush_list'] = list(range(2, 11))
            labels['cbush_count'] = 9
        
        # Legacy compatibility
        if len(labels.get('cbush_list', [])) == 1:
            labels['varied_cbush'] = labels['cbush_list'][0]
        else:
            labels['varied_cbush'] = -1  # Multi-CBUSH indicator
        
        return labels
    
    def _auto_detect_cbush_involvement(self, design_id, study_info):
        """
        Auto-detect which CBUSHes are involved based on study patterns.
        
        This is a heuristic approach - for more accurate detection,
        actual parameter sweep files would need to be analyzed.
        """
        study_type = study_info.get('type', 'unknown')
        total_designs = study_info.get('total_designs', 0)
        design_numbers = study_info.get('design_numbers', [])
        
        if design_id not in design_numbers:
            return []
        
        design_index = design_numbers.index(design_id)
        
        # Heuristic CBUSH detection based on known patterns
        if study_type == 'single_cbush_complete':
            # Single CBUSH - detect which one based on design position
            # 64 designs = 4^3 combinations, cycling through CBUSHes
            cbush_num = (design_index // 64) % 9 + 2  # CBUSHes 2-10
            return [cbush_num]
            
        elif study_type == 'adjacent_pairs':
            # Adjacent pairs - detect pair based on design position
            pair_index = (design_index // 16) % 8
            cbush1 = pair_index + 2
            cbush2 = pair_index + 3
            return [cbush1, cbush2]
            
        elif study_type == 'multi_bolt_batch':
            # Multi-bolt - detect combination based on design position
            if total_designs == 192:  # S3-Limited pattern
                combo_index = (design_index // 64) % 3
                combos = [(2, 5, 8), (3, 6, 9), (4, 7, 10)]
                return list(combos[combo_index])
            else:
                # For massive S3 studies, more complex detection needed
                # Default to distributed pattern
                return [2, 5, 8]  # Default first combination
                
        elif study_type in ['random_study', 'massive_study', 'custom_study']:
            # Random studies - assume all CBUSHes potentially involved
            # Could be refined by analyzing actual parameter values
            return list(range(2, 11))
            
        else:
            # Unknown pattern - assume all CBUSHes
            return list(range(2, 11))
    
    def process_design_batch(self, design_numbers, study_info, heeds_directory):
        """
        Process a batch of designs efficiently with enhanced error handling.
        
        Args:
            design_numbers: List of design numbers to process
            study_info: Study information dictionary
            heeds_directory: HEEDS output directory path
            
        Returns:
            list: Processed design data
        """
        from feature_matrix_builder import BoltAnomalyFeatureExtractor
        
        # Initialize extractor
        extractor = BoltAnomalyFeatureExtractor(heeds_directory)
        
        processed_data = []
        failed_count = 0
        
        for design_id in design_numbers:
            # Check for stop/pause
            if self.should_stop:
                print(f"🛑 Processing stopped by user at design {design_id}")
                break
            
            while self.is_paused:
                time.sleep(0.1)
                if self.should_stop:
                    break
            
            try:
                # Construct file paths
                design_folder = f"Design{design_id}"
                accel_file = os.path.join(heeds_directory, "POST_0", design_folder, 
                                        "Analysis_1", "acceleration_results.csv")
                disp_file = os.path.join(heeds_directory, "POST_0", design_folder,
                                       "Analysis_1", "displacement_results.csv")
                
                # Check if files exist
                if not (os.path.exists(accel_file) and os.path.exists(disp_file)):
                    if self.log_failed_designs:
                        print(f"⚠️  Design {design_id}: Missing files")
                    failed_count += 1
                    continue
                
                # Extract features
                features = extractor.extract_features_from_files(accel_file, disp_file, design_id)
                
                if features:
                    # Add study-specific labels
                    study_labels = self.create_study_labels_flexible(design_id, study_info)
                    features.update(study_labels)
                    processed_data.append(features)
                else:
                    if self.log_failed_designs:
                        print(f"⚠️  Design {design_id}: Feature extraction failed")
                    failed_count += 1
                
            except Exception as e:
                if self.log_failed_designs:
                    print(f"❌ Design {design_id}: Error - {str(e)}")
                failed_count += 1
                
                if not self.skip_failed_designs:
                    raise e
            
            # Check failure rate
            total_processed = len(processed_data) + failed_count
            if total_processed > 0:
                failure_rate = failed_count / total_processed
                if failure_rate > self.max_failure_rate and total_processed > 100:
                    raise RuntimeError(f"High failure rate ({failure_rate:.1%}) - stopping processing")
        
        print(f"📊 Batch complete: {len(processed_data)} processed, {failed_count} failed")
        return processed_data
    
    def store_processed_study(self, study_name, study_data):
        """Store processed study data with memory management."""
        if study_data:
            # Convert to DataFrame
            study_df = pd.DataFrame(study_data)
            
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
    
    def get_processed_studies_summary(self):
        """Get summary of all processed studies."""
        summary = {}
        
        for study_name, study_info in self.processed_data.items():
            summary[study_name] = {
                'type': study_info['data']['study_type'].iloc[0] if 'study_type' in study_info['data'].columns else 'unknown',
                'design_count': study_info['design_count'],
                'feature_count': study_info['feature_count'],
                'memory_mb': study_info['memory_mb'],
                'processing_time': study_info['processing_time'],
                'quality_score': study_info['quality_score']
            }
        
        return summary
    
    def create_master_matrix(self, strategy="smart_combined", max_designs=50000, 
                           max_memory_gb=16.0, sampling_strategy="random"):
        """
        Create master matrix with massive dataset support and intelligent strategies.
        
        Args:
            strategy: Matrix creation strategy
            max_designs: Maximum designs to include
            max_memory_gb: Maximum memory limit
            sampling_strategy: How to sample if size limiting
            
        Returns:
            pandas.DataFrame: Master matrix
        """
        print(f"\n🔨 Creating master matrix using {strategy} strategy...")
        print(f"   Limits: {max_designs:,} designs, {max_memory_gb}GB memory")
        
        if not self.processed_data:
            print("❌ No processed data available")
            return None
        
        try:
            if strategy == "smart_combined":
                return self._create_smart_combined_matrix(max_designs, sampling_strategy)
            elif strategy == "complete_combined":
                return self._create_complete_combined_matrix()
            elif strategy == "type_specific":
                return self._create_type_specific_matrix(max_designs)
            elif strategy == "size_limited":
                return self._create_size_limited_matrix(max_designs, sampling_strategy)
            elif strategy == "custom":
                return self._create_custom_matrix()
            else:
                print(f"❌ Unknown strategy: {strategy}")
                return None
                
        except Exception as e:
            print(f"❌ Error creating matrix: {e}")
            return None
    
    def _create_smart_combined_matrix(self, max_designs, sampling_strategy):
        """Create smart combined matrix that balances different study types."""
        all_dataframes = []
        total_designs = 0
        
        # Get all study data
        study_data = [(name, info['data']) for name, info in self.processed_data.items()]
        study_data.sort(key=lambda x: len(x[1]))  # Sort by size
        
        # Calculate how to distribute designs across studies
        total_available = sum(len(data) for _, data in study_data)
        
        if total_available <= max_designs:
            # Use all data
            for study_name, data in study_data:
                all_dataframes.append(data)
                total_designs += len(data)
        else:
            # Intelligently sample from each study
            remaining_budget = max_designs
            
            for i, (study_name, data) in enumerate(study_data):
                study_size = len(data)
                remaining_studies = len(study_data) - i
                
                # Allocate designs fairly, with preference for smaller studies
                if remaining_studies > 0:
                    if study_size <= remaining_budget // remaining_studies * 2:
                        # Small study - take all
                        sample_size = study_size
                    else:
                        # Large study - take fair share
                        sample_size = max(remaining_budget // remaining_studies, 
                                        min(1000, remaining_budget))  # At least 1000 or remaining
                else:
                    sample_size = remaining_budget
                
                sample_size = min(sample_size, study_size, remaining_budget)
                
                if sample_size > 0:
                    if sampling_strategy == "random":
                        sampled_data = data.sample(n=sample_size, random_state=42)
                    elif sampling_strategy == "stratified":
                        sampled_data = self._stratified_sample(data, sample_size)
                    else:  # systematic
                        step = max(1, study_size // sample_size)
                        sampled_data = data.iloc[::step][:sample_size]
                    
                    all_dataframes.append(sampled_data)
                    total_designs += len(sampled_data)
                    remaining_budget -= len(sampled_data)
                    
                    print(f"   {study_name}: {len(sampled_data):,}/{study_size:,} designs ({len(sampled_data)/study_size:.1%})")
                
                if remaining_budget <= 0:
                    break
        
        if all_dataframes:
            master_df = pd.concat(all_dataframes, ignore_index=True)
            master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            print(f"✅ Smart combined matrix created: {len(master_df):,} designs")
            return master_df
        else:
            return None
    
    def _create_complete_combined_matrix(self):
        """Create complete combined matrix with all processed data."""
        all_dataframes = []
        
        for study_name, study_info in self.processed_data.items():
            all_dataframes.append(study_info['data'])
            print(f"   Including {study_name}: {len(study_info['data']):,} designs")
        
        if all_dataframes:
            master_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"✅ Complete combined matrix created: {len(master_df):,} designs")
            return master_df
        else:
            return None
    
    def _create_size_limited_matrix(self, max_designs, sampling_strategy):
        """Create size-limited matrix using specified sampling strategy."""
        # Get all data
        all_data = []
        for study_name, study_info in self.processed_data.items():
            all_data.append(study_info['data'])
        
        if not all_data:
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        if len(combined_df) <= max_designs:
            return combined_df
        
        # Sample down to size limit
        if sampling_strategy == "random":
            sampled_df = combined_df.sample(n=max_designs, random_state=42)
        elif sampling_strategy == "stratified":
            sampled_df = self._stratified_sample(combined_df, max_designs)
        else:  # systematic
            step = len(combined_df) // max_designs
            sampled_df = combined_df.iloc[::step][:max_designs]
        
        print(f"✅ Size-limited matrix created: {len(sampled_df):,}/{len(combined_df):,} designs")
        return sampled_df
    
    def _stratified_sample(self, df, sample_size):
        """Perform stratified sampling based on study type or CBUSH involvement."""
        try:
            # Try stratifying by study type first
            if 'study_type' in df.columns:
                strata = df['study_type']
            elif 'study_name' in df.columns:
                strata = df['study_name']
            else:
                # Fall back to random sampling
                return df.sample(n=sample_size, random_state=42)
            
            # Calculate samples per stratum
            strata_counts = strata.value_counts()
            strata_proportions = strata_counts / len(df)
            
            sampled_dfs = []
            remaining_samples = sample_size
            
            for stratum, proportion in strata_proportions.items():
                stratum_data = df[strata == stratum]
                stratum_sample_size = int(proportion * sample_size)
                stratum_sample_size = min(stratum_sample_size, len(stratum_data), remaining_samples)
                
                if stratum_sample_size > 0:
                    sampled_dfs.append(stratum_data.sample(n=stratum_sample_size, random_state=42))
                    remaining_samples -= stratum_sample_size
            
            if sampled_dfs:
                return pd.concat(sampled_dfs, ignore_index=True)
            else:
                return df.sample(n=sample_size, random_state=42)
                
        except Exception:
            # Fall back to random sampling if stratified fails
            return df.sample(n=sample_size, random_state=42)
    
    def pause_processing(self):
        """Pause current processing operation."""
        self.is_paused = True
        print("⏸️ Processing paused")
    
    def resume_processing(self):
        """Resume processing operation."""
        self.is_paused = False
        print("▶️ Processing resumed")
    
    def stop_processing(self):
        """Stop current processing operation."""
        self.should_stop = True
        self.is_paused = False
        print("⏹️ Processing stop requested")
    
    def reset_processing_state(self):
        """Reset processing control state."""
        self.is_paused = False
        self.should_stop = False
        self.current_operation = None
    
    def get_memory_usage(self):
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Calculate memory used by processed data
            data_memory_mb = sum(info['memory_mb'] for info in self.processed_data.values())
            
            return {
                'total_memory_mb': memory_mb,
                'data_memory_mb': data_memory_mb,
                'system_memory_mb': memory_mb - data_memory_mb,
                'memory_limit_mb': self.max_memory_gb * 1024
            }
        except Exception:
            return {'total_memory_mb': 0, 'data_memory_mb': 0, 'system_memory_mb': 0, 'memory_limit_mb': self.max_memory_gb * 1024}
    
    def export_processed_data(self, output_directory, export_format='csv'):
        """Export all processed data to files."""
        os.makedirs(output_directory, exist_ok=True)
        
        for study_name, study_info in self.processed_data.items():
            if export_format == 'csv':
                filename = f"{study_name}_processed.csv"
                filepath = os.path.join(output_directory, filename)
                study_info['data'].to_csv(filepath, index=False)
                print(f"✅ Exported {study_name} to {filepath}")
            elif export_format == 'parquet':
                filename = f"{study_name}_processed.parquet"
                filepath = os.path.join(output_directory, filename)
                study_info['data'].to_parquet(filepath, index=False)
                print(f"✅ Exported {study_name} to {filepath}")
        
        print(f"📁 All processed data exported to {output_directory}")


def main():
    """Example usage of FlexibleCBUSHProcessor."""
    print("Flexible CBUSH Processor ready for massive dataset processing!")
    print("Features:")
    print("  ✅ Flexible HEEDS directory scanning")
    print("  ✅ Auto-detection of study types and patterns")
    print("  ✅ Failed design handling and reporting")
    print("  ✅ Massive dataset support (11K-111K designs)")
    print("  ✅ Memory management and batch processing")
    print("  ✅ Intelligent matrix creation strategies")


if __name__ == "__main__":
    main()
