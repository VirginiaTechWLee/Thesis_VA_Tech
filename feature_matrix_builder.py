import pandas as pd
import numpy as np
import os
import glob
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BoltAnomalyFeatureExtractor:
    """
    Efficient feature extractor for bolt anomaly detection from NASTRAN random vibration results.
    Designed to handle large datasets with work software limitations.
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
        
        # Define CBUSH mapping (Design number to which CBUSH was varied)
        self.cbush_mapping = self._create_cbush_mapping()
        
    def _create_cbush_mapping(self):
        """
        Create mapping from design number to bolt parameters.
        Based on Excel analysis: 9 sets × 64 designs = 576 total designs
        Each set varies one CBUSH through all K4×K5×K6 combinations (4×4×4=64)
        """
        mapping = {}
        
        # Value to stiffness mapping
        value_to_stiffness = {
            1: {'exp': 4, 'value': 1e4},   # Very loose
            4: {'exp': 7, 'value': 1e7},   # Loose
            7: {'exp': 10, 'value': 1e10}, # Medium  
            10: {'exp': 13, 'value': 1e13} # Very tight
        }
        
        # Set to CBUSH mapping
        set_to_cbush = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
        
        for design_id in range(1, 577):  # Designs 1-576
            # Determine which set (1-9) and sweep (1-64) within that set
            set_num = math.ceil(design_id / 64)
            sweep_num = ((design_id - 1) % 64) + 1
            
            # Get the varied CBUSH for this set
            varied_cbush = set_to_cbush[set_num]
            
            # Decode sweep number to K4, K5, K6 values (4×4×4 combinations)
            sweep_index = sweep_num - 1  # 0-based
            stiffness_values = [1, 4, 7, 10]  # The 4 varied values
            
            k6_index = sweep_index % 4
            k5_index = (sweep_index // 4) % 4  
            k4_index = sweep_index // 16
            
            k4_val = stiffness_values[k4_index]
            k5_val = stiffness_values[k5_index] 
            k6_val = stiffness_values[k6_index]
            
            mapping[design_id] = {
                'set_num': set_num,
                'sweep_num': sweep_num,
                'varied_cbush': varied_cbush,
                'k4_value': k4_val,
                'k5_value': k5_val,
                'k6_value': k6_val,
                'k4_stiffness_exp': value_to_stiffness[k4_val]['exp'],
                'k5_stiffness_exp': value_to_stiffness[k5_val]['exp'],
                'k6_stiffness_exp': value_to_stiffness[k6_val]['exp'],
                'k4_stiffness_value': value_to_stiffness[k4_val]['value'],
                'k5_stiffness_value': value_to_stiffness[k5_val]['value'],
                'k6_stiffness_value': value_to_stiffness[k6_val]['value'],
                'is_baseline': False  # All designs have at least one loose bolt
            }
        
        return mapping
    
    def extract_features_from_files(self, accel_delta_file, disp_delta_file, design_id):
        """
        Extract features from acceleration and displacement delta CSV files.
        
        Args:
            accel_delta_file: Path to acceleration_results_delta.csv
            disp_delta_file: Path to displacement_results_delta.csv  
            design_id: Design identifier
            
        Returns:
            Dictionary of features
        """
        try:
            # Load the data
            accel_df = pd.read_csv(accel_delta_file)
            disp_df = pd.read_csv(disp_delta_file)
            
            # Initialize feature dictionary
            features = {'design_id': design_id}
            
            # Add label information if available
            if design_id in self.cbush_mapping:
                bolt_info = self.cbush_mapping[design_id]
                features.update({
                    'set_num': bolt_info['set_num'],
                    'sweep_num': bolt_info['sweep_num'],
                    'varied_cbush': bolt_info['varied_cbush'],
                    'k4_value': bolt_info['k4_value'],
                    'k5_value': bolt_info['k5_value'],
                    'k6_value': bolt_info['k6_value'],
                    'k4_stiffness_exp': bolt_info['k4_stiffness_exp'],
                    'k5_stiffness_exp': bolt_info['k5_stiffness_exp'],
                    'k6_stiffness_exp': bolt_info['k6_stiffness_exp']
                })
            
            # Get node columns (exclude 'Measurement')
            node_columns = [col for col in accel_df.columns if col.startswith('Node_')]
            
            # Extract features for each key measurement
            for measurement in self.key_measurements:
                # Find the row for this measurement
                if measurement.startswith('ACCE_'):
                    data_row = accel_df[accel_df['Measurement'] == measurement]
                elif measurement.startswith('DISP_'):
                    data_row = disp_df[disp_df['Measurement'] == measurement]
                
                if not data_row.empty:
                    # Extract values for each node
                    for node_col in node_columns:
                        feature_name = f"{measurement}_{node_col}"
                        features[feature_name] = data_row.iloc[0][node_col]
                else:
                    # If measurement not found, fill with zeros
                    for node_col in node_columns:
                        feature_name = f"{measurement}_{node_col}"
                        features[feature_name] = 0.0
            
            return features
            
        except Exception as e:
            print(f"Error processing design {design_id}: {e}")
            return None
    
    def process_design_batch(self, design_numbers, progress_callback=None):
        """
        Process a batch of designs efficiently.
        
        Args:
            design_numbers: List of design numbers to process
            progress_callback: Optional function to call with progress updates
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        
        for i, design_num in enumerate(design_numbers):
            if progress_callback:
                progress_callback(i + 1, len(design_numbers), design_num)
            
            # Construct file paths
            design_folder = f"Design{design_num}"
            accel_file = os.path.join(self.base_path, "POST_0", design_folder, 
                                    "Analysis_1", "acceleration_results_delta.csv")
            disp_file = os.path.join(self.base_path, "POST_0", design_folder,
                                   "Analysis_1", "displacement_results_delta.csv")
            
            # Check if files exist
            if os.path.exists(accel_file) and os.path.exists(disp_file):
                features = self.extract_features_from_files(accel_file, disp_file, design_num)
                if features:
                    features_list.append(features)
            else:
                print(f"Warning: Files not found for Design{design_num}")
        
        return features_list
    
    def create_master_matrix(self, design_numbers=None, batch_size=50, output_file='master_feature_matrix.csv'):
        """
        Create master feature matrix by processing designs in batches.
        
        Args:
            design_numbers: List of design numbers to process (default: all 576)
            batch_size: Number of designs to process at once (adjust for memory)
            output_file: Output CSV file name
            
        Returns:
            pandas DataFrame with all features
        """
        if design_numbers is None:
            design_numbers = list(range(1, 577))  # All 576 designs
        
        print(f"Creating master feature matrix for {len(design_numbers)} designs...")
        print(f"Processing in batches of {batch_size}")
        
        all_features = []
        
        # Process in batches
        for i in range(0, len(design_numbers), batch_size):
            batch = design_numbers[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}: Designs {batch[0]}-{batch[-1]}")
            
            def progress_cb(current, total, design_id):
                if current % 10 == 0 or current == total:
                    print(f"  Progress: {current}/{total} - Design {design_id}")
            
            batch_features = self.process_design_batch(batch, progress_cb)
            all_features.extend(batch_features)
            
            # Save intermediate results
            if i > 0 and i % (batch_size * 5) == 0:  # Every 5 batches
                temp_df = pd.DataFrame(all_features)
                temp_file = f"temp_features_batch_{i//batch_size}.csv"
                temp_df.to_csv(temp_file, index=False)
                print(f"  Saved temporary results to {temp_file}")
        
        # Create final DataFrame
        if all_features:
            master_df = pd.DataFrame(all_features)
            master_df = master_df.sort_values('design_id').reset_index(drop=True)
            
            # Save to file
            master_df.to_csv(output_file, index=False)
            print(f"\n✅ Master feature matrix saved to {output_file}")
            print(f"   Shape: {master_df.shape}")
            print(f"   Features per design: {master_df.shape[1] - 4}")  # Exclude ID columns
            
            return master_df
        else:
            print("❌ No features extracted!")
            return None
    
    def get_feature_summary(self, master_df):
        """Generate summary statistics of the feature matrix."""
        print("\n=== FEATURE MATRIX SUMMARY ===")
        print(f"Total designs: {len(master_df)}")
        print(f"Total features: {master_df.shape[1]}")
        
        # CBUSH distribution
        if 'varied_cbush' in master_df.columns:
            cbush_counts = master_df['varied_cbush'].value_counts().sort_index()
            print(f"\nCBUSH distribution:")
            for cbush_id, count in cbush_counts.items():
                print(f"  CBUSH {cbush_id}: {count} designs")
        
        # Set distribution  
        if 'set_num' in master_df.columns:
            set_counts = master_df['set_num'].value_counts().sort_index()
            print(f"\nSet distribution:")
            for set_num, count in set_counts.items():
                print(f"  Set {set_num}: {count} designs")
                
        # Stiffness distribution for K4 (representative)
        if 'k4_stiffness_exp' in master_df.columns:
            stiff_counts = master_df['k4_stiffness_exp'].value_counts().sort_index()
            print(f"\nK4 Stiffness distribution:")
            for exp, count in stiff_counts.items():
                print(f"  1e{exp}: {count} designs")
        
        # Feature statistics
        feature_cols = [col for col in master_df.columns 
                       if col.startswith(('ACCE_', 'DISP_'))]
        print(f"\nFeature statistics:")
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Non-zero features: {(master_df[feature_cols] != 0).sum().sum()}")
        
        # Most important features (T1_Area)
        t1_area_cols = [col for col in feature_cols if 'T1_Area' in col]
        if t1_area_cols:
            t1_stats = master_df[t1_area_cols].abs().mean().sort_values(ascending=False)
            print(f"\nTop 5 most responsive nodes (by T1_Area):")
            for i, (col, val) in enumerate(t1_stats.head().items()):
                node = col.split('_')[-1]
                print(f"  {i+1}. {node}: {val:.0f} average absolute change")


# Usage example for work environment with limited resources
def main_processing_example():
    """
    Example of how to use the feature extractor in a work environment.
    Based on actual parameter sweep: 9 sets × 64 designs = 576 total designs
    """
    # Initialize the extractor
    base_path = r"C:\Users\m33002\Documents\Beam_Model\Square_Beam_scenario_short_rev_Study_1"
    extractor = BoltAnomalyFeatureExtractor(base_path)
    
    # Option 1: Test with first designs from each set (recommended start)
    print("=== OPTION 1: Test with representative designs ===")
    # Take first design from each set to test all CBUSHes
    test_designs = [1, 65, 129, 193, 257, 321, 385, 449, 513]  # One from each set
    test_matrix = extractor.create_master_matrix(
        design_numbers=test_designs, 
        batch_size=5,
        output_file='test_all_cbush.csv'
    )
    
    if test_matrix is not None:
        extractor.get_feature_summary(test_matrix)
    
    # Option 2: Process one complete set (64 designs)
    print("\n=== OPTION 2: Process complete Set_1 (CBUSH 2) ===")
    set1_designs = list(range(1, 65))  # Designs 1-64 (CBUSH 2 varied)
    set1_matrix = extractor.create_master_matrix(
        design_numbers=set1_designs,
        batch_size=20,
        output_file='set1_cbush2_features.csv'
    )
    
    # Option 3: Process by sets for memory management
    print("\n=== OPTION 3: Process all sets individually ===")
    set_ranges = {
        1: (1, 65),      # CBUSH 2
        2: (65, 129),    # CBUSH 3  
        3: (129, 193),   # CBUSH 4
        4: (193, 257),   # CBUSH 5
        5: (257, 321),   # CBUSH 6
        6: (321, 385),   # CBUSH 7
        7: (385, 449),   # CBUSH 8
        8: (449, 513),   # CBUSH 9
        9: (513, 577)    # CBUSH 10
    }
    
    # Process first 3 sets as example
    for set_num in [1, 2, 3]:
        start, end = set_ranges[set_num]
        designs = list(range(start, end))
        print(f"\nProcessing Set_{set_num} (CBUSH {set_num + 1}): Designs {start}-{end-1}")
        
        set_matrix = extractor.create_master_matrix(
            design_numbers=designs,
            batch_size=25,
            output_file=f'set{set_num}_features.csv'
        )
    
    # Option 4: Combine all set files (after processing all sets)
    print("\n=== OPTION 4: Combine all sets (run after processing all) ===")
    print("# Uncomment when all sets are processed:")
    print("# all_files = [f'set{i}_features.csv' for i in range(1, 10)]")
    print("# combined_df = pd.concat([pd.read_csv(f) for f in all_files])")
    print("# combined_df.to_csv('master_feature_matrix_576.csv', index=False)")


if __name__ == "__main__":
    main_processing_example()