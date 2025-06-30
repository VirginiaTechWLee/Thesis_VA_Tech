import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

class MultiCBUSHProcessor:
    """
    Enhanced processor for multi-CBUSH bolt anomaly detection studies.
    Handles multiple study folders with individual POST_0 directories.
    """
    
    def __init__(self):
        """Initialize the multi-CBUSH processor."""
        # Study folder paths (to be set by user)
        self.study_paths = {}
        
        # Study type definitions
        self.study_types = {
            'single': 'Single-CBUSH Sweeps',
            'adjacent': 'Adjacent Bolt Pairs', 
            'multibolt': 'Multi-Bolt Distributed'
        }
        
        # Study ID mappings
        self.study_mappings = {
            # Single-CBUSH (Study 1)
            'S1A': {'type': 'single', 'cbush': [2], 'name': 'SINGLE_CB2', 'study_folder': 'Study1'},
            'S1B': {'type': 'single', 'cbush': [3], 'name': 'SINGLE_CB3', 'study_folder': 'Study1'},
            'S1C': {'type': 'single', 'cbush': [4], 'name': 'SINGLE_CB4', 'study_folder': 'Study1'},
            'S1D': {'type': 'single', 'cbush': [5], 'name': 'SINGLE_CB5', 'study_folder': 'Study1'},
            'S1E': {'type': 'single', 'cbush': [6], 'name': 'SINGLE_CB6', 'study_folder': 'Study1'},
            'S1F': {'type': 'single', 'cbush': [7], 'name': 'SINGLE_CB7', 'study_folder': 'Study1'},
            'S1G': {'type': 'single', 'cbush': [8], 'name': 'SINGLE_CB8', 'study_folder': 'Study1'},
            'S1H': {'type': 'single', 'cbush': [9], 'name': 'SINGLE_CB9', 'study_folder': 'Study1'},
            'S1I': {'type': 'single', 'cbush': [10], 'name': 'SINGLE_CB10', 'study_folder': 'Study1'},
            
            # Adjacent pairs (Study 2)
            'S2A': {'type': 'adjacent', 'cbush': [2, 3], 'name': 'ADJ_CB2-3', 'study_folder': 'Study2'},
            'S2B': {'type': 'adjacent', 'cbush': [3, 4], 'name': 'ADJ_CB3-4', 'study_folder': 'Study2'},
            'S2C': {'type': 'adjacent', 'cbush': [4, 5], 'name': 'ADJ_CB4-5', 'study_folder': 'Study2'},
            'S2D': {'type': 'adjacent', 'cbush': [5, 6], 'name': 'ADJ_CB5-6', 'study_folder': 'Study2'},
            'S2E': {'type': 'adjacent', 'cbush': [6, 7], 'name': 'ADJ_CB6-7', 'study_folder': 'Study2'},
            'S2F': {'type': 'adjacent', 'cbush': [7, 8], 'name': 'ADJ_CB7-8', 'study_folder': 'Study2'},
            'S2G': {'type': 'adjacent', 'cbush': [8, 9], 'name': 'ADJ_CB8-9', 'study_folder': 'Study2'},
            'S2H': {'type': 'adjacent', 'cbush': [9, 10], 'name': 'ADJ_CB9-10', 'study_folder': 'Study2'},
            
            # Multi-bolt distributed (Study 3)
            'S3A': {'type': 'multibolt', 'cbush': [2, 5, 8], 'name': 'DIST_CB2-5-8', 'study_folder': 'Study3'},
            'S3B': {'type': 'multibolt', 'cbush': [3, 6, 9], 'name': 'DIST_CB3-6-9', 'study_folder': 'Study3'},
            'S3C': {'type': 'multibolt', 'cbush': [4, 7, 10], 'name': 'DIST_CB4-7-10', 'study_folder': 'Study3'}
        }
        
        self.processed_data = {}
        
    def set_study_path(self, study_folder, path):
        """Set the path for a specific study folder."""
        self.study_paths[study_folder] = path
        print(f"✅ Set {study_folder} path: {path}")
    
    def get_required_study_folders(self):
        """Get list of required study folders."""
        return ['Study1', 'Study2', 'Study3']
    
    def validate_study_paths(self):
        """Validate that all required study paths are set and exist."""
        required_folders = self.get_required_study_folders()
        missing_folders = []
        invalid_paths = []
        
        for folder in required_folders:
            if folder not in self.study_paths:
                missing_folders.append(folder)
            else:
                path = self.study_paths[folder]
                if not os.path.exists(path):
                    invalid_paths.append(f"{folder}: {path}")
                else:
                    post_0_path = os.path.join(path, "POST_0")
                    if not os.path.exists(post_0_path):
                        invalid_paths.append(f"{folder}: POST_0 not found in {path}")
        
        if missing_folders:
            raise ValueError(f"Missing study paths: {missing_folders}")
        if invalid_paths:
            raise ValueError(f"Invalid paths: {invalid_paths}")
        
        return True
    
    def create_study_labels(self, study_id, design_data):
        """Create multi-label classification labels for different study types."""
        study_info = self.study_mappings.get(study_id, {})
        study_type = study_info.get('type', 'unknown')
        cbush_list = study_info.get('cbush', [])
        
        labels = {
            'study_id': study_id,
            'study_type': study_type,
            'study_name': study_info.get('name', 'UNKNOWN'),
            'cbush_count': len(cbush_list),
            'cbush_list': cbush_list
        }
        
        # Create binary labels for each CBUSH (for multi-label classification)
        for cbush_num in range(2, 11):  # CBUSH 2-10
            labels[f'cbush_{cbush_num}_loose'] = 1 if cbush_num in cbush_list else 0
        
        # Add traditional single-CBUSH compatibility
        if len(cbush_list) == 1:
            labels['varied_cbush'] = cbush_list[0]
        else:
            labels['varied_cbush'] = -1  # Multi-CBUSH indicator
        
        return labels
    
    def process_design_results(self, design_id, study_id):
        """Process individual design results from appropriate study folder."""
        try:
            study_info = self.study_mappings.get(study_id, {})
            study_folder = study_info.get('study_folder', 'Study1')
            
            if study_folder not in self.study_paths:
                raise ValueError(f"Study path not set for {study_folder}")
            
            base_path = self.study_paths[study_folder]
            
            # Import the original extractor
            from feature_matrix_builder import BoltAnomalyFeatureExtractor
            extractor = BoltAnomalyFeatureExtractor(base_path)
            
            # Process single design
            design_folder = f"Design{design_id}"
            accel_file = f"{base_path}/POST_0/{design_folder}/Analysis_1/acceleration_results.csv"
            disp_file = f"{base_path}/POST_0/{design_folder}/Analysis_1/displacement_results.csv"
            
            # Extract features
            features = extractor.extract_features_from_files(accel_file, disp_file, design_id)
            
            if features:
                # Add study-specific labels
                study_labels = self.create_study_labels(study_id, features)
                features.update(study_labels)
                
            return features
            
        except Exception as e:
            print(f"Error processing Design {design_id} for study {study_id}: {e}")
            return None
    
    def get_design_range_for_study(self, study_id):
        """Get design range based on study type and data availability."""
        study_info = self.study_mappings.get(study_id, {})
        study_type = study_info.get('type', 'single')
        
        if study_type == 'single':
            return range(1, 65)  # 64 designs for single-CBUSH (1-64)
        elif study_type == 'adjacent':
            return range(1, 17)  # 16 designs for adjacent pairs (1-16)
        elif study_type == 'multibolt':
            return range(1, 65)  # 64 designs for multi-bolt (1-64)
        else:
            return range(1, 17)  # Default fallback
    
    def process_study_batch(self, study_id, design_range=None, batch_size=25):
        """Process a batch of designs for a specific study."""
        print(f"\n=== PROCESSING STUDY {study_id} ===")
        
        study_info = self.study_mappings.get(study_id)
        if not study_info:
            print(f"❌ Unknown study ID: {study_id}")
            return None
        
        # Validate study path is available
        study_folder = study_info.get('study_folder', 'Study1')
        if study_folder not in self.study_paths:
            print(f"❌ Study path not set for {study_folder}")
            return None
        
        print(f"Study: {study_info['name']}")
        print(f"Type: {study_info['type']}")
        print(f"CBUSHes: {study_info['cbush']}")
        print(f"Folder: {self.study_paths[study_folder]}")
        
        # Determine design range
        if design_range is None:
            design_range = self.get_design_range_for_study(study_id)
        
        features_list = []
        start_time = time.time()
        error_count = 0
        
        for i, design_idx in enumerate(design_range):
            if (i + 1) % 10 == 0:
                print(f"  Processing design {design_idx} ({i+1}/{len(design_range)})")
            
            # Process design
            features = self.process_design_results(design_idx, study_id)
            if features:
                features_list.append(features)
            else:
                error_count += 1
        
        if features_list:
            # Convert to DataFrame
            study_df = pd.DataFrame(features_list)
            
            # Store processed data
            self.processed_data[study_id] = study_df
            
            elapsed = time.time() - start_time
            print(f"✅ Study {study_id} completed in {elapsed:.1f} seconds")
            print(f"   Processed: {len(study_df)} designs")
            if error_count > 0:
                print(f"   Errors: {error_count} designs failed")
            
            return study_df
        else:
            print(f"❌ No valid data processed for study {study_id}")
            print(f"   All {len(design_range)} designs failed processing")
            return None
    
    def create_master_matrix(self, study_ids=None, matrix_type="combined"):
        """Create master matrix from selected studies."""
        print(f"\n=== CREATING MASTER MATRIX ({matrix_type.upper()}) ===")
        
        if study_ids is None:
            study_ids = list(self.processed_data.keys())
        
        # Filter by matrix type if specified
        if matrix_type == "single":
            study_ids = [sid for sid in study_ids if sid.startswith('S1')]
        elif matrix_type == "adjacent":
            study_ids = [sid for sid in study_ids if sid.startswith('S2')]
        elif matrix_type == "multibolt":
            study_ids = [sid for sid in study_ids if sid.startswith('S3')]
        # "combined" includes all study_ids as-is
        
        print(f"Selected studies: {study_ids}")
        
        dataframes = []
        total_designs = 0
        study_breakdown = {}
        
        for study_id in study_ids:
            if study_id in self.processed_data:
                df = self.processed_data[study_id]
                dataframes.append(df)
                total_designs += len(df)
                study_breakdown[study_id] = len(df)
                print(f"✅ {study_id}: {len(df)} designs")
            else:
                print(f"❌ {study_id}: Not processed yet")
        
        if not dataframes:
            print("❌ No data available for master matrix")
            return None
        
        # Combine all dataframes
        master_df = pd.concat(dataframes, ignore_index=True)
        
        # Sort by study_id and design_id for organization
        master_df = master_df.sort_values(['study_id', 'design_id']).reset_index(drop=True)
        
        print(f"\n✅ Master matrix created:")
        print(f"   Total designs: {len(master_df)}")
        print(f"   Studies included: {len(study_ids)}")
        print(f"   Columns: {len(master_df.columns)}")
        print(f"   Study breakdown:")
        for study_id, count in study_breakdown.items():
            study_info = self.study_mappings.get(study_id, {})
            print(f"     {study_id} ({study_info.get('name', 'Unknown')}): {count} designs")
        
        # Multi-label summary
        cbush_involvement = {}
        for cbush_num in range(2, 11):
            col_name = f'cbush_{cbush_num}_loose'
            if col_name in master_df.columns:
                cbush_involvement[cbush_num] = master_df[col_name].sum()
        
        print(f"   CBUSH involvement (total designs per CBUSH):")
        for cbush_num, count in cbush_involvement.items():
            print(f"     CBUSH {cbush_num}: {count} designs")
        
        return master_df
    
    def export_master_matrix(self, study_ids=None, matrix_type="combined", output_file=None):
        """Export master matrix to CSV file."""
        master_df = self.create_master_matrix(study_ids, matrix_type)
        
        if master_df is not None and output_file:
            master_df.to_csv(output_file, index=False)
            print(f"✅ Exported to {output_file}")
            return True
        return False
    
    def get_available_studies(self):
        """Get list of available studies by type."""
        studies_by_type = {
            'single': [],
            'adjacent': [],
            'multibolt': []
        }
        
        for study_id, info in self.study_mappings.items():
            study_type = info['type']
            studies_by_type[study_type].append({
                'id': study_id,
                'name': info['name'],
                'cbush': info['cbush'],
                'processed': study_id in self.processed_data,
                'study_folder': info['study_folder']
            })
        
        return studies_by_type
    
    def get_study_summary(self):
        """Get comprehensive summary of all processed studies."""
        summary = {
            'total_studies': len(self.processed_data),
            'total_designs': sum(len(df) for df in self.processed_data.values()),
            'by_type': {},
            'study_paths': self.study_paths.copy()
        }
        
        for study_type in ['single', 'adjacent', 'multibolt']:
            type_studies = [sid for sid, info in self.study_mappings.items() 
                           if info['type'] == study_type and sid in self.processed_data]
            type_designs = sum(len(self.processed_data[sid]) for sid in type_studies)
            
            summary['by_type'][study_type] = {
                'studies': len(type_studies),
                'designs': type_designs,
                'study_ids': type_studies
            }
        
        return summary
    
    def create_spatial_analysis(self, study_ids=None, save_plots=True):
        """Create spatial response pattern analysis for studies."""
        print(f"\n=== CREATING SPATIAL ANALYSIS ===")
        
        if study_ids is None:
            study_ids = list(self.processed_data.keys())
        
        # Get feature data
        master_df = self.create_master_matrix(study_ids)
        if master_df is None:
            print("❌ No data available for spatial analysis")
            return None
        
        # Import plotting libraries
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get T1_Area features (primary spatial signature)
        t1_area_cols = [col for col in master_df.columns if 'ACCE_T1_Area' in col]
        if not t1_area_cols:
            print("❌ No T1_Area features found for spatial analysis")
            return None
        
        # Node positions along beam (mm)
        node_positions = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
        
        plots_created = []
        
        # Plot 1: Individual study patterns
        for study_type in ['single', 'adjacent', 'multibolt']:
            type_studies = [sid for sid in study_ids 
                           if self.study_mappings.get(sid, {}).get('type') == study_type]
            
            if not type_studies:
                continue
                
            fig, ax = plt.subplots(figsize=(14, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, len(type_studies)))
            
            for i, study_id in enumerate(type_studies):
                study_data = master_df[master_df['study_id'] == study_id]
                if len(study_data) == 0:
                    continue
                
                # Get extreme loose case for visualization
                extreme_case = study_data.iloc[0]  # First design (usually most extreme)
                response = extreme_case[t1_area_cols].values
                
                study_info = self.study_mappings[study_id]
                label = f"{study_id} (CBUSH {'-'.join(map(str, study_info['cbush']))})"
                
                ax.plot(node_positions, response, 'o-', label=label, 
                       color=colors[i], linewidth=2, markersize=6)
            
            ax.set_xlabel('Position along beam (mm)', fontsize=12)
            ax.set_ylabel('T1_Area Response (Delta from baseline)', fontsize=12)
            ax.set_title(f'Spatial Response Patterns - {study_type.title()} Studies\n(Extreme loose condition)', fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                filename = f'spatial_analysis_{study_type}_studies.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plots_created.append(filename)
                print(f"✅ Saved: {filename}")
            
            plt.close()
        
        # Plot 2: Combined all studies
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Group by unique CBUSH combinations
        cbush_combinations = master_df['cbush_list'].apply(lambda x: tuple(sorted(x)) if isinstance(x, list) else (x,)).unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(cbush_combinations)))
        
        for i, cbush_combo in enumerate(cbush_combinations):
            # Find studies with this CBUSH combination
            matching_data = master_df[master_df['cbush_list'].apply(
                lambda x: tuple(sorted(x)) if isinstance(x, list) else (x,)) == cbush_combo]
            
            if len(matching_data) == 0:
                continue
            
            # Use first design as representative
            response = matching_data.iloc[0][t1_area_cols].values
            label = f"CBUSH {'-'.join(map(str, cbush_combo))}"
            
            ax.plot(node_positions, response, 'o-', label=label, 
                   color=colors[i], linewidth=2, markersize=6)
        
        ax.set_xlabel('Position along beam (mm)', fontsize=12)
        ax.set_ylabel('T1_Area Response (Delta from baseline)', fontsize=12)
        ax.set_title(f'Combined Spatial Response Patterns - All Studies\n(Total: {len(master_df)} designs)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            filename = 'spatial_analysis_combined_all.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plots_created.append(filename)
            print(f"✅ Saved: {filename}")
        
        plt.close()
        
        print(f"✅ Spatial analysis complete. Created {len(plots_created)} plots.")
        return plots_created


def main():
    """Example usage of MultiCBUSHProcessor."""
    print("Enhanced MultiCBUSHProcessor ready for GUI integration!")
    print("Supports multiple study folders with individual POST_0 directories")
    print("Features:")
    print("  ✅ Multiple study folder support")
    print("  ✅ Multi-label classification labels")
    print("  ✅ Combined matrix creation (896 designs)")
    print("  ✅ Spatial analysis and visualization")
    print("  ✅ Comprehensive error handling")


if __name__ == "__main__":
    main()