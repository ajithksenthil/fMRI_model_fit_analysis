import numpy as np
import nibabel as nib
from scipy import stats
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class FSLModelComparison:
    def __init__(self, subject_id='01', run_id='01'):
        """
        Initialize FSL Model Comparison with subject and run information.
        
        Parameters:
        -----------
        subject_id : str
            Subject ID (e.g., '01')
        run_id : str
            Run ID (e.g., '01')
        """
        self.subject_id = subject_id
        self.run_id = run_id
        self.results = {}
        
        # Get the current working directory
        self.base_dir = Path.cwd()
        
        # Define the model types and their corresponding folder patterns
        self.model_folders = {
            'linear': f'sub-{subject_id}_run-{run_id}_linear_analysis++.feat',
            'quadratic': f'sub-{subject_id}_run-{run_id}_quadratic_analysis++.feat',
            'cubic': f'sub-{subject_id}_run-{run_id}_cubic_analysis++.feat',
            'exponential': f'sub-{subject_id}_run-{run_id}_exponential_analysis++.feat'
        }
        
        # Validate folder existence
        self.validate_folders()
    
    def validate_folders(self):
        """
        Check if the required FEAT folders exist and contain necessary files.
        """
        for model, folder in self.model_folders.items():
            feat_dir = self.base_dir / folder
            if not feat_dir.exists():
                print(f"Warning: {folder} not found")
                continue
                
            # Check for required files
            required_files = [
                'stats/res4d.nii.gz',
                'design.mat',
                'mask.nii.gz'
            ]
            
            missing_files = [f for f in required_files 
                           if not (feat_dir / Path(f)).exists()]
            
            if missing_files:
                print(f"Warning: Missing files in {folder}:")
                for file in missing_files:
                    print(f"  - {file}")

    def load_feat_data(self, model_type):
        """
        Load necessary files from FEAT directory.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('linear', 'quadratic', 'cubic', or 'exponential')
        
        Returns:
        --------
        dict : Dictionary containing loaded data
        """
        feat_dir = self.base_dir / self.model_folders[model_type]
        
        try:
            # Load residuals
            res4d = nib.load(feat_dir / 'stats' / 'res4d.nii.gz')
            residuals = res4d.get_fdata()
            
            # Load design matrix
            design = np.loadtxt(feat_dir / 'design.mat', skiprows=5)
            
            # Load mask
            mask = nib.load(feat_dir / 'mask.nii.gz').get_fdata()
            
            # Load parameter estimates
            copes = []
            cope_files = sorted((feat_dir / 'stats').glob('cope*.nii.gz'))
            for cope_file in cope_files:
                cope = nib.load(cope_file).get_fdata()
                copes.append(cope)
            
            # Load Z-statistics
            zstats = []
            zstat_files = sorted((feat_dir / 'stats').glob('zstat*.nii.gz'))
            for zstat_file in zstat_files:
                zstat = nib.load(zstat_file).get_fdata()
                zstats.append(zstat)
            
            return {
                'residuals': residuals,
                'design': design,
                'mask': mask,
                'copes': np.array(copes),
                'zstats': np.array(zstats)
            }
            
        except Exception as e:
            print(f"Error loading {model_type} model data: {str(e)}")
            raise

    def calculate_log_likelihood(self, residuals, mask):
        """
        Calculate log-likelihood for the model with added diagnostics.
        
        Parameters:
        -----------
        residuals : numpy.ndarray
            4D array of model residuals (x x y x z x time)
        mask : numpy.ndarray
            3D brain mask (x x y x z)
        
        Returns:
        --------
        float : Log-likelihood value
        """
        # Print shapes for debugging
        print(f"Original residuals shape: {residuals.shape}")
        print(f"Original mask shape: {mask.shape}")
        
        # Transpose residuals to get time as first dimension
        residuals = np.transpose(residuals, (3, 0, 1, 2))
        print(f"Transposed residuals shape: {residuals.shape}")
        
        # Get the time dimension
        n_timepoints = residuals.shape[0]
        
        # Reshape residuals to 2D (time × voxels)
        reshaped_residuals = residuals.reshape(n_timepoints, -1)
        
        # Reshape mask to 1D
        reshaped_mask = mask.reshape(-1) > 0
        
        print(f"Reshaped residuals shape: {reshaped_residuals.shape}")
        print(f"Reshaped mask shape: {reshaped_mask.shape}")
        
        # Apply mask
        masked_residuals = reshaped_residuals[:, reshaped_mask]
        print(f"Final masked residuals shape: {masked_residuals.shape}")
        
        # Enhanced Diagnostics Section
        print("\n=== Enhanced Residuals Diagnostics ===")
        
        # 1. Scale and Distribution Checks
        print("\n1. Scale and Distribution Analysis:")
        mean_res = np.mean(masked_residuals)
        std_res = np.std(masked_residuals)
        print(f"Mean of residuals: {mean_res:.4f}")
        print(f"Standard deviation: {std_res:.4f}")
        print(f"Mean absolute residual: {np.mean(np.abs(masked_residuals)):.4f}")
        
        # Check if scale is reasonable for fMRI data (typically between -100 and 100)
        if np.abs(mean_res) > 100:
            print("WARNING: Mean residual magnitude seems large for fMRI data")
        
        # 2. Variability Analysis
        print("\n2. Variability Analysis:")
        temporal_std = np.std(masked_residuals, axis=0)
        spatial_std = np.std(masked_residuals, axis=1)
        print(f"Median temporal standard deviation: {np.median(temporal_std):.4f}")
        print(f"Median spatial standard deviation: {np.median(spatial_std):.4f}")
        print(f"Temporal/Spatial variability ratio: {np.median(temporal_std)/np.median(spatial_std):.4f}")
        
        # 3. Outlier Detection
        print("\n3. Outlier Analysis:")
        q1, q3 = np.percentile(masked_residuals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        outliers = np.logical_or(masked_residuals < lower_bound, masked_residuals > upper_bound)
        outlier_percentage = (np.sum(outliers) / outliers.size) * 100
        
        print(f"Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
        print(f"Outlier bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"Percentage of outliers: {outlier_percentage:.2f}%")
        print(f"Min residual: {np.min(masked_residuals):.4f}")
        print(f"Max residual: {np.max(masked_residuals):.4f}")
        
        # 4. RSS Analysis
        print("\n4. RSS Analysis:")
        rss_per_voxel = np.sum(masked_residuals ** 2, axis=0)
        rss_per_timepoint = np.sum(masked_residuals ** 2, axis=1)
        
        print(f"Mean RSS per voxel: {np.mean(rss_per_voxel):.4f}")
        print(f"Median RSS per voxel: {np.median(rss_per_voxel):.4f}")
        print(f"Mean RSS per timepoint: {np.mean(rss_per_timepoint):.4f}")
        print(f"Median RSS per timepoint: {np.median(rss_per_timepoint):.4f}")
        
        # Check if RSS values are within typical fMRI range
        typical_range = (1e3, 1e6)
        if not typical_range[0] <= np.median(rss_per_voxel) <= typical_range[1]:
            print("WARNING: RSS values may be outside typical fMRI range")
        
        # 5. Normality Assessment
        print("\n5. Normality Check:")
        random_sample = np.random.choice(masked_residuals.flatten(), size=1000)
        _, normality_p = stats.normaltest(random_sample)
        print(f"Normality test p-value: {normality_p:.4e}")
        
        print("\n" + "="*50)
        
        # Calculate RSS with stability checks
        rss = np.sum(masked_residuals ** 2, axis=0)
        
        # Handle numerical stability
        eps = np.finfo(float).eps
        rss = np.maximum(rss, eps)
        
        # Calculate log-likelihood
        log_likelihood = -0.5 * n_timepoints * (np.log(2 * np.pi) + 
                                            np.log(rss / n_timepoints) + 1)
        
        return np.sum(log_likelihood)
    
    def calculate_information_criteria(self, log_likelihood, n_params, n_timepoints):
        """
        Calculate AIC and BIC.
        """
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n_timepoints) * n_params - 2 * log_likelihood
        return aic, bic
    
    def compare_models(self):
        """
        Perform model comparison analysis.
        
        Returns:
        --------
        dict : Complete results of model comparison
        """
        available_models = []
        
        # Analyze each available model
        for model_type in self.model_folders:
            feat_dir = self.base_dir / self.model_folders[model_type]
            if not feat_dir.exists():
                continue
                
            available_models.append(model_type)
            data = self.load_feat_data(model_type)
            
            # Calculate metrics
            n_timepoints = data['residuals'].shape[0]
            n_params = data['design'].shape[1]
            log_like = self.calculate_log_likelihood(data['residuals'], data['mask'])
            aic, bic = self.calculate_information_criteria(log_like, n_params, n_timepoints)
            
            # Store results
            self.results[model_type] = {
                'log_likelihood': log_like,
                'n_params': n_params,
                'aic': aic,
                'bic': bic,
                'data': data
            }
        
        # Perform likelihood ratio tests for available nested models
        nested_pairs = [
            ('linear', 'quadratic'),
            ('quadratic', 'cubic'),
            ('linear', 'exponential')
        ]
        
        lr_results = {}
        for simple, complex in nested_pairs:
            if simple in available_models and complex in available_models:
                lr_stat = 2 * (self.results[complex]['log_likelihood'] - 
                              self.results[simple]['log_likelihood'])
                df = (self.results[complex]['n_params'] - 
                      self.results[simple]['n_params'])
                p_value = 1 - stats.chi2.cdf(lr_stat, df)
                
                lr_results[f'{simple}_vs_{complex}'] = {
                    'lr_statistic': lr_stat,
                    'p_value': p_value,
                    'df': df
                }
        
        self.results['lr_tests'] = lr_results
        return self.results

    def plot_model_comparison(self, save_path=None):
        """
        Create visualization of model comparison results.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot information criteria
        models = [m for m in self.results.keys() if m != 'lr_tests']
        
        aic_values = [self.results[m]['aic'] for m in models]
        bic_values = [self.results[m]['bic'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, aic_values, width, label='AIC')
        ax1.bar(x + width/2, bic_values, width, label='BIC')
        ax1.set_ylabel('Information Criterion Value')
        ax1.set_title('Model Comparison: Information Criteria')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        
        # Plot log-likelihoods
        log_likes = [self.results[m]['log_likelihood'] for m in models]
        ax2.bar(models, log_likes)
        ax2.set_ylabel('Log-Likelihood')
        ax2.set_title('Model Comparison: Log-Likelihood')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig

# Example usage
if __name__ == "__main__":
    # Initialize with subject and run IDs
    comparator = FSLModelComparison(subject_id='01', run_id='01')
    
    # Run comparison
    results = comparator.compare_models()
    
    # Create visualization
    fig = comparator.plot_model_comparison(save_path='model_comparison_results.png')
    
    # Print summary
    print("\nModel Comparison Results:")
    print("-" * 50)
    
    for model in results:
        if model != 'lr_tests':
            print(f"\n{model.capitalize()} Model:")
            print(f"AIC: {results[model]['aic']:.2f}")
            print(f"BIC: {results[model]['bic']:.2f}")
            print(f"Log-likelihood: {results[model]['log_likelihood']:.2f}")
            print(f"Number of parameters: {results[model]['n_params']}")
    
    print("\nLikelihood Ratio Tests:")
    print("-" * 50)
    for test, stats in results['lr_tests'].items():
        print(f"\n{test}:")
        print(f"LR statistic: {stats['lr_statistic']:.2f}")
        print(f"p-value: {stats['p_value']:.4f}")
        print(f"degrees of freedom: {stats['df']}")