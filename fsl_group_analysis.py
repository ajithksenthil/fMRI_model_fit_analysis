import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class FSLGroupModelComparison:
    def __init__(self, subject_ids, run_id='01'):
        """
        Initialize group-level model comparison.
        
        Parameters:
        -----------
        subject_ids : list
            List of subject IDs (e.g., ['01', '02', '03'])
        run_id : str
            Run ID to analyze
        """
        self.subject_ids = subject_ids
        self.run_id = run_id
        self.individual_results = {}
        self.group_results = {}
        
    def run_individual_analyses(self):
        """Run model comparison for each subject."""
        for subject_id in self.subject_ids:
            print(f"\nAnalyzing subject {subject_id}...")
            try:
                # Run individual subject analysis
                comparator = FSLModelComparison(subject_id=subject_id, run_id=self.run_id)
                results = comparator.compare_models()
                
                # Store results
                self.individual_results[subject_id] = results
                
                # Create and save individual plots
                fig = comparator.plot_model_comparison(
                    save_path=f'sub-{subject_id}_model_comparison.png'
                )
                plt.close(fig)
                
            except Exception as e:
                print(f"Error analyzing subject {subject_id}: {str(e)}")
    
    def aggregate_results(self):
        """Aggregate results across subjects."""
        if not self.individual_results:
            raise ValueError("No individual results to aggregate. Run individual analyses first.")
        
        # Initialize containers for group metrics
        model_metrics = {
            'linear': {'aic': [], 'bic': [], 'log_likelihood': [], 'n_params': []},
            'quadratic': {'aic': [], 'bic': [], 'log_likelihood': [], 'n_params': []},
            'cubic': {'aic': [], 'bic': [], 'log_likelihood': [], 'n_params': []},
            'exponential': {'aic': [], 'bic': [], 'log_likelihood': [], 'n_params': []}
        }
        
        # Collect metrics across subjects
        for subject_id, results in self.individual_results.items():
            for model in results:
                if model != 'lr_tests':
                    try:
                        model_metrics[model]['aic'].append(results[model]['aic'])
                        model_metrics[model]['bic'].append(results[model]['bic'])
                        model_metrics[model]['log_likelihood'].append(results[model]['log_likelihood'])
                        model_metrics[model]['n_params'].append(results[model]['n_params'])
                    except KeyError:
                        continue
        
        # Calculate group statistics
        for model in model_metrics:
            if model_metrics[model]['aic']:  # Check if we have data for this model
                self.group_results[model] = {
                    'aic_mean': np.mean(model_metrics[model]['aic']),
                    'aic_std': np.std(model_metrics[model]['aic']),
                    'bic_mean': np.mean(model_metrics[model]['bic']),
                    'bic_std': np.std(model_metrics[model]['bic']),
                    'log_likelihood_mean': np.mean(model_metrics[model]['log_likelihood']),
                    'log_likelihood_std': np.std(model_metrics[model]['log_likelihood']),
                    'n_subjects': len(model_metrics[model]['aic'])
                }
        
        return self.group_results
    
    def plot_group_results(self, save_path=None):
        """Create group-level visualization."""
        if not self.group_results:
            self.aggregate_results()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get available models
        models = list(self.group_results.keys())
        
        # 1. Plot mean AIC with error bars
        aic_means = [self.group_results[m]['aic_mean'] for m in models]
        aic_stds = [self.group_results[m]['aic_std'] for m in models]
        ax1.bar(models, aic_means, yerr=aic_stds, capsize=5)
        ax1.set_title('Group Average AIC')
        ax1.set_ylabel('AIC Value')
        
        # 2. Plot mean BIC with error bars
        bic_means = [self.group_results[m]['bic_mean'] for m in models]
        bic_stds = [self.group_results[m]['bic_std'] for m in models]
        ax2.bar(models, bic_means, yerr=bic_stds, capsize=5)
        ax2.set_title('Group Average BIC')
        ax2.set_ylabel('BIC Value')
        
        # 3. Plot mean log-likelihood with error bars
        ll_means = [self.group_results[m]['log_likelihood_mean'] for m in models]
        ll_stds = [self.group_results[m]['log_likelihood_std'] for m in models]
        ax3.bar(models, ll_means, yerr=ll_stds, capsize=5)
        ax3.set_title('Group Average Log-Likelihood')
        ax3.set_ylabel('Log-Likelihood Value')
        
        # 4. Plot individual values for best metric (AIC)
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        for i, model in enumerate(models):
            subject_values = [self.individual_results[s][model]['aic'] 
                            for s in self.individual_results 
                            if model in self.individual_results[s]]
            subjects = [f'sub-{s}' for s in self.subject_ids 
                       if model in self.individual_results[s]]
            
            ax4.scatter(subjects, subject_values, 
                       label=model, color=model_colors[i], alpha=0.7)
        
        ax4.set_title('Individual Subject AIC Values')
        ax4.set_ylabel('AIC Value')
        ax4.legend()
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def print_summary(self):
        """Print comprehensive summary of group-level results."""
        if not self.group_results:
            self.aggregate_results()
        
        print("\nGroup-Level Model Comparison Results")
        print("=" * 40)
        
        # Print summary for each model
        for model in self.group_results:
            print(f"\n{model.capitalize()} Model (n={self.group_results[model]['n_subjects']}):")
            print("-" * 40)
            print(f"AIC: {self.group_results[model]['aic_mean']:.2f} ± {self.group_results[model]['aic_std']:.2f}")
            print(f"BIC: {self.group_results[model]['bic_mean']:.2f} ± {self.group_results[model]['bic_std']:.2f}")
            print(f"Log-Likelihood: {self.group_results[model]['log_likelihood_mean']:.2f} ± {self.group_results[model]['log_likelihood_std']:.2f}")
        
        # Determine best model based on AIC and BIC
        best_aic = min(self.group_results.items(), key=lambda x: x[1]['aic_mean'])[0]
        best_bic = min(self.group_results.items(), key=lambda x: x[1]['bic_mean'])[0]
        
        print("\nModel Selection:")
        print("-" * 40)
        print(f"Best model (AIC): {best_aic}")
        print(f"Best model (BIC): {best_bic}")

# Example usage
if __name__ == "__main__":
    # Define subjects to analyze
    subjects = ['01', '02', '03']  # Add all your subject IDs
    
    # Initialize group analysis
    group_analysis = FSLGroupModelComparison(subjects, run_id='01')
    
    # Run individual analyses
    group_analysis.run_individual_analyses()
    
    # Get group results
    group_results = group_analysis.aggregate_results()
    
    # Create and save group-level visualization
    fig = group_analysis.plot_group_results(save_path='group_model_comparison.png')
    
    # Print summary
    group_analysis.print_summary()
