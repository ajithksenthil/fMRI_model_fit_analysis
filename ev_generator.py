import pandas as pd
import numpy as np

def create_fsleev_files(csv_path, output_prefix):
    """
    Create FSL EV files for different models from timing CSV
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Extract relevant columns and remove any non-trial rows
    df = df[df['NoiseLevel'].notna()]
    
    # Get onset times, durations, and noise levels
    onsets = df['clip.started'].values
    durations = df['clip.stopped'].values - df['clip.started'].values
    noise_levels = df['NoiseLevel'].values / 100.0  # Normalize to 0-1
    
    # Define models
    models = {
        'linear': lambda x: x,
        'quadratic': lambda x: x**2,
        'cubic': lambda x: x**3,
        'exponential': lambda x: 1 - np.exp(-2 * x)
    }
    
    # Create EV file for each model
    for model_name, model_func in models.items():
        weights = model_func(noise_levels)
        
        # Write 3-column format file
        filename = f"{output_prefix}_{model_name}.txt"
        with open(filename, 'w') as f:
            for onset, duration, weight in zip(onsets, durations, weights):
                f.write(f"{onset:.3f}\t{duration:.3f}\t{weight:.6f}\n")
        
        print(f"Created {filename}")


create_fsleev_files('data/GAS1_Stimuli_001_2024-11-14_09h55.49.607.csv', 'sub-01_run-01')
