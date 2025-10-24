# -----------------------------------------------------------------
# full_pipeline.py
# 
# This script contains the complete 4-step software pipeline
# for the SNOUT project.
#
# -----------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Simulator Imports
from SNOUTSimulator import SNOUTSimulator
from FeatureExtractor import FeatureExtractor
from DataProcessor import DataProcessor


# Suppress warnings from peak finding on flat signals
warnings.filterwarnings("ignore", message="Values in x were all equal")

# --- STEP 4: AI MODEL TRAINING & EXECUTION (NEW) ---

def generate_training_data(simulator, processor, extractor, n_samples=200):
    """
    Helper function to generate a labeled dataset for the AI model.
    """
    
    simMin = simulator.baseline_level - simulator.drift_amplitude
    simMax = simulator.max_val
    print(f"\n--- Generating {n_samples * 2} training samples ---")
    features_list = []
    
    # Define our two "compounds"
    # Compound 0: High peak, fast decay (like Benzene)
    compound_0_params = {'peak_range': (25000, 35000), 'decay_range': (0.01, 0.05)}
    # Compound 1: Medium peak, slow decay (like Toluene)
    compound_1_params = {'peak_range': (10000, 20000), 'decay_range': (0.001, 0.009)}

    for label, params in enumerate([compound_0_params, compound_1_params]):
        for _ in range(n_samples):
            # 1. Simulate
            peak = np.random.uniform(*params['peak_range'])
            decay = np.random.uniform(*params['decay_range'])
            start_time = np.random.randint(200, 300) # Add some randomness
            
            raw_data = simulator.generate_stream(
                stream_length=1500, # Shorter streams for faster training
                events=[(start_time, peak, decay)]
            )
            
            # TODO: need to get the minimum in the simulator accounting for baseline noise
            
            # 2. Process
            proc = DataProcessor(raw_data, simMin, simMax)
            processed_data = proc.process(filter_window_size=15)
            
            # 3. Extract Features
            # We only expect one peak, so we take the first row [0]
            features_df = extractor.extract_features(processed_data)
            
            if not features_df.empty:
                # Get the first peak's features
                peak_features = features_df.iloc[0]
                
                # Add the label and append
                features_list.append({
                    'peak_height': peak_features['peak_height'],
                    'peak_width': peak_features['peak_width'],
                    'label': label  # This is our 'y' value
                })
                
    print("--- Training data generation complete ---")
    return pd.DataFrame(features_list)

def main():
    """
    Runs the entire SNOUT software pipeline from start to finish.
    """
    # --- PART 1: VISUAL DEMO OF THE FULL PIPELINE (Steps 1-3) ---
    print("--- PART 1: Running Visual Pipeline Demo ---")
    
    # 1. Simulate
    sim = SNOUTSimulator(baseline_level=2000, read_noise_amp=60, drift_amplitude=150)
    demo_events = [
        (300, 30000, 0.01),   # Compound 0 (high peak, fast decay)
        (1100, 15000, 0.005)  # Compound 1 (mid peak, slow decay)
    ]
    demo_raw_data = sim.generate_stream(stream_length=2000, events=demo_events)
    
    # 2. Process
    demo_processor = DataProcessor(demo_raw_data, (sim.baseline_level - sim.drift_amplitude), sim.max_val)
    demo_normalized_data = demo_processor.process(filter_window_size=15)
    demo_filtered_data = demo_processor.filtered_data # Get intermediate step for plotting
    
    # 3. Extract Features
    demo_extractor = FeatureExtractor(height_threshold=0.2, peak_distance=200)
    demo_features = demo_extractor.extract_features(demo_normalized_data)
    
    print("\n[Demo] Features Extracted:")
    print(demo_features.to_markdown(index=False))
    
    # --- Plot the visual demo ---
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Raw Data
    axs[0].plot(demo_raw_data, label='Raw Sensor Data', color='gray', alpha=0.8)
    axs[0].set_title('Step 1: Raw Simulated Data (with Noise & Drift)', fontsize=14)
    axs[0].set_ylabel(f'ADC Reading (0-{sim.max_val})')
    axs[0].legend()
    
    # Plot 2: Filtered Data
    axs[1].plot(demo_filtered_data, label='Filtered Data', color='royalblue', linewidth=2)
    axs[1].set_title('Step 2: Filtered Data (Moving Average)', fontsize=14)
    axs[1].set_ylabel('Filtered ADC Reading')
    axs[1].legend()

    # Plot 3: Normalized Data & Features
    axs[2].plot(demo_normalized_data, label='Normalized Data', color='green', linewidth=2)
    axs[2].set_title('Step 3: Normalized Data & Extracted Features', fontsize=14)
    axs[2].set_ylabel('Normalized Intensity (0-1)')
    
    # Plot extracted peaks and widths on the normalized graph
    for _, row in demo_features.iterrows():
        idx = int(row['peak_index'])
        h = row['peak_height']
        w = row['peak_width']
        
        # Plot peak marker
        axs[2].plot(idx, h, 'x', color='red', markersize=10, label='Detected Peak')
        
        # Plot width line (at 50% height)
        half_height = h * 0.5
        width_start = idx - (w / 2)
        width_end = idx + (w / 2)
        axs[2].hlines(y=half_height, xmin=width_start, xmax=width_end, color='red',
                        linestyle='--', label=f'Peak Width (proxy for decay)')

    axs[2].set_xlabel('Time (samples)', fontsize=12)
    # Avoid duplicate labels in legend
    handles, labels = axs[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[2].legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()

    
    # --- PART 2: AI MODEL TRAINING & EVALUATION (Step 4) ---
    print("\n\n--- PART 2: Running AI Model Training (Step 4) ---")
    
    # Create the training data
    # (We can re-use the same simulator and extractor instances)
    training_df = generate_training_data(sim, DataProcessor, demo_extractor, n_samples=100)
    
    print(f"\nGenerated {len(training_df)} total labeled samples.")
    print("Training Data Head:")
    print(training_df.head().to_markdown(index=False))
    
    # Define our X (features) and y (labels)
    X = training_df[['peak_height', 'peak_width']]
    y = training_df['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nSplitting data: {len(X_train)} train, {len(X_test)} test samples.")
    
    # Initialize and train the Random Forest Classifier
    # This is the model specified in your project slides!
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("...Model training complete!")
    
    # Make predictions on the test set and evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- MODEL EVALUATION ---")
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    
    # --- Use the trained model to predict the demo events ---
    print("\n--- FINAL PREDICTION DEMO ---")
    print("Using our trained model to classify the peaks from the visual demo...")
    
    # Get features from the first demo (X_demo)
    X_demo = demo_features[['peak_height', 'peak_width']]
    demo_predictions = rf_model.predict(X_demo)
    demo_features['predicted_compound'] = demo_predictions
    
    print("Prediction Results:")
    print(demo_features.to_markdown(index=False))
    print("\n(Compound 0 = High/Fast, Compound 1 = Mid/Slow)")
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()