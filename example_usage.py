"""
Example usage of the Fluorescence Classifier
"""

import numpy as np
from fluorescence_classifier import FluorescenceClassifier


def generate_synthetic_data():
    """Generate synthetic fluorescence data for testing"""
    np.random.seed(42)

    n_samples_per_class = 100
    n_channels = 9
    n_intervals = 50
    n_sensors = 2
    n_excitation = 2

    X_list = []
    y_list = []

    # Compound 1: Blue-green fluorescence under 254nm
    for i in range(n_samples_per_class):
        sample = np.random.randn(n_channels, n_intervals, n_sensors, n_excitation) * 0.1 + 0.5
        sample[2:5, :, :, 0] += np.random.uniform(0.8, 1.2)
        sample[2:5, :, :, 1] += np.random.uniform(0.3, 0.5)

        concentration_factor = np.random.uniform(0.5, 2.0)
        sample *= concentration_factor

        decay = np.linspace(1.0, 0.9, n_intervals)
        sample *= decay[np.newaxis, :, np.newaxis, np.newaxis]

        sample = np.clip(sample, 0, None)
        X_list.append(sample)
        y_list.append(0)

    # Compound 2: Red-orange fluorescence
    for i in range(n_samples_per_class):
        sample = np.random.randn(n_channels, n_intervals, n_sensors, n_excitation) * 0.1 + 0.4
        sample[6:8, :, :, 0] += np.random.uniform(1.0, 1.5)
        sample[6:8, :, :, 1] += np.random.uniform(0.9, 1.4)

        concentration_factor = np.random.uniform(0.5, 2.0)
        sample *= concentration_factor

        decay = np.linspace(1.0, 0.85, n_intervals)
        sample *= decay[np.newaxis, :, np.newaxis, np.newaxis]

        sample = np.clip(sample, 0, None)
        X_list.append(sample)
        y_list.append(1)

    # Compound 3: UV under 254nm, yellow-green under 365nm
    for i in range(n_samples_per_class):
        sample = np.random.randn(n_channels, n_intervals, n_sensors, n_excitation) * 0.1 + 0.3
        sample[0:2, :, :, 0] += np.random.uniform(1.2, 1.8)
        sample[4:6, :, :, 1] += np.random.uniform(1.0, 1.5)

        concentration_factor = np.random.uniform(0.5, 2.0)
        sample *= concentration_factor

        decay = np.linspace(1.0, 0.92, n_intervals)
        sample *= decay[np.newaxis, :, np.newaxis, np.newaxis]

        sample = np.clip(sample, 0, None)
        X_list.append(sample)
        y_list.append(2)

    return np.array(X_list), np.array(y_list)


def main():
    print("=" * 60)
    print("Fluorescence Classifier - Example Usage")
    print("=" * 60)

    # Generate synthetic data
    print("\nGenerating synthetic fluorescence data...")
    X, y = generate_synthetic_data()
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")

    # Initialize classifier
    classifier = FluorescenceClassifier(n_sensors=2)

    # Visualize some spectra
    print("\nVisualizing emission spectra...")
    classifier.plot_emission_spectra(X, y, n_samples=3)

    # Analyze individual sample
    print("\nAnalyzing compound signature...")
    classifier.analyze_compound_signature(X[0], compound_name="Compound 1 (Example)")

    # Train classifier
    print("\nTraining classifier...")
    results = classifier.train(
        X, y,
        test_size=0.2,
        strategy='fluorescence_enhanced',
        n_estimators=200
    )

    # Plot feature importance
    print("\nPlotting feature importance...")
    classifier.plot_feature_importance(top_n=25)
    classifier.plot_wavelength_importance()

    # Test predictions
    print("\nTesting predictions on new samples...")
    X_test = X[:5]
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    print(f"\nTrue labels:     {y[:5]}")
    print(f"Predictions:     {predictions}")
    print(f"\nPrediction probabilities:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Sample {i}: Class {pred} with confidence {prob[pred]:.3f}")

    # Save model
    print("\nSaving model...")
    classifier.save_model('fluorescence_model.pkl')

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()