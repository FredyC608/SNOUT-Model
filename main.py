import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import skew, kurtosis


class FluorescenceClassifier:
    """
    Enhanced pipeline for classifying compounds based on fluorescence spectra

    Setup:
    - 2 excitation sources: 254nm and 365nm UV lights
    - 2 identical spectrometers measuring emission
    - 9 detection channels: purple, dark_blue, blue, light_blue, green, yellow, orange, red, IR
    - Data shape: (n_samples, 9 channels, time_intervals, 2 sensors, 2 excitation_sources)
    """

    def __init__(self, n_sensors=2):
        self.n_channels = 9
        self.n_sensors = n_sensors
        self.n_excitation_sources = 2  # 254nm and 365nm

        self.channel_names = [
            'purple',  # ~400-450nm
            'dark_blue',  # ~450-475nm
            'blue',  # ~475-495nm
            'light_blue',  # ~495-520nm
            'green',  # ~520-565nm
            'yellow',  # ~565-590nm
            'orange',  # ~590-625nm
            'red',  # ~625-750nm
            'IR'  # ~750-1000nm
        ]

        self.excitation_names = ['254nm', '365nm']

        # Approximate emission wavelengths for each channel (center wavelength)
        self.emission_wavelengths = {
            'purple': 425,
            'dark_blue': 462,
            'blue': 485,
            'light_blue': 507,
            'green': 542,
            'yellow': 577,
            'orange': 607,
            'red': 687,
            'IR': 875
        }

        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.model = None
        self.feature_names = None

    def preprocess_data(self, data, strategy='fluorescence_enhanced'):
        """
        Convert fluorescence data to feature matrix

        Parameters:
        -----------
        data : numpy array of shape (n_samples, 9, interval, 2, 2)
               [samples, channels, time, sensors, excitation_sources]
               OR (n_samples, 9, interval, 2) if single excitation source
        strategy : str
            - 'fluorescence_enhanced': Specialized for fluorescence (RECOMMENDED)
            - 'spectral_ratios': Focus on relative intensities (concentration-independent)
            - 'excitation_comparison': Compare 254nm vs 365nm responses
            - 'statistical': Basic statistical features
        """

        if strategy == 'fluorescence_enhanced':
            features = self._extract_fluorescence_features(data)
        elif strategy == 'spectral_ratios':
            features = self._extract_spectral_ratios(data)
        elif strategy == 'excitation_comparison':
            features = self._extract_excitation_comparison(data)
        elif strategy == 'statistical':
            features = self._extract_statistical_features(data)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return features

    def _extract_fluorescence_features(self, data):
        """
        Extract features specific to fluorescence spectroscopy

        Key features:
        1. Emission spectrum shape (concentration-independent)
        2. Peak positions and ratios
        3. Spectral width
        4. Stokes shift indicators
        5. Response differences between excitation wavelengths
        """

        # Handle both data formats
        if len(data.shape) == 5:  # (samples, channels, time, sensors, excitation)
            has_dual_excitation = True
        else:  # (samples, channels, time, sensors)
            has_dual_excitation = False

        n_samples = data.shape[0]
        features_list = []
        feature_names = []

        for sample_idx in range(n_samples):
            sample_features = []

            # Process each excitation source
            n_excitations = 2 if has_dual_excitation else 1

            for exc_idx in range(n_excitations):
                exc_name = self.excitation_names[exc_idx] if has_dual_excitation else '254nm'

                # Average across sensors for more stable measurements
                if has_dual_excitation:
                    avg_spectrum = np.mean(data[sample_idx, :, :, :, exc_idx],
                                           axis=(1, 2))  # Average over time and sensors
                else:
                    avg_spectrum = np.mean(data[sample_idx, :, :, :], axis=(1, 2))

                # === 1. Normalized Spectrum (concentration-independent) ===
                total_intensity = np.sum(avg_spectrum)
                if total_intensity > 0:
                    normalized_spectrum = avg_spectrum / total_intensity
                else:
                    normalized_spectrum = avg_spectrum

                # === 2. Spectral Shape Features ===
                peak_channel = np.argmax(avg_spectrum)
                peak_intensity = avg_spectrum[peak_channel]
                peak_wavelength = self.emission_wavelengths[self.channel_names[peak_channel]]

                # Spectral centroid (center of mass of spectrum)
                wavelengths = np.array([self.emission_wavelengths[ch] for ch in self.channel_names])
                if total_intensity > 0:
                    spectral_centroid = np.sum(wavelengths * avg_spectrum) / total_intensity
                else:
                    spectral_centroid = 0

                # Spectral width (variance)
                spectral_variance = np.sum(((wavelengths - spectral_centroid) ** 2) * normalized_spectrum)
                spectral_width = np.sqrt(spectral_variance)

                # Spectral skewness and kurtosis
                spectral_skew = skew(avg_spectrum)
                spectral_kurt = kurtosis(avg_spectrum)

                # === 3. Channel Ratios (very discriminative and concentration-independent) ===
                # Blue/Green ratio
                blue_intensity = avg_spectrum[2]  # blue channel
                green_intensity = avg_spectrum[4]  # green channel
                blue_green_ratio = blue_intensity / (green_intensity + 1e-10)

                # Red/Blue ratio
                red_intensity = avg_spectrum[7]  # red channel
                red_blue_ratio = red_intensity / (blue_intensity + 1e-10)

                # UV/Visible ratio (purple vs green-red)
                uv_intensity = np.sum(avg_spectrum[0:2])  # purple + dark_blue
                visible_intensity = np.sum(avg_spectrum[4:7])  # green + yellow + orange
                uv_visible_ratio = uv_intensity / (visible_intensity + 1e-10)

                # === 4. Quantum Yield Proxy (total emission) ===
                # Higher quantum yield = more fluorescence
                total_emission = np.sum(avg_spectrum)
                mean_emission = np.mean(avg_spectrum)
                max_emission = np.max(avg_spectrum)

                # === 5. Temporal Features (photobleaching, stability) ===
                if has_dual_excitation:
                    temporal_data = np.mean(data[sample_idx, :, :, :, exc_idx],
                                            axis=(0, 2))  # Average over channels and sensors
                else:
                    temporal_data = np.mean(data[sample_idx, :, :, :], axis=(0, 2))

                # Trend over time (negative slope indicates photobleaching)
                time_points = np.arange(len(temporal_data))
                if len(temporal_data) > 1:
                    temporal_slope = np.polyfit(time_points, temporal_data, 1)[0]
                    temporal_stability = np.std(temporal_data) / (np.mean(temporal_data) + 1e-10)
                else:
                    temporal_slope = 0
                    temporal_stability = 0

                # === Compile Features ===
                feats = [
                    # Spectral position
                    peak_wavelength,
                    spectral_centroid,
                    spectral_width,
                    spectral_skew,
                    spectral_kurt,

                    # Intensity features
                    total_emission,
                    mean_emission,
                    max_emission,

                    # Spectral ratios (concentration-independent!)
                    blue_green_ratio,
                    red_blue_ratio,
                    uv_visible_ratio,

                    # Normalized intensities for each channel
                    *normalized_spectrum,

                    # Temporal features
                    temporal_slope,
                    temporal_stability,
                ]

                sample_features.extend(feats)

                # Generate feature names (only once)
                if sample_idx == 0:
                    feat_names = [
                        f'{exc_name}_peak_wavelength',
                        f'{exc_name}_spectral_centroid',
                        f'{exc_name}_spectral_width',
                        f'{exc_name}_spectral_skew',
                        f'{exc_name}_spectral_kurtosis',
                        f'{exc_name}_total_emission',
                        f'{exc_name}_mean_emission',
                        f'{exc_name}_max_emission',
                        f'{exc_name}_blue_green_ratio',
                        f'{exc_name}_red_blue_ratio',
                        f'{exc_name}_uv_visible_ratio',
                    ]

                    # Add normalized channel intensities
                    for ch in self.channel_names:
                        feat_names.append(f'{exc_name}_{ch}_normalized')

                    feat_names.extend([
                        f'{exc_name}_temporal_slope',
                        f'{exc_name}_temporal_stability',
                    ])

                    feature_names.extend(feat_names)

            # === 6. Cross-Excitation Features (if dual excitation available) ===
            if has_dual_excitation:
                # Compare responses between 254nm and 365nm
                spectrum_254 = np.mean(data[sample_idx, :, :, :, 0], axis=(1, 2))
                spectrum_365 = np.mean(data[sample_idx, :, :, :, 1], axis=(1, 2))

                # Ratio of total emissions
                emission_ratio = np.sum(spectrum_254) / (np.sum(spectrum_365) + 1e-10)

                # Spectral shift between excitations
                centroid_254 = np.sum(wavelengths * spectrum_254) / (np.sum(spectrum_254) + 1e-10)
                centroid_365 = np.sum(wavelengths * spectrum_365) / (np.sum(spectrum_365) + 1e-10)
                centroid_shift = centroid_254 - centroid_365

                # Correlation between emission patterns
                correlation = np.corrcoef(spectrum_254, spectrum_365)[0, 1]

                cross_feats = [
                    emission_ratio,
                    centroid_shift,
                    correlation,
                ]

                sample_features.extend(cross_feats)

                if sample_idx == 0:
                    cross_feat_names = [
                        'excitation_emission_ratio',
                        'excitation_centroid_shift',
                        'excitation_correlation',
                    ]
                    feature_names.extend(cross_feat_names)

            features_list.append(sample_features)

        self.feature_names = feature_names
        return np.array(features_list)

    def _extract_spectral_ratios(self, data):
        """
        Focus on spectral ratios - these are largely independent of concentration
        This is CRUCIAL for robust classification in fluorescence
        """
        has_dual_excitation = len(data.shape) == 5
        n_samples = data.shape[0]
        features_list = []
        feature_names = []

        for sample_idx in range(n_samples):
            sample_features = []

            n_excitations = 2 if has_dual_excitation else 1

            for exc_idx in range(n_excitations):
                exc_name = self.excitation_names[exc_idx] if has_dual_excitation else '254nm'

                if has_dual_excitation:
                    spectrum = np.mean(data[sample_idx, :, :, :, exc_idx], axis=(1, 2))
                else:
                    spectrum = np.mean(data[sample_idx, :, :, :], axis=(1, 2))

                # Normalize
                total = np.sum(spectrum)
                if total > 0:
                    spectrum_norm = spectrum / total
                else:
                    spectrum_norm = spectrum

                # All pairwise ratios between channels
                for i in range(self.n_channels):
                    for j in range(i + 1, self.n_channels):
                        ratio = spectrum[i] / (spectrum[j] + 1e-10)
                        sample_features.append(ratio)

                        if sample_idx == 0:
                            feature_names.append(
                                f'{exc_name}_{self.channel_names[i]}_to_{self.channel_names[j]}_ratio'
                            )

                # Add normalized intensities
                sample_features.extend(spectrum_norm)

                if sample_idx == 0:
                    for ch in self.channel_names:
                        feature_names.append(f'{exc_name}_{ch}_normalized')

            features_list.append(sample_features)

        self.feature_names = feature_names
        return np.array(features_list)

    def _extract_excitation_comparison(self, data):
        """
        Specifically compare how compound responds to 254nm vs 365nm
        """
        if len(data.shape) != 5:
            raise ValueError("Excitation comparison requires dual excitation data")

        n_samples = data.shape[0]
        features_list = []
        feature_names = []

        for sample_idx in range(n_samples):
            sample_features = []

            spectrum_254 = np.mean(data[sample_idx, :, :, :, 0], axis=(1, 2))
            spectrum_365 = np.mean(data[sample_idx, :, :, :, 1], axis=(1, 2))

            # For each channel, compare response
            for ch_idx, ch_name in enumerate(self.channel_names):
                intensity_254 = spectrum_254[ch_idx]
                intensity_365 = spectrum_365[ch_idx]

                # Ratio
                ratio = intensity_254 / (intensity_365 + 1e-10)

                # Difference
                diff = intensity_254 - intensity_365

                # Relative difference
                rel_diff = diff / (intensity_254 + intensity_365 + 1e-10)

                sample_features.extend([intensity_254, intensity_365, ratio, diff, rel_diff])

                if sample_idx == 0:
                    feature_names.extend([
                        f'{ch_name}_254nm',
                        f'{ch_name}_365nm',
                        f'{ch_name}_ratio_254_365',
                        f'{ch_name}_diff_254_365',
                        f'{ch_name}_rel_diff',
                    ])

            features_list.append(sample_features)

        self.feature_names = feature_names
        return np.array(features_list)

    def _extract_statistical_features(self, data):
        """Basic statistical features - similar to original implementation"""
        has_dual_excitation = len(data.shape) == 5
        n_samples = data.shape[0]
        features_list = []
        feature_names = []

        for sample_idx in range(n_samples):
            sample_features = []

            n_excitations = 2 if has_dual_excitation else 1

            for exc_idx in range(n_excitations):
                exc_name = self.excitation_names[exc_idx] if has_dual_excitation else '254nm'

                for ch_idx in range(self.n_channels):
                    ch_name = self.channel_names[ch_idx]

                    for sensor_idx in range(self.n_sensors):
                        if has_dual_excitation:
                            intensity_series = data[sample_idx, ch_idx, :, sensor_idx, exc_idx]
                        else:
                            intensity_series = data[sample_idx, ch_idx, :, sensor_idx]

                        feats = [
                            np.mean(intensity_series),
                            np.std(intensity_series),
                            np.min(intensity_series),
                            np.max(intensity_series),
                            np.median(intensity_series),
                        ]

                        sample_features.extend(feats)

                        if sample_idx == 0:
                            feat_names = [
                                f'{exc_name}_{ch_name}_s{sensor_idx}_mean',
                                f'{exc_name}_{ch_name}_s{sensor_idx}_std',
                                f'{exc_name}_{ch_name}_s{sensor_idx}_min',
                                f'{exc_name}_{ch_name}_s{sensor_idx}_max',
                                f'{exc_name}_{ch_name}_s{sensor_idx}_median',
                            ]
                            feature_names.extend(feat_names)

            features_list.append(sample_features)

        self.feature_names = feature_names
        return np.array(features_list)

    def train(self, X, y, test_size=0.2, random_state=42, strategy='fluorescence_enhanced',
              optimize_hyperparams=False, **rf_params):
        """
        Train the classifier

        Parameters:
        -----------
        X : numpy array
            Shape: (n_samples, 9, interval, 2, 2) with dual excitation
            OR (n_samples, 9, interval, 2) with single excitation
        y : numpy array of labels (compound types)
        strategy : preprocessing strategy
        optimize_hyperparams : bool, whether to run grid search
        """
        print(f"Input data shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution: {np.bincount(y)}")

        # Preprocess
        print(f"\nPreprocessing with strategy: {strategy}")
        X_processed = self.preprocess_data(X, strategy=strategy)
        print(f"Processed feature shape: {X_processed.shape}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale (RobustScaler is better for fluorescence data with outliers)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Hyperparameter optimization
        if optimize_hyperparams:
            print("\nOptimizing hyperparameters...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

            rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train_scaled, y_train)

            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Default parameters
            default_params = {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': random_state,
                'n_jobs': -1,
                'class_weight': 'balanced'  # Important if classes are imbalanced
            }
            default_params.update(rf_params)

            self.model = RandomForestClassifier(**default_params)
            self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        print(f"\n{'=' * 50}")
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        print(f"{'=' * 50}")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        return {
            'train_score': train_score,
            'test_score': test_score,
            'cv_scores': cv_scores,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }

    def predict(self, X, strategy='fluorescence_enhanced'):
        """Predict compound class"""
        X_processed = self.preprocess_data(X, strategy=strategy)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict(X_scaled)

    def predict_proba(self, X, strategy='fluorescence_enhanced'):
        """Get prediction probabilities"""
        X_processed = self.preprocess_data(X, strategy=strategy)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict_proba(X_scaled)

    def plot_feature_importance(self, top_n=25):
        """Plot most important features"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(14, 8))
        plt.title(f'Top {top_n} Most Important Features for Compound Classification')

        labels = [self.feature_names[i] if self.feature_names else f'Feature {i}'
                  for i in indices]

        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), labels)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        return dict(zip(labels, importances[indices]))

    def plot_emission_spectra(self, X, y, n_samples=3):
        """
        Plot example emission spectra for each compound class
        Useful for understanding what the classifier is seeing
        """
        has_dual_excitation = len(X.shape) == 5
        unique_classes = np.unique(y)

        fig, axes = plt.subplots(len(unique_classes),
                                 2 if has_dual_excitation else 1,
                                 figsize=(14, 4 * len(unique_classes)))

        if not has_dual_excitation:
            axes = axes.reshape(-1, 1)

        wavelengths = [self.emission_wavelengths[ch] for ch in self.channel_names]

        for class_idx, class_label in enumerate(unique_classes):
            # Get samples from this class
            class_samples = X[y == class_label][:n_samples]

            for exc_idx in range(2 if has_dual_excitation else 1):
                ax = axes[class_idx, exc_idx]

                for sample in class_samples:
                    if has_dual_excitation:
                        spectrum = np.mean(sample[:, :, :, exc_idx], axis=(1, 2))
                    else:
                        spectrum = np.mean(sample[:, :, :], axis=(1, 2))

                    ax.plot(wavelengths, spectrum, alpha=0.6, marker='o')

                exc_name = self.excitation_names[exc_idx] if has_dual_excitation else '254nm'
                ax.set_title(f'Class {class_label} - {exc_name} Excitation')
                ax.set_xlabel('Emission Wavelength (nm)')
                ax.set_ylabel('Fluorescence Intensity')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(wavelengths)
                ax.set_xticklabels(self.channel_names, rotation=45)

        plt.tight_layout()
        plt.show()

    def analyze_compound_signature(self, X_sample, compound_name="Unknown"):
        """
        Detailed analysis of a single compound's fluorescence signature
        """
        has_dual_excitation = len(X_sample.shape) == 4

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        wavelengths = [self.emission_wavelengths[ch] for ch in self.channel_names]

        # Plot emission spectra
        ax1 = fig.add_subplot(gs[0, :])

        if has_dual_excitation:
            spectrum_254 = np.mean(X_sample[:, :, :, 0], axis=(1, 2))
            spectrum_365 = np.mean(X_sample[:, :, :, 1], axis=(1, 2))

            ax1.plot(wavelengths, spectrum_254, marker='o', linewidth=2,
                     label='254nm Excitation', color='purple')
            ax1.plot(wavelengths, spectrum_365, marker='s', linewidth=2,
                     label='365nm Excitation', color='blue')
        else:
            spectrum = np.mean(X_sample[:, :, :], axis=(1, 2))
            ax1.plot(wavelengths, spectrum, marker='o', linewidth=2)

        ax1.set_title(f'Fluorescence Signature: {compound_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emission Wavelength (nm)', fontsize=12)
        ax1.set_ylabel('Fluorescence Intensity', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(wavelengths)
        ax1.set_xticklabels(self.channel_names, rotation=45)

        # Temporal behavior
        ax2 = fig.add_subplot(gs[1, 0])
        if has_dual_excitation:
            temporal_254 = np.mean(X_sample[:, :, :, 0], axis=(0, 2))
            ax2.plot(temporal_254, label='254nm', color='purple')
            temporal_365 = np.mean(X_sample[:, :, :, 1], axis=(0, 2))
            ax2.plot(temporal_365, label='365nm', color='blue')
        else:
            temporal = np.mean(X_sample[:, :, :], axis=(0, 2))
            ax2.plot(temporal)

        ax2.set_title('Temporal Stability (Photobleaching Check)')
        ax2.set_xlabel('Time Frame')
        ax2.set_ylabel('Average Intensity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Normalized spectrum (concentration-independent)
        ax3 = fig.add_subplot(gs[1, 1])
        if has_dual_excitation:
            norm_254 = spectrum_254 / np.sum(spectrum_254)
            norm_365 = spectrum_365 / np.sum(spectrum_365)

            x = np.arange(len(self.channel_names))
            width = 0.35
            ax3.bar(x - width / 2, norm_254, width, label='254nm', color='purple', alpha=0.7)
            ax3.bar(x + width / 2, norm_365, width, label='365nm', color='blue', alpha=0.7)
        else:
            norm = spectrum / np.sum(spectrum)
            ax3.bar(self.channel_names, norm, color='green', alpha=0.7)

        ax3.set_title('Normalized Emission Profile')
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Relative Intensity')
        ax3.legend()
        ax3.set_xticklabels(self.channel_names, rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Sensor agreement
        ax4 = fig.add_subplot(gs[2, :])
        if has_dual_excitation:
            sensor1 = np.mean(X_sample[:, :, 0, 0], axis=1)
            sensor2 = np.mean(X_sample[:, :, 1, 0], axis=1)
        else:
            sensor1 = np.mean(X_sample[:, :, 0], axis=1)
            sensor2 = np.mean(X_sample[:, :, 1], axis=1)

        ax4.scatter(sensor1, sensor2, s=100, alpha=0.6)
        ax4.plot([sensor1.min(), sensor1.max()], [sensor1.min(), sensor1.max()],
                 'r--', label='Perfect Agreement')
        ax4.set_title('Sensor 1 vs Sensor 2 Agreement')
        ax4.set_xlabel('Sensor 1 Intensity')
        ax4.set_ylabel('Sensor 2 Intensity')
        correlation = np.corrcoef(sensor1, sensor2)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=ax4.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print key statistics
        print(f"\n{'=' * 60}")
        print(f"Fluorescence Analysis: {compound_name}")
        print(f"{'=' * 60}")

        if has_dual_excitation:
            print(f"\n254nm Excitation:")
            print(f"  Peak channel: {self.channel_names[np.argmax(spectrum_254)]}")
            print(f"  Peak wavelength: {wavelengths[np.argmax(spectrum_254)]} nm")
            print(f"  Total intensity: {np.sum(spectrum_254):.2f}")

            print(f"\n365nm Excitation:")
            print(f"  Peak channel: {self.channel_names[np.argmax(spectrum_365)]}")
            print(f"  Peak wavelength: {wavelengths[np.argmax(spectrum_365)]} nm")
            print(f"  Total intensity: {np.sum(spectrum_365):.2f}")

            print(f"\nExcitation Comparison:")
            print(f"  Intensity ratio (254/365): {np.sum(spectrum_254) / np.sum(spectrum_365):.2f}")
        else:
            print(f"\nPeak channel: {self.channel_names[np.argmax(spectrum)]}")
            print(f"Peak wavelength: {wavelengths[np.argmax(spectrum)]} nm")
            print(f"Total intensity: {np.sum(spectrum):.2f}")

        print(f"\nSensor Agreement: {correlation:.3f}")
        print(f"{'=' * 60}\n")


# Example usage with simulated fluorescence data
if __name__ == "__main__":
    np.random.seed(42)

    # Simulate fluorescence data for 3 different compounds
    n_samples_per_class = 100
    n_channels = 9
    n_intervals = 50  # 50 time frames
    n_sensors = 2
    n_excitation = 2  # 254nm and 365nm

    # Create data: (samples, channels, time, sensors, excitation)
    X_list = []
    y_list = []

    # Compound 1: Strong blue-green fluorescence under 254nm, weaker under 365nm
    for i in range(n_samples_per_class):
        sample = np.random.randn(n_channels, n_intervals, n_sensors, n_excitation) * 0.1 + 0.5

        # 254nm excitation - strong blue-green
        sample[2:5, :, :, 0] += np.random.uniform(0.8, 1.2)  # blue, light_blue, green

        # 365nm excitation - moderate response
        sample[2:5, :, :, 1] += np.random.uniform(0.3, 0.5)

        # Add concentration variation
        concentration_factor = np.random.uniform(0.5, 2.0)
        sample *= concentration_factor

        # Add photobleaching
        decay = np.linspace(1.0, 0.9, n_intervals)
        sample *= decay[np.newaxis, :, np.newaxis, np.newaxis]

        sample = np.clip(sample, 0, None)
        X_list.append(sample)
        y_list.append(0)

    # Compound 2: Strong red-orange fluorescence under both wavelengths
    for i in range(n_samples_per_class):
        sample = np.random.randn(n_channels, n_intervals, n_sensors, n_excitation) * 0.1 + 0.4

        # Both excitations - strong red-orange
        sample[6:8, :, :, 0] += np.random.uniform(1.0, 1.5)  # orange, red
        sample[6:8, :, :, 1] += np.random.uniform(0.9, 1.4)

        concentration_factor = np.random.uniform(0.5, 2.0)
        sample *= concentration_factor

        decay = np.linspace(1.0, 0.85, n_intervals)
        sample *= decay[np.newaxis, :, np.newaxis, np.newaxis]

        sample = np.clip(sample, 0, None)
        X_list.append(sample)
        y_list.append(1)

    # Compound 3: Strong UV response under 254nm, yellow-green under 365nm
    for i in range(n_samples_per_class):
        sample = np.random.randn(n_channels, n_intervals, n_sensors, n_excitation) * 0.1 + 0.3

        # 254nm - UV channels
        sample[0:2, :, :, 0] += np.random.uniform(1.2, 1.8)  # purple, dark_blue

        # 365nm - yellow-green
        sample[4:6, :, :, 1] += np.random.uniform(1.0, 1.5)  # green, yellow

        concentration_factor = np.random.uniform(0.5, 2.0)
        sample *= concentration_factor

        decay = np.linspace(1.0, 0.92, n_intervals)
        sample *= decay[np.newaxis, :, np.newaxis, np.newaxis]

        sample = np.clip(sample, 0, None)
        X_list.append(sample)
        y_list.append(2)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Generated fluorescence dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Compounds: {np.unique(y)}")
    print(f"  Samples per compound: {n_samples_per_class}")
    print(f"  Excitation wavelengths: 254nm, 365nm")
    print(
        f"  Detection channels: {', '.join(['purple', 'dark_blue', 'blue', 'light_blue', 'green', 'yellow', 'orange', 'red', 'IR'])}")

    # Initialize classifier
    classifier = FluorescenceClassifier(n_sensors=2)

    # Visualize example spectra
    print("\nVisualizing emission spectra...")
    classifier.plot_emission_spectra(X, y, n_samples=3)

    # Analyze individual compound
    print("\nAnalyzing Compound 1 signature...")
    classifier.analyze_compound_signature(X[0], compound_name="Compound 1 (Example)")

    # Train classifier
    print("\nTraining classifier...")
    results = classifier.train(
        X, y,
        test_size=0.2,
        strategy='fluorescence_enhanced',
        optimize_hyperparams=False,
        n_estimators=200
    )

    # Plot feature importance
    print("\nAnalyzing feature importance...")
    classifier.plot_feature_importance(top_n=25)

    # Test prediction
    X_new = X[:5]  # Take first 5 samples
    predictions = classifier.predict(X_new)
    probabilities = classifier.predict_proba(X_new)

    print(f"\nTest predictions:")
    print(f"  True labels: {y[:5]}")
    print(f"  Predictions: {predictions}")
    print(f"  Confidence scores:\n{probabilities}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR REAL FLUORESCENCE DATA:")
    print("=" * 60)
    print("""
    1. Use 'fluorescence_enhanced' or 'spectral_ratios' strategy
       - These are concentration-independent

    2. Collect diverse training data:
       - Multiple concentrations of each compound
       - Different environmental conditions
       - Fresh and photobleached samples

    3. Calibration:
       - Measure dark current (no excitation)
       - Measure blank (solvent only)
       - Subtract background

    4. Quality checks:
       - Verify sensor agreement (correlation > 0.95)
       - Check for photobleaching (temporal slope)
       - Monitor excitation source stability

    5. For best results:
       - Average multiple measurements
       - Use RobustScaler for preprocessing
       - Consider ensemble methods
       - Validate with known standards
    """)