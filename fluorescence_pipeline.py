"""
Fluorescence Data Pipeline
Preprocessing and feature extraction for fluorescence spectroscopy data
"""

import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis


class FluorescenceDataPipeline:
    """
    Data processing pipeline for fluorescence spectrometry

    Handles:
    - 9 detection channels (purple through IR)
    - 2 identical sensors
    - Dual excitation sources (254nm and 365nm UV)
    - Feature extraction and preprocessing
    """

    def __init__(self, n_sensors=2):
        self.n_channels = 9
        self.n_sensors = n_sensors
        self.n_excitation_sources = 2

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

        self.scaler = RobustScaler()
        self.feature_names = None

    def fit_transform(self, data, strategy='fluorescence_enhanced'):
        """
        Fit scaler and transform data

        Parameters:
        -----------
        data : numpy array
            Shape: (n_samples, 9, interval, 2, 2) with dual excitation
            OR (n_samples, 9, interval, 2) with single excitation
        strategy : str
            Feature extraction strategy

        Returns:
        --------
        features : numpy array, shape (n_samples, n_features)
            Scaled feature matrix
        """
        features = self.extract_features(data, strategy=strategy)
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled

    def transform(self, data, strategy='fluorescence_enhanced'):
        """
        Transform new data using fitted scaler

        Parameters:
        -----------
        data : numpy array
            Shape: (n_samples, 9, interval, 2, 2) or (n_samples, 9, interval, 2)
        strategy : str
            Feature extraction strategy (must match training)

        Returns:
        --------
        features : numpy array, shape (n_samples, n_features)
            Scaled feature matrix
        """
        features = self.extract_features(data, strategy=strategy)
        features_scaled = self.scaler.transform(features)
        return features_scaled

    def extract_features(self, data, strategy='fluorescence_enhanced'):
        """
        Extract features from raw fluorescence data

        Parameters:
        -----------
        data : numpy array
            Shape: (n_samples, 9, interval, 2, 2) with dual excitation
            OR (n_samples, 9, interval, 2) with single excitation
        strategy : str
            - 'fluorescence_enhanced': Specialized for fluorescence (RECOMMENDED)
            - 'spectral_ratios': Focus on relative intensities (concentration-independent)
            - 'excitation_comparison': Compare 254nm vs 365nm responses
            - 'statistical': Basic statistical features

        Returns:
        --------
        features : numpy array, shape (n_samples, n_features)
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
        has_dual_excitation = len(data.shape) == 5

        n_samples = data.shape[0]
        features_list = []
        feature_names = []

        wavelengths = np.array([self.emission_wavelengths[ch] for ch in self.channel_names])

        for sample_idx in range(n_samples):
            sample_features = []

            # Process each excitation source
            n_excitations = 2 if has_dual_excitation else 1

            for exc_idx in range(n_excitations):
                exc_name = self.excitation_names[exc_idx] if has_dual_excitation else '254nm'

                # Average across sensors for more stable measurements
                if has_dual_excitation:
                    avg_spectrum = np.mean(data[sample_idx, :, :, :, exc_idx], axis=(1, 2))
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
                peak_wavelength = self.emission_wavelengths[self.channel_names[peak_channel]]

                # Spectral centroid (center of mass of spectrum)
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

                # === 3. Channel Ratios (concentration-independent) ===
                blue_intensity = avg_spectrum[2]
                green_intensity = avg_spectrum[4]
                red_intensity = avg_spectrum[7]

                blue_green_ratio = blue_intensity / (green_intensity + 1e-10)
                red_blue_ratio = red_intensity / (blue_intensity + 1e-10)

                uv_intensity = np.sum(avg_spectrum[0:2])
                visible_intensity = np.sum(avg_spectrum[4:7])
                uv_visible_ratio = uv_intensity / (visible_intensity + 1e-10)

                # === 4. Quantum Yield Proxy ===
                total_emission = np.sum(avg_spectrum)
                mean_emission = np.mean(avg_spectrum)
                max_emission = np.max(avg_spectrum)

                # === 5. Temporal Features (photobleaching, stability) ===
                if has_dual_excitation:
                    temporal_data = np.mean(data[sample_idx, :, :, :, exc_idx], axis=(0, 2))
                else:
                    temporal_data = np.mean(data[sample_idx, :, :, :], axis=(0, 2))

                time_points = np.arange(len(temporal_data))
                if len(temporal_data) > 1:
                    temporal_slope = np.polyfit(time_points, temporal_data, 1)[0]
                    temporal_stability = np.std(temporal_data) / (np.mean(temporal_data) + 1e-10)
                else:
                    temporal_slope = 0
                    temporal_stability = 0

                # === Compile Features ===
                feats = [
                    peak_wavelength,
                    spectral_centroid,
                    spectral_width,
                    spectral_skew,
                    spectral_kurt,
                    total_emission,
                    mean_emission,
                    max_emission,
                    blue_green_ratio,
                    red_blue_ratio,
                    uv_visible_ratio,
                    *normalized_spectrum,
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

                    for ch in self.channel_names:
                        feat_names.append(f'{exc_name}_{ch}_normalized')

                    feat_names.extend([
                        f'{exc_name}_temporal_slope',
                        f'{exc_name}_temporal_stability',
                    ])

                    feature_names.extend(feat_names)

            # === 6. Cross-Excitation Features ===
            if has_dual_excitation:
                spectrum_254 = np.mean(data[sample_idx, :, :, :, 0], axis=(1, 2))
                spectrum_365 = np.mean(data[sample_idx, :, :, :, 1], axis=(1, 2))

                emission_ratio = np.sum(spectrum_254) / (np.sum(spectrum_365) + 1e-10)

                centroid_254 = np.sum(wavelengths * spectrum_254) / (np.sum(spectrum_254) + 1e-10)
                centroid_365 = np.sum(wavelengths * spectrum_365) / (np.sum(spectrum_365) + 1e-10)
                centroid_shift = centroid_254 - centroid_365

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
        Focus on spectral ratios - largely independent of concentration
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
        Compare compound response to 254nm vs 365nm excitation
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

            for ch_idx, ch_name in enumerate(self.channel_names):
                intensity_254 = spectrum_254[ch_idx]
                intensity_365 = spectrum_365[ch_idx]

                ratio = intensity_254 / (intensity_365 + 1e-10)
                diff = intensity_254 - intensity_365
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
        """Basic statistical features"""
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

    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names if self.feature_names else []

    def get_channel_info(self):
        """Return channel configuration information"""
        return {
            'channels': self.channel_names,
            'wavelengths': self.emission_wavelengths,
            'excitation_sources': self.excitation_names
        }