"""
Fluorescence Classifier
Random Forest classifier for compound identification via fluorescence spectroscopy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from fluorescence_pipeline import FluorescenceDataPipeline


class FluorescenceClassifier:
    """
    Random Forest classifier for identifying compounds based on fluorescence signatures

    Uses dual UV excitation (254nm and 365nm) with 9-channel spectrometer detection
    """

    def __init__(self, n_sensors=2):
        """
        Initialize classifier

        Parameters:
        -----------
        n_sensors : int
            Number of spectrometer sensors (default: 2)
        """
        self.pipeline = FluorescenceDataPipeline(n_sensors=n_sensors)
        self.model = None
        self.strategy = None
        self.classes_ = None

    def train(self, X, y, test_size=0.2, random_state=42,
              strategy='fluorescence_enhanced', optimize_hyperparams=False, **rf_params):
        """
        Train the classifier

        Parameters:
        -----------
        X : numpy array
            Shape: (n_samples, 9, interval, 2, 2) with dual excitation
            OR (n_samples, 9, interval, 2) with single excitation
        y : numpy array
            Labels (compound types)
        test_size : float
            Proportion of data for testing
        random_state : int
            Random seed for reproducibility
        strategy : str
            Feature extraction strategy:
            - 'fluorescence_enhanced': Specialized features (RECOMMENDED)
            - 'spectral_ratios': Concentration-independent ratios
            - 'excitation_comparison': Compare 254nm vs 365nm
            - 'statistical': Basic statistical features
        optimize_hyperparams : bool
            Whether to run grid search for hyperparameter tuning
        **rf_params : dict
            Additional parameters for RandomForestClassifier

        Returns:
        --------
        results : dict
            Training results including scores, predictions, and confusion matrix
        """

        self.strategy = strategy
        self.classes_ = np.unique(y)

        print(f"Input data shape: {X.shape}")
        print(f"Number of classes: {len(self.classes_)}")
        print(f"Class distribution: {np.bincount(y)}")

        # Extract and scale features
        print(f"\nExtracting features with strategy: {strategy}")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train = self.pipeline.fit_transform(X_train_raw, strategy=strategy)
        X_test = self.pipeline.transform(X_test_raw, strategy=strategy)

        print(f"Feature matrix shape: {X_train.shape}")

        # Hyperparameter optimization
        if optimize_hyperparams:
            print("\nOptimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_train, y_train, random_state)
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
                'class_weight': 'balanced'
            }
            default_params.update(rf_params)

            print("\nTraining Random Forest...")
            self.model = RandomForestClassifier(**default_params)
            self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"\n{'=' * 60}")
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Testing Accuracy: {test_score:.4f}")
        print(f"{'=' * 60}")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, self.classes_)

        return {
            'train_score': train_score,
            'test_score': test_score,
            'cv_scores': cv_scores,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm
        }

    def _optimize_hyperparameters(self, X_train, y_train, random_state):
        """Run grid search for hyperparameter optimization"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def predict(self, X):
        """
        Predict compound class for new samples

        Parameters:
        -----------
        X : numpy array
            Shape: (n_samples, 9, interval, 2, 2) or (n_samples, 9, interval, 2)

        Returns:
        --------
        predictions : numpy array
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")

        X_transformed = self.pipeline.transform(X, strategy=self.strategy)
        return self.model.predict(X_transformed)

    def predict_proba(self, X):
        """
        Get prediction probabilities for new samples

        Parameters:
        -----------
        X : numpy array
            Shape: (n_samples, 9, interval, 2, 2) or (n_samples, 9, interval, 2)

        Returns:
        --------
        probabilities : numpy array
            Prediction probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Call train() first.")

        X_transformed = self.pipeline.transform(X, strategy=self.strategy)
        return self.model.predict_proba(X_transformed)

    def plot_feature_importance(self, top_n=25):
        """
        Plot most important features

        Parameters:
        -----------
        top_n : int
            Number of top features to display

        Returns:
        --------
        importance_dict : dict
            Dictionary of feature names and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        importances = self.model.feature_importances_
        feature_names = self.pipeline.get_feature_names()

        if not feature_names:
            feature_names = [f'Feature {i}' for i in range(len(importances))]

        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(14, 8))
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')

        labels = [feature_names[i] for i in indices]
        values = importances[indices]

        plt.barh(range(top_n), values, color='steelblue')
        plt.yticks(range(top_n), labels, fontsize=10)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()

        return dict(zip(labels, values))

    def plot_wavelength_importance(self):
        """
        Plot aggregated importance by wavelength channel
        Shows which wavelength bands are most important
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        importances = self.model.feature_importances_
        feature_names = self.pipeline.get_feature_names()
        channel_names = self.pipeline.channel_names

        # Aggregate importance by channel
        channel_importance = {channel: 0.0 for channel in channel_names}

        if feature_names:
            for feat_name, importance in zip(feature_names, importances):
                for channel in channel_names:
                    if channel in feat_name:
                        channel_importance[channel] += importance
                        break

        # Plot
        plt.figure(figsize=(12, 6))
        channels = list(channel_importance.keys())
        values = list(channel_importance.values())

        colors = ['purple', 'darkblue', 'blue', 'lightblue',
                  'green', 'yellow', 'orange', 'red', 'darkred']

        plt.bar(channels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        plt.title('Feature Importance by Wavelength Channel', fontsize=14, fontweight='bold')
        plt.xlabel('Wavelength Channel', fontsize=12)
        plt.ylabel('Cumulative Importance', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

        return channel_importance

    def plot_emission_spectra(self, X, y, n_samples=3):
        """
        Plot example emission spectra for each compound class

        Parameters:
        -----------
        X : numpy array
            Fluorescence data
        y : numpy array
            Labels
        n_samples : int
            Number of samples to plot per class
        """
        has_dual_excitation = len(X.shape) == 5
        unique_classes = np.unique(y)

        fig, axes = plt.subplots(
            len(unique_classes),
            2 if has_dual_excitation else 1,
            figsize=(14, 4 * len(unique_classes))
        )

        if len(unique_classes) == 1:
            axes = axes.reshape(1, -1)
        elif not has_dual_excitation:
            axes = axes.reshape(-1, 1)

        channel_info = self.pipeline.get_channel_info()
        wavelengths = [channel_info['wavelengths'][ch] for ch in channel_info['channels']]

        for class_idx, class_label in enumerate(unique_classes):
            class_samples = X[y == class_label][:n_samples]

            for exc_idx in range(2 if has_dual_excitation else 1):
                ax = axes[class_idx, exc_idx] if len(unique_classes) > 1 else axes[exc_idx]

                for sample in class_samples:
                    if has_dual_excitation:
                        spectrum = np.mean(sample[:, :, :, exc_idx], axis=(1, 2))
                    else:
                        spectrum = np.mean(sample[:, :, :], axis=(1, 2))

                    ax.plot(wavelengths, spectrum, alpha=0.6, marker='o', linewidth=2)

                exc_name = channel_info['excitation_sources'][exc_idx] if has_dual_excitation else '254nm'
                ax.set_title(f'Class {class_label} - {exc_name} Excitation',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('Emission Wavelength (nm)', fontsize=11)
                ax.set_ylabel('Fluorescence Intensity', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_xticks(wavelengths)
                ax.set_xticklabels(channel_info['channels'], rotation=45)

        plt.tight_layout()
        plt.show()

    def analyze_compound_signature(self, X_sample, compound_name="Unknown"):
        """
        Detailed analysis of a single compound's fluorescence signature

        Parameters:
        -----------
        X_sample : numpy array
            Single sample data, shape (9, interval, 2, 2) or (9, interval, 2)
        compound_name : str
            Name for the compound being analyzed
        """
        has_dual_excitation = len(X_sample.shape) == 4
        channel_info = self.pipeline.get_channel_info()
        wavelengths = [channel_info['wavelengths'][ch] for ch in channel_info['channels']]

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Plot 1: Emission spectra
        ax1 = fig.add_subplot(gs[0, :])

        if has_dual_excitation:
            spectrum_254 = np.mean(X_sample[:, :, :, 0], axis=(1, 2))
            spectrum_365 = np.mean(X_sample[:, :, :, 1], axis=(1, 2))

            ax1.plot(wavelengths, spectrum_254, marker='o', linewidth=2,
                     label='254nm Excitation', color='purple', markersize=8)
            ax1.plot(wavelengths, spectrum_365, marker='s', linewidth=2,
                     label='365nm Excitation', color='blue', markersize=8)
        else:
            spectrum = np.mean(X_sample[:, :, :], axis=(1, 2))
            ax1.plot(wavelengths, spectrum, marker='o', linewidth=2, markersize=8)

        ax1.set_title(f'Fluorescence Signature: {compound_name}',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emission Wavelength (nm)', fontsize=12)
        ax1.set_ylabel('Fluorescence Intensity', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        ax1.set_xticks(wavelengths)
        ax1.set_xticklabels(channel_info['channels'], rotation=45)

        # Plot 2: Temporal behavior
        ax2 = fig.add_subplot(gs[1, 0])
        if has_dual_excitation:
            temporal_254 = np.mean(X_sample[:, :, :, 0], axis=(0, 2))
            temporal_365 = np.mean(X_sample[:, :, :, 1], axis=(0, 2))
            ax2.plot(temporal_254, label='254nm', color='purple', linewidth=2)
            ax2.plot(temporal_365, label='365nm', color='blue', linewidth=2)
        else:
            temporal = np.mean(X_sample[:, :, :], axis=(0, 2))
            ax2.plot(temporal, linewidth=2)

        ax2.set_title('Temporal Stability (Photobleaching Check)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Frame', fontsize=11)
        ax2.set_ylabel('Average Intensity', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Normalized spectrum
        ax3 = fig.add_subplot(gs[1, 1])
        if has_dual_excitation:
            norm_254 = spectrum_254 / np.sum(spectrum_254)
            norm_365 = spectrum_365 / np.sum(spectrum_365)

            x = np.arange(len(channel_info['channels']))
            width = 0.35
            ax3.bar(x - width / 2, norm_254, width, label='254nm', color='purple', alpha=0.7)
            ax3.bar(x + width / 2, norm_365, width, label='365nm', color='blue', alpha=0.7)
            ax3.set_xticks(x)
        else:
            norm = spectrum / np.sum(spectrum)
            ax3.bar(channel_info['channels'], norm, color='green', alpha=0.7)

        ax3.set_title('Normalized Emission Profile', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Channel', fontsize=11)
        ax3.set_ylabel('Relative Intensity', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.set_xticklabels(channel_info['channels'], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Sensor agreement
        ax4 = fig.add_subplot(gs[2, :])
        if has_dual_excitation:
            sensor1 = np.mean(X_sample[:, :, 0, 0], axis=1)
            sensor2 = np.mean(X_sample[:, :, 1, 0], axis=1)
        else:
            sensor1 = np.mean(X_sample[:, :, 0], axis=1)
            sensor2 = np.mean(X_sample[:, :, 1], axis=1)

        ax4.scatter(sensor1, sensor2, s=100, alpha=0.6, c=range(len(sensor1)), cmap='viridis')
        lims = [min(sensor1.min(), sensor2.min()), max(sensor1.max(), sensor2.max())]
        ax4.plot(lims, lims, 'r--', label='Perfect Agreement', linewidth=2)
        ax4.set_title('Sensor 1 vs Sensor 2 Agreement', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Sensor 1 Intensity', fontsize=11)
        ax4.set_ylabel('Sensor 2 Intensity', fontsize=11)

        correlation = np.corrcoef(sensor1, sensor2)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                 transform=ax4.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=11)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        print(f"\n{'=' * 60}")
        print(f"Fluorescence Analysis: {compound_name}")
        print(f"{'=' * 60}")

        if has_dual_excitation:
            print(f"\n254nm Excitation:")
            print(f"  Peak channel: {channel_info['channels'][np.argmax(spectrum_254)]}")
            print(f"  Peak wavelength: {wavelengths[np.argmax(spectrum_254)]} nm")
            print(f"  Total intensity: {np.sum(spectrum_254):.2f}")

            print(f"\n365nm Excitation:")
            print(f"  Peak channel: {channel_info['channels'][np.argmax(spectrum_365)]}")
            print(f"  Peak wavelength: {wavelengths[np.argmax(spectrum_365)]} nm")
            print(f"  Total intensity: {np.sum(spectrum_365):.2f}")

            ratio = np.sum(spectrum_254) / np.sum(spectrum_365)
            print(f"\nExcitation Comparison:")
            print(f"  Intensity ratio (254/365): {ratio:.2f}")
        else:
            print(f"\nPeak channel: {channel_info['channels'][np.argmax(spectrum)]}")
            print(f"Peak wavelength: {wavelengths[np.argmax(spectrum)]} nm")
            print(f"Total intensity: {np.sum(spectrum):.2f}")

        print(f"\nSensor Agreement: {correlation:.3f}")
        print(f"{'=' * 60}\n")

    def _plot_confusion_matrix(self, cm, classes):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """Save trained model to file"""
        import pickle

        if self.model is None:
            raise ValueError("No model to save! Train the model first.")

        model_data = {
            'model': self.model,
            'pipeline': self.pipeline,
            'strategy': self.strategy,
            'classes': self.classes_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model from file"""
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.pipeline = model_data['pipeline']
        self.strategy = model_data['strategy']
        self.classes_ = model_data['classes']

        print(f"Model loaded from {filepath}")