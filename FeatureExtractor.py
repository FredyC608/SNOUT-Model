import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

"""
Functions in this class:

extract_features()

"""

# --- STEP 3: AI MODEL TRAINING & EXECUTION (NEW) ---

class FeatureExtractor:
    """
    Extracts key features from the processed (filtered, normalized) data stream.
    """
    def __init__(self, height_threshold=0.1, peak_distance=50):
        """
        Initializes the feature extractor.
        
        Args:
            height_threshold (float): The minimum normalized height (0-1) to
                                      consider a signal a "peak".
            peak_distance (int): Minimum number of samples between peaks.
        """
        self.height_threshold = height_threshold
        self.peak_distance = peak_distance
        print(f"FeatureExtractor initialized: peak threshold={height_threshold}, distance={peak_distance}")

    def extract_features(self, normalized_data):
        """
        Finds all peaks and extracts their features.

        Args:
            normalized_data (np.ndarray): The 0-1 scaled data from the DataProcessor.

        Returns:
            pd.DataFrame: A DataFrame with features for each detected peak.
        """
        # Use SciPy's find_peaks, a powerful and standard tool for this.
        """
        find_peaks() 
        
        - takes in normalized_data index represents time and value at index represents the intensity
        
        - only considers possible peaks above a threshold
        
        - only considers peaks at a distance of atleast 50 units of time (indexes)
        
        """
        peaks, properties = find_peaks(
            normalized_data,
            height=self.height_threshold,  # Ignores baseline noise
            distance=self.peak_distance  # Avoids finding multiple peaks on one event
        )
        
        if not peaks.any():
            return pd.DataFrame(columns=['peak_height', 'peak_width'])

        # Calculate the width of each peak at 50% of its height (Full-Width Half-Max)
        # This is an excellent proxy for the "decay rate" feature
        
        """
        peak_widths() 
        
        - given normalized_data from DataProcessor and peaks which are the indices of each peak detected
        - it returns widths at half height of the peak
    
        """
        widths = peak_widths(normalized_data, peaks, rel_height=0.5)
        
        # 'widths[0]' contains the width values
        peak_width_values = widths[0]
        
        # 'properties["peak_height"]' contains the height of each peak
        peak_height_values = properties['peak_heights']

        # 'peaks' contains the x-index (time) of each peak
        peak_indices = peaks

        # Assemble features into a clean DataFrame
        feature_data = {
            'peak_index': peak_indices,
            'peak_height': peak_height_values,
            'peak_width': peak_width_values
        }
        
        return pd.DataFrame(feature_data)