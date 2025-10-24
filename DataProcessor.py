import numpy as np
import pandas as pd

"""
Functions in this class:
filter_moving_avereage()
normalize_min_max()
process()

"""
# --- STEP 2: PREPROCESSING MODULE (from DataProcessor.py) ---

class DataProcessor:
    """
    Handles the ingestion, filtering, and normalization of raw SNOUT sensor data.
    """
    def __init__(self, raw_data, allMin, allMax):
        """
        Initializes the processor with a raw data stream.

        Args:
            raw_data (np.ndarray): A 1D numpy array of raw sensor readings.
        """
        self.raw_data = raw_data
        self.filtered_data = None
        self.normalized_data = None
        self.globalMin = allMin
        self.globalMax = allMax
        print("DataProcessor initialized with a data stream of length:", len(raw_data))

    def filter_moving_average(self, window_size=10):
        """
        Applies a moving average filter to the raw data to reduce noise.

        Args:
            window_size (int): The number of data points to include in the moving average window.
                               A larger window size results in more smoothing.

        Returns:
            np.ndarray: The filtered (smoothed) data array.
        """
        # Using pandas is highly efficient and readable for this task.
        # We convert the numpy array to a pandas Series to use the .rolling() method.
        series = pd.Series(self.raw_data)
        
        # .rolling() creates a rolling window view of the data.
        # .mean() calculates the average of the values within each window.
        rolling_mean = series.rolling(window=window_size).mean()
        
        # The first 'window_size - 1' values will be NaN because the window is not full yet.
        # We use backfill ('bfill') to propagate the first valid calculation backward.
        rolling_mean = rolling_mean.fillna(method='bfill')
        
        self.filtered_data = rolling_mean.to_numpy()
        print(f"Applied moving average filter with window size {window_size}.")
        return self.filtered_data

    def normalize_min_max(self):
        """
        Applies Min-Max normalization to the filtered data, scaling it to a [0, 1] range.
        
        This should be called *after* filtering the data.
        
        Returns:
            np.ndarray: The normalized data array.
        """
        if self.filtered_data is None:
            raise ValueError("Data must be filtered before normalization. Run .filter_moving_average() first.")
            
        min_val = self.globalMin
        max_val = self.globalMax
        
        # Avoid division by zero if the data is flat (all values are the same)
        if max_val - min_val == 0:
            self.normalized_data = np.zeros_like(self.filtered_data)
        else:
            # The Min-Max formula: (value - min) / (max - min)
            self.normalized_data = (self.filtered_data - min_val) / (max_val - min_val)
            
        print("Applied Min-Max normalization to scale data between 0 and 1.")
        return self.normalized_data

    def process(self, filter_window_size=10):
        """
        A convenience method to run the full preprocessing pipeline in order.

        Args:
            filter_window_size (int): The window size for the moving average filter.
        
        Returns:
            np.ndarray: The final, processed (filtered and normalized) data.
        """
        print("\n--- Starting Data Processing Pipeline ---")
        # Step 1: Filter
        self.filter_moving_average(window_size=filter_window_size)
        
        # Step 2: Normalize
        self.normalize_min_max()
        print("--- Pipeline Finished ---")
        
        return self.normalized_data
    
    
"""
must improve normalization strategy
Z-score normalization seems good. 
"""