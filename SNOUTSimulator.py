import numpy as np
import matplotlib.pyplot as plt

"""
Functions in this class:

generate_event()
generate_stream()

"""
# --- STEP 1: SIMULATION MODULE (from SNOUTSimulator.py) ---

class SNOUTSimulator:
    """
    Simulates a realistic data stream from the SNOUT sensor, including
    drifting baseline and signal-dependent shot noise.
    """

    def __init__(self, bit_depth=16, baseline_level=1000,
                 read_noise_amp=50, shot_noise_factor=0.6,
                 drift_amplitude=75, drift_frequency=0.0001):
        """
        Initializes the simulator with advanced sensor parameters.

        Args:
            bit_depth (int): The bit-depth of the ADC (e.g., 16 for 0-65535).
            baseline_level (int): The central value of the baseline.
            read_noise_amp (int): The amplitude (std dev) of constant electronic noise.
            shot_noise_factor (float): Multiplier for signal-dependent shot noise.
                                       Set to 0 to disable.
            drift_amplitude (int): The amount the baseline will drift up/down.
                                   Set to 0 to disable.
            drift_frequency (float): How quickly the baseline drifts.
        """
        self.bit_depth = bit_depth
        self.max_val = (2**bit_depth) - 1
        self.baseline_level = baseline_level
        self.read_noise_amp = read_noise_amp
        self.shot_noise_factor = shot_noise_factor
        self.drift_amplitude = drift_amplitude
        self.drift_frequency = drift_frequency

    def generate_event(self, length, peak_intensity, decay_rate):
        """
        Generates a single, clean fluorescence event curve.
        (This method is unchanged)
        """
        time_points = np.arange(length)
        event_curve = peak_intensity * np.exp(-decay_rate * time_points)
        return event_curve

    def generate_stream(self, stream_length, events):
        """
        Generates a complete data stream with advanced noise and drift.

        Args:
            stream_length (int): The total number of data points for the stream.
            events (list of tuples): Format: (start_time, peak, decay_rate)

        Returns:
            np.ndarray: The final simulated data stream.
        """
        # --- Step 1: Generate the "clean" signal without any noise ---
        time_points = np.arange(stream_length)

        # Create a slow, meandering baseline drift
        drift = self.drift_amplitude * np.sin(2 * np.pi * self.drift_frequency * time_points)
        clean_signal = self.baseline_level + drift

        # Add the clean event curves to the drifting baseline
        for start_time, peak, decay in events:
            event_length = stream_length - start_time
            if event_length > 0:
                event_curve = self.generate_event(event_length, peak, decay)
                clean_signal[start_time:] += event_curve

        # --- Step 2: Calculate the total noise based on the clean signal ---

        # Ensure signal is non-negative before taking the square root for shot noise
        non_negative_signal = np.clip(clean_signal, 0, None)
        shot_noise_amp = self.shot_noise_factor * np.sqrt(non_negative_signal)

        # Total noise is the combination of constant read noise and variable shot noise
        total_noise_amp = self.read_noise_amp + shot_noise_amp

        # --- Step 3: Generate and add the noise to the clean signal ---
        # Generate noise where the standard deviation can be different for every point
        noise = np.random.normal(loc=0, scale=total_noise_amp, size=stream_length)
        noisy_signal = clean_signal + noise

        # --- Step 4: Finalize the stream ---
        # Clip to ADC limits and convert to integer
        final_stream = np.clip(noisy_signal, 0, self.max_val)

        return final_stream.astype(int)