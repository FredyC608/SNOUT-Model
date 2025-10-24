from SNOUTSimulator import SNOUTSimulator as sm 
import numpy as np
import matplotlib.pyplot as plt
# 1. Initialize the ADVANCED simulator
adv_simulator = sm(
    bit_depth=16,
    baseline_level=2000,
    read_noise_amp=60,         # Constant background noise
    shot_noise_factor=0.9,     # How strong the signal-dependent noise is
    drift_amplitude=150,       # Baseline will drift by +/- 150
    drift_frequency=0.0002     # How fast the baseline drifts
)

# 2. Define events (same as before)
event_list = [
    (300, 30000, 0.01),
    (1100, 15000, 0.005)
]

# 3. Generate the data stream
simulated_data = adv_simulator.generate_stream(stream_length=2000, events=event_list)

# 4. Plot the results
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(simulated_data, label='Realistic Simulated SNOUT Data', color='royalblue', linewidth=1.5)
ax.set_title('Advanced SNOUT Simulation with Drift and Shot Noise', fontsize=16)
ax.set_xlabel('Time (samples)', fontsize=12)
ax.set_ylabel(f'ADC Reading (0-{adv_simulator.max_val})', fontsize=12)

# Highlight the visible effects
ax.annotate('Baseline Drift', xy=(0, 2150), xytext=(100, 3500),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
ax.annotate('Higher noise on peak\n(Shot Noise)', xy=(400, 32000), xytext=(500, 32000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))
ax.annotate('Lower noise on baseline\n(Read Noise)', xy=(1800, 2000), xytext=(1500, 8000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))


ax.legend()
plt.show()