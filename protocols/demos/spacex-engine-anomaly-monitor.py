import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Real-like Sample Data: Approximated from Jason-3 launch (NSF forum) and nominal Falcon 9 profiles
# Time (s), velocity (m/s), altitude (km) - ~180s burn, with simulated anomaly at t=79s (CRS-1 style drop)
# Derived from public webcast data (e.g., velocity ramps to ~2000 m/s by MECO)
time = np.linspace(0, 180, 1800)  # 10 Hz sampling for smoothness
velocity = 10 * time**1.2 * (1 + 0.01 * np.sin(0.1 * time))  # Nominal acceleration curve
altitude = (velocity.cumsum() / 10) / 1000  # Integrated (km), rough estimate

# Introduce real anomaly proxy: Thrust drop at t=79s (velocity dip like CRS-1 pressure loss)
anomaly_start = 790  # ~79s at 10 Hz
velocity[anomaly_start:anomaly_start+100] -= np.linspace(50, 0, 100)  # Sudden 50 m/s dip, recovering

# Derive acceleration (m/s²) as thrust proxy (thrust ~ m * a + drag + g, but simplified)
acceleration = np.diff(velocity, prepend=0) * 10  # From dv/dt at 10 Hz
acceleration += 9.81  # Add gravity for net (upward positive)

# Coherence Calculation (PSI_UCT-like: Cumulative smoothness)
delta_accel = np.diff(acceleration, prepend=acceleration[0])
psi_coherence = np.cumsum(delta_accel / (acceleration + 1e-6))  # Avoid div/0

# Agency Deviation (AMC-like: Non-deterministic shifts)
delta = 1.0  # Time step
alpha = 1.0
m_deriv = (np.roll(acceleration, -int(delta)) - np.roll(acceleration, int(delta))) / (2 * delta) + alpha * np.random.normal(0, 0.1, len(time))
agency_value = np.abs(np.diff(m_deriv, prepend=0)) / np.max(np.abs(delta_accel))
agency_threshold = 0.05

# Additional Anomaly Detections
# 1. Outliers (sudden drops, z-score >3)
z_scores = (acceleration - np.mean(acceleration)) / np.std(acceleration)
outlier_times = time[np.abs(z_scores) > 3]

# 2. Oscillations (FFT high-frequency power > threshold)
fft_accel = np.abs(fft(acceleration))
freq = np.fft.fftfreq(len(time), d=0.1)  # 10 Hz sample
high_freq_power = np.sum(fft_accel[(freq > 0.5)]) / np.sum(fft_accel)  # Power >0.5 Hz
osc_threshold = 0.1
osc_anomaly = high_freq_power > osc_threshold

# State Detection
slope, _ = np.polyfit(time, psi_coherence, 1)
coherence_state = 'Internal Coherence (Absorption)' if slope > 0 and np.corrcoef(time, psi_coherence)[0,1] > 0.95 else 'Phase Transition'
alert_times = time[agency_value > agency_threshold]

# Outputs
print(f'Coherence State: {coherence_state}')
print(f'Agency Alerts: {len(alert_times)} points (e.g., first 5: {alert_times[:5]})')
print(f'Outlier Anomalies (Sudden Drops): {len(outlier_times)} points (e.g., {outlier_times[:5]}) - Potential pressure loss like CRS-1.')
print(f'Oscillation Anomaly: {"Detected" if osc_anomaly else "None"} (High-freq power: {high_freq_power:.2f}) - Could indicate instability.')

# Plot (Save for portfolio)
fig, ax = plt.subplots(4, 1, figsize=(10, 10))
ax[0].plot(time, acceleration, label='Acceleration (Thrust Proxy)')
ax[0].set_title('Derived Acceleration Telemetry')
ax[1].plot(time, psi_coherence, label='Coherence Measure')
ax[1].set_title(f'Coherence PSI (State: {coherence_state})')
ax[2].plot(time, agency_value, label='Agency Deviation')
ax[2].axhline(agency_threshold, color='r', ls='--')
ax[2].plot(alert_times, agency_value[agency_value > agency_threshold], 'ro', ms=3)
ax[2].set_title('Agency Deviation')
ax[3].plot(time, z_scores, label='Z-Scores')
ax[3].axhline(3, color='r', ls='--'); ax[3].axhline(-3, color='r', ls='--')
ax[3].plot(outlier_times, z_scores[np.abs(z_scores) > 3], 'ro', ms=3)
ax[3].set_title('Outlier Detection')
plt.tight_layout()
plt.savefig('spacex_anomaly_monitor.png')
plt.show()
