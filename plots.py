from enum import auto
import math
import sys
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.fft import rfft, rfftfreq
import addcopyfighandler
 
LIMIT = 100
 
first_timestamp = None
very_first_ts = None
strip_time = 0.1
amp_plot_y = []
pha_plot_y = []
rssis = []
plot_x = []
 
for line in sys.stdin:
    if not line.startswith("CSI_DATA"):
        continue
 
    imaginary = []
    real = []
    amplitudes = []
    phases = []
 
    data = line.split(",")
    timestamp = float(data[23])
 
    if first_timestamp is None:
        very_first_ts = timestamp
        first_timestamp = timestamp
    
    if strip_time and timestamp - very_first_ts < strip_time:
        continue
    elif strip_time:
        strip_time = None
        first_timestamp = None
        continue
 
    delta_time = timestamp - first_timestamp
    for i, x in enumerate(plot_x):
        if delta_time < x:
            plot_x = plot_x[:i]
            amp_plot_y = amp_plot_y[:i]
            pha_plot_y = pha_plot_y[:i]
            rssis = rssis[:i]
            break
 
    # Parse string to create integer list
    rssi = int(data[3])
    csi_string = data[25].strip()[1:-2].strip()
    csi_raw = [int(x) for x in csi_string.split(" ") if x != ""]
 
    # Create list of imaginary and real numbers from CSI
    for i in range(len(csi_raw)):
        if i % 2 == 0:
            imaginary.append(csi_raw[i])
        else:
            real.append(csi_raw[i])
 
    # Transform imaginary and real into amplitude and phase
    for i in range(int(len(csi_raw) / 2)):
        amplitudes.append(math.sqrt(imaginary[i] ** 2 + real[i] ** 2))
        phases.append(math.atan2(imaginary[i], real[i]))
    
    pha_plot_y.append(phases)
    amp_plot_y.append(amplitudes)
    plot_x.append(delta_time)
    rssis.append(rssi)
 
    if delta_time - plot_x[0] > LIMIT:
        break
 
rssis = np.array(rssis)
plot_x = np.array(plot_x)
amp_plot_y = np.array(amp_plot_y).transpose()
pha_plot_y = np.array(pha_plot_y).transpose()
amp_plot_y = amp_plot_y[41] # np.average(amp_plot_y, axis=0)
 
# amp_plot_y = 20*np.log10(amp_plot_y/12)
amp_plot_y = ((amp_plot_y - np.min(amp_plot_y)) / (np.max(amp_plot_y) - np.min(amp_plot_y)))*rssis
 
pha_plot_y = pha_plot_y[41]
c_phase, c_phase_final = np.zeros_like(pha_plot_y), np.zeros_like(pha_plot_y)
c_phase[0] = pha_plot_y[0]
diff = 0
for i in range(1, pha_plot_y.shape[0]):
    temp = pha_plot_y[i] - pha_plot_y[i - 1]
    if abs(temp) > np.pi:
        diff = diff + 1 * np.sign(temp)
    c_phase[i] = pha_plot_y[i] - diff * 2 * np.pi
k = (c_phase[-1] - c_phase[0]) / (pha_plot_y.shape[0] - 1)
b = np.mean(c_phase)
for i in range(pha_plot_y.shape[0]):
    c_phase_final[i] = c_phase[i] - k * i - b
 
pha_plot_y = c_phase_final
 
# # Remove amp samples with mag. >= 10
# # Remove phase samples with mag. <= 0
# sel = (amp_plot_y > 4) # & (pha_plot_y > 0)
# # amp_plot_y = amp_plot_y[sel]
# plot_x = plot_x[sel]
# amp_plot_y = amp_plot_y[sel]
# # pha_plot_x = pha_plot_y[sel]
 
 
N = len(plot_x)  # No. of samples
T = plot_x[-1] - plot_x[0]  # Total Time
S = N/T  # Sample Rate
 
print(f"{N=}, {S=}, {T=}")
 
# Low pass filter (Butterworth)
# sos = signal.cheby1(5, 5, 5, fs=S, output="sos")
amp_plot_y2 = signal.savgol_filter(amp_plot_y, 51, 5)
sos = signal.butter(5, 10, fs=S, output="sos")
amp_plot_y2 = signal.sosfiltfilt(sos, amp_plot_y2)
 
pha_plot_y2 = signal.savgol_filter(pha_plot_y, 51, 5)
sos = signal.butter(5, 10, fs=S, output="sos")
pha_plot_y2 = signal.sosfiltfilt(sos, pha_plot_y2)
 
fig = plt.figure(figsize=(19, 10), dpi=100)
# ax1 = fig.add_subplot(311)
# ax1.set_title("Raw data")
# ax1.set_xticks(np.arange(0, 40, 1))
ax2 = fig.add_subplot(211)
ax2.set_title("Amplitude")
ax2.set_xticks(np.arange(0, 35, 1))
# ax1.scatter(plot_x, amp_plot_y)
ax2.plot(plot_x, amp_plot_y2)
ax2.grid()
ax3 = fig.add_subplot(212)
ax3.set_title("Phase")
ax3.set_xticks(np.arange(0, 35, 1))
ax3.plot(plot_x, pha_plot_y2)
ax3.grid()
# ax1.plot(freq_plot_x, np.abs(freq_plot_y))
plt.show()
