# effective_rate.py
import numpy as np
import matplotlib.pyplot as plt

# 1) Εισαγωγή baseline BER δεδομένων (2x1 και 4x1) από το paper
SNR_dBs    = np.arange(0, 21, 2)
ber_2x1    = np.array([0.3955, 0.3987, 0.4069, 0.4103, 0.4106,
                       0.4071, 0.4149, 0.4147, 0.4143, 0.4208, 0.4103])
ber_4x1    = np.array([0.4393, 0.4365, 0.4398, 0.4439, 0.4378,
                       0.4389, 0.4432, 0.4406, 0.4422, 0.4410, 0.4372])

# 2) Υπολογισμός Effective Rate = (n_spatial+1)*(1 - BER)
rate_max_2x1 = 2  # 1 spatial + 1 BPSK bit
rate_max_4x1 = 3  # 2 spatial + 1 BPSK bit

rate_eff_2x1 = rate_max_2x1 * (1 - ber_2x1)
rate_eff_4x1 = rate_max_4x1 * (1 - ber_4x1)

# 3) Plot Effective Rate vs SNR
plt.figure()
plt.plot(SNR_dBs, rate_eff_2x1, marker='o', label='2×1 Effective Rate')
plt.plot(SNR_dBs, rate_eff_4x1, marker='s', label='4×1 Effective Rate')
plt.xlabel('SNR [dB]')
plt.ylabel('Effective Rate [bits/use]')
plt.title('Effective Spectral Efficiency vs SNR')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 4) Save & show
plt.savefig('effective_rate.png')
plt.show()

# 5) Save data to CSV
import csv
with open('effective_rate.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['SNR_dB', 'Rate_2x1', 'Rate_4x1'])
    for i, snr in enumerate(SNR_dBs):
        writer.writerow([snr, rate_eff_2x1[i], rate_eff_4x1[i]])
print("Saved effective_rate.png and effective_rate.csv")
