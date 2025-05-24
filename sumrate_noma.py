# sumrate_noma.py
import numpy as np
import matplotlib.pyplot as plt
import csv

# Παράμετροι
SNR_dBs = np.arange(0,21,2)
P = 1.0
alpha = 0.8        # power fraction to User 1
N = 5000           # Monte Carlo draws

# Προετοιμασία αποθήκευσης
R1_list, R2_list, Rsum_list = [], [], []

for snr_db in SNR_dBs:
    snr = 10**(snr_db/10)
    N0  = 1/snr
    # Generate N Rayleigh gains for each user
    h1 = (np.random.randn(N)+1j*np.random.randn(N))/np.sqrt(2)
    h2 = (np.random.randn(N)+1j*np.random.randn(N))/np.sqrt(2)
    # SINRs
    sinr1 = alpha * np.abs(h1)**2 / ((1-alpha)*np.abs(h1)**2 + N0)
    sinr2 = (1-alpha) * np.abs(h2)**2 / N0
    # Rates
    R1 = np.log2(1 + sinr1)
    R2 = np.log2(1 + sinr2)
    # Μέσοι όροι
    R1_list.append(R1.mean())
    R2_list.append(R2.mean())
    Rsum_list.append((R1 + R2).mean())

# Plot
plt.figure()
plt.plot(SNR_dBs, R1_list, marker='o', label='R1 (strong)')
plt.plot(SNR_dBs, R2_list, marker='s', label='R2 (weak)')
plt.plot(SNR_dBs, Rsum_list, marker='^', label='R_sum')
plt.xlabel('SNR [dB]')
plt.ylabel('Rate [bits/s/Hz]')
plt.title('Sum-Rate NOMA 2-user vs SNR (α=0.8)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('sumrate_noma.png')
plt.show()

# Save CSV
with open('sumrate_noma.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['SNR_dB','R1','R2','Rsum'])
    for i, snr_db in enumerate(SNR_dBs):
        writer.writerow([snr_db, R1_list[i], R2_list[i], Rsum_list[i]])
print("Saved sumrate_noma.png and sumrate_noma.csv")
