# mi_estimate.py
import numpy as np
import matplotlib.pyplot as plt
import csv

def estimate_mutual_info(snr_db, n_ant, n_samples=5000):
    """
    Approximate I(X;Y) for SM-MIMO with n_ant transmit antennas (2 or 4),
    using uniform input over 2*n_ant symbols, AWGN channel.
    """
    snr   = 10**(snr_db/10)
    sigma = np.sqrt(1/(2 * snr))
    symbols = [(i, b) for i in range(n_ant) for b in (+1, -1)]
    M = len(symbols)
    total = 0.0
    for i, (ant_true, b_true) in enumerate(symbols):
        # sample h once per symbol
        h = (np.random.randn(n_ant) + 1j*np.random.randn(n_ant)) / np.sqrt(2)
        for _ in range(n_samples):
            y = h[ant_true]*b_true + sigma*(np.random.randn()+1j*np.random.randn())
            # p(y|x') for all x'
            pyx = np.array([
                (1/(np.pi*sigma**2)) * np.exp(-abs(y - h[ant2]*b2)**2 / sigma**2)
                for ant2, b2 in symbols
            ])
            py = pyx.mean()
            total += np.log2(pyx[i] / py)
    return total / (M * n_samples)

if __name__ == "__main__":
    # 1) Define SNR range and prepare storage
    SNRs = np.arange(0, 21, 5)          # [0,5,10,15,20]
    I2 = []  # I for 2×1
    I4 = []  # I for 4×1

    # 2) Estimate MI
    for snr_db in SNRs:
        print(f"Estimating I at {snr_db} dB...")
        i2 = estimate_mutual_info(snr_db, n_ant=2, n_samples=2000)
        i4 = estimate_mutual_info(snr_db, n_ant=4, n_samples=2000)
        I2.append(i2)
        I4.append(i4)
        print(f"  I_2x1={i2:.3f}, I_4x1={i4:.3f}")

    # 3) Plot
    plt.figure()
    plt.plot(SNRs, I2, marker='o', label='I 2×1')
    plt.plot(SNRs, I4, marker='s', label='I 4×1')
    plt.xlabel('SNR [dB]')
    plt.ylabel('Mutual Information [bits/use]')
    plt.title('Estimated I(X;Y) vs SNR (SM–MIMO)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mi_vs_snr.png')
    plt.show()

    # 4) Save to CSV
    with open('mi_vs_snr.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR_dB', 'I_2x1', 'I_4x1'])
        for i, snr_db in enumerate(SNRs):
            writer.writerow([snr_db, I2[i], I4[i]])
    print("Saved mi_vs_snr.png and mi_vs_snr.csv")
