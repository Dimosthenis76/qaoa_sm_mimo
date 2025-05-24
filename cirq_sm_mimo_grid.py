# cirq_sm_mimo_grid.py
import cirq
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import csv

# --- Παράμετροι grid-search και sweep ---
shots_tune  = 500     # μειωμένες μετρήσεις για tuning
shots_sweep = 5000    # πλήρεις μετρήσεις για SNR sweep
SNR_tune    = 20      # SNR που χρησιμοποιείται για tuning
SNR_sweep   = np.arange(0, 21, 2)
cases       = {2:1, 3:2}  # n_qubits -> n_spatial bits

# Grid τιμές για γ και β
gamma_list = np.linspace(0, np.pi/2, 5)  # [0, π/8, π/4, 3π/8, π/2]
beta_list  = np.linspace(0, np.pi/2, 5)


def cost_hamiltonian(y, h):
    n_ant = len(h)
    dim   = 2 * n_ant
    C = np.zeros(dim)
    idx = 0
    for ant in range(n_ant):
        for bval in (+1, -1):
            C[idx] = abs(y - h[ant] * bval)**2
            idx += 1
    return np.diag(C)


def build_qaoa_circuit(n_qubits, gamma, beta, HC):
    qubits = cirq.LineQubit.range(n_qubits)
    qc = cirq.Circuit()
    qc.append(cirq.H.on_each(*qubits))
    Uc = expm(-1j * gamma * HC)
    qc.append(cirq.MatrixGate(Uc).on(*qubits))
    for q in qubits:
        qc.append(cirq.rx(2 * beta).on(q))
    qc.append(cirq.measure(*qubits, key='m'))
    return qc


def simulate_single(n_qubits, n_spatial, gamma, beta, snr_db, shots):
    """Επιστρέφει BER σε ένα μόνο SNR, με συγκεκριμένες gamma,beta, shots."""
    n_ant = 2 ** n_spatial
    errors = 0
    snr    = 10**(snr_db / 10)
    sigma  = np.sqrt(1 / (2 * snr))
    sim    = cirq.Simulator()

    for _ in range(shots):
        h  = (np.random.randn(n_ant) + 1j*np.random.randn(n_ant)) / np.sqrt(2)
        y  = np.sum(h) + sigma * (np.random.randn() + 1j*np.random.randn())
        HC = cost_hamiltonian(y, h)
        qc = build_qaoa_circuit(n_qubits, gamma, beta, HC)
        res= sim.run(qc, repetitions=1)
        bits = res.measurements['m'][0]
        ant_est  = int("".join(str(b) for b in bits[:-1]), 2)
        bpsk_est = bits[-1]
        ant_true = np.argmax(np.abs(h))
        bpsk_true= 0 if np.real(np.conj(h[ant_true]) * y) >= 0 else 1
        errors += (ant_est != ant_true) + (bpsk_est != bpsk_true)

    return errors / (2 * shots)


def main():
    # --- 1) Grid-search tuning ---
    best_angles = {}
    for n_qubits, n_spatial in cases.items():
        print(f"Tuning for {n_qubits} qubits at {SNR_tune} dB:")
        min_ber = 1.0
        best    = (None, None)
        for g in gamma_list:
            for b in beta_list:
                print(f"  Testing γ={g:.3f}, β={b:.3f} ... ", end='', flush=True)
                ber = simulate_single(n_qubits, n_spatial, g, b, SNR_tune, shots_tune)
                print(f"BER={ber:.3f}")
                if ber < min_ber:
                    min_ber = ber
                    best    = (g, b)
        best_angles[n_qubits] = best
        print(f"-> Best: γ*={best[0]:.3f}, β*={best[1]:.3f}, BER*={min_ber:.3f}\n")

    # --- 2) Full SNR sweep με tuned angles ---
    all_data = {'SNR_dB': list(SNR_sweep)}
    plt.figure()
    for n_qubits, n_spatial in cases.items():
        g_opt, b_opt = best_angles[n_qubits]
        bers = []
        for snr_db in SNR_sweep:
            ber = simulate_single(n_qubits, n_spatial, g_opt, b_opt, snr_db, shots_sweep)
            bers.append(ber)
        label = f"{2**n_spatial}x1 tuned"
        plt.semilogy(SNR_sweep, bers, marker='o', label=label)
        all_data[label] = bers

    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("Grid-search Fixed-angle QAOA SM–MIMO (Cirq)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # --- Αποθήκευση diagram ---
    plt.savefig('ber_vs_snr_tuned.png')
    plt.show()

    # --- Αποθήκευση σε CSV (Excel) ---
    with open('ber_vs_snr_tuned.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['SNR_dB'] + [key for key in all_data if key!='SNR_dB']
        writer.writerow(headers)
        for i in range(len(SNR_sweep)):
            row = [all_data['SNR_dB'][i]] + [all_data[k][i] for k in headers[1:]]
            writer.writerow(row)
    print("Saved 'ber_vs_snr_tuned.png' and 'ber_vs_snr_tuned.csv'.")


if __name__ == "__main__":
    main()
