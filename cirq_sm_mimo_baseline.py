# cirq_sm_mimo_baseline.py
import cirq
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- Παράμετροι πειράματος ---
SNR_dBs   = np.arange(0, 21, 2)
shots     = 5000
gamma     = np.pi/4
beta      = np.pi/4
# cases: n_qubits → n_spatial bits
cases     = {2: 1, 3: 2}


def cost_hamiltonian(y, h):
    """Διαγώνια cost-Hamiltonian για λαμβανόμενο y και channel vector h."""
    n_ant = len(h)
    dim   = 2 * n_ant
    C = np.zeros(dim)
    idx = 0
    for ant in range(n_ant):
        for bval in (+1, -1):
            C[idx] = abs(y - h[ant] * bval) ** 2
            idx += 1
    return np.diag(C)


def build_qaoa_circuit(n_qubits, HC):
    """p=1 fixed-angle QAOA circuit σε Cirq."""
    qubits = cirq.LineQubit.range(n_qubits)
    qc = cirq.Circuit()
    qc.append(cirq.H.on_each(*qubits))                        # Init
    Uc = expm(-1j * gamma * HC)
    qc.append(cirq.MatrixGate(Uc).on(*qubits))                # Cost layer
    for q in qubits:
        qc.append(cirq.rx(2 * beta).on(q))                    # Mixer layer
    qc.append(cirq.measure(*qubits, key='m'))
    return qc


def simulate_case(n_qubits, n_spatial):
    """Γυρνά BER vs SNR για δεδομένο p=1 fixed-angle QAOA."""
    n_ant = 2 ** n_spatial
    bers  = []
    sim   = cirq.Simulator()

    for snr_db in SNR_dBs:
        errors = 0
        snr    = 10 ** (snr_db / 10)
        sigma  = np.sqrt(1 / (2 * snr))
        for _ in range(shots):
            h = (np.random.randn(n_ant) + 1j * np.random.randn(n_ant)) / np.sqrt(2)
            y = np.sum(h) + sigma * (np.random.randn() + 1j * np.random.randn())
            HC = cost_hamiltonian(y, h)
            qc = build_qaoa_circuit(n_qubits, HC)
            result = sim.run(qc, repetitions=1)
            bits = result.measurements['m'][0]
            ant_est  = int("".join(str(b) for b in bits[:-1]), 2)
            bpsk_est = bits[-1]
            ant_true = np.argmax(np.abs(h))
            bpsk_true= 0 if np.real(np.conj(h[ant_true]) * y) >= 0 else 1
            errors += (ant_est != ant_true) + (bpsk_est != bpsk_true)
        bers.append(errors / (2 * shots))
        print(f"{n_qubits}q p=1 SNR={snr_db} dB -> BER={bers[-1]:.3e}")
    return bers


def main():
    plt.figure()
    for n_qubits, n_spatial in cases.items():
        bers = simulate_case(n_qubits, n_spatial)
        label = f"{2**n_spatial}x1"
        plt.semilogy(SNR_dBs, bers, marker='o', label=label)
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("Fixed-angle QAOA SM–MIMO (Cirq) – Baseline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ber_vs_snr_baseline.png')
    plt.show()


if __name__ == "__main__":
    main()
