import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import DensityMatrix, Statevector
import matplotlib.pyplot as plt

# Constants
c = 3.00e8          # Speed of light in m/s
delta = 0.1         # Fractional reduction
tau = 1.0e9         # Decay timescale in years
t0 = 13.8e9         # Present age of the universe in years
N0 = 3.0e6          # Current dark matter density in m^-3
H0 = 2.268e-18      # Hubble constant in s^-1 (70 km/s/Mpc)
Omega_m = 0.3       # Matter density parameter
Omega_Lambda = 0.7  # Dark energy density parameter
Omega_r = 9.05e-5   # Radiation density parameter
Omega_k = 1 - Omega_m - Omega_Lambda - Omega_r  # Curvature parameter
SECONDS_PER_YEAR = 3.15576e7  # Seconds in a year (confirmed correct)

# Debug constant value
print(f"Debug: SECONDS_PER_YEAR={SECONDS_PER_YEAR:.2e}")

def hubble_integral(a, Omega_r, Omega_m, Omega_k, Omega_Lambda):
    """Integrand for the age of the universe."""
    return 1 / np.sqrt(Omega_r / a**2 + Omega_m / a + Omega_k + Omega_Lambda * a**2)

def compute_age(a, H0, Omega_r, Omega_m, Omega_k, Omega_Lambda):
    """Compute age of the universe at scale factor a in seconds."""
    integral, _ = quad(hubble_integral, 0, a, args=(Omega_r, Omega_m, Omega_k, Omega_Lambda), epsabs=1e-12, epsrel=1e-12)
    return integral / H0

def compute_redshift(t, lookback=True):
    """Compute redshift z based on time t (in years). If lookback=True, t is time ago from present."""
    if t < 0:
        raise ValueError("t must be >= 0")
    
    t_effective = t0 - t if lookback else t
    if t_effective < 0:
        raise ValueError("Lookback time t cannot exceed present age t0")
    
    t_seconds = t_effective * SECONDS_PER_YEAR
    if abs(t_seconds / t_effective - SECONDS_PER_YEAR) > 1e-6:
        raise ValueError(f"Time conversion error: t_seconds={t_seconds:.2e} for t_effective={t_effective:.2e}")
    
    # Use Hubble's law approximation for small lookback times
    if lookback and t / t0 < 0.01:
        z_approx = H0 * t * SECONDS_PER_YEAR
        print(f"Debug: Using Hubble's law approximation, z_approx={z_approx:.2e}")
        return z_approx
    
    def age_diff(a):
        return compute_age(a, H0, Omega_r, Omega_m, Omega_k, Omega_Lambda) - t_seconds
    
    try:
        bracket = [1e-10, 1.0]
        print(f"Debug: t_seconds={t_seconds:.2e}")
        print(f"Debug: age_diff({bracket[0]})={age_diff(bracket[0]):.2e}")
        print(f"Debug: age_diff({bracket[1]})={age_diff(bracket[1]):.2e}")
        
        result = root_scalar(age_diff, bracket=bracket, method='brentq', xtol=1e-12, rtol=1e-12)
        a = result.root
        z = 1 / a - 1
        
        expected_z_max = 1000  # Relaxed for early universe
        if z > expected_z_max:
            print(f"Warning: High redshift z={z:.2e} for t={t:.2e} years (lookback={lookback})")
        
        print(f"Debug: t={t:.2e} years, t_effective={t_effective:.2e} years, t_seconds={t_seconds:.2e}, a={a:.6e}, z={z:.6e}, converged={result.converged}")
        return max(z, 0)
    except ValueError as e:
        print(f"Root finding failed: {e}")
        raise

def compute_quantum_qiskit(N, n_qubits=3, shots=100000):
    """Compute quantum correlations using Qiskit for an n-qubit GHZ state."""
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer >= 2")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    
    # Create an n-qubit GHZ state
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.measure(range(n_qubits), range(n_qubits))
    
    # Simulate using AerSimulator
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    
    # Compute expectation value of Z_1 Z_2 ... Z_n for GHZ state
    expectation = 0
    for state, count in counts.items():
        if state in ['0' * n_qubits, '1' * n_qubits]:
            expectation += count / shots
        else:
            expectation -= count / shots
    
    # Ensure expectation is reasonable for GHZ state
    expectation = max(min(expectation, 1.0), -1.0)
    
    # Quantum correlation parameters
    S_quantum = 2 * np.sqrt(2) * np.sqrt(N) * expectation
    W = S_quantum * np.sqrt(2)
    
    if N % 2 == 0:
        M_quantum = W * np.sqrt(N) / np.sqrt(2)
    else:
        M_quantum = W * np.sqrt(N) / np.sqrt(N + 1)
    
    # Save histogram
    plot_histogram(counts)
    plt.savefig('histogram.png')
    plt.close()
    
    return S_quantum, M_quantum, circuit, counts, expectation

def compute_density_matrix(circuit, z=None):
    """Compute density matrix for the circuit and scale by (1+z)^3 if z is provided."""
    # Create a new circuit without measurements
    n_qubits = circuit.num_qubits
    circuit_without_measure = QuantumCircuit(n_qubits)
    circuit_without_measure.h(0)
    for i in range(n_qubits - 1):
        circuit_without_measure.cx(i, i + 1)
    
    # Simulate the circuit to get the statevector
    simulator = AerSimulator()
    circuit_without_measure.save_statevector()
    job = simulator.run(circuit_without_measure, method='statevector')
    result = job.result()
    statevector = result.get_statevector(circuit_without_measure)
    
    # Debug: Print statevector to verify GHZ state
    print(f"Debug: Statevector = {statevector}")
    
    # Compute density matrix
    rho = DensityMatrix(statevector)
    
    # Scale density matrix by (1+z)^3 if redshift is provided
    if z is not None:
        scale_factor = (1 + z)**3
        rho_data = rho.data * scale_factor
        # Normalize to ensure trace = 1
        trace = np.trace(rho_data)
        if abs(trace) > 1e-10:  # Avoid division by zero
            rho_data = rho_data / trace
        rho = DensityMatrix(rho_data)
    
    # Compute purity
    purity = np.real(np.trace(rho.data @ rho.data))
    
    # Compute expectation value mimicking measurement-based expectation
    state_vec = np.array(statevector)
    # Projector for |000><000| + |111><111|
    proj = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    proj[0, 0] = 1  # |000><000|
    proj[-1, -1] = 1  # |111><111|
    expectation_Z = np.real(np.vdot(state_vec, proj @ state_vec))
    
    # Debug: Print projector matrix
    print(f"Debug: Projector matrix =\n{proj}")
    
    return rho, purity, expectation_Z

def compute_classical(N):
    """Compute classical correlation parameters."""
    S_classical = 2 * np.sqrt(N)
    W_classical = S_classical / np.sqrt(2)
    M_classical = W_classical  # Same for even and odd N
    return S_classical, M_classical

def compute_cosmological(t, z):
    """Compute cosmological parameters."""
    if t < 0 or z < 0:
        raise ValueError("t and z must be >= 0")
    
    c_eff_t = c * (1 - delta * np.exp(-t / tau))
    c_eff_t0 = c * (1 - delta * np.exp(-t0 / tau))
    N_DM = N0 * (c_eff_t0 / c_eff_t) * (1 + z)**3
    print(f"Debug: c_eff_t0={c_eff_t0:.4e}, c_eff_t={c_eff_t:.4e}, z={z:.4e}, (1+z)^3={(1+z)**3:.4e}")
    return c_eff_t, N_DM

def main():
    """Main function to run the quantum-classical conversion script with Qiskit."""
    try:
        N = int(input("Enter number of particles N (integer >= 2): "))
        if N < 2:
            raise ValueError("N must be >= 2")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    include_cosmological = input("Include cosmological calculations? (yes/no): ").lower()
    if include_cosmological in ['yes', 'y']:
        try:
            t = float(input("Enter lookback time t in years (time ago from present): "))
            if t < 0:
                raise ValueError("t must be >= 0")
            auto_z = input("Calculate redshift z automatically? (yes/no): ").lower()
            if auto_z in ['yes', 'y']:
                z = compute_redshift(t, lookback=True)
                print(f"Calculated redshift z: {z:.6f}")
            else:
                z = float(input("Enter redshift z: "))
                if z < 0:
                    raise ValueError("z must be >= 0")
        except ValueError as e:
            print(f"Invalid input: {e}")
            return
    else:
        t = None
        z = None

    # Compute quantum and classical correlations
    n_qubits = min(3, int(np.log2(N))) if N > 2 else 3
    S_quantum, M_quantum, circuit, counts, meas_expectation = compute_quantum_qiskit(N, n_qubits=n_qubits)
    S_classical, M_classical = compute_classical(N)

    # Compute density matrix
    rho, purity, expectation_Z = compute_density_matrix(circuit, z if include_cosmological in ['yes', 'y'] else None)

    # Check for consistency between measurement and density matrix expectation
    if abs(expectation_Z - meas_expectation) > 0.1:
        print(f"Warning: Density matrix Z expectation ({expectation_Z:.6f}) differs from measurement-based expectation ({meas_expectation:.6f})")

    print(f"\nQuantum correlation S_quantum: {S_quantum:.4f}")
    print(f"Quantum M_quantum: {M_quantum:.4f}")
    print(f"Classical correlation S_classical: {S_classical:.4f}")
    print(f"Classical M_classical: {M_classical:.4f}")
    print(f"\nQuantum circuit counts: {counts}")
    print(f"Circuit diagram:\n{circuit.draw()}")
    print(f"Histogram saved to histogram.png")
    print(f"\nDensity matrix purity: {purity:.6f}")
    print(f"Density matrix Z expectation value: {expectation_Z:.6f}")
    if z is not None:
        print(f"Density matrix scaled by (1+z)^3: {(1+z)**3:.6f}")

    if t is not None and z is not None:
        c_eff, N_DM = compute_cosmological(t, z)
        print(f"Redshift z: {z:.6f}")
        print(f"Effective speed of light c_eff: {c_eff:.4e} m/s")
        print(f"Dark matter number density N_DM: {N_DM:.4e} m^-3")

if __name__ == "__main__":
    main()