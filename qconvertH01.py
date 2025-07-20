import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
import matplotlib.pyplot as plt
from functools import lru_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
c = 3.00e8          # Speed of light in m/s
delta = 0.1         # Fractional reduction for quantum resistance
tau = 1.0e9         # Decay timescale in years
t0 = 12.8e9         # Present age of the universe in years
c_eff_t0 = c * (1 - delta * np.exp(-t0 / tau))  # Precompute c_eff at t0
N0 = 3.0e6          # Current dark matter density in m^-3
H0_default = 2.268e-18  # Hubble constant in s^-1 (70 km/s/Mpc)
H0_alternatives = [2.184e-18, 2.366e-18]  # 67.4 and 73 km/s/Mpc for Hubble tension
Omega_m = 0.3       # Matter density parameter
Omega_Lambda = 0.7  # Dark energy density parameter
Omega_r = 9.05e-5   # Radiation density parameter
Omega_total = 1.000090  # Total density for closed universe
Omega_k = 1 - Omega_m - Omega_Lambda - Omega_r  # Curvature parameter
SECONDS_PER_YEAR = 3.15576e7  # Seconds in a year
KM_PER_MPC = 3.0857e19  # Kilometers per megaparsec

# Conversion factor from s^-1 to km/s/Mpc
conversion_factor = 70 / H0_default  # Based on H0_default = 2.268e-18 s^-1 = 70 km/s/Mpc

# Debug constant values
logger.info(f"SECONDS_PER_YEAR={SECONDS_PER_YEAR:.2e}")
logger.info(f"Omega_r + Omega_m + Omega_Lambda = {Omega_r + Omega_m + Omega_Lambda:.6f} (closed universe, Omega_k={Omega_k:.6e})")
logger.info(f"Precomputed c_eff_t0={c_eff_t0:.4e} m/s")

@lru_cache(maxsize=10000)
def hubble_integral(a, Omega_r, Omega_m, Omega_k, Omega_Lambda):
    """Integrand for the age of the universe, adjusted for closed universe."""
    term = Omega_r / a**2 + Omega_m / a + Omega_k + Omega_Lambda * a**2
    if term <= 0:
        logger.warning(f"Negative or zero integrand at a={a:.6e}, setting term to small positive value")
        term = 1e-20
    return 1 / np.sqrt(term)

def compute_age(a, H0, Omega_r, Omega_m, Omega_k, Omega_Lambda):
    """Compute age of the universe at scale factor a in seconds."""
    if a < 1e-10:
        integral1, err1 = quad(hubble_integral, 0, 1e-10, args=(Omega_r, Omega_m, Omega_k, Omega_Lambda), epsabs=1e-15, epsrel=1e-15, limit=500)
        integral2, err2 = quad(hubble_integral, 1e-10, a, args=(Omega_r, Omega_m, Omega_k, Omega_Lambda), epsabs=1e-15, epsrel=1e-15, limit=500)
        integral = integral1 + integral2
        err = err1 + err2
    elif a < 1e-5:
        integral1, err1 = quad(hubble_integral, 0, 1e-5, args=(Omega_r, Omega_m, Omega_k, Omega_Lambda), epsabs=1e-15, epsrel=1e-15, limit=500)
        integral2, err2 = quad(hubble_integral, 1e-5, a, args=(Omega_r, Omega_m, Omega_k, Omega_Lambda), epsabs=1e-15, epsrel=1e-15, limit=500)
        integral = integral1 + integral2
        err = err1 + err2
    else:
        integral, err = quad(hubble_integral, 0, a, args=(Omega_r, Omega_m, Omega_k, Omega_Lambda), epsabs=1e-15, epsrel=1e-15, limit=1000)
    
    if err > 1e-8:
        logger.warning(f"High integration error at a={a:.6e}, err={err:.2e}")
    return integral / H0

def compute_redshift(t, lookback=True, H0_value=H0_default):
    """Compute redshift z based on time t (in years). If lookback=True, t is time ago from present."""
    if t < 0:
        raise ValueError("t must be >= 0")
    
    t_effective = t0 - t if lookback else t
    if t_effective < 0:
        raise ValueError("Lookback time t cannot exceed present age t0")
    
    t_seconds = t_effective * SECONDS_PER_YEAR
    if abs(t_seconds / t_effective - SECONDS_PER_YEAR) > 1e-6:
        raise ValueError(f"Time conversion error: t_seconds={t_seconds:.2e} for t_effective={t_effective:.2e}")
    
    if lookback and t / t0 < 0.01:
        z_approx = H0_value * t * SECONDS_PER_YEAR
        logger.info(f"Using Hubble's law approximation, z_approx={z_approx:.2e}")
        return z_approx
    
    def age_diff(a):
        return compute_age(a, H0_value, Omega_r, Omega_m, Omega_k, Omega_Lambda) - t_seconds
    
    try:
        bracket = [1e-10, 1.0]
        logger.info(f"t={t:.2e} years, t_effective={t_effective:.2e} years, t_seconds={t_seconds:.2e}")
        result = root_scalar(age_diff, bracket=bracket, method='brentq', xtol=1e-12, rtol=1e-12)
        a = result.root
        z = 1 / a - 1
        
        expected_z_max = 1000
        if z > expected_z_max:
            logger.warning(f"High redshift z={z:.2e} for t={t:.2e} years (lookback={lookback})")
        
        logger.info(f"a={a:.6e}, z={z:.6e}, converged={result.converged}, iterations={result.iterations}")
        return max(z, 0)
    except ValueError as e:
        logger.error(f"Root finding failed: {e}")
        raise

def compute_quantum_qiskit(N, n_qubits=3, shots=100000):
    """Compute quantum correlations using Qiskit for an n-qubit GHZ state."""
    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer >= 2")
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.measure(range(n_qubits), range(n_qubits))
    
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit)
    
    expectation = 0
    for state, count in counts.items():
        if state in ['0' * n_qubits, '1' * n_qubits]:
            expectation += count / shots
        else:
            expectation -= count / shots
    
    expectation = max(min(expectation, 1.0), -1.0)
    
    S_quantum = 2 * np.sqrt(2) * np.sqrt(N) * expectation
    W = S_quantum / np.sqrt(2)
    
    if N % 2 == 0:
        M_quantum = W * np.sqrt(N)
    else:
        M_quantum = W * np.sqrt(N) / np.sqrt(N + 1)
    
    plot_histogram(counts)
    plt.savefig('histogram.png')
    plt.close()
    
    return S_quantum, M_quantum, circuit, counts

def compute_density_matrix(circuit, z=None):
    """Compute density matrix for the circuit and scale by (1+z)^3 if z is provided."""
    n_qubits = circuit.num_qubits
    circuit_without_measure = QuantumCircuit(n_qubits)
    circuit_without_measure.h(0)
    for i in range(n_qubits - 1):
        circuit_without_measure.cx(i, i + 1)
    
    simulator = AerSimulator()
    circuit_without_measure.save_statevector()
    job = simulator.run(circuit_without_measure, method='statevector')
    result = job.result()
    statevector = result.get_statevector(circuit_without_measure)
    
    logger.info(f"Statevector = {statevector}")
    
    rho = DensityMatrix(statevector)
    
    if z is not None:
        scale_factor = (1 + z)**3
        rho_data = rho.data * scale_factor
        trace = np.trace(rho_data)
        if abs(trace) > 1e-10:
            rho_data = rho_data / trace
        rho = DensityMatrix(rho_data)
    
    purity = np.real(np.trace(rho.data @ rho.data))
    
    state = Statevector(statevector)
    Z = np.array([[1, 0], [0, -1]])
    Z_op = Operator.from_label('Z')  # Define the Pauli Z operator
    Z_total = Z_op
    for _ in range(n_qubits - 1):
        Z_total = Z_total.tensor(Z_op)  # Tensor product for all qubits
    expectation_Z = state.expectation_value(Z_total)
    
    logger.info(f"Z_total matrix =\n{Z_total.data}")
    

    return rho, purity, expectation_Z, Z_total  # Add Z_total to return values


def compute_classical(N):
    """Compute classical correlation parameters."""
    S_classical = 2 * np.sqrt(N)
    W_classical = S_classical / np.sqrt(2)
    M_classical = W_classical
    return S_classical, M_classical

def compute_cosmological(t, z):
    """Compute cosmological parameters."""
    if t < 0 or z < 0:
        raise ValueError("t and z must be >= 0")
    
    c_eff_t = c * (1 - delta * np.exp(-t / tau))
    N_DM = N0 * (c_eff_t0 / c_eff_t) * (1 + z)**3
    logger.info(f"c_eff_t0={c_eff_t0:.4e}, c_eff_t={c_eff_t:.4e}, z={z:.4e}, (1+z)^3={(1+z)**3:.4e}")
    return c_eff_t, N_DM

def main():
    """Main function to run the quantum-classical conversion script with Qiskit."""
    results = []
    try:
        N_input = input("Enter number of particles N (integer >= 2, default 2): ").strip()
        N = int(N_input) if N_input else 2
        if N < 2:
            raise ValueError("N must be >= 2")
    except ValueError as e:
        logger.error(f"Invalid N input: {e}")
        return

    # Dynamically assign n_qubits based on N
    n_qubits = N
    logger.info(f"Selected n_qubits={n_qubits} for N={N}")
    
    # Warn if simulating many qubits
    if n_qubits > 10:
        logger.warning(f"Simulating {n_qubits} qubits may be computationally intensive on classical hardware.")

    include_cosmological = input("Include cosmological calculations? (yes/no, default no): ").lower()
    t, z_values = None, []
    if include_cosmological in ['yes', 'y']:
        try:
            t_input = input("Enter lookback time t in years (time ago from present, default 1e9): ").strip()
            t = float(t_input) if t_input else 1e9
            if t < 0:
                raise ValueError("t must be >= 0")
            auto_z = input("Calculate redshift z automatically? (yes/no, default yes): ").lower()
            if auto_z in ['yes', 'y', '']:
                for h0 in [H0_default] + H0_alternatives:
                    z = compute_redshift(t, lookback=True, H0_value=h0)
                    h0_km_s_Mpc = h0 * conversion_factor
                    z_values.append((h0, z, h0_km_s_Mpc))
                    logger.info(f"Calculated redshift z={z:.6f} for H0={h0_km_s_Mpc:.2f} km/s/Mpc")
            else:
                z_input = input("Enter redshift z (default 0.130646): ").strip()
                z = float(z_input) if z_input else 0.130646
                if z < 0:
                    raise ValueError("z must be >= 0")
                h0_km_s_Mpc = H0_default * conversion_factor
                z_values.append((H0_default, z, h0_km_s_Mpc))
        except ValueError as e:
            logger.error(f"Invalid cosmological input: {e}")
            return
    else:
        z_values.append((H0_default, None, H0_default * conversion_factor))

    S_quantum, M_quantum, circuit, counts = compute_quantum_qiskit(N, n_qubits=n_qubits)
    S_classical, M_classical = compute_classical(N)
    rho, purity, expectation_Z, Z_total = compute_density_matrix(circuit, z_values[0][1] if z_values and z_values[0][1] is not None else None)
    eigenvalues = np.linalg.eigvals(Z_total.data)
    logger.info(f"Z_total eigenvalues: {eigenvalues}")
    meas_expectation = sum(counts.get(state, 0) / sum(counts.values()) for state in ['0' * n_qubits, '1' * n_qubits])
    if abs(expectation_Z - meas_expectation) > 0.1:
        eigenvalues = np.linalg.eigvals(Z_total.data)
        logger.info(f"Z_total eigenvalues: {eigenvalues}")

    result = {
        'N': N,
        'n_qubits': n_qubits,
        'S_quantum': S_quantum,
        'M_quantum': M_quantum,
        'S_classical': S_classical,
        'M_classical': M_classical,
        'circuit_counts': counts,
        'circuit_diagram': str(circuit.draw()),
        'purity': purity,
        'expectation_Z': expectation_Z,
        'meas_expectation': meas_expectation
    }
    logger.info(f"Quantum correlation S_quantum: {S_quantum:.4f}")
    logger.info(f"Quantum M_quantum: {M_quantum:.4f}")
    logger.info(f"Classical correlation S_classical: {S_classical:.4f}")
    logger.info(f"Classical M_classical: {M_classical:.4f}")
    logger.info(f"Quantum circuit counts: {counts}")
    logger.info(f"Circuit diagram:\n{circuit.draw()}")
    logger.info(f"Histogram saved to histogram.png")
    logger.info(f"Density matrix purity: {purity:.6f}")
    logger.info(f"Density matrix Z expectation value: {expectation_Z:.6f}")
    if z_values and z_values[0][1] is not None:
        logger.info(f"Density matrix scaled by (1+z)^3: {(1+z_values[0][1])**3:.6f}")
    results.append(result)

    if t is not None and z_values:
        for h0, z, h0_km_s_Mpc in z_values:
            if z is not None:
                c_eff, N_DM = compute_cosmological(t, z)
                cosmo_result = {
                    'H0_s_inv': h0,
                    'H0_km_s_Mpc': h0_km_s_Mpc,
                    'z': z,
                    'c_eff': c_eff,
                    'N_DM': N_DM
                }
                logger.info(f"H0={h0_km_s_Mpc:.2f} km/s/Mpc")  # Fixed typo from "= 70"
                logger.info(f"Effective speed of light c_eff: {c_eff:.4e} m/s")
                logger.info(f"Dark matter number density N_DM: {N_DM:.4e} m^-3")
                results.append(cosmo_result)

    # Save results to file (unchanged except for updated n_qubits)
    with open('qconvert_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Results for N={N}, n_qubits={n_qubits}\n")
        f.write(f"Quantum S_quantum: {S_quantum:.4f}\n")
        f.write(f"Quantum M_quantum: {M_quantum:.4f}\n")
        f.write(f"Classical S_classical: {S_classical:.4f}\n")
        f.write(f"Classical M_classical: {M_classical:.4f}\n")
        f.write(f"Quantum circuit counts: {counts}\n")
        f.write(f"Circuit diagram:\n{circuit.draw()}\n")
        f.write(f"Density matrix purity: {purity:.6f}\n")
        f.write(f"Density matrix Z expectation value: {expectation_Z:.6f}\n")
        if z_values and z_values[0][1] is not None:
            f.write(f"Density matrix scaled by (1+z)^3: {(1+z_values[0][1])**3:.6f}\n")
        if t is not None and z_values:
            f.write("\nCosmological Results:\n")
            for cosmo_result in [r for r in results if 'H0_km_s_Mpc' in r]:
                f.write(f"\nH0={cosmo_result['H0_km_s_Mpc']:.2f} km/s/Mpc\n")
                f.write(f"Redshift z: {cosmo_result['z']:.6f}\n")
                f.write(f"Effective speed of light c_eff: {cosmo_result['c_eff']:.4e} m/s\n")
                f.write(f"Dark matter number density N_DM: {cosmo_result['N_DM']:.4e} m^-3\n")
            z_min = min(z for h0, z, h0_km_s_Mpc in z_values if z is not None)
            z_max = max(z for h0, z, h0_km_s_Mpc in z_values if z is not None)
            z_diff_percent = (z_max - z_min) / z_min * 100 if z_min != 0 else 0
            h0_min_z = min(z_values, key=lambda x: x[1])[2]
            h0_max_z = max(z_values, key=lambda x: x[1])[2]
            f.write(f"\nHubble Tension Analysis:\n")
            f.write(f"Redshift variation: {z_min:.6f} (H0={h0_min_z:.2f} km/s/Mpc) to {z_max:.6f} (H0={h0_max_z:.2f} km/s/Mpc)\n")
            f.write(f"Percentage difference: {z_diff_percent:.2f}%\n")
    logger.info("Results saved to qconvert_results.txt")


    result = {
        'N': N,
        'n_qubits': n_qubits,
        'S_quantum': S_quantum,
        'M_quantum': M_quantum,
        'S_classical': S_classical,
        'M_classical': M_classical,
        'circuit_counts': counts,
        'circuit_diagram': str(circuit.draw()),
        'purity': purity,
        'expectation_Z': expectation_Z,
        'meas_expectation': meas_expectation
    }
    logger.info(f"Quantum correlation S_quantum: {S_quantum:.4f}")
    logger.info(f"Quantum M_quantum: {M_quantum:.4f}")
    logger.info(f"Classical correlation S_classical: {S_classical:.4f}")
    logger.info(f"Classical M_classical: {M_classical:.4f}")
    logger.info(f"Quantum circuit counts: {counts}")
    logger.info(f"Circuit diagram:\n{circuit.draw()}")
    logger.info(f"Histogram saved to histogram.png")
    logger.info(f"Density matrix purity: {purity:.6f}")
    logger.info(f"Density matrix Z expectation value: {expectation_Z:.6f}")
    if z_values and z_values[0][1] is not None:
        logger.info(f"Density matrix scaled by (1+z)^3: {(1+z_values[0][1])**3:.6f}")
    results.append(result)

    if t is not None and z_values:
        for h0, z, h0_km_s_Mpc in z_values:
            if z is not None:
                c_eff, N_DM = compute_cosmological(t, z)
                cosmo_result = {
                    'H0_s_inv': h0,
                    'H0_km_s_Mpc': h0_km_s_Mpc,
                    'z': z,
                    'c_eff': c_eff,
                    'N_DM': N_DM
                }
                logger.info(f"H0={h0_km_s_Mpc:.2f} km/s/MPC = 70")
                logger.info(f"Effective speed of light c_eff: {c_eff:.4e} m/s")
                logger.info(f"Dark matter number density N_DM: {N_DM:.4e} m^-3")
                results.append(cosmo_result)

    # Save results to file
    with open('qconvert_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Results for N={N}, n_qubits={n_qubits}\n")
        f.write(f"Quantum S_quantum: {S_quantum:.4f}\n")
        f.write(f"Quantum M_quantum: {M_quantum:.4f}\n")
        f.write(f"Classical S_classical: {S_classical:.4f}\n")
        f.write(f"Classical M_classical: {M_classical:.4f}\n")
        f.write(f"Quantum circuit counts: {counts}\n")
        f.write(f"Circuit diagram:\n{circuit.draw()}\n")
        f.write(f"Density matrix purity: {purity:.6f}\n")
        f.write(f"Density matrix Z expectation value: {expectation_Z:.6f}\n")
        if z_values and z_values[0][1] is not None:
            f.write(f"Density matrix scaled by (1+z)^3: {(1+z_values[0][1])**3:.6f}\n")
        if t is not None and z_values:
            f.write("\nCosmological Results:\n")
            for cosmo_result in [r for r in results if 'H0_km_s_Mpc' in r]:
                f.write(f"\nH0={cosmo_result['H0_km_s_Mpc']:.2f} km/s/Mpc\n")
                f.write(f"Redshift z: {cosmo_result['z']:.6f}\n")
                f.write(f"Effective speed of light c_eff: {cosmo_result['c_eff']:.4e} m/s\n")
                f.write(f"Dark matter number density N_DM: {cosmo_result['N_DM']:.4e} m^-3\n")
            # Hubble tension analysis
            z_min = min(z for h0, z, h0_km_s_Mpc in z_values if z is not None)
            z_max = max(z for h0, z, h0_km_s_Mpc in z_values if z is not None)
            z_diff_percent = (z_max - z_min) / z_min * 100 if z_min != 0 else 0
            h0_min_z = min(z_values, key=lambda x: x[1])[2]
            h0_max_z = max(z_values, key=lambda x: x[1])[2]
            f.write(f"\nHubble Tension Analysis:\n")
            f.write(f"Redshift variation: {z_min:.6f} (H0={h0_min_z:.2f} km/s/Mpc) to {z_max:.6f} (H0={h0_max_z:.2f} km/s/Mpc)\n")
            f.write(f"Percentage difference: {z_diff_percent:.2f}%\n")
    logger.info("Results saved to qconvert_results.txt")

if __name__ == "__main__":
    main()