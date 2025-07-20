import bodo
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from scipy.integrate import quad
from scipy.optimize import root_scalar

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

# ===== BODO-COMPATIBLE FUNCTIONS =====

@bodo.jit(cache=True)
def hubble_integral_bodo(a, Omega_r, Omega_m, Omega_k, Omega_Lambda):
	"""Bodo-optimized integrand for the age of the universe."""
	return 1 / np.sqrt(Omega_r / a**2 + Omega_m / a + Omega_k + Omega_Lambda * a**2)

@bodo.jit(cache=True)
def compute_classical_bodo(N):
	"""Bodo-optimized classical correlation parameters."""
	S_classical = 2 * np.sqrt(N)
	W_classical = S_classical / np.sqrt(2)
	M_classical = W_classical  # Same for even and odd N
	return S_classical, M_classical

@bodo.jit(cache=True)
def compute_cosmological_bodo(t, z, c_val, delta_val, tau_val, t0_val, N0_val):
	"""Bodo-optimized cosmological parameters."""
	c_eff_t = c_val * (1 - delta_val * np.exp(-t / tau_val))
	c_eff_t0 = c_val * (1 - delta_val * np.exp(-t0_val / tau_val))
	N_DM = N0_val * (c_eff_t0 / c_eff_t) * (1 + z)**3
	return c_eff_t, N_DM

@bodo.jit(cache=True)
def quantum_correlation_params_bodo(N, expectation):
	"""Bodo-optimized quantum correlation parameter calculations."""
	S_quantum = 2 * np.sqrt(2) * np.sqrt(N) * expectation
	W = S_quantum * np.sqrt(2)
	
	if N % 2 == 0:
		M_quantum = W * np.sqrt(N) / np.sqrt(2)
	else:
		M_quantum = W * np.sqrt(N) / np.sqrt(N + 1)
	
	return S_quantum, M_quantum

@bodo.jit(cache=True)
def simple_redshift_approximation_bodo(t, t0_val, H0_val, SECONDS_PER_YEAR_val):
	"""Bodo-optimized Hubble's law approximation for small lookback times."""
	return H0_val * t * SECONDS_PER_YEAR_val

@bodo.jit(cache=True)
def hubble_integral(a, Omega_r, Omega_m, Omega_k, Omega_Lambda):
	"""Original integrand for the age of the universe."""
	return 1 / np.sqrt(Omega_r / a**2 + Omega_m / a + Omega_k + Omega_Lambda * a**2)

@bodo.jit(cache=True)
def compute_age(a, H0, Omega_r, Omega_m, Omega_k, Omega_Lambda):
	"""Compute age of the universe at scale factor a in seconds."""
	integral, _ = quad(hubble_integral, 0, a, args=(Omega_r, Omega_m, Omega_k, Omega_Lambda), epsabs=1e-12, epsrel=1e-12)
	return integral / H0

# ===== NON-BODO FUNCTIONS (using original scipy/qiskit) =====

def compute_redshift(t, lookback=True):
	"""Compute redshift z based on time t (in years). If lookback=True, t is time ago from present."""
	if t < 0:
		raise ValueError("t must be >= 0")
	
	t_effective = t0 - t if lookback else t
	if t_effective < 0:
		raise ValueError("Lookback time t cannot exceed present age t0")
	
	t_seconds = t_effective * SECONDS_PER_YEAR
	
	# Use Bodo-optimized Hubble's law approximation for small lookback times
	if lookback and t / t0 < 0.01:
		z_approx = simple_redshift_approximation_bodo(t, t0, H0, SECONDS_PER_YEAR)
		print(f"Debug: Using Bodo-optimized Hubble's law approximation, z_approx={z_approx:.2e}")
		return z_approx
	
	def age_diff(a):
		return compute_age(a, H0, Omega_r, Omega_m, Omega_k, Omega_Lambda) - t_seconds
	
	try:
		bracket = [1e-10, 1.0]
		result = root_scalar(age_diff, bracket=bracket, method='brentq', xtol=1e-12, rtol=1e-12)
		a = result.root
		z = 1 / a - 1
		
		expected_z_max = 1000
		if z > expected_z_max:
			print(f"Warning: High redshift z={z:.2e} for t={t:.2e} years (lookback={lookback})")
		
		print(f"Debug: t={t:.2e} years, t_effective={t_effective:.2e} years, z={z:.6e}")
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
	
	expectation = max(min(expectation, 1.0), -1.0)
	
	# Use Bodo-optimized quantum correlation calculations
	S_quantum, M_quantum = quantum_correlation_params_bodo(N, expectation)
	
	# Save histogram (non-Bodo operation)
	plot_histogram(counts)
	plt.savefig('histogram.png')
	plt.close()
	
	return S_quantum, M_quantum, circuit, counts, expectation

# ===== HYBRID MAIN FUNCTION =====

def main_hybrid():
	"""Main function using Bodo optimization where possible."""
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
				z = compute_redshift(t, lookback=True)  # Uses original SciPy version
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
	
	# Use Bodo-optimized classical calculations
	S_classical, M_classical = compute_classical_bodo(N)

	print(f"\nBodo-optimized calculations:")
	print(f"Classical correlation S_classical: {S_classical:.4f}")
	print(f"Classical M_classical: {M_classical:.4f}")
	print(f"Quantum correlation S_quantum: {S_quantum:.4f}")
	print(f"Quantum M_quantum: {M_quantum:.4f}")
	
	if t is not None and z is not None:
		# Use Bodo-optimized cosmological calculations
		c_eff, N_DM = compute_cosmological_bodo(t, z, c, delta, tau, t0, N0)
		print(f"Redshift z: {z:.6f}")
		print(f"Bodo-optimized effective speed of light c_eff: {c_eff:.4e} m/s")
		print(f"Bodo-optimized dark matter number density N_DM: {N_DM:.4e} m^-3")

def main_original():
	"""Original main function for comparison."""
	#* Your original main function code here 
	pass

if __name__ == "__main__":
	print("Choose execution mode:")
	print("1. Hybrid (Bodo + Original)")
	print("2. Original only")
	
	choice = input("Enter choice (1 or 2): ")
	
	if choice == "1":
		print("Running hybrid Bodo-optimized version...")
		main_hybrid()
	elif choice == "2":
		print("Running original version...")
		main_original()
	else:
		print("Invalid choice. Running hybrid version by default...")
		main_hybrid()
