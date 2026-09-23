"""1D proton Schrodinger solver + vibronic overlap.

Given a proton potential energy surface V(q_H) on an evenly-spaced grid, solves the
1D Schrodinger equation
    [-hbar^2/(2m) d^2/dq^2 + V(q)] chi_mu = eps_mu chi_mu
for a chosen isotope mass m, using a 3-point finite-difference scheme (tridiagonal
eigen-solver). Returns lowest-K eigenstates and their eigenenergies.

For nonadiabatic PCET the vibronic overlap between reactant proton state chi_mu^(D)
and product proton state chi_nu^(A) is the Franck-Condon-like integral
    S_{mu,nu} = integral chi_mu^(D)* chi_nu^(A) dq
which we compute by trapezoidal integration on the common grid.

Units: q in Angstrom; V in kcal/mol; masses in atomic mass units (H=1.008, D=2.014, T=3.016).
Eigenenergies returned in kcal/mol; wavefunctions normalised to integral(|chi|^2) dq = 1.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal

# Constants
HBAR_J_S = 1.054571817e-34
J_PER_KCAL_MOL = 4184.0 / 6.02214076e23      # J per (kcal/mol) unit-energy per particle
AMU_KG = 1.66053906660e-27                    # kg per amu
ANGSTROM_M = 1e-10                            # m per A

# Convert kinetic-energy prefactor hbar^2/(2 m) into kcal/mol * A^2
# T[units kcal/mol * A^2] = HBAR^2 / (2 * m_kg * A^2 * kcalmol_per_J)
def kinetic_prefactor_kcal_A2(m_amu):
    m_kg = m_amu * AMU_KG
    return (HBAR_J_S**2 / (2.0 * m_kg * ANGSTROM_M**2)) / J_PER_KCAL_MOL


def solve_1d(q_grid, V_kcal, m_amu, K=8):
    """Return (eigenenergies_kcal, wavefunctions[K, n_grid]).
    q_grid: 1D array of grid points in A (must be evenly spaced).
    V_kcal: 1D array of V at each grid point in kcal/mol.
    m_amu: isotope mass in amu.
    K: number of lowest eigenstates to return.
    """
    assert q_grid.ndim == 1 and V_kcal.shape == q_grid.shape
    dq = q_grid[1] - q_grid[0]
    assert np.allclose(np.diff(q_grid), dq), 'grid must be evenly spaced'
    T_pref = kinetic_prefactor_kcal_A2(m_amu)                   # kcal/mol * A^2
    T_diag = 2.0 * T_pref / dq**2                                # kcal/mol
    T_offd = -T_pref / dq**2
    H_diag = T_diag + V_kcal                                     # kcal/mol
    H_offd = np.full(len(q_grid) - 1, T_offd)
    eps, psi = eigh_tridiagonal(H_diag, H_offd,
                                select='i', select_range=(0, K - 1))
    # Normalise: integral |psi|^2 dq = 1
    for k in range(K):
        norm = np.sqrt(np.trapezoid(np.abs(psi[:, k])**2, q_grid))
        psi[:, k] /= norm
    return eps, psi.T   # shape (K, n_grid)


def vibronic_overlap(q_grid, psi_D, psi_A):
    """Compute S_{mu, nu} = integral chi_mu^(D)* chi_nu^(A) dq on the shared grid.
    psi_D, psi_A: (K, n_grid) arrays of the two sets of vibrational eigenstates."""
    S = np.zeros((psi_D.shape[0], psi_A.shape[0]))
    for mu in range(psi_D.shape[0]):
        for nu in range(psi_A.shape[0]):
            S[mu, nu] = np.trapezoid(psi_D[mu] * psi_A[nu], q_grid)
    return S


def double_well(q, a=0.5, b=3.0):
    """Symmetric double well V(q) = a*(q^2 - b^2/4)^2 / (b^2/4)^2 [kcal/mol; barrier a]."""
    return a * (q**2 - b**2 / 4)**2 / (b**2 / 4)**2


def harmonic(q, k=100.0, q0=0.0):
    """Harmonic V = 0.5 k (q - q0)^2 [kcal/mol]. k in kcal/mol/A^2."""
    return 0.5 * k * (q - q0)**2


# ---- self-tests ----
def _test_harmonic():
    """Verify: harmonic frequency omega = sqrt(k/m) gives ZPE = hbar omega / 2.
    For k = 100 kcal/mol/A^2, m_H = 1.008: omega = sqrt(100/1.008) = 9.960 kcal/(mol A^2 amu),
    convert to angular freq: omega [rad/s] = sqrt(k_J_m2 / m_kg).
    """
    import math
    q = np.linspace(-1.0, 1.0, 401)
    k_kcal_A2 = 100.0
    V = harmonic(q, k=k_kcal_A2)
    for m_amu, name in [(1.008, 'H'), (2.014, 'D')]:
        eps, psi = solve_1d(q, V, m_amu, K=4)
        # Theoretical ZPE
        k_J_m2 = k_kcal_A2 * 0.694770 * 1.0    # N/m = J/m^2
        m_kg = m_amu * AMU_KG
        omega = math.sqrt(k_J_m2 / m_kg)
        zpe_J = 0.5 * HBAR_J_S * omega
        zpe_kcal = zpe_J / J_PER_KCAL_MOL
        print(f'{name}: numerical ZPE = {eps[0]:.4f} kcal/mol  theory = {zpe_kcal:.4f}  '
              f'level spacing = {eps[1]-eps[0]:.4f}  theory hbar*omega = {2*zpe_kcal:.4f}')
        assert abs(eps[0] - zpe_kcal) / zpe_kcal < 0.02, 'ZPE mismatch'
    print('harmonic solver self-test OK')


def _test_double_well_KIE():
    """Symmetric double well: check that H tunnels more than D > T.
    Tunneling splitting Delta = 2 * H_tunnel ~ hbar*omega/pi * exp(-S/hbar),
    S = integral sqrt(2m(V-E)) dq -> Delta_H/Delta_D ratio grows exponentially with barrier
    because S scales as sqrt(m).  Ratio of ratios approximates a KIE."""
    q = np.linspace(-2.0, 2.0, 1001)
    for barrier in [2.0, 4.0, 6.0, 10.0]:
        V = double_well(q, a=barrier, b=2.0)
        print(f'\nbarrier = {barrier:.1f} kcal/mol:')
        deltas = {}
        for m_amu, name in [(1.008, 'H'), (2.014, 'D'), (3.016, 'T')]:
            eps, psi = solve_1d(q, V, m_amu, K=4)
            Delta = eps[1] - eps[0]
            deltas[name] = Delta
            print(f'  {name}: ZPE(donor)={eps[0]:.4f}  tunnel-splitting={Delta:.6e} kcal/mol')
        if deltas['D'] > 1e-15:
            print(f'  Delta_H / Delta_D = {deltas["H"]/deltas["D"]:.3f}  (proxy KIE)')


if __name__ == '__main__':
    _test_harmonic()
    _test_double_well_KIE()
