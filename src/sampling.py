"""Sampling of the generator conditions (initial energy, number of interactions, direction).

The generator is conditioned on three quantities (see the README): the initial kinetic
energy of the positron, the number of interactions of the path, and the initial direction.
When no GATE data is available, ``sample_conditions`` draws them from approximations.
These approximations are NOT part of the published method, see the README for details.
"""
import numpy as np

F18_ENDPOINT_MEV = 0.6335  # beta+ end-point energy of F-18
F18_DAUGHTER_Z = 8         # O-18
_ELECTRON_MASS_MEV = 0.51099895
_ALPHA = 1 / 137.036


def sample_f18_energies(n, rng, endpoint=F18_ENDPOINT_MEV, z=F18_DAUGHTER_Z):
    """Sample positron kinetic energies (MeV) from an allowed beta+ spectrum.

    Spectrum shape: p * W * (W0 - W)^2 * F(Z, W), with F the non-relativistic Fermi function
    for positrons. Sampling is done by inverse-CDF on a fine grid.
    """
    t = np.linspace(1e-5, endpoint - 1e-5, 20000)  # kinetic energy, MeV
    w = t / _ELECTRON_MASS_MEV + 1.0               # total energy / m_e
    w0 = endpoint / _ELECTRON_MASS_MEV + 1.0
    p = np.sqrt(w ** 2 - 1.0)
    eta = z * _ALPHA * w / p                       # positive: Coulomb repulsion of positrons
    fermi = 2 * np.pi * eta / (np.exp(2 * np.pi * eta) - 1.0)
    pdf = p * w * (w0 - w) ** 2 * fermi
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, t)


# Stand-in rule N = round(a + b * E / E_max), clipped to [3, 18], per material.
# (a, b) were picked by a coarse grid search so that the generated R_mean and R_max of the released
# generators are close to the GATE values reported in the thesis (Table of R_mean / R_max, F-18). They are
# therefore NOT an independent validation of the generators, and they are not the GATE
# energy-to-interactions histogram used in the paper.
_DEMO_RULE = {"Water": (5, 4), "RibBone": (5, 2), "Lung": (7, 8)}


def energy_to_num_interactions(energy, material="Water", endpoint=F18_ENDPOINT_MEV, n_min=3, n_max=18):
    a, b = _DEMO_RULE[material]
    n = np.rint(a + b * np.asarray(energy) / endpoint)
    return np.clip(n, n_min, n_max).astype(np.int64)


def sample_isotropic_directions(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sample_conditions(n, rng, material="Water"):
    energy = sample_f18_energies(n, rng)
    return energy, energy_to_num_interactions(energy, material), sample_isotropic_directions(n, rng)
