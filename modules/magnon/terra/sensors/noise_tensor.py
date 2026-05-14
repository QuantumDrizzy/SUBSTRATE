"""
TERRA-QCI — Electromagnetic Noise Tensor Constructor
====================================================

Level 1 Bridge: Raw sensor data → H_noise(t) quantum operator.

Captures electromagnetic noise from two sources:
    1. SDR (RTL-SDR dongle): RF power spectral density in the
       biologically relevant band (0.1 — 100 MHz)
    2. NOAA: Geomagnetic field data (Kp index, Dst, real-time
       magnetometer from observatories)

The raw spectral data is NOT graphed directly. Instead, it is
transformed into a time-dependent perturbation Hamiltonian:

    H_noise(t) = γ_e · B_noise(t) · (S₁ + S₂)

Where B_noise(t) is the effective magnetic field fluctuation
derived from the RF power spectral density via:

    B_rms = √(2μ₀ · S_psd · Δf) / c

This Hamiltonian is then injected into the radical pair
Lindblad solver to measure biological decoherence.

Units: SI throughout. Frequencies in Hz, fields in Tesla,
       energies in Joules, times in seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import signal as sig

# ── Physical constants ───────────────────────────────────────────────

GYROMAGNETIC_E = 1.760_859_630_23e11  # rad/(s·T) — free electron
MU_BOHR = 9.274_010_078_3e-24         # J/T — Bohr magneton
HBAR = 1.054_571_817e-34              # J·s
MU_0 = 1.256_637_062_12e-6            # N/A² — vacuum permeability
C_LIGHT = 2.997_924_58e8              # m/s
B_EARTH = 50e-6                        # T — Earth's field (Spain, ~50 μT)
KB = 1.380_649e-23                     # J/K — Boltzmann constant


# ── Pauli spin matrices (2×2) ────────────────────────────────────────

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class NoiseCapture:
    """Raw electromagnetic noise measurement from sensors."""
    timestamp: float                    # UNIX timestamp
    source: str                         # 'sdr', 'noaa', 'synthetic'
    center_freq_hz: float               # Center frequency of capture
    bandwidth_hz: float                 # Capture bandwidth
    psd: np.ndarray                     # Power spectral density (W/Hz)
    freqs: np.ndarray                   # Frequency axis (Hz)
    duration_s: float                   # Capture duration
    location: str = "Aljucer,Murcia"    # Geographic location


@dataclass
class NoiseTensor:
    """
    Formatted noise tensor ready for injection into the radical pair
    Hamiltonian. This is H_noise(t) in operator form.
    """
    timestamp: float
    # Effective magnetic noise field components (Tesla)
    b_noise_x: float
    b_noise_y: float
    b_noise_z: float
    b_noise_rms: float                  # RMS magnitude
    # Spectral characteristics
    dominant_freq_hz: float             # Strongest noise frequency
    total_power_dbm: float              # Total integrated power
    spectral_entropy: float             # Shannon entropy of PSD
    # The operator itself (4×4 for two-electron system)
    hamiltonian: np.ndarray             # H_noise as matrix
    # Metadata
    source: str
    location: str
    capture_duration: float


@dataclass
class GeomagneticState:
    """Real-time geomagnetic field state from NOAA."""
    timestamp: float
    b_total: float = B_EARTH            # Total field magnitude (T)
    b_x: float = 0.0                    # North component (T)
    b_y: float = 0.0                    # East component (T)
    b_z: float = 0.0                    # Down component (T)
    kp_index: float = 0.0               # Planetary K-index [0-9]
    dst_index: float = 0.0              # Disturbance Storm Time (nT)
    source: str = "noaa"


# ── SDR Noise Capture ────────────────────────────────────────────────

def capture_sdr_noise(
    center_freq: float = 100e6,
    sample_rate: float = 2.4e6,
    duration: float = 1.0,
    gain: float = 20.0,
) -> NoiseCapture:
    """
    Capture RF noise from RTL-SDR dongle.

    Falls back to synthetic noise if no hardware is available.
    The capture focuses on the biologically relevant RF band
    where radical pair decoherence is most sensitive.

    Args:
        center_freq: Center frequency in Hz (default: 100 MHz)
        sample_rate: Sample rate in Hz (default: 2.4 MHz)
        duration: Capture duration in seconds
        gain: SDR gain in dB

    Returns:
        NoiseCapture with power spectral density
    """
    n_samples = int(sample_rate * duration)

    try:
        from rtlsdr import RtlSdr

        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.gain = gain

        # Capture IQ samples
        iq_samples = sdr.read_samples(n_samples)
        sdr.close()

    except (ImportError, Exception):
        # Synthetic noise: urban EM environment simulation
        iq_samples = _generate_synthetic_noise(n_samples, sample_rate)

    # Compute power spectral density via Welch's method
    nperseg = min(4096, n_samples // 4)
    freqs, psd = sig.welch(
        iq_samples,
        fs=sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        scaling='density',
    )

    # Shift to actual frequencies
    freqs = freqs + center_freq

    return NoiseCapture(
        timestamp=time.time(),
        source='sdr',
        center_freq_hz=center_freq,
        bandwidth_hz=sample_rate,
        psd=np.abs(psd).astype(np.float64),
        freqs=freqs.astype(np.float64),
        duration_s=duration,
    )


def _generate_synthetic_noise(n_samples: int, fs: float) -> np.ndarray:
    """
    Generate synthetic urban electromagnetic noise.

    Models a realistic urban RF environment with:
    - Broadband thermal noise floor
    - FM broadcast carriers (~88-108 MHz)
    - TETRA police/emergency band (~380-400 MHz harmonics)
    - 5G NR sub-6 GHz leakage
    - Power line harmonics (50 Hz × n)
    """
    rng = np.random.default_rng(seed=int(time.time()) % 2**31)
    t = np.arange(n_samples) / fs

    # Thermal noise floor (-174 dBm/Hz + receiver noise figure)
    noise = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    noise *= 1e-6  # ~ -60 dBm

    # FM broadcast interference (narrowband carriers)
    for f_offset in [-0.8e6, -0.3e6, 0.2e6, 0.7e6, 1.1e6]:
        amplitude = rng.uniform(1e-4, 5e-4)
        phase = rng.uniform(0, 2 * np.pi)
        noise += amplitude * np.exp(1j * (2 * np.pi * f_offset * t + phase))

    # TETRA-like pulsed interference
    tetra_freq = 0.5e6  # offset from center
    tetra_period = 1.0 / 17.65  # TETRA TDMA frame rate
    tetra_pulse = (np.mod(t, tetra_period) < tetra_period * 0.25).astype(float)
    noise += 2e-4 * tetra_pulse * np.exp(1j * 2 * np.pi * tetra_freq * t)

    # Power line harmonics (broadband, from switching power supplies)
    for harmonic in range(1, 20):
        f_harm = 50.0 * harmonic  # 50 Hz × n (European grid)
        noise += rng.uniform(1e-5, 5e-5) * np.exp(1j * 2 * np.pi * f_harm * t)

    # Random bursts (simulating phone transmissions, WiFi, etc.)
    n_bursts = rng.integers(3, 10)
    for _ in range(n_bursts):
        start = rng.integers(0, n_samples - int(fs * 0.01))
        length = rng.integers(int(fs * 0.001), int(fs * 0.01))
        burst_freq = rng.uniform(-fs / 2, fs / 2)
        burst_amp = rng.uniform(1e-4, 1e-3)
        burst_t = np.arange(length) / fs
        noise[start:start + length] += burst_amp * np.exp(
            1j * 2 * np.pi * burst_freq * burst_t
        )

    return noise


# ── NOAA Geomagnetic Data ────────────────────────────────────────────

def fetch_geomagnetic_state() -> GeomagneticState:
    """
    Fetch real-time geomagnetic data from NOAA SWPC.

    Falls back to nominal Earth field if network is unavailable.
    Respects the zero-cloud constraint: this is the ONLY external
    HTTP call in the system, and it's to a public government API.
    """
    try:
        import requests

        # NOAA SWPC real-time planetary Kp index
        url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if data:
            latest = data[-1]
            kp = float(latest.get("kp_index", 0))
        else:
            kp = 0.0

        # NOAA real-time magnetometer (Boulder, CO — closest proxy)
        mag_url = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-1-day.json"
        mag_resp = requests.get(mag_url, timeout=5)
        mag_data = mag_resp.json()

        if mag_data:
            latest_mag = mag_data[-1]
            # GOES magnetometer gives Hp, He, Hn in nT
            bx = float(latest_mag.get("Hp", 0)) * 1e-9  # nT → T
            by = float(latest_mag.get("He", 0)) * 1e-9
            bz = float(latest_mag.get("Hn", 0)) * 1e-9
            b_total = np.sqrt(bx**2 + by**2 + bz**2)
        else:
            bx, by, bz = 0.0, 0.0, 0.0
            b_total = B_EARTH

        return GeomagneticState(
            timestamp=time.time(),
            b_total=max(b_total, B_EARTH),
            b_x=bx, b_y=by, b_z=bz,
            kp_index=kp,
            source="noaa_swpc",
        )

    except Exception:
        # Nominal Earth field — Aljucer, Murcia (~44 μT horizontal)
        return GeomagneticState(
            timestamp=time.time(),
            b_total=B_EARTH,
            b_x=24.5e-6,  # North component
            b_y=0.5e-6,   # East component
            b_z=-39.0e-6,  # Down component (negative = upward)
            kp_index=1.0,
            source="nominal",
        )


# ── PSD → Magnetic Noise Field ──────────────────────────────────────

def psd_to_magnetic_field(capture: NoiseCapture) -> tuple[float, float, float]:
    """
    Convert RF power spectral density to an effective oscillating
    magnetic field that perturbs the radical pair.

    Physics:
        For a plane wave, the magnetic field amplitude is related
        to the electric field (and thus power) by:

            B_rms = √(2·μ₀·S_psd·Δf) / c

        where S_psd is the power spectral density and Δf is the
        bandwidth of interest.

        We decompose into 3 spatial components assuming isotropic
        noise (urban environment — no preferred direction).

    Returns:
        (B_x, B_y, B_z) in Tesla — noise field components
    """
    # Integrate PSD over the biologically relevant band
    # Radical pairs are sensitive to RF in the 1-50 MHz range
    # (Larmor frequency of electron in Earth's field ~1.4 MHz)
    df = np.mean(np.diff(capture.freqs)) if len(capture.freqs) > 1 else 1.0
    total_power = np.sum(capture.psd) * abs(df)

    # Convert power density to RMS magnetic field
    # P = c·B²/(2·μ₀) → B = √(2·μ₀·P/c)
    b_rms = np.sqrt(2 * MU_0 * max(total_power, 1e-30) / C_LIGHT)

    # Decompose isotropically with random phase
    # (in reality, would use polarization data from dual-pol SDR)
    rng = np.random.default_rng(seed=int(capture.timestamp * 1e3) % 2**31)
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)

    return (
        float(b_rms * direction[0]),
        float(b_rms * direction[1]),
        float(b_rms * direction[2]),
    )


# ── Spectral Entropy ────────────────────────────────────────────────

def spectral_entropy(psd: np.ndarray) -> float:
    """
    Shannon entropy of the normalized PSD.

    Low entropy = dominated by few frequencies (coherent interference)
    High entropy = broadband noise (thermal-like)

    Biologically relevant: coherent interference at the Larmor
    frequency is far more destructive to radical pair coherence
    than broadband thermal noise.
    """
    psd_norm = psd / (np.sum(psd) + 1e-30)
    psd_safe = psd_norm[psd_norm > 1e-30]
    return float(-np.sum(psd_safe * np.log2(psd_safe)))


# ── H_noise(t) Construction ─────────────────────────────────────────

def build_noise_hamiltonian(
    b_noise_x: float,
    b_noise_y: float,
    b_noise_z: float,
) -> np.ndarray:
    """
    Construct the noise perturbation Hamiltonian for the radical pair.

    The noise acts on BOTH electron spins (S₁ and S₂) via Zeeman
    coupling to the fluctuating magnetic field:

        H_noise = γ_e · B_noise · (S₁ + S₂)

    In the two-electron product basis {|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩},
    the total spin operators are:

        S_total_α = S₁_α ⊗ I₂ + I₂ ⊗ S₂_α    (α = x, y, z)

    Returns:
        4×4 complex Hermitian matrix (two-electron Hilbert space)
    """
    # Two-electron spin operators: S_total = S₁ ⊗ I + I ⊗ S₂
    # Factor of 1/2 because σ = 2S
    S1x = np.kron(SIGMA_X / 2, I2)
    S1y = np.kron(SIGMA_Y / 2, I2)
    S1z = np.kron(SIGMA_Z / 2, I2)

    S2x = np.kron(I2, SIGMA_X / 2)
    S2y = np.kron(I2, SIGMA_Y / 2)
    S2z = np.kron(I2, SIGMA_Z / 2)

    # Total spin
    Sx = S1x + S2x
    Sy = S1y + S2y
    Sz = S1z + S2z

    # H_noise = -γ_e · (B_x·Sx + B_y·Sy + B_z·Sz)
    # (negative sign: energy = -μ·B, μ = -γ_e·S)
    H_noise = -GYROMAGNETIC_E * (
        b_noise_x * Sx +
        b_noise_y * Sy +
        b_noise_z * Sz
    )

    return H_noise


# ── Main Bridge: Sensor → Tensor ────────────────────────────────────

def capture_and_tensorize(
    sdr_center_freq: float = 100e6,
    sdr_duration: float = 0.5,
) -> NoiseTensor:
    """
    THE BRIDGE: Capture EM noise → format as quantum Hamiltonian.

    This is the Level 1 pipeline that feeds the radical pair
    Lindblad solver. Called once per simulation frame.

    Pipeline:
        1. Capture RF noise (SDR or synthetic)
        2. Compute power spectral density
        3. Convert PSD → effective B_noise(t)
        4. Build H_noise operator (4×4)
        5. Package as NoiseTensor

    Returns:
        NoiseTensor ready for injection into the quantum engine
    """
    # Step 1+2: Capture and compute PSD
    capture = capture_sdr_noise(
        center_freq=sdr_center_freq,
        duration=sdr_duration,
    )

    # Step 3: PSD → magnetic field components
    bx, by, bz = psd_to_magnetic_field(capture)
    b_rms = np.sqrt(bx**2 + by**2 + bz**2)

    # Step 4: Build the 4×4 noise Hamiltonian
    H_noise = build_noise_hamiltonian(bx, by, bz)

    # Spectral analysis
    s_entropy = spectral_entropy(capture.psd)

    # Dominant frequency (loudest noise source)
    peak_idx = np.argmax(capture.psd)
    dominant_freq = float(capture.freqs[peak_idx])

    # Total power in dBm
    df = np.mean(np.diff(capture.freqs)) if len(capture.freqs) > 1 else 1.0
    total_power_w = np.sum(capture.psd) * abs(df)
    total_power_dbm = 10 * np.log10(max(total_power_w, 1e-30) * 1e3)

    return NoiseTensor(
        timestamp=capture.timestamp,
        b_noise_x=bx,
        b_noise_y=by,
        b_noise_z=bz,
        b_noise_rms=float(b_rms),
        dominant_freq_hz=dominant_freq,
        total_power_dbm=float(total_power_dbm),
        spectral_entropy=s_entropy,
        hamiltonian=H_noise,
        source=capture.source,
        location=capture.location,
        capture_duration=capture.duration_s,
    )
