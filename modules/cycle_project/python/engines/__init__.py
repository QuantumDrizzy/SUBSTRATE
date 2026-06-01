"""
SUBSTRATE — Python science engines
===================================
Llamados desde Rust via PyO3. Cada módulo expone funciones puras:
entrada = parámetros numéricos / paths, salida = dict serializable.

Engines disponibles:
  data            — carga y alineación de proxies paleoclimáticos
  geomagnetic     — detección de anomalías GNN (PCA + graph diffusion)
  spectral        — análisis espectral FFT + CWT
  decay           — modelos de decaimiento VADM (4 modelos + CI)
  fingerprint     — detección de pre-excursión geomagnética
  forecast        — ensemble LSTM con MC-dropout
  lbm             — simulación litosférica Lattice Boltzmann D2Q9
  radical_pair    — dinámica de spin en criptocromos
  lindblad        — ecuación maestra GKSL para sistemas cuánticos abiertos
  tensor_network  — MPS/GPU para espacios de Hilbert grandes
  rf_noise        — perturbación RF en coherencia de spin
  myth_corpus     — corpus de mitos de catástrofe (fuentes académicas)
  myth_correlate  — correlación temporal mito-geomagnética
"""
