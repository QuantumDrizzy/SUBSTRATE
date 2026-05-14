import numpy as np

def generate_cmb_seed():
    """
    Simula un fragmento del mapa de anisotropías del CMB (Planck).
    Retorna un valor de fluctuación de temperatura primordial.
    """
    print("[CMB] Generando semilla cosmológica (Planck Legacy)...")
    
    # El CMB tiene fluctuaciones de ~1 parte en 100,000
    # Generamos un mapa de ruido gaussiano filtrado para simular el espectro de potencia
    size = 4
    raw_map = np.random.normal(0, 1e-5, (size, size))
    
    # Aplicamos un "filtro de escala" para simular los picos acústicos
    # (En una versión real, esto usaría los archivos FITS de Planck)
    cmb_factor = np.mean(raw_map) * 1e6
    
    print(f"[CMB] Fluctuación primordial detectada: {cmb_factor:.6f} μK")
    return cmb_factor

if __name__ == "__main__":
    seed = generate_cmb_seed()
    print(f"Semilla: {seed}")
