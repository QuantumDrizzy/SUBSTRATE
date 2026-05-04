"""
P2_WBOSON/data_generator.py
Simula la distribución de masa transversal del Bosón W (Pico Jacobiano)
y aplica la Matriz de Respuesta del detector.
"""
import numpy as np

def generate_jacobian_peak(n_bins=20, m_w=80.4):
    """
    Genera un espectro sintético del pico jacobiano (Truth).
    Representa el límite abrupto en la masa cinemática transversal.
    """
    masses = np.linspace(60, 100, n_bins)
    truth = np.zeros(n_bins)
    
    for i, m in enumerate(masses):
        if m < m_w:
            # Pico jacobiano: diverge teóricamente en M_W, suavizado empíricamente
            truth[i] = 1.0 / np.sqrt((m_w**2 - m**2) + 0.1)
        else:
            # Caída exponencial extremadamente rápida post-pico
            truth[i] = np.exp(-(m - m_w)*5)
            
    # Normalizamos y escalamos a "conteo de eventos" representativo
    truth = (truth / np.sum(truth)) * 10000
    return masses, truth

def generate_response_matrix(n_bins=20, blur=1.0):
    """
    Genera la matriz de respuesta (Smeared) del detector.
    Aplica un Gaussian blur que dispersa los eventos verdaderos entre bins cercanos.
    """
    R = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            # Matriz de transición probabilística (emborronamiento)
            R[i, j] = np.exp(-0.5 * ((i - j) / blur)**2)
        # Normalización para conservación de probabilidad (cada evento generado debe caer en algún bin)
        R[:, i] /= np.sum(R[:, i])
    return R

def get_toy_data(n_bins=20):
    """
    Orquesta la generación del Truth, aplica el detector, e inyecta fluctuaciones estadísticas.
    """
    masses, truth = generate_jacobian_peak(n_bins)
    R = generate_response_matrix(n_bins)
    
    # Smearing puro
    smeared_ideal = R @ truth
    
    # Ruido estadístico (Fluctuaciones Poissonianas) típico en detectores
    np.random.seed(42)
    measured = np.random.poisson(smeared_ideal).astype(float)
    
    return masses, truth, measured, R

if __name__ == "__main__":
    masses, truth, measured, R = get_toy_data()
    print("Truth vs Measured (Primeros 5 bins):")
    for i in range(5):
        print(f"M={masses[i]:.1f} | Truth: {truth[i]:.1f} | Measured: {measured[i]:.1f}")
