import numpy as np

class QuantumFoam:
    """
    Simula la espuma cuántica como fluctuaciones del espacio-tiempo en una rejilla 4x4.
    """
    def __init__(self, size=4, correlation_length=1.5, use_ligo=True):
        self.size = size
        self.correlation_length = correlation_length
        self.lattice = np.zeros((size, size))
        self.use_ligo = use_ligo
        self.ligo_data = None
        
        if use_ligo:
            from ligo_loader import download_ligo_data
            from cmb_loader import generate_cmb_seed
            self.ligo_data = download_ligo_data()
            # La semilla del CMB modula la longitud de correlación del vacío
            self.correlation_length = 1.0 + abs(generate_cmb_seed())

    def generate_fluctuation(self):
        """
        Genera un campo de fluctuaciones basado en datos reales de LIGO o ruido sintético.
        """
        if self.use_ligo and self.ligo_data is not None:
            # Tomamos un fragmento aleatorio de la serie temporal de LIGO
            idx = np.random.randint(0, len(self.ligo_data) - 16)
            chunk = self.ligo_data[idx:idx+16].reshape((self.size, self.size))
            self.lattice = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-9)
            return self.lattice
            
        # Fallback a ruido blanco sintético
        white_noise = np.random.normal(0, 1, (self.size, self.size))
        
        # 2. Aplicar correlación espacial (espuma cuántica)
        # Usamos un kernel gaussiano simple para simular la estructura del vacío
        x = np.arange(self.size)
        y = np.arange(self.size)
        xx, yy = np.meshgrid(x, y)
        
        # Generar matriz de distancias
        fluctuations = np.zeros_like(white_noise)
        for i in range(self.size):
            for j in range(self.size):
                dist = np.sqrt((xx - i)**2 + (yy - j)**2)
                kernel = np.exp(-dist**2 / (2 * self.correlation_length**2))
                fluctuations[i, j] = np.sum(white_noise * kernel)
                
        # Normalizar
        self.lattice = (fluctuations - np.mean(fluctuations)) / np.std(fluctuations)
        return self.lattice

    def get_local_density(self, x, y):
        """
        Retorna la densidad de energía del vacío en un punto específico.
        """
        return self.lattice[x % self.size, y % self.size]

if __name__ == "__main__":
    foam = QuantumFoam()
    signal = foam.generate_fluctuation()
    print("[FOAM] Espuma cuántica generada (Lattice 4x4):")
    print(signal)
    print(f"[FOAM] Densidad media: {np.mean(signal):.4f}")
    print(f"[FOAM] Varianza del vacío: {np.var(signal):.4f}")
