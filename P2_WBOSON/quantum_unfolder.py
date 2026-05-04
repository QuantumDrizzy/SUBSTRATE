"""
P2_WBOSON/quantum_unfolder.py
Mapeo del unfolding a QUBO mediante ventana deslizante (Sliding Window).
Usa D-Wave dimod (Simulated Annealing) para desplegar fragmentos de espectro.
"""
import numpy as np
import dimod
import neal

class QuantumUnfolder:
    def __init__(self, R, y, x_prior, window_size=5, bits_per_bin=8, lmbda=0.1):
        """
        R: Response matrix [N, N]
        y: Measured spectrum [N]
        x_prior: Prior estimate from classical unfolding [N]
        window_size: Número de bins a desplegar simultáneamente
        bits_per_bin: Profundidad de cuantización para discretización
        lmbda: Regularización de Tikhonov (diferencias finitas)
        """
        self.R = R
        self.y = y
        self.x_prior = x_prior
        self.n_bins = len(y)
        self.window_size = window_size
        self.bits = bits_per_bin
        self.lmbda = lmbda
        
        # Sampler clásico para optimización QUBO (Simulated Annealing de D-Wave Ocean)
        self.sampler = neal.SimulatedAnnealingSampler()

    def _build_qubo_for_window(self, start_idx):
        end_idx = min(start_idx + self.window_size, self.n_bins)
        w_len = end_idx - start_idx
        
        # Extraer sub-matriz de respuesta y sub-vectores medidos para esta ventana
        R_w = self.R[start_idx:end_idx, start_idx:end_idx]
        y_w = self.y[start_idx:end_idx]
        x_prior_w = self.x_prior[start_idx:end_idx]
        
        # Matriz de diferencias finitas (D) para penalizar variaciones bruscas
        D = np.zeros((w_len - 1, w_len))
        for i in range(w_len - 1):
            D[i, i] = -1
            D[i, i+1] = 1
        L = D.T @ D
        if w_len == 1:
            L = np.zeros((1, 1))
            
        # Formulación Delta: y_new = y - R * x_prior
        y_new = y_w - R_w @ x_prior_w
        
        # Rango simétrico para delta_x: [-S, S]
        # x_tilde in [0, 2S] -> delta_x = x_tilde - S
        # Fijamos S como el 20% del máximo del prior, o mínimo 500
        S = max(np.max(self.x_prior) * 0.2, 500.0)
        
        # Como delta_x = x_tilde - S, R * delta_x = R * x_tilde - S * R * 1
        # Así que y_target = y_new + S * R * 1
        y_target = y_new + S * (R_w @ np.ones(w_len))
        
        # Objetivo matricial: || R x_tilde - y_target ||^2 + lambda || D x_tilde ||^2
        # (Nota: D * S * 1 = 0, por lo que || D delta_x ||^2 = || D x_tilde ||^2)
        A = R_w.T @ R_w + self.lmbda * L
        b = -2 * y_target.T @ R_w
        
        # Mapeo a binario
        scale = (2 * S) / (2**self.bits - 1)
        
        Q = {}
        for i in range(w_len):
            for k in range(self.bits):
                idx1 = i * self.bits + k
                
                # Términos lineales (Diagonal de QUBO)
                coeff_lin = (scale * 2**k) * b[i] + (scale * 2**k)**2 * A[i, i]
                Q[(idx1, idx1)] = coeff_lin
                
                # Términos cuadráticos intra-bin (interacciones entre bits del mismo bin)
                for l in range(k + 1, self.bits):
                    idx2 = i * self.bits + l
                    Q[(idx1, idx2)] = 2 * (scale * 2**k) * (scale * 2**l) * A[i, i]
                    
                # Términos cuadráticos inter-bin (interacciones entre bins distintos)
                for j in range(i + 1, w_len):
                    for l in range(self.bits):
                        idx2 = j * self.bits + l
                        coeff_quad = 2 * (scale * 2**k) * (scale * 2**l) * A[i, j]
                        Q[(idx1, idx2)] = coeff_quad
                        
        return Q, scale, S

    def unfold(self):
        print(f"[QUBO] Iniciando Sliding Window Unfolding ({self.window_size} bins/ventana | {self.bits} bits/bin)...")
        print(f"[QUBO] Qubits por ventana: {self.window_size * self.bits}")
        
        x_unfolded = np.zeros(self.n_bins)
        counts = np.zeros(self.n_bins)
        
        # Desplazamiento de la ventana con un overlap para suavizar bordes
        step = max(self.window_size - 2, 1)
        
        for i in range(0, self.n_bins, step):
            # Prevenir que la última ventana se salga de los límites
            if i + self.window_size > self.n_bins and i != 0:
                i = self.n_bins - self.window_size
                
            Q, scale, S = self._build_qubo_for_window(i)
            
            # Resolver QUBO con Simulated Annealing
            response = self.sampler.sample_qubo(Q, num_reads=500)
            best_sample = response.first.sample
            
            # Decodificación del resultado binario de vuelta a conteos físicos
            end_idx = min(i + self.window_size, self.n_bins)
            w_len = end_idx - i
            x_w = np.zeros(w_len)
            
            for j in range(w_len):
                val = 0
                for k in range(self.bits):
                    idx = j * self.bits + k
                    val += int(best_sample[idx]) * (2**k)
                x_w[j] = (val * scale) - S
                
            # Stitching: Acumulamos los residuos
            x_unfolded[i:end_idx] += x_w
            counts[i:end_idx] += 1
            
            if end_idx == self.n_bins:
                break
                
        # Promediar las zonas donde las ventanas se superpusieron y aplicar al prior
        final_delta = x_unfolded / np.maximum(counts, 1)
        final_x = self.x_prior + final_delta
        print("[QUBO] Unfolding completado.")
        return np.maximum(final_x, 0)
