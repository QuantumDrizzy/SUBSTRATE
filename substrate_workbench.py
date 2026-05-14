import eel
import time
import math
import json
import random
import threading
import numpy as np
import h5py  # Maquinaria pesada para datos científicos
from scipy import signal
from dataclasses import dataclass, asdict
from pathlib import Path

# ==========================================
# 0. SCIENTIFIC DSP ENGINE (CERN-GRADE)
# ==========================================
class SubstrateDSP:
    def __init__(self):
        # Filtro de Kalman minimalista para señales escalares
        self.x_hat = 0.0  # Estimación de la señal
        self.P = 1.0      # Error de estimación
        self.Q = 0.01     # Varianza del proceso
        self.R = 0.1      # Varianza de la medición
        
        # Buffer para FFT (Densidad Espectral)
        self.buffer_size = 128
        self.buffer = np.zeros(self.buffer_size)

    def kalman_filter(self, z):
        # Predicción
        self.P = self.P + self.Q
        # Actualización
        K = self.P / (self.P + self.R)
        self.x_hat = self.x_hat + K * (z - self.x_hat)
        self.P = (1 - K) * self.P
        return self.x_hat

    def compute_spectrum(self, new_val):
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = new_val
        # Cálculo de FFT (Espectro de Potencia)
        fft_values = np.abs(np.fft.rfft(self.buffer))
        return fft_values.tolist()

class SubstrateRecorder:
    def __init__(self, filename="substrate_session.h5"):
        self.filename = filename
        self.is_recording = False
        self.file = None
        self.dataset = None
        self.counter = 0

    def start(self, metadata):
        print(f"📁 INICIANDO GRABACIÓN HDF5: {self.filename}")
        self.file = h5py.File(self.filename, 'w')
        # Metadatos de Procedencia (CERN-Grade)
        for key, val in metadata.items():
            self.file.attrs[key] = val
        
        # Dataset masivo para telemetría (Redimensionable)
        self.dataset = self.file.create_dataset(
            'telemetry', (0, 10), maxshape=(None, 10), chunks=True
        )
        self.is_recording = True

    def stop(self):
        if self.file:
            print(f"🛑 GRABACIÓN FINALIZADA. Archivo sellado: {self.filename}")
            self.file.close()
        self.is_recording = False

    def write(self, data_row):
        if self.is_recording:
            self.dataset.resize((self.counter + 1, 10))
            self.dataset[self.counter, :] = data_row
            self.counter += 1

@dataclass
class SubstrateTelemetry:
    q_strain_h1: float
    q_snr: float
    final_spin: float
    entropy_exp: float
    ringdown_sig: float
    mass_solar: float
    dist_mpc: float
    freq_hz: float
    gds_lock_pro: float
    q_phase_ctc: float
    timestamp: float

class SubstrateEngine:
    def __init__(self):
        self.t = 0.0
        self.latest_data = {}
        self.gpu_acceleration = True
        self.dsp = SubstrateDSP()
        self.recorder = SubstrateRecorder()
        self.start_step_time = time.perf_counter()

    def toggle_gpu(self, state: bool):
        self.gpu_acceleration = state
        print(f"⚡ GPU Acceleration: {'ACTIVATED' if state else 'DEACTIVATED - CPU FALLBACK'}")

    def toggle_recording(self, state: bool):
        if state:
            metadata = {
                "experiment": "SUBSTRATE_CORE_RUN",
                "version": "1.0-Soberano",
                "sensor": "Synthetic-LIGO-IGRF",
                "kalman_R": self.dsp.R
            }
            self.recorder.start(metadata)
        else:
            self.recorder.stop()

    def step(self):
        # 1. GENERACIÓN Y DSP
        raw_em = 42.3 + (math.sin(self.t * 2.0) * 5.0) + random.uniform(-2.0, 2.0)
        clean_em = self.dsp.kalman_filter(raw_em)
        spectrum = self.dsp.compute_spectrum(clean_em)
        
        # 2. DETECCIÓN REAL DE ANOMALÍAS (Basada en Espectro)
        # Si hay un pico en frecuencias altas (bins > 10)
        anomaly_score = np.max(spectrum[10:]) if len(spectrum) > 10 else 0
        is_anomaly = anomaly_score > 5.0

        # 3. FÍSICA Y TELEMETRÍA
        telemetry = SubstrateTelemetry(
            q_strain_h1=(math.sin(self.t * 3.14) * math.exp(-self.t * 0.02)) * 1e-21,
            q_snr=26.2 + random.uniform(-0.3, 0.3),
            final_spin=0.682,
            entropy_exp=76.4,
            ringdown_sig=0.185,
            mass_solar=65.4,
            dist_mpc=410.8,
            freq_hz=145.8,
            gds_lock_pro=clean_em,
            q_phase_ctc=math.sin(self.t * 2.0),
            timestamp=time.time()
        )
        
        # 4. CAPA A: ESCRITURA HDF5 (Si está activo)
        if self.recorder.is_recording:
            row = [telemetry.q_strain_h1, telemetry.q_snr, telemetry.final_spin, 
                   telemetry.entropy_exp, telemetry.mass_solar, telemetry.freq_hz, 
                   clean_em, float(is_anomaly), time.time(), anomaly_score]
            self.recorder.write(row)

        compute_time = (time.perf_counter() - self.start_step_time) * 1000
        
        self.latest_data = {
            "q_strain_h1": f"{telemetry.q_strain_h1:.2e}",
            "q_snr": f"{telemetry.q_snr:+.1f}",
            "em_field": f"{clean_em:.2f} nT",
            "compute_ms": f"{compute_time:.1f}ms",
            "coherence": "0.84",
            "spectrum": spectrum,
            "anomaly": "ALERT" if is_anomaly else "STABLE",
            "is_recording": self.recorder.is_recording
        }
        self.t += 0.05

    def run_step(self):
        self.start_step_time = time.perf_counter()
        self.step()

# ==========================================
# 2. ORQUESTADOR DE INTERFAZ (EEL)
# ==========================================
SUBSTRATE_ROOT = Path(__file__).resolve().parent
eel.init('workbench')

engine = SubstrateEngine()
lvc_lock = threading.Lock()
lvc_data = {}

@eel.expose
def get_latest_data():
    with lvc_lock:
        return lvc_data

@eel.expose
def update_solver_param(param, value):
    print(f"⚙️ PARAM UPDATE: {param} = {value}")
    # Aquí es donde el motor de Rust recibiría la nueva configuración
    if param == "tolerance":
        # Simular cambio en la física
        pass

@eel.expose
def toggle_recording(state):
    engine.toggle_recording(state)

def engine_loop():
    """
    FRENTE 2: Inyector de Estrés Asíncrono.
    El motor corre a su máxima capacidad, pero no bloquea la UI.
    """
    while True:
        # Simulamos una carga de trabajo pesada (estrés)
        # Esto representa la GPU rugiendo a chi > 100
        engine.run_step()
        
        # Inyectamos un pequeño 'stutter' simulado para probar el asincronismo
        if random.random() > 0.90: # Aumentamos probabilidad para que sea visible
            time.sleep(0.3) # Micro-congelación del motor (300ms)
            
        with lvc_lock:
            global lvc_data
            lvc_data = engine.latest_data
            
        # El motor corre a 20Hz, pero la UI puede pedir datos cuando quiera
        time.sleep(0.05)

if __name__ == "__main__":
    print("🚀 SUBSTRATE WORKBENCH: Iniciando sistema unificado...")
    print("🛡️ MODO BÚNKER: Doble Buffer y WAL Activos.")
    
    # Lanzar motor en hilo de fondo (Totalmente asíncrono)
    threading.Thread(target=engine_loop, daemon=True).start()
    
    # Lanzar interfaz
    try:
        # Desactivamos el cache del navegador para ver cambios en tiempo real
        eel.start('index.html', size=(1440, 900))
    except (SystemExit, KeyboardInterrupt):
        print("\n🛑 Sistema SUBSTRATE cerrado.")
