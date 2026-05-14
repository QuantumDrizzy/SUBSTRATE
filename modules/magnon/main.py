"""
TERRA-QCI — Main Orchestrator
============================
Integrates SDR/NOAA sensors with the Radical Pair Quantum Engine.
"""

import time
import logging
import numpy as np
from terra.sensors.noise_tensor import capture_and_tensorize, fetch_geomagnetic_state, capture_sdr_noise
from terra.quantum.radical_pair import measure_decoherence
from terra.db import TerraDB

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] TERRA-QCI: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

import struct
import mmap
import os

def main():
    logger.info("Iniciando Nodo Táctico TERRA-QCI...")
    db = TerraDB()

    # Inicializar Memoria Compartida (SHM)
    SHM_PATH = "data/terra_qci.shm"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(SHM_PATH):
        with open(SHM_PATH, "wb") as f:
            f.write(b'\x00' * 80)
    
    shm_file = open(SHM_PATH, "r+b")
    shm_mem = mmap.mmap(shm_file.fileno(), 80)
    shm_format = "<QQddddd24s"
    seq_counter = 0
    
    # Registro de inicio en auditoría
    db.append_audit("system", "NODE_STARTUP", {"status": "initializing", "version": "0.1.0"})
    
    SDR_FREQ = 100e6  # 100 MHz
    
    try:
        while True:
            t_start = time.perf_counter()
            
            # 1. CAPTURA RAW (SDR)
            raw_capture = capture_sdr_noise(center_freq=SDR_FREQ, duration=0.1)
            cap_id = db.log_sensor_capture(raw_capture)
            
            # 2. TENSORIZACIÓN (H_noise)
            noise_tensor = capture_and_tensorize(sdr_center_freq=SDR_FREQ)
            tensor_id = db.log_noise_tensor(cap_id, noise_tensor)
            
            # 3. ESTADO GEOMAGNÉTICO
            geo_state = fetch_geomagnetic_state()
            db.log_geomagnetic_state(geo_state)
            
            # 4. SIMULACIÓN CUÁNTICA (Decoherencia)
            logger.info(f"Procesando Noise Tensor #{tensor_id} | RMS: {noise_tensor.b_noise_rms:.2e} T")
            observables = measure_decoherence(
                h_noise=noise_tensor.hamiltonian,
                b_earth=(geo_state.b_x, geo_state.b_y, geo_state.b_z)
            )
            
            # 5. PERSISTENCIA
            db.log_coherence_state(tensor_id, observables)
            
            # 6. PUBLICACIÓN SHM (Dashboard)
            seq_counter += 1
            ts_ns = int(time.time_ns())
            shm_data = struct.pack(
                shm_format,
                seq_counter,
                ts_ns,
                observables.bloch_x,
                observables.bloch_y,
                observables.bloch_z,
                observables.fidelity,
                0.0, # gamma (placeholder)
                b'\x00' * 24
            )
            shm_mem.seek(0)
            shm_mem.write(shm_data)

            # Auditoría periódica de integridad
            if tensor_id % 10 == 0:
                db.append_audit("quantum", "COHERENCE_CHECK", {
                    "last_fidelity": observables.fidelity,
                    "avg_solve_time": observables.solve_time_ms
                })

            # Diagnóstico en consola
            status = "STABLE" if observables.fidelity > 0.9 else "DECOHERING"
            if observables.fidelity < 0.7: status = "COLLAPSED"
            
            logger.info(f"FIDELIDAD: {observables.fidelity:.4f} | T2_eff: {observables.t2_effective*1e6:.2f}µs | {status}")
            
            # Control de frecuencia del bucle
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        db.append_audit("system", "NODE_SHUTDOWN", {"reason": "manual_interrupt"})
        logger.info("Nodo TERRA-QCI detenido.")

if __name__ == "__main__":
    main()
