import eel
import os
import threading
import time
from pathlib import Path
from substrate_core_emitter import SubstrateEngine

# 1. Configuración de Rutas
SUBSTRATE_ROOT = Path(__file__).resolve().parent
eel.init('workbench')

# 2. Orquestación del Motor (Background Thread)
latest_telemetry = {}

def run_engine_background():
    """Ejecuta el motor de física en un hilo separado"""
    engine = SubstrateEngine()
    # Modificamos el run del motor para que guarde en la variable global
    # en lugar de solo emitir por ZMQ (para redundancia local)
    try:
        while True:
            data = engine.calculate_quantum_fields()
            global latest_telemetry
            latest_telemetry = {
                "q_strain_h1": f"{data.q_strain_h1:.2e}",
                "q_snr": f"{data.q_snr:+.1f}",
                "final_spin": f"{data.final_spin:.3f}",
                "entropy_exp": f"{data.entropy_exp:+.1f}",
                "ringdown_sig": f"{data.ringdown_sig:.3f}",
                "mass_solar": f"{data.mass_solar:.1f}",
                "dist_mpc": f"{data.dist_mpc:.1f}",
                "freq_hz": f"{data.freq_hz:.1f}",
                "gds_lock_pro": f"{data.gds_lock_pro:.3f}",
                "q_phase_ctc": f"{data.q_phase_ctc/3.14:.3f}π",
                "timestamp": data.timestamp
            }
            # También lo enviamos por ZMQ por si otros procesos escuchan
            import json
            from dataclasses import asdict
            engine.publisher.send_string(f"SUBSTRATE_STATE {json.dumps(asdict(data))}")
            
            engine.t += 0.05
            time.sleep(0.05)
    except Exception as e:
        print(f"❌ Error en el motor: {e}")

# Iniciar el motor inmediatamente
threading.Thread(target=run_engine_background, daemon=True).start()

# 3. Exposición de Datos al UI
@eel.expose
def get_latest_data():
    """Devuelve la telemetría viva del motor de fondo"""
    return latest_telemetry

# 4. Lanzamiento del Workbench
print("🚀 LANZANDO SISTEMA UNIFICADO SUBSTRATE...")
try:
    # Abrir el Workbench en modo App (sin barras de navegador para máxima inmersión)
    eel.start('index.html', size=(1440, 900), mode='chrome')
except Exception as e:
    print(f"Error al lanzar UI: {e}")
    eel.start('index.html', size=(1440, 900))
