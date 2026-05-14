import zmq
import time
import math
import json
import random

def iniciar_puente():
    # 1. Inicializar el publicador ZeroMQ
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    
    print("🌌  PUENTE SUBSTRATE-CHRONOS ACTIVADO.")
    print("📡  Transmitiendo telemetría TRISOLARIS (3-Body Waveform) por el puerto 5555...")
    
    t = 0.0
    dt = 0.05 # Resolución temporal
    
    try:
        while True:
            # 2. FÓRMULA DEL CAOS TERNARIO (Trisolaris Strain)
            f1 = 2.1
            f2 = 2.4
            f3 = 1.8
            
            # Superposición caótica + ruido cuántico
            wave = (math.sin(t * f1) + math.sin(t * f2) * 0.8 + math.cos(t * f3) * 0.6)
            noise = (random.random() - 0.5) * 0.1
            
            # Decaimiento lento
            decay = math.exp(-t * 0.02)
            strain = (wave + noise) * decay
            
            # 3. Empaquetar los datos en JSON
            payload = {
                "time": t,
                "strain": strain,
                "status": "CHAOTIC_INSPIRAL" if t < 50 else "MERGER_IMMINENT",
                "snr": 15.0 + math.sin(t * 0.5) * 5.0
            }
            
            # 4. Enviar a Rust con prefijo RINGDOWN
            mensaje = f"RINGDOWN {json.dumps(payload)}"
            socket.send_string(mensaje)
            
            t += dt
            time.sleep(0.016) # ~60fps
            
    except KeyboardInterrupt:
        print("\n🛑  Puente cerrado por el Operador.")
        socket.close()
        context.term()

if __name__ == "__main__":
    iniciar_puente()
