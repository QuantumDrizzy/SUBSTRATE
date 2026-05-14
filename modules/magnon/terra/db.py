import sqlite3
import time
import json
import os
import hashlib
from datetime import datetime, timezone

class TerraDB:
    def __init__(self, db_name="terra_qci.sqlite"):
        # Establecemos la ruta base del proyecto
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(self.project_root, "data", db_name)
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._initialize_schema()

    def _initialize_schema(self):
        schema_path = os.path.join(self.project_root, "migrations", "0001_initial_schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                self.conn.executescript(f.read())
        self.conn.commit()

    # ── Auditoría (SHA-256 Chained) ───────────────────────────────────

    def append_audit(self, component, event_type, payload=None):
        cursor = self.conn.cursor()
        cursor.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1")
        last_row = cursor.fetchone()
        prev_hash = last_row[0] if last_row else "0" * 64
        
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        payload_str = json.dumps(payload) if payload else ""
        
        # Chain hash: prev_hash + ts + component + type + payload
        hash_input = f"{prev_hash}{timestamp}{component}{event_type}{payload_str}"
        current_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        cursor.execute("""
            INSERT INTO audit_log (timestamp_utc, component, event_type, payload_json, hash_prev, hash)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, component, event_type, payload_str, prev_hash, current_hash))
        self.conn.commit()
        return cursor.lastrowid

    # ── Ingesta de Sensores y Ruido ──────────────────────────────────

    def log_sensor_capture(self, capture):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sensor_captures (source, center_freq_hz, bandwidth_hz, duration_s, n_samples, psd_blob, freqs_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (capture.source, capture.center_freq_hz, capture.bandwidth_hz, capture.duration_s, 
              len(capture.psd), capture.psd.tobytes(), capture.freqs.tobytes()))
        self.conn.commit()
        return cursor.lastrowid

    def log_noise_tensor(self, capture_id, tensor):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO noise_tensors (capture_id, b_noise_x, b_noise_y, b_noise_z, b_noise_rms, 
                                      dominant_freq_hz, total_power_dbm, spectral_entropy, hamiltonian_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (capture_id, tensor.b_noise_x, tensor.b_noise_y, tensor.b_noise_z, tensor.b_noise_rms,
              tensor.dominant_freq_hz, tensor.total_power_dbm, tensor.spectral_entropy, tensor.hamiltonian.tobytes()))
        self.conn.commit()
        return cursor.lastrowid

    def log_geomagnetic_state(self, state):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO geomagnetic_states (source, b_total, b_x, b_y, b_z, kp_index)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (state.source, state.b_total, state.b_x, state.b_y, state.b_z, state.kp_index))
        self.conn.commit()
        return cursor.lastrowid

    # ── Coherencia Cuántica ──────────────────────────────────────────

    def log_coherence_state(self, tensor_id, obs):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO coherence_states (noise_tensor_id, singlet_yield, triplet_yield, coherence_magnitude,
                                         t2_effective, fidelity, purity, entropy, compass_sensitivity, 
                                         noise_power_ratio, solve_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tensor_id, obs.singlet_yield, obs.triplet_yield, obs.coherence_magnitude,
              obs.t2_effective, obs.fidelity, obs.purity, obs.entropy, obs.compass_sensitivity,
              obs.noise_power_ratio, obs.solve_time_ms))
        self.conn.commit()
        return cursor.lastrowid
