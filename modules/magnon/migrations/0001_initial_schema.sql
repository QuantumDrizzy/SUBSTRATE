-- TERRA-QCI: Initial Schema — Biological Decoherence Monitor
-- SQLite with WAL mode
--
-- Tables:
--   1. sensor_captures    — Raw EM noise telemetry from SDR/NOAA
--   2. noise_tensors      — Processed H_noise(t) operators
--   3. geomagnetic_states — Earth field measurements from NOAA
--   4. coherence_states   — Quantum observables from Lindblad solver
--   5. audit_log          — SHA-256 chained integrity log

-- ═══════════════════════════════════════════════════════════════
-- 1. SENSOR CAPTURES — Raw RF spectral data
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS sensor_captures (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    source          TEXT    NOT NULL CHECK(source IN ('sdr','noaa','synthetic')),
    center_freq_hz  REAL    NOT NULL,
    bandwidth_hz    REAL    NOT NULL,
    duration_s      REAL    NOT NULL,
    n_samples       INTEGER NOT NULL,
    location        TEXT    DEFAULT 'Aljucer,Murcia',
    -- PSD stored as BLOB (float64 array, little-endian)
    psd_blob        BLOB,
    freqs_blob      BLOB
);

-- ═══════════════════════════════════════════════════════════════
-- 2. NOISE TENSORS — H_noise(t) formatted for the quantum engine
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS noise_tensors (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    capture_id          INTEGER REFERENCES sensor_captures(id),
    -- Effective magnetic noise field (Tesla)
    b_noise_x           REAL    NOT NULL,
    b_noise_y           REAL    NOT NULL,
    b_noise_z           REAL    NOT NULL,
    b_noise_rms         REAL    NOT NULL,
    -- Spectral characteristics
    dominant_freq_hz    REAL    NOT NULL,
    total_power_dbm     REAL    NOT NULL,
    spectral_entropy    REAL    NOT NULL,
    -- The 4×4 Hamiltonian stored as BLOB (complex128[4,4], 256 bytes)
    hamiltonian_blob    BLOB    NOT NULL
);

-- ═══════════════════════════════════════════════════════════════
-- 3. GEOMAGNETIC STATES — Earth field from NOAA SWPC
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS geomagnetic_states (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    source          TEXT    NOT NULL DEFAULT 'noaa_swpc',
    b_total         REAL    NOT NULL,
    b_x             REAL    NOT NULL,
    b_y             REAL    NOT NULL,
    b_z             REAL    NOT NULL,
    kp_index        REAL    DEFAULT 0.0,
    dst_index       REAL    DEFAULT 0.0
);

-- ═══════════════════════════════════════════════════════════════
-- 4. COHERENCE STATES — Quantum observables from Lindblad solver
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS coherence_states (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    noise_tensor_id         INTEGER REFERENCES noise_tensors(id),
    -- Radical pair observables
    singlet_yield           REAL    NOT NULL,
    triplet_yield           REAL    NOT NULL,
    coherence_magnitude     REAL    NOT NULL,
    t2_effective            REAL    NOT NULL,
    fidelity                REAL    NOT NULL CHECK(fidelity >= 0.0 AND fidelity <= 1.01),
    purity                  REAL    NOT NULL CHECK(purity >= 0.0 AND purity <= 1.01),
    entropy                 REAL    NOT NULL,
    compass_sensitivity     REAL    NOT NULL,
    noise_power_ratio       REAL    NOT NULL,
    -- Performance
    solve_time_ms           REAL    NOT NULL,
    solver_method           TEXT    DEFAULT 'lindblad_euler'
);

-- ═══════════════════════════════════════════════════════════════
-- 5. AUDIT LOG — SHA-256 chained (tamper-evident)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS audit_log (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    component       TEXT    NOT NULL CHECK(component IN ('sensor','quantum','system')),
    event_type      TEXT    NOT NULL,
    payload_json    TEXT,
    hash_prev       TEXT    NOT NULL,
    hash            TEXT    NOT NULL
);

-- ═══════════════════════════════════════════════════════════════
-- 6. INDICES
-- ═══════════════════════════════════════════════════════════════

CREATE INDEX IF NOT EXISTS idx_captures_ts     ON sensor_captures(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_captures_src    ON sensor_captures(source);
CREATE INDEX IF NOT EXISTS idx_noise_ts        ON noise_tensors(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_geomag_ts       ON geomagnetic_states(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_coherence_ts    ON coherence_states(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_coherence_fid   ON coherence_states(fidelity);
CREATE INDEX IF NOT EXISTS idx_audit_comp      ON audit_log(component);
