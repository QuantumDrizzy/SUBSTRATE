/// Protocolo de Memoria Compartida TERRA-QCI
/// 
/// Layout binario (little-endian, 80 bytes total):
///   offset 0:  seq            u64
///   offset 8:  timestamp_ns   u64
///   offset 16: bloch_x        f64
///   offset 24: bloch_y        f64
///   offset 32: bloch_z        f64
///   offset 40: fidelity       f64
///   offset 48: lindblad_gamma f64
///   offset 56: reserved       [u8; 24]
///
/// Python escribe con: struct.pack("<QQddddd24s", ...)
/// Rust lee con: from_bytes()

pub const SHM_SIZE: usize = 80;

#[derive(Debug, Clone, Copy)]
pub struct TerraQciState {
    pub seq: u64,
    pub timestamp_ns: u64,
    pub bloch_x: f64,
    pub bloch_y: f64,
    pub bloch_z: f64,
    pub fidelity: f64,
    pub lindblad_gamma: f64,
}

impl TerraQciState {
    pub fn from_bytes(buf: &[u8]) -> Self {
        assert!(buf.len() >= 56, "SHM buffer too small");
        Self {
            seq:            u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            timestamp_ns:   u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            bloch_x:        f64::from_le_bytes(buf[16..24].try_into().unwrap()),
            bloch_y:        f64::from_le_bytes(buf[24..32].try_into().unwrap()),
            bloch_z:        f64::from_le_bytes(buf[32..40].try_into().unwrap()),
            fidelity:       f64::from_le_bytes(buf[40..48].try_into().unwrap()),
            lindblad_gamma: f64::from_le_bytes(buf[48..56].try_into().unwrap()),
        }
    }
}

impl Default for TerraQciState {
    fn default() -> Self {
        Self {
            seq: 0, timestamp_ns: 0,
            bloch_x: 0.0, bloch_y: 0.0, bloch_z: 1.0,
            fidelity: 1.0, lindblad_gamma: 0.0,
        }
    }
}
