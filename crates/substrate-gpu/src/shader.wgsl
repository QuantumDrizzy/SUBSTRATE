// SUBSTRATE — iNFAMØUS Field Renderer
// Isometric wireframe quantum field with Pearson-driven height + glow

const N:  f32 = 24.0;
const PI: f32 = 3.14159265;

// Uniform layout matches FieldUniforms in lib.rs exactly.
// All array<vec4<f32>, K> have 16-byte stride in WGSL uniform space.
struct Uniforms {
    time:      f32,
    coherence: f32,
    n_grid:    f32,
    _pad0:     f32,
    scores:    array<vec4<f32>, 3>,   // 10 scores packed → 12 floats
    corr:      array<vec4<f32>, 25>,  // 10×10 matrix flattened → 100 floats
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) height:  f32,
    @location(1) fi:      f32,
    @location(2) fj:      f32,
}

// ── Index helpers ─────────────────────────────────────────────────────────

fn get_score(k: u32) -> f32 {
    return u.scores[k / 4u][k % 4u];
}

fn get_corr(i: u32, j: u32) -> f32 {
    let flat = i * 10u + j;
    return u.corr[flat / 4u][flat % 4u];
}

// ── Grid topology: LineList, procedural ───────────────────────────────────
// Horizontal lines: N rows, each with (N-1) segments → N*(N-1)*2 verts
// Vertical   lines: N cols, each with (N-1) segments → N*(N-1)*2 verts

fn grid_ij(idx: u32) -> vec2<f32> {
    let Ni   = u32(N);
    let segs = Ni - 1u;
    let h_verts = Ni * segs * 2u;   // total verts for horizontal lines

    if idx < h_verts {
        let seg = idx / 2u;
        let end = idx % 2u;
        let row = seg / segs;
        let col = seg % segs;
        return vec2<f32>(f32(row), f32(col + end));
    } else {
        let idx2 = idx - h_verts;
        let seg  = idx2 / 2u;
        let end  = idx2 % 2u;
        let col  = seg / segs;
        let row  = seg % segs;
        return vec2<f32>(f32(row + end), f32(col));
    }
}

// ── Field height ──────────────────────────────────────────────────────────
// Combines time-driven waves with per-layer scores

fn field_height(fi: f32, fj: f32) -> f32 {
    // Base wave — multi-frequency interference
    var h: f32 = 0.0;
    h += 0.22 * sin(fi * 3.8 * PI + u.time * 0.27);
    h += 0.18 * cos(fj * 4.2 * PI - u.time * 0.21);
    h += 0.12 * sin((fi + fj) * 5.6 * PI + u.time * 0.44);
    h += 0.07 * cos(fi * fj * 7.8 * PI - u.time * 0.34);

    // Layer score modulation: each score warps a local region of the field
    for (var k: u32 = 0u; k < 10u; k++) {
        let sc  = get_score(k);
        let fi0 = f32(k % 5u) / 4.0;   // 5 cols
        let fj0 = f32(k / 5u) / 1.0;   // 2 rows (0 or 1)
        let d   = sqrt((fi - fi0) * (fi - fi0) + (fj - fj0) * (fj - fj0));
        h += sc * 0.18 * exp(-d * d * 4.0);
    }

    // Global coherence lifts baseline
    h += u.coherence * 0.14;

    // Pearson coupling — correlated layer pairs create constructive interference
    // at their midpoint in the field grid. r > 0 → bump, r < 0 → dip.
    for (var i: u32 = 0u; i < 10u; i++) {
        for (var j: u32 = i + 1u; j < 10u; j++) {
            let r     = get_corr(i, j);
            let fi0   = f32(i % 5u) / 4.0;
            let fj0   = f32(i / 5u);
            let fi1   = f32(j % 5u) / 4.0;
            let fj1   = f32(j / 5u);
            let fmid  = (fi0 + fi1) * 0.5;
            let fjmid = (fj0 + fj1) * 0.5;
            let d     = sqrt((fi - fmid) * (fi - fmid) + (fj - fjmid) * (fj - fjmid));
            h += r * 0.11 * exp(-d * d * 5.0);
        }
    }

    return clamp(h + 0.40, 0.0, 1.0);
}

// ── Color — blue→magenta→red→gold with glow at peaks ─────────────────────

fn field_color(h: f32) -> vec3<f32> {
    var col: vec3<f32>;
    if h < 0.35 {
        let t = h / 0.35;
        col = mix(vec3<f32>(0.04, 0.10, 0.60), vec3<f32>(0.50, 0.05, 0.95), t);
    } else if h < 0.65 {
        let t = (h - 0.35) / 0.30;
        col = mix(vec3<f32>(0.50, 0.05, 0.95), vec3<f32>(1.00, 0.12, 0.28), t);
    } else if h < 0.85 {
        let t = (h - 0.65) / 0.20;
        col = mix(vec3<f32>(1.00, 0.12, 0.28), vec3<f32>(1.00, 0.65, 0.10), t);
    } else {
        let t = (h - 0.85) / 0.15;
        col = mix(vec3<f32>(1.00, 0.65, 0.10), vec3<f32>(1.00, 0.95, 0.80), t);
    }
    // Peak glow: add brightness and shift toward white on highest points
    let glow = pow(max(h - 0.70, 0.0) / 0.30, 2.0) * 0.70;
    col += vec3<f32>(glow * 0.8, glow * 0.5, glow * 0.3);
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Vertex shader ─────────────────────────────────────────────────────────

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VOut {
    let ij = grid_ij(idx);
    let fi = ij.x / (N - 1.0);
    let fj = ij.y / (N - 1.0);
    let h  = field_height(fi, fj);

    // Isometric projection → NDC
    // x: (i-j) * sx,  y: -(i+j)*sy + h*sz + y_bias
    let sx     = 0.082;
    let sy     = 0.038;
    let sz     = 0.48;
    let y_bias = 0.18;   // shift mesh up in viewport

    let x_ndc = (ij.x - ij.y) * sx;
    let y_ndc = -(ij.x + ij.y) * sy + h * sz + y_bias;

    var out: VOut;
    out.clip   = vec4<f32>(x_ndc, y_ndc, 0.5, 1.0);
    out.height = h;
    out.fi     = fi;
    out.fj     = fj;
    return out;
}

// ── Fragment shader ───────────────────────────────────────────────────────

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let col   = field_color(in.height);
    let alpha = 0.70 + in.height * 0.30;

    // Subtle scanline pattern (horizontal banding every ~4px in UV space)
    let scan = 0.5 + 0.5 * sin(in.fi * N * PI * 2.0);
    let dim  = mix(1.0, 0.88, scan * 0.15);

    return vec4<f32>(col * dim, alpha);
}
