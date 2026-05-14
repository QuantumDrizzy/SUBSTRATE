//! substrate-gpu — wgpu pipeline for the SUBSTRATE iNFAMØUS field renderer
//!
//! Integrates with egui via `egui_wgpu::CallbackTrait`.
//! The `FieldPipeline` is initialized once in `SubstrateGui::new()` and stored
//! in `egui_wgpu::CallbackResources`. Each frame a `FieldCallback` is created
//! with current telemetry data, inserted as a `PaintCallback`, and the pipeline
//! updates its uniform buffer and draws.

use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

// ══════════════════════ CONSTANTS ═════════════════════════════════════════ //

pub const GRID_N: u32 = 24;

// LineList: N rows * (N-1) segs * 2 verts  +  N cols * (N-1) segs * 2 verts
pub const VERT_COUNT: u32 = GRID_N * (GRID_N - 1) * 2 * 2;

// ══════════════════════ UNIFORM BUFFER ════════════════════════════════════ //

/// Must match `struct Uniforms` in shader.wgsl **exactly** (std140 alignment).
///
/// Layout:
///   offset   0 : time, coherence, n_grid, _pad0          → 16 bytes
///   offset  16 : scores [3 × vec4<f32>]                  → 48 bytes
///   offset  64 : corr   [25 × vec4<f32>]  (10×10 flat)   → 400 bytes
///   total      :                                            464 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FieldUniforms {
    pub time:      f32,
    pub coherence: f32,
    pub n_grid:    f32,
    pub _pad0:     f32,
    pub scores:    [[f32; 4]; 3],    // 10 scores, zero-padded to 12
    pub corr:      [[f32; 4]; 25],   // 10×10 → 100 f32, zero-padded to 100
}

impl FieldUniforms {
    pub fn from_telemetry(
        time:      f32,
        coherence: f64,
        scores:    &[f64],
        corr:      &[Vec<f64>],
    ) -> Self {
        let mut s = [[0.0f32; 4]; 3];
        for (k, &sc) in scores.iter().take(10).enumerate() {
            s[k / 4][k % 4] = sc as f32;
        }
        let mut c = [[0.0f32; 4]; 25];
        for i in 0..10 {
            for j in 0..10 {
                let flat = i * 10 + j;
                let val  = if i < corr.len() && j < corr[i].len() { corr[i][j] } else { 0.0 };
                c[flat / 4][flat % 4] = val as f32;
            }
        }
        Self {
            time,
            coherence: coherence as f32,
            n_grid:    GRID_N as f32,
            _pad0:     0.0,
            scores:    s,
            corr:      c,
        }
    }
}

// ══════════════════════ PIPELINE ══════════════════════════════════════════ //

pub struct FieldPipeline {
    pipeline:    wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group:  wgpu::BindGroup,
}

impl FieldPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("substrate_field_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shader.wgsl").into()
            ),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("field_uniforms"),
            size:               std::mem::size_of::<FieldUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("field_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("field_bg"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("field_pl"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("field_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[],   // fully procedural — vertex index only
            },
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format:     target_format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        Self { pipeline, uniform_buf, bind_group }
    }

    pub fn update(&self, queue: &wgpu::Queue, uniforms: &FieldUniforms) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
    }

    pub fn paint<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..VERT_COUNT, 0..1);
    }
}

// ══════════════════════ EGUI CALLBACK ═════════════════════════════════════ //

/// Created every frame with fresh telemetry; stored as a `PaintCallback`.
pub struct FieldCallback {
    pub uniforms: FieldUniforms,
}

impl egui_wgpu::CallbackTrait for FieldCallback {
    fn prepare(
        &self,
        _device:     &wgpu::Device,
        queue:       &wgpu::Queue,
        _screen:     &egui_wgpu::ScreenDescriptor,
        _encoder:    &mut wgpu::CommandEncoder,
        resources:   &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(pipeline) = resources.get::<FieldPipeline>() {
            pipeline.update(queue, &self.uniforms);
        }
        vec![]
    }

    fn paint<'a>(
        &'a self,
        _info:       egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'a>,
        resources:   &'a egui_wgpu::CallbackResources,
    ) {
        if let Some(pipeline) = resources.get::<FieldPipeline>() {
            pipeline.paint(render_pass);
        }
    }
}
