use eframe::egui::{self, Color32, Pos2, Vec2, Stroke, Sense};

pub struct BlochSphere {
    x: f32, y: f32, z: f32,
    fidelity: f32,
}

impl BlochSphere {
    pub fn new(x: f32, y: f32, z: f32, fidelity: f32) -> Self {
        Self { x, y, z, fidelity }
    }

    pub fn ui(self, ui: &mut egui::Ui) {
        let (rect, _response) = ui.allocate_exact_size(
            Vec2::splat(600.0),
            Sense::hover()
        );
        let painter = ui.painter_at(rect);
        let c = rect.center();
        let r = rect.width().min(rect.height()) * 0.4;

        let proj = |x: f32, y: f32, z: f32| -> Pos2 {
            let scale = 0.82_f32;
            Pos2::new(
                c.x + r * (x * scale + y * 0.5),
                c.y - r * (z * scale - y * 0.3),
            )
        };

        let n_lines = 12;
        for i in 0..n_lines {
            let theta = std::f32::consts::PI * i as f32 / n_lines as f32;
            let mut prev = None;
            for j in 0..=n_lines {
                let phi = 2.0 * std::f32::consts::PI * j as f32 / n_lines as f32;
                let p = proj(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos());
                if let Some(pr) = prev {
                    painter.line_segment([pr, p], Stroke::new(0.8, Color32::from_rgb(40, 60, 80)));
                }
                prev = Some(p);
            }
        }

        let axis = |dx, dy, dz, col| {
            let end = proj(dx, dy, dz);
            painter.line_segment([c, end], Stroke::new(2.0, col));
            painter.circle_filled(end, 4.0, col);
        };
        axis(1.3, 0.0, 0.0, Color32::from_rgb(255, 100, 100));
        axis(0.0, 1.3, 0.0, Color32::from_rgb(100, 255, 100));
        axis(0.0, 0.0, 1.3, Color32::from_rgb(100, 100, 255));

        let tip = proj(self.x, self.y, self.z);
        let color_vec = Color32::from_rgb(
            (255.0 * (1.0 - self.fidelity)) as u8,
            (255.0 * self.fidelity) as u8,
            200,
        );
        painter.line_segment([c, tip], Stroke::new(3.5, color_vec));
        painter.circle_filled(tip, 6.0, color_vec);
        
        painter.circle_stroke(c, r * self.fidelity, Stroke::new(2.0, Color32::from_rgb(0, 255, 136)));
    }
}
