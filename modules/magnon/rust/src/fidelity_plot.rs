use eframe::egui::{self, Color32, Pos2, Stroke, Vec2};

pub struct FidelityHistory<'a> {
    data: &'a [(f64, f64)],
}

impl<'a> FidelityHistory<'a> {
    pub fn new(data: &'a [(f64, f64)]) -> Self { Self { data } }
    
    pub fn ui(self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(460.0, 240.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        
        if self.data.len() < 2 { return; }
        
        let t_max = self.data.last().unwrap().0;
        
        let x_map = |t: f64| -> f32 {
            rect.left() + rect.width() * ((t - t_max + 60.0) / 60.0) as f32
        };
        let y_map = |f: f64| -> f32 {
            rect.bottom() - rect.height() * f as f32
        };

        for i in 0..=5 {
            let y = rect.bottom() - rect.height() * (i as f32 / 5.0);
            painter.line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                Stroke::new(0.5, Color32::from_rgb(30, 40, 50)),
            );
        }

        let points: Vec<Pos2> = self.data.iter()
            .filter(|(t, _)| *t >= t_max - 60.0)
            .map(|(t, f)| Pos2::new(x_map(*t), y_map(f.clamp(0.0, 1.0))))
            .collect();

        if points.len() >= 2 {
            painter.add(egui::Shape::line(points, Stroke::new(1.5, Color32::from_rgb(0, 255, 136))));
        }
    }
}
