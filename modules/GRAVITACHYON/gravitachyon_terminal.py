import pygame
import numpy as np
import random
import math
from quantum_foam import QuantumFoam
from penrose_collapse import PenroseCollapse
from graviton_detector import GravitonDetector

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1280, 720
FPS = 60
BG_COLOR = (5, 10, 15)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 150)

class GravitachyonTerminal:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("GRAVITACHYON COMMAND CENTER | Deep Substrate Monitor")
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Courier New", 20, bold=True)
        self.font_small = pygame.font.SysFont("Courier New", 14)
        
        self.foam = QuantumFoam(use_ligo=True)
        self.penrose = PenroseCollapse()
        self.detector = GravitonDetector()
        
        self.running = True
        self.angle = 0
        self.history_stability = []

    def project_3d(self, x, y, z):
        # Proyección 3D simple a 2D
        scale = 150
        off_x = WIDTH // 2
        off_y = HEIGHT // 2 + 50
        
        # Rotación
        rx = x * math.cos(self.angle) - z * math.sin(self.angle)
        rz = x * math.sin(self.angle) + z * math.cos(self.angle)
        
        # Perspectiva
        fov = 400
        factor = fov / (fov + rz + 2)
        px = rx * factor * scale + off_x
        py = y * factor * scale + off_y
        return int(px), int(py)

    def draw_hud(self, stability, signal, events):
        # Marco del terminal
        pygame.draw.rect(self.screen, (20, 30, 40), (10, 10, WIDTH-20, HEIGHT-20), 1)
        
        # Títulos
        title = self.font_main.render(">> GRAVITACHYON: DEEP SUBSTRATE MONITOR [LIGO GW150914]", True, CYAN)
        self.screen.blit(title, (30, 30))
        
        # Readouts
        st_text = self.font_main.render(f"STABILITY: {stability*100:.2f}%", True, CYAN if stability > 0.5 else MAGENTA)
        self.screen.blit(st_text, (30, 70))
        
        sig_text = self.font_main.render(f"DETECTOR SIGNAL: {signal:.4f}", True, GREEN)
        self.screen.blit(sig_text, (30, 100))
        
        # Log de eventos
        log_title = self.font_small.render("TEMPORAL EVENT LOG:", True, (100, 100, 100))
        self.screen.blit(log_title, (WIDTH-250, 30))
        for i, ev in enumerate(events[-10:]):
            ev_text = self.font_small.render(f"> T+{i}: {ev:.4f}", True, GREEN if ev > 0.5 else CYAN)
            self.screen.blit(ev_text, (WIDTH-250, 60 + i*20))

    def run(self):
        event_history = []
        while self.running:
            self.screen.fill(BG_COLOR)
            self.angle += 0.01
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # Obtener datos del sustrato
            substrate = self.foam.generate_fluctuation()
            density = np.mean(substrate)
            stability = self.penrose.calculate_decoherence_time(density)
            signal = self.detector.simulate_exchange(density)
            event_history.append(signal)
            
            # Dibujar la malla 3D (Substrato)
            size = 4
            mesh_points = []
            for i in range(size):
                row = []
                for j in range(size):
                    # Coordenadas 3D (x, y=densidad, z)
                    z_val = substrate[i, j] * 1.5
                    px, py = self.project_3d(i - size/2, z_val, j - size/2)
                    row.append((px, py))
                mesh_points.append(row)
            
            # Dibujar líneas de la malla
            color = CYAN if stability > 0.3 else MAGENTA
            for i in range(size):
                for j in range(size):
                    if i < size - 1:
                        pygame.draw.line(self.screen, color, mesh_points[i][j], mesh_points[i+1][j], 1)
                    if j < size - 1:
                        pygame.draw.line(self.screen, color, mesh_points[i][j], mesh_points[i][j+1], 1)
                    # Puntos de la malla (Glow effect)
                    pygame.draw.circle(self.screen, color, mesh_points[i][j], 3)
            
            # HUD
            self.draw_hud(stability, signal, event_history)
            
            # Scanline effect
            for y in range(0, HEIGHT, 4):
                pygame.draw.line(self.screen, (0, 0, 0, 50), (0, y), (WIDTH, y), 1)

            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    terminal = GravitachyonTerminal()
    terminal.run()
