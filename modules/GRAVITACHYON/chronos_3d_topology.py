import pyvista as pv
import numpy as np
import time
from datetime import datetime

# --- CONFIGURATION ---
GRID_SIZE = 50

# --- SIMULATION STATE ---
class ChronosState:
    def __init__(self):
        self.strain = 0.0
        self.snr = 26.2
        self.ctc_phase = 0.15
        self.logs = [
            "CHRONOS_V8_3D_ENGINE_READY",
            "VOID_LATTICE_INITIALIZED",
            "SENSING_REAL_GW150914"
        ]
        self.last_log_time = time.time()
        self.start_time = time.time()

    def update(self):
        t = time.time() - self.start_time
        # Simulate oscillating strain (-5 to +5)
        self.strain = 5.0 * np.sin(t * 1.5) + np.random.uniform(-0.2, 0.2)
        self.snr = 24.0 + 3.0 * np.sin(t * 0.3) + np.random.uniform(-0.1, 0.1)
        self.ctc_phase = 0.1 + 0.1 * np.abs(np.sin(t * 0.5))
        
        # Periodic Logs
        if time.time() - self.last_log_time > 7.0:
            events = ["CTC_CHANNEL_SYNC", "VOID_STABILITY_LOCK", "RETROCAUSAL_FEEDBACK", "GRAV_LOCK_ACTIVE"]
            new_log = f"[{datetime.now().strftime('%H:%M:%S')}] {np.random.choice(events)}"
            self.logs.append(new_log)
            if len(self.logs) > 6: self.logs.pop(0)
            self.last_log_time = time.time()

# --- INITIALIZATION ---
state = ChronosState()

# Create a structured grid
x = np.linspace(-10, 10, GRID_SIZE)
y = np.linspace(-10, 10, GRID_SIZE)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)

# Create the plotter
plotter = pv.Plotter(title="CHRONOS // 3D TOPOLOGY // STABLE_FINAL_V8.5")
plotter.set_background("black")

# Create the mesh object
mesh = pv.StructuredGrid(x, y, z)
mesh["Z_Height"] = z.flatten()

# Add mesh to plotter
actor = plotter.add_mesh(
    mesh, 
    scalars="Z_Height", 
    cmap="bwr", 
    show_scalar_bar=False,
    lighting=True,
    smooth_shading=True,
    clim=[-6, 6]
)

# Add Telemetry Overlays
telemetry_text = plotter.add_text(
    "CHRONOS TELEMETRY...", 
    position="upper_right", 
    font_size=10, 
    color="cyan", 
    font="courier"
)

log_text = plotter.add_text(
    "SYSTEM LOGS...", 
    position="lower_left", 
    font_size=8, 
    color="orange", 
    font="courier"
)

# --- ANIMATION CALLBACK ---
def update_scene(step):
    # Update internal state
    state.update()
    t = time.time() - state.start_time
    
    # Update Z heights
    amp = state.strain
    new_z = amp * np.sin(np.sqrt(x**2 + y**2) - t * 3.0) * np.exp(-0.04 * (x**2 + y**2))
    
    # Update mesh data
    mesh.points[:, 2] = new_z.flatten()
    mesh["Z_Height"] = new_z.flatten()
    
    # Update Text
    telemetry_str = (
        f"CHRONOS // SOVEREIGN_MONITOR\n"
        f"---------------------------\n"
        f"STRAIN:    {state.strain:+.4f}\n"
        f"SNR:       {state.snr:.2f}\n"
        f"CTC_PHASE: {state.ctc_phase:.4f}\n"
    )
    telemetry_text.set_text(telemetry_str)
    
    log_str = "SYSTEM_EVENT_LOG:\n" + "\n".join([f"> {msg}" for msg in state.logs])
    log_text.set_text(log_str)
    
    # Refresh the plotter
    plotter.render()

# --- RUN LOOP ---
print("CHRONOS 3D Topology Engine Active.")
print("Interact with mouse: Left(Rotate), Right(Zoom), Middle(Pan)")

# Add a timer event to call the update function every 50ms
plotter.add_timer_event(max_steps=None, duration=50, callback=update_scene)

# Show the plotter (this will block until the window is closed)
plotter.show()

print("CHRONOS Shutdown Sequence Complete.")
