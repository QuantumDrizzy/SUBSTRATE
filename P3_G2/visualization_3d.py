"""
P3_G2/visualization_3d.py
Renderizado local de alto rendimiento del campo gauge U(1) (La sombra del vacío).
Usa PyVista (VTK backend / OpenGL) para levantar un entorno interactivo nativo.
"""
import numpy as np
try:
    import pyvista as pv
except ImportError:
    pv = None
from scipy.ndimage import zoom

def plot_vacuum_3d(theta_field, title="Vacuum Shadow (U(1) Lattice)"):
    """
    Toma un campo de fases theta (L, L), calcula su energía de plaqueta,
    lo interpola masivamente, y lo renderiza en 3D usando OpenGL/VTK.
    """
    if pv is None:
        print("[Visualización 3D] PyVista no está instalado. Ejecuta 'pip install pyvista'.")
        return

    print(f"\n[Visualización 3D] Preparando renderizado bare-metal VTK para '{title}'...")
    
    L = theta_field.shape[1]
    theta_0 = theta_field[0]
    theta_1 = theta_field[1]
    
    # 1. Calcular energía de plaqueta local: E(x,y) = cos(theta_x,y + theta_y,x+1 - theta_x,y+1 - theta_y,x)
    plaquette_energy = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            # Condiciones de contorno periódicas (Toroide)
            ip1 = (i + 1) % L
            jp1 = (j + 1) % L
            
            p_val = theta_0[i, j] + theta_1[ip1, j] - theta_0[i, jp1] - theta_1[i, j]
            # La energía está usualmente relacionada con cos(phi)
            plaquette_energy[i, j] = 1.0 - np.cos(p_val)
            
    # 2. Upsampling masivo para renderizado suave (Interpolación Spline)
    scale_factor = 256 // L  # Queremos un grid ~256x256
    # Aplicar zoom
    high_res_energy = zoom(plaquette_energy, scale_factor, order=3)
    
    # 3. Construir la Malla Espacial 3D (StructuredGrid)
    # Dimensiones
    dim_x, dim_y = high_res_energy.shape
    x = np.linspace(-10, 10, dim_x)
    y = np.linspace(-10, 10, dim_y)
    x, y = np.meshgrid(x, y)
    
    # El eje Z será la energía de plaqueta escalada para impacto visual
    # Suavizamos y exageramos los picos
    z = high_res_energy * 5.0
    
    # Crear la malla estructurada para VTK
    grid = pv.StructuredGrid(x, y, z)
    grid["Energy Density"] = high_res_energy.flatten(order="F")
    
    # 4. Renderizado OpenGL
    print("[Visualización 3D] Levantando motor gráfico. Cierra la ventana 3D para continuar...")
    
    # Configurar el Plotter
    plotter = pv.Plotter(title=title)
    
    # Añadir la malla. Usamos un colormap agresivo (ej. 'magma' o 'inferno')
    plotter.add_mesh(
        grid, 
        scalars="Energy Density", 
        cmap="magma", 
        show_edges=False, 
        smooth_shading=True,
        specular=1.0,      # Brillo especular
        ambient=0.2,       # Iluminación base
        diffuse=0.8,
        render_points_as_spheres=False
    )
    
    # Añadir un plano base negro para simular el abismo
    base_z = np.min(z) - 2.0
    base_grid = pv.StructuredGrid(x, y, np.full_like(z, base_z))
    plotter.add_mesh(base_grid, color="black", opacity=0.8)
    
    # Estética del entorno
    plotter.set_background("black")
    plotter.add_text(f"Quantum Lattice Labs\n{title}", font_size=12, color="white")
    
    # Posicionar cámara para un aspecto dramático
    plotter.camera_position = [
        (25.0, -25.0, 15.0),  # Posición de la cámara
        (0.0, 0.0, np.mean(z)),    # Punto al que mira
        (0.0, 0.0, 1.0)       # Vector Up
    ]
    
    # Lanzar la ventana interactiva
    plotter.show()
    print("[Visualización 3D] Contexto OpenGL cerrado con éxito.")

if __name__ == "__main__":
    # Test rápido
    dummy_theta = np.random.uniform(-np.pi, np.pi, (2, 8, 8))
    plot_vacuum_3d(dummy_theta, "Test Calibration")
