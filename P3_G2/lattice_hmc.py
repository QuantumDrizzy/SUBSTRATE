"""
P3_G2/lattice_hmc.py
Generación de configuraciones gauge U(1) en 2D usando HMC vectorizado con JAX.
"""
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

@jax.jit
def plaquette(theta):
    """
    Calcula la plaqueta P_01(x) = theta_0(x) + theta_1(x+hat{0}) - theta_0(x+hat{1}) - theta_1(x)
    theta shape: (2, L, L)
    """
    theta_0 = theta[0]
    theta_1 = theta[1]
    
    theta_1_shifted_0 = jnp.roll(theta_1, shift=-1, axis=0) # x + hat{0}
    theta_0_shifted_1 = jnp.roll(theta_0, shift=-1, axis=1) # x + hat{1}
    
    P = theta_0 + theta_1_shifted_0 - theta_0_shifted_1 - theta_1
    return P

@jax.jit
def action(theta, beta):
    """
    Acción de Wilson para U(1).
    S = -beta * sum(cos(P))
    """
    P = plaquette(theta)
    return -beta * jnp.sum(jnp.cos(P))

# La Fuerza es el gradiente escalar de la acción respecto al campo
force = jax.jit(jax.grad(action, argnums=0))

@partial(jax.jit, static_argnums=(2, 3))
def hmc_step(theta, key, n_steps, eps, beta):
    """
    Un paso completo de Hybrid Monte Carlo usando integración Leapfrog.
    """
    key, subkey = jax.random.split(key)
    # Inicializar momento conjugado p ~ N(0, 1)
    p = jax.random.normal(subkey, theta.shape)
    
    # Hamiltoniano Inicial
    H_init = 0.5 * jnp.sum(p**2) + action(theta, beta)
    
    theta_new = theta
    p_new = p
    
    # Leapfrog (Medio paso inicial de momento)
    f = -force(theta_new, beta)
    p_new = p_new + 0.5 * eps * f
    
    # Pasos completos
    def body_fun(i, val):
        t, p = val
        t = t + eps * p
        f = -force(t, beta)
        p = p + eps * f
        return t, p
        
    theta_new, p_new = jax.lax.fori_loop(0, n_steps - 1, body_fun, (theta_new, p_new))
    
    # Último medio paso
    theta_new = theta_new + eps * p_new
    f = -force(theta_new, beta)
    p_new = p_new + 0.5 * eps * f
    
    # Hamiltoniano Final
    H_final = 0.5 * jnp.sum(p_new**2) + action(theta_new, beta)
    delta_H = H_final - H_init
    
    # Aceptación Metropolis-Hastings
    key, subkey = jax.random.split(key)
    accept_prob = jnp.minimum(1.0, jnp.exp(-delta_H))
    accept = jax.random.uniform(subkey) < accept_prob
    
    theta_next = jnp.where(accept, theta_new, theta)
    return theta_next, accept, key

def generate_u1_configs(L=8, beta=1.0, n_configs=200, n_steps=10, eps=0.1):
    print(f"[HMC] Iniciando termalización U(1) en lattice {L}x{L} con beta={beta}...")
    key = jax.random.PRNGKey(42)
    theta = jnp.zeros((2, L, L))
    
    # Termalización (Quemado inicial)
    for _ in range(500):
        theta, _, key = hmc_step(theta, key, n_steps, eps, beta)
        
    configs = []
    acceptances = 0
    
    # Muestreo
    for i in range(n_configs):
        # Pasos de descorrelación en el tiempo de Markov
        for _ in range(10):
            theta, acc, key = hmc_step(theta, key, n_steps, eps, beta)
            acceptances += acc
            
        # Normalizar theta al rango [-pi, pi) por limpieza
        theta_norm = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        configs.append(np.array(theta_norm))
        
        if (i+1) % 50 == 0:
            print(f"  Generadas {i+1}/{n_configs} configs...")
            
    print(f"[HMC] Tasa de aceptación global: {acceptances / (n_configs * 10):.2f}")
    return np.array(configs)
