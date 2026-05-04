"""
P3_G2/lattice_hmc.py
2D U(1) gauge configuration generation using vectorized HMC with JAX.
"""
import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

@jax.jit
def plaquette(theta):
    """
    Calculates the plaquette P_01(x) = theta_0(x) + theta_1(x+hat{0}) - theta_0(x+hat{1}) - theta_1(x)
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
    Wilson action for U(1).
    S = -beta * sum(cos(P))
    """
    P = plaquette(theta)
    return -beta * jnp.sum(jnp.cos(P))

# The Force is the scalar gradient of the action with respect to the field
force = jax.jit(jax.grad(action, argnums=0))

@partial(jax.jit, static_argnums=(2, 3))
def hmc_step(theta, key, n_steps, eps, beta):
    """
    Full Hybrid Monte Carlo step using Leapfrog integration.
    """
    key, subkey = jax.random.split(key)
    # Initialize conjugate momentum p ~ N(0, 1)
    p = jax.random.normal(subkey, theta.shape)
    
    # Initial Hamiltonian
    H_init = 0.5 * jnp.sum(p**2) + action(theta, beta)
    
    theta_new = theta
    p_new = p
    
    # Leapfrog (Initial half-step for momentum)
    f = -force(theta_new, beta)
    p_new = p_new + 0.5 * eps * f
    
    # Full steps
    def body_fun(i, val):
        t, p = val
        t = t + eps * p
        f = -force(t, beta)
        p = p + eps * f
        return t, p
        
    theta_new, p_new = jax.lax.fori_loop(0, n_steps - 1, body_fun, (theta_new, p_new))
    
    # Final half-step
    theta_new = theta_new + eps * p_new
    f = -force(theta_new, beta)
    p_new = p_new + 0.5 * eps * f
    
    # Final Hamiltonian
    H_final = 0.5 * jnp.sum(p_new**2) + action(theta_new, beta)
    delta_H = H_final - H_init
    
    # Metropolis-Hastings Acceptance
    key, subkey = jax.random.split(key)
    accept_prob = jnp.minimum(1.0, jnp.exp(-delta_H))
    accept = jax.random.uniform(subkey) < accept_prob
    
    theta_next = jnp.where(accept, theta_new, theta)
    return theta_next, accept, key

def generate_u1_configs(L=8, beta=1.0, n_configs=200, n_steps=10, eps=0.1):
    print(f"[HMC] Starting U(1) thermalization on {L}x{L} lattice with beta={beta}...")
    key = jax.random.PRNGKey(42)
    theta = jnp.zeros((2, L, L))
    
    # Thermalization (Burn-in)
    for _ in range(500):
        theta, _, key = hmc_step(theta, key, n_steps, eps, beta)
        
    configs = []
    acceptances = 0
    
    # Sampling
    for i in range(n_configs):
        # Decorrelation steps in Markov time
        for _ in range(10):
            theta, acc, key = hmc_step(theta, key, n_steps, eps, beta)
            acceptances += acc
            
        # Normalize theta to [-pi, pi) range for cleanliness
        theta_norm = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        configs.append(np.array(theta_norm))
        
        if (i+1) % 50 == 0:
            print(f"  Generated {i+1}/{n_configs} configs...")
            
    print(f"[HMC] Global acceptance rate: {acceptances / (n_configs * 10):.2f}")
    return np.array(configs)
