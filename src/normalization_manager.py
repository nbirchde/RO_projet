import numpy as np
import math
import logging

log = logging.getLogger(__name__)

def calculate_analytical_factors(n):
    """
    Calculates the analytical mean (μ) and standard deviation (σ) for HS, PS, and MD assuming a random Home/Away assignment. Formulas now use the corrected σ_PS = √[n(n−2)] / 2.
    """
    if n < 2:
        log.warning(f"Cannot calculate analytical factors for n={n}. Must be n >= 2.")
        # Return placeholder values that won't cause division by zero
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)

    R = n - 1

    # 1. HomeStrength (HS)
    # E[HS] = n*(n-1)*(n+1)/12
    # sigma_HS = n*sqrt(n^2 - 1)/(4*sqrt(3))
    mu_HS = n * (n - 1) * (n + 1) / 12
    sigma_HS = n * math.sqrt(n**2 - 1) / (4 * math.sqrt(3))

    # 2. Penalty Sequence (PS)
    #   Mean   : μ_PS = n (n − 2) / 2
    #   Std‑dev: σ_PS = √[ n (n − 2) ] / 2   (valid for n > 2, else use 1.0)
    mu_PS = n * (n - 2) / 2
    sigma_PS = math.sqrt(n * (n - 2)) / 2 if n > 2 else 1.0  # Handle n≤2 gracefully

    # 3. Max Deviation (MD)
    # E[MD] ~ sqrt(R)/2 * sqrt(2*ln n)
    # sigma_MD ~ sqrt(R)/2 * pi / sqrt(12*ln n)
    if n > 1: # ln(1) is undefined, MD is 0 for n=1
        mu_MD = 0.5 * math.sqrt(R) * math.sqrt(2 * math.log(n))
        sigma_MD = 0.5 * math.sqrt(R) * math.pi / math.sqrt(12 * math.log(n)) if n > 1 else 1.0 # Handle n=1 case
    else:
        mu_MD = 0.0
        sigma_MD = 1.0 # Default sigma for n=1

    # Ensure sigmas are not zero or very small
    sigma_HS = max(sigma_HS, 1e-9)
    sigma_PS = max(sigma_PS, 1e-9)
    sigma_MD = max(sigma_MD, 1e-9)

    log.info(f"Calculated analytical factors for n={n}:")
    log.info(f"  HS: Mu={mu_HS:.4f}, Sigma={sigma_HS:.4f}")
    log.info(f"  PS: Mu={mu_PS:.4f}, Sigma={sigma_PS:.4f}")
    log.info(f"  MD: Mu={mu_MD:.4f}, Sigma={sigma_MD:.4f}")

    return (mu_HS, sigma_HS), (mu_PS, sigma_PS), (mu_MD, sigma_MD)

def z_to_scaled(z, z_range=50.0):
    """Maps z ∈ ℝ to [0,1] using smooth sigmoid centered at 0."""
    return 1 / (1 + math.exp(z / z_range))


# This function will be used by solvers to calculate the normalized score
# It now uses analytical factors directly and calculates scaled scores
def calculate_normalized_score(raw_hs, raw_ps, raw_md, alpha, beta, n, z_range=50.0):
    """
    Calculates the combined normalized score using analytical factors and scaled scores.
    Normalization: z = (raw - mu) / sigma
    Scaled Score: scaled = 1 / (1 + exp(z / z_range)), where the default range is 50.0.
    Objective (Z-score sum): Z = z_HS + alpha * z_PS + beta * z_MD
    Objective (Scaled sum): Scaled_Z = scaled_HS + alpha * scaled_PS + beta * scaled_MD
    """
    (mu_hs, sigma_hs), (mu_ps, sigma_ps), (mu_md, sigma_md) = calculate_analytical_factors(n)

    # Calculate z-scores
    obj_hs = (raw_hs - mu_hs) / sigma_hs
    obj_ps = (raw_ps - mu_ps) / sigma_ps
    obj_md = (raw_md - mu_md) / sigma_md

    # Calculate scaled scores
    scaled_hs = z_to_scaled(obj_hs, z_range)
    scaled_ps = z_to_scaled(obj_ps, z_range)
    scaled_md = z_to_scaled(obj_md, z_range)

    # Calculate combined objective scores
    total_normalized_score = obj_hs + alpha * obj_ps + beta * obj_md # Keep the original z-score sum
    scaled_score = scaled_hs + alpha * scaled_ps + beta * scaled_md # New scaled score sum

    # Return both raw z-scores and scaled scores, plus their sums
    return total_normalized_score, obj_hs, obj_ps, obj_md, scaled_score, scaled_hs, scaled_ps, scaled_md
