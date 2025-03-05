import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

"""
Probability Distributions Implementation

This file contains implementations of probability density functions (PDFs) and 
cumulative distribution functions (CDFs) for three common distributions:
1. Gaussian (Normal) Distribution
2. Poisson Distribution
3. Uniform Distribution

Each implementation includes:
- Mathematical derivation
- Custom implementation
- Verification against SciPy's implementation
- Visualization
"""

# ============================================================================
# 1. GAUSSIAN (NORMAL) DISTRIBUTION
# ============================================================================
"""
Mathematical Derivation of Gaussian Distribution:

The probability density function (PDF) of a Gaussian distribution is:
f(x; μ, σ) = (1 / (σ * sqrt(2π))) * exp(-(x - μ)² / (2σ²))

Where:
- μ is the mean (expected value)
- σ is the standard deviation
- σ² is the variance

The cumulative distribution function (CDF) is:
F(x; μ, σ) = (1/2) * [1 + erf((x - μ) / (σ * sqrt(2)))]

Where erf is the error function:
erf(z) = (2/sqrt(π)) * ∫(0 to z) exp(-t²) dt
"""

def gaussian_pdf(x, mu=0, sigma=1):
    """
    Gaussian (Normal) Probability Density Function
    
    Parameters:
    - x: point(s) at which to evaluate the PDF
    - mu: mean of the distribution
    - sigma: standard deviation of the distribution
    
    Returns:
    - PDF value(s) at x
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)

def gaussian_cdf(x, mu=0, sigma=1):
    """
    Gaussian (Normal) Cumulative Distribution Function
    
    Parameters:
    - x: point(s) at which to evaluate the CDF
    - mu: mean of the distribution
    - sigma: standard deviation of the distribution
    
    Returns:
    - CDF value(s) at x
    """
    z = (x - mu) / (sigma * np.sqrt(2))
    return 0.5 * (1 + np.vectorize(math.erf)(z))

# ============================================================================
# 2. POISSON DISTRIBUTION
# ============================================================================
"""
Mathematical Derivation of Poisson Distribution:

The Poisson distribution models the number of events occurring in a fixed time interval,
given the average rate of occurrence.

The probability mass function (PMF) is:
P(X = k; λ) = (λ^k * e^(-λ)) / k!

Where:
- λ (lambda) is the average rate of events per interval
- k is the number of events (non-negative integer)
- e is Euler's number (≈ 2.71828)
- k! is the factorial of k

The cumulative distribution function (CDF) is:
F(k; λ) = ∑(i=0 to k) (λ^i * e^(-λ)) / i!

This is the sum of PMF values from 0 to k.
"""

def poisson_pmf(k, lam):
    """
    Poisson Probability Mass Function
    
    Parameters:
    - k: number of events (non-negative integer)
    - lam: average rate of events (lambda)
    
    Returns:
    - PMF value at k
    """
    if isinstance(k, np.ndarray):
        # Handle array input
        result = np.zeros_like(k, dtype=float)
        valid_indices = (k >= 0) & (k == np.floor(k))
        k_valid = k[valid_indices]
        
        if len(k_valid) > 0:
            result[valid_indices] = np.exp(-lam) * (lam ** k_valid) / np.array([math.factorial(int(ki)) for ki in k_valid])
        return result
    elif k >= 0 and k == int(k):
        # Handle scalar input
        return np.exp(-lam) * (lam ** k) / math.factorial(int(k))
    else:
        return 0

def poisson_cdf(k, lam):
    """
    Poisson Cumulative Distribution Function
    
    Parameters:
    - k: number of events (non-negative integer)
    - lam: average rate of events (lambda)
    
    Returns:
    - CDF value at k
    """
    if isinstance(k, np.ndarray):
        # Handle array input
        result = np.zeros_like(k, dtype=float)
        for i in range(len(k)):
            if k[i] >= 0 and k[i] == int(k[i]):
                result[i] = sum(poisson_pmf(np.arange(int(k[i]) + 1), lam))
        return result
    elif k >= 0 and k == int(k):
        # Handle scalar input
        return sum(poisson_pmf(np.arange(int(k) + 1), lam))
    else:
        return 0

# ============================================================================
# 3. UNIFORM DISTRIBUTION
# ============================================================================
"""
Mathematical Derivation of Uniform Distribution:

The uniform distribution represents equal probability over a range [a, b].

The probability density function (PDF) is:
f(x; a, b) = 1 / (b - a) for a ≤ x ≤ b
f(x; a, b) = 0 otherwise

Where:
- a is the lower bound
- b is the upper bound

The cumulative distribution function (CDF) is:
F(x; a, b) = 0 for x < a
F(x; a, b) = (x - a) / (b - a) for a ≤ x ≤ b
F(x; a, b) = 1 for x > b
"""

def uniform_pdf(x, a=0, b=1):
    """
    Uniform Probability Density Function
    
    Parameters:
    - x: point(s) at which to evaluate the PDF
    - a: lower bound of the distribution
    - b: upper bound of the distribution
    
    Returns:
    - PDF value(s) at x
    """
    if isinstance(x, np.ndarray):
        # Handle array input
        result = np.zeros_like(x, dtype=float)
        valid_indices = (x >= a) & (x <= b)
        result[valid_indices] = 1 / (b - a)
        return result
    else:
        # Handle scalar input
        return 1 / (b - a) if a <= x <= b else 0

def uniform_cdf(x, a=0, b=1):
    """
    Uniform Cumulative Distribution Function
    
    Parameters:
    - x: point(s) at which to evaluate the CDF
    - a: lower bound of the distribution
    - b: upper bound of the distribution
    
    Returns:
    - CDF value(s) at x
    """
    if isinstance(x, np.ndarray):
        # Handle array input
        result = np.zeros_like(x, dtype=float)
        below_indices = x < a
        between_indices = (x >= a) & (x <= b)
        above_indices = x > b
        
        result[below_indices] = 0
        result[between_indices] = (x[between_indices] - a) / (b - a)
        result[above_indices] = 1
        return result
    else:
        # Handle scalar input
        if x < a:
            return 0
        elif x <= b:
            return (x - a) / (b - a)
        else:
            return 1

# ============================================================================
# VISUALIZATION AND VERIFICATION
# ============================================================================

def plot_distributions():
    """
    Plot and verify all implemented distributions against SciPy implementations
    """
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    # 1. Gaussian Distribution
    x_gaussian = np.linspace(-5, 5, 1000)
    
    # Plot PDF
    axs[0, 0].plot(x_gaussian, gaussian_pdf(x_gaussian), 'b-', label='Custom Implementation')
    axs[0, 0].plot(x_gaussian, stats.norm.pdf(x_gaussian), 'r--', label='SciPy Implementation')
    axs[0, 0].set_title('Gaussian PDF (μ=0, σ=1)')
    axs[0, 0].legend()
    
    # Plot CDF
    axs[0, 1].plot(x_gaussian, gaussian_cdf(x_gaussian), 'b-', label='Custom Implementation')
    axs[0, 1].plot(x_gaussian, stats.norm.cdf(x_gaussian), 'r--', label='SciPy Implementation')
    axs[0, 1].set_title('Gaussian CDF (μ=0, σ=1)')
    axs[0, 1].legend()
    
    # 2. Poisson Distribution
    lam = 5
    x_poisson = np.arange(0, 15)
    
    # Plot PMF
    axs[1, 0].stem(x_poisson, poisson_pmf(x_poisson, lam), 'b', markerfmt='bo', label='Custom Implementation')
    axs[1, 0].stem(x_poisson, stats.poisson.pmf(x_poisson, lam), 'r', markerfmt='rx', label='SciPy Implementation')
    axs[1, 0].set_title(f'Poisson PMF (λ={lam})')
    axs[1, 0].legend()
    
    # Plot CDF
    axs[1, 1].step(x_poisson, poisson_cdf(x_poisson, lam), 'b-', where='post', label='Custom Implementation')
    axs[1, 1].step(x_poisson, stats.poisson.cdf(x_poisson, lam), 'r--', where='post', label='SciPy Implementation')
    axs[1, 1].set_title(f'Poisson CDF (λ={lam})')
    axs[1, 1].legend()
    
    # 3. Uniform Distribution
    a, b = 2, 6
    x_uniform = np.linspace(0, 8, 1000)
    
    # Plot PDF
    axs[2, 0].plot(x_uniform, uniform_pdf(x_uniform, a, b), 'b-', label='Custom Implementation')
    axs[2, 0].plot(x_uniform, stats.uniform.pdf(x_uniform, a, b-a), 'r--', label='SciPy Implementation')
    axs[2, 0].set_title(f'Uniform PDF (a={a}, b={b})')
    axs[2, 0].legend()
    
    # Plot CDF
    axs[2, 1].plot(x_uniform, uniform_cdf(x_uniform, a, b), 'b-', label='Custom Implementation')
    axs[2, 1].plot(x_uniform, stats.uniform.cdf(x_uniform, a, b-a), 'r--', label='SciPy Implementation')
    axs[2, 1].set_title(f'Uniform CDF (a={a}, b={b})')
    axs[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('probability_distributions.png')
    plt.show()

if __name__ == "__main__":
    # Verify implementations against SciPy
    print("Verifying implementations against SciPy...")
    
    # Gaussian
    x_test = 1.5
    print(f"\nGaussian at x={x_test}:")
    print(f"Custom PDF: {gaussian_pdf(x_test):.6f}, SciPy PDF: {stats.norm.pdf(x_test):.6f}")
    print(f"Custom CDF: {gaussian_cdf(x_test):.6f}, SciPy CDF: {stats.norm.cdf(x_test):.6f}")
    
    # Poisson
    k_test = 3
    lam_test = 2.5
    print(f"\nPoisson at k={k_test}, λ={lam_test}:")
    print(f"Custom PMF: {poisson_pmf(k_test, lam_test):.6f}, SciPy PMF: {stats.poisson.pmf(k_test, lam_test):.6f}")
    print(f"Custom CDF: {poisson_cdf(k_test, lam_test):.6f}, SciPy CDF: {stats.poisson.cdf(k_test, lam_test):.6f}")
    
    # Uniform
    x_test = 3.5
    a_test, b_test = 2, 5
    print(f"\nUniform at x={x_test}, a={a_test}, b={b_test}:")
    print(f"Custom PDF: {uniform_pdf(x_test, a_test, b_test):.6f}, SciPy PDF: {stats.uniform.pdf(x_test, a_test, b_test-a_test):.6f}")
    print(f"Custom CDF: {uniform_cdf(x_test, a_test, b_test):.6f}, SciPy CDF: {stats.uniform.cdf(x_test, a_test, b_test-a_test):.6f}")
    
    # Generate plots
    plot_distributions() 