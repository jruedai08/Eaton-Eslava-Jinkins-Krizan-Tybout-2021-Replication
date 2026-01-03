import numpy as np
from scipy import sparse
from scipy.stats import norm

def make_q(L, D, n):
    """
    Generate Intensity Matrix.
    
    Args:
        L (float): Jump hazard (Lambda).
        D (float): Jump size (Delta).
        n (int): Half-size grid, 2*n + 1.
        
    Returns:
        Q (sparse.csr_matrix): (2n+1)x(2n+1) matrix.
        st_vec (np.array): (2n+1) state vector, grid points [-nD, ..., nD].
    """
    dim = 2 * n + 1
    
    # state vector
    grid_indices = np.arange(-n, n + 1) # [-n, -n+1, ..., 0, ..., n]
    st_vec = grid_indices * D
    
    # Intensity Matrix, 
    rates_up   = L * 0.5 * (1 - grid_indices / n)
    rates_down = L * 0.5 * (1 + grid_indices / n)
    diag_main = -np.ones(dim) * L # diagonal
    diag_upper = rates_up[:-1]
    diag_lower = rates_down[1:]
    
    Q = sparse.diags(
        [diag_lower, diag_main, diag_upper], 
        offsets=[-1, 0, 1], 
        shape=(dim, dim),
        format='csr' # for linear equation system
    )
    
    return Q, st_vec

def make_erg(L, D, X):
    """
    Calculates steady state of the OU Process from Shimer (2005)

    Args:
        L (float): Arrival rate (Jump hazard, Lambda).
        D (float): Jump size (Delta).
        X (np.array): State vector/Grid points (1D array).

    Returns:
        np.array: Normalized probability vector over the states.
    """
    n = len(X)
    gamma = L / n # Mean reversion parameter
    sig = np.sqrt(L) * D # Diffusion parameter
    variance = (sig ** 2) / (2 * gamma) # steady state variance
    
    # PDF
    std_dev = np.sqrt(variance)
    erg = norm.pdf(X, loc=0, scale=std_dev)
    erg = erg / np.sum(erg) # Discretization
    
    return erg