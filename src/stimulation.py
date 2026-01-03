import numpy as np

def simulate_macro_trajectories(params, policy):
    """
    Generate time series for exogenous macro states (Home and Foreign).
    
    Args:
        mm: Model configuration object (needs mm.periods, mm.x_size)
        policy: Policy object (needs pmat_cum_msf, pmat_cum_msh)
        
    Returns:
        macro_state_f (np.ndarray): Foreign macro state trajectory [T,]
        macro_state_h (np.ndarray): Home macro state trajectory [T,]
    """
    
    # Initialize arrays
    macro_state_f = np.zeros(params.periods, dtype=int)
    macro_state_h = np.zeros(params.periods, dtype=int)
    start_idx = params.x_size # starting at midpoint of distribution
    macro_state_f[0] = start_idx
    macro_state_h[0] = start_idx
    
    # Simulate forward (Markov Chain)
    for t in range(1, params.periods):
        # --- Foreign Market ---
        prev_state_f = macro_state_f[t-1]
        rand_val_f = np.random.rand()
        macro_state_f[t] = np.searchsorted(policy.pmat_cum_msf[prev_state_f], rand_val_f)
        
        # --- Home Market ---
        prev_state_h = macro_state_h[t-1]
        rand_val_h = np.random.rand()
        macro_state_h[t] = np.searchsorted(policy.pmat_cum_msh[prev_state_h], rand_val_h)
        
    return macro_state_f, macro_state_h