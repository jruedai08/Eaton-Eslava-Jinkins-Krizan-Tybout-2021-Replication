from dataclasses import dataclass
import numpy as np


@dataclass
class Policy:
    """
    restore calculated matrix
    """
    # --- Formula 6 (from solveExpectedMatchProfit) ---
    pi_f: np.ndarray = None       # Expected match profit (n_z, n_phi, n_x)
    c_val_f: np.ndarray = None    # Continuation value (n_phi, n_x)
    pi_h: np.ndarray = None       # Home
    c_val_h: np.ndarray = None    # Home

    # --- Formula 9 (from makeForeignSuccessPosteriors) ---
    post_probs: np.ndarray = None # (n_size, n_size)

    # --- Formula 10 (from solvePolicyForeign & solvePolicyHome) ---
    value_f: np.ndarray = None    # V(a, n, x, phi)
    lambda_f: np.ndarray = None   # Optimal search s*(a, n, x, phi)
    value_h: np.ndarray = None
    lambda_h: np.ndarray = None

    # --- Simulation Helpers (from makeExporterTransitionProbabilities) ---
    prob_meet_f: np.ndarray = None
    prob_succ_f: np.ndarray = None

    # --- Formula 13 (from make_exogenous_firm_transition_probabilities) ---
    firm_type_macro_succ_prod: np.ndarray = None
    pmat_cum_msf: np.ndarray = None  # Foreign Macro
    pmat_cum_msh: np.ndarray = None  # Home Macro
    pmat_cum_z: np.ndarray = None    # Productivity