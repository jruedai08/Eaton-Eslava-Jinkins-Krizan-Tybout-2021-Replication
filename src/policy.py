import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from tqdm import tqdm

def make_exporter_transition_probabilities(policy, params):
    """
    Constructs the lookup table for firm types and calculates exogenous 
    transition probabilities for the simulation step.
    
    Corresponds to: makeExporterTransitionProbabilities.m
    """
    print("  > Constructing State Space and Exogenous Transitions...")

    # Firm type is determined by Phi, theta and X
    n_phi = params.n_phi
    n_theta = params.dim1
    n_x = params.x_size
    total_states = n_phi * n_theta * n_x
    # index
    col_x = np.repeat(np.arange(n_x), n_phi * n_theta) # slowest
    col_theta = np.tile(np.tile(np.arange(n_theta), n_phi), n_x) # fastest
    col_phi = np.tile(np.repeat(np.arange(n_phi), n_theta), n_x) # middle
    
    # restore
    firm_type_matrix = np.column_stack((
        np.arange(total_states),
        col_x,
        col_theta,
        col_phi
    ))
    
    policy.firm_type_macro_succ_prod = firm_type_matrix.astype(int)

    # ensogenous transition probabilities ((n,a) or A)
    policy = intensity_to_probability_foreign(params, policy)
    policy = intensity_to_probability_home(params, policy)
    
    # exogenous transition probabilities (X)
    policy = make_exogenous_firm_transition_probabilities(params, policy)
    
    return policy




def intensity_to_probability_foreign(params, policy):
    """
    Constructs the transition probability matrix for Foreign Market states (n, a).
    
    Logic:
    1. Iterate over each firm type (Macro, Theta, Phi).
    2. Build a Q-matrix (Generator) representing transitions between (n, a) states.
       - Success: (n, a) -> (n+1, a+1)
       - Failure: (n, a) -> (n+1, a)
    3. Compute P = expm(Q * dt) to get discrete probabilities.
    4. Store cumulative probabilities for simulation sampling.
    """
    print("  > Building Foreign Transition Matrices (Endogenous)...")

    n_types = policy.firm_type_macro_succ_prod.shape[0] # Total firm types
    n_max = params.n_size
    
    # Lower Triangle Only
    n_states_learning = (n_max + 1) * (n_max + 2) // 2
    
    # initialize the restore
    policy.pmat_cum_f = [] 
    
    # mapping from (n, a) to linear index
    state_map = {}
    idx_counter = 0
    for n in range(n_max + 1):
        for a in range(n + 1):
            state_map[(n, a)] = idx_counter
            idx_counter += 1

    # Loop over all firm types
    for typ_idx in tqdm(range(n_types), desc="Foreign Trans Probs"):
        
        # Get Firm Specs
        # Row format: [ID, Macro_Idx, Theta_Idx, Phi_Idx]
        row = policy.firm_type_macro_succ_prod[typ_idx]
        macro_idx = row[1]
        theta_idx = row[2]
        phi_idx   = row[3]
        succ_prob = params.theta2[theta_idx] # True Theta Probability
        
        # restore Intensity Matrix Q, [(n_size*a_size),(n_size*a_size)]
        data = []
        rows = []
        cols = []
        
        # Iterate through all source states (n, a)
        for n in range(n_max + 1):
            for a in range(n + 1):
                curr_linear_idx = state_map[(n, a)] # index
                if n < n_max:
                    s = policy.lambda_f[n, a, a, phi_idx, macro_idx] # optimal search intensity
                    
                    target_succ = state_map[(n+1, a+1)]
                    rate_succ = s * succ_prob # success
                    target_fail = state_map[(n+1, a)]
                    rate_fail = s * (1.0 - succ_prob) # fail
                    
                    data.append(rate_succ); rows.append(curr_linear_idx); cols.append(target_succ)
                    data.append(rate_fail); rows.append(curr_linear_idx); cols.append(target_fail)
                    data.append(-s); rows.append(curr_linear_idx); cols.append(curr_linear_idx) # diagonal
                    
                else: # reach the boundary, can't match more
                    #data.append(-params.firm_death_haz); rows.append(curr_linear_idx); cols.append(curr_linear_idx)
                    pass

        # Construct Sparse Matrix
        Q_sparse = sp.coo_matrix((data, (rows, cols)), shape=(n_states_learning, n_states_learning))
        Q_dense = Q_sparse.toarray()
        P_mat = expm(Q_dense) # converge into probability
        ## normalization
        row_sums = P_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # safe guard
        P_mat = P_mat / row_sums
        
        P_cum = np.cumsum(P_mat, axis=1)
        policy.pmat_cum_f.append(P_cum) # restore culmulative probability
        
    return policy



def intensity_to_probability_home(params, policy):
    """
    Constructs the transition probability matrix for Home Market states (Network Size).
    
    Logic:
    Home market state is just Network Size (m).
    Transitions: m -> m+1 (Success) with rate s * theta.
    Failure keeps m -> m.
    """
    print("  > Building Home Transition Matrices (Endogenous)...")

    n_types = policy.firm_type_macro_succ_prod.shape[0] # firm types

    net_max = params.net_size
    n_states = net_max + 1

    # Initialize storage
    policy.pmat_cum_h = []
    
    for typ_idx in tqdm(range(n_types), desc="Home Trans Probs"):
        
        # 1. Get Firm Specs
        row = policy.firm_type_macro_succ_prod[typ_idx]
        macro_idx = row[1]
        theta_idx = row[2] # Home Theta Type
        phi_idx   = row[3]
        
        theta_val = params.theta1[theta_idx] # exactly the same as theta2
        
        data = []
        rows = []
        cols = []
        
        # Iterate over Network Size m (0 to net_max)
        for m in range(n_states):
            lambda_m_idx = min(m, policy.lambda_h.shape[1] - 1) # safe guard
            s = policy.lambda_h[theta_idx, lambda_m_idx, phi_idx, macro_idx] # optimal search intensity
            rate_succ = s * theta_val # success rate
            
            if m < net_max:
                data.append(rate_succ); rows.append(m); cols.append(m+1)
                data.append(-rate_succ); rows.append(m); cols.append(m) # diagonal
                
            else:
                pass # Pure absorbing state
                
        Q_sparse = sp.coo_matrix((data, (rows, cols)), shape=(n_states, n_states))
        Q_dense = Q_sparse.toarray()
        P_mat = expm(Q_dense) # transform into Probability
        # Normalize
        row_sums = P_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # safe guard
        P_mat = P_mat / row_sums
        P_cum = np.cumsum(P_mat, axis=1)
        policy.pmat_cum_h.append(P_cum) # restore culmulative probability

    return policy


def make_exogenous_firm_transition_probabilities(params, policy):
    """
    Converts continuous-time Q matrices to discrete-time cumulative probability matrices.
    """
    dt = 1 / params.pd_per_yr
    # --- Foreign Market ---
    # convert into dense matrix
    Q_f_dense = params.Q_f.toarray() if hasattr(params.Q_f, "toarray") else params.Q_f
    pmat_msf = expm(Q_f_dense*dt)
    pmat_msf = pmat_msf / np.sum(pmat_msf, axis=1, keepdims=True) # normalize
    policy.pmat_cum_msf = np.cumsum(pmat_msf, axis=1) # culmulative probability

    # --- Home Market ---
    Q_h_dense = params.Q_h.toarray() if hasattr(params.Q_h, "toarray") else params.Q_h
    pmat_msh = expm(Q_h_dense*dt)
    pmat_msh = pmat_msh / np.sum(pmat_msh, axis=1, keepdims=True) # normalize
    policy.pmat_cum_msh = np.cumsum(pmat_msh, axis=1) # culmulative probability


    # --- Productivity Shock ---
    # Note: mm.Q_z usually governs Phi transitions
    Q_z_dense = params.Q_z.toarray() if hasattr(params.Q_z, "toarray") else params.Q_z
    pmat_z = expm(Q_z_dense) # already transformed into monthly data
    pmat_z = pmat_z / np.sum(pmat_z, axis=1, keepdims=True) # normalize
    policy.pmat_cum_z = np.cumsum(pmat_z, axis=1) # culmulative probability
    
    return policy