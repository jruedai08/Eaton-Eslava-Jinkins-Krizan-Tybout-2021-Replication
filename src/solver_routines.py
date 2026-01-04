import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from src.static_funcs import get_optimal_search_intensity, get_search_cost

from tqdm import tqdm

# ==============================================================================
# Expected Match Profit and Continuation Value Solver
# ==============================================================================

def solve_expected_match_profit(params, pie_scale, L_b, X_grid, Q_macro, Q_macro_d, F):
    """
    Computes the expected profit and continuation value of a match (formula 6).
    
    Args:
        params: ModelConfig object (needs r, delta, eta, Phi, Z, Q_z, etc.)
        pie_scale: Profit scalar (mm.scale_f or mm.scale_h)
        L_b: Shipment hazard rate (lambda_b)
        X_grid: Macro state grid (lnX)
        Q_macro: Macro transition matrix (Q_f or Q_h)
        Q_macro_d: Macro transition matrix without diagonal
        F: Fixed cost to maintain a match (F_f or F_h)
        
    Returns:
        expected_match_profit: (n_phi, n_x) matrix
        continuation_value: (n_z, n_phi, n_x) tensor
    """
    
    # Event Hazard Rate (Discount + Destruction + Arrivals + Jump Rates)
    event_hazard = params.r + params.delta + L_b + abs(Q_macro[0, 0]) + abs(params.Q_z[0, 0])
    
    # Construct Shipment Payoff (pi_phi)
    ## Shape of profit from shipment: (n_y, n_phi, n_x)
    payoff_no_z = np.exp(pie_scale + X_grid[:, None] + (params.eta - 1) * params.Phi[None, :])
    payoff_tensor = np.exp(params.Z)[:, None, None] * payoff_no_z.T[None, :, :]
    ## shape
    n_z = params.Q_z.shape[0]
    n_phi = len(params.Phi)
    n_x = len(X_grid)
    
    continuation_value = np.zeros((n_z, n_phi, n_x))
    expected_match_profit = np.zeros((n_phi, n_x))
    
    # Bellman Iterative Solver for Expected Profit
    for j in range(n_phi): # given Phi fixed
        current_payoff = payoff_tensor[:, j, :]
        #print(f" > with Phi equals to {params.Phi[j]:.4f}",current_payoff)
        
        # Initial Guess
        ## c_val = 0 implies what if expected profit is 0 in the state
        pi_z = current_payoff / (params.delta + params.r) # pi tilde
        c_val = np.zeros((n_z, n_x)) # pi hat (option value)
        
        iteration = 0
        error = 1e12
        
        while (error > params.pi_tolerance) and (iteration <= params.max_iter):
            iteration += 1
            
            # For continuation value, it's solving the Belleman Equation
            # For the value of the match, it's a fixed point iteration
            # Convergence implies the marginal profit can be ignored

            # Macro Transitions: switch from x to x'
            # c_val is (n_z, n_x), Q_macro_d is (n_x, n_x)
            term_macro = (Q_macro_d @ c_val.T).T
            # Micro Transitions (Q_z_d * C): switch from y to y'
            # Q_z_d is (n_z, n_z), c_val is (n_z, n_x)
            term_micro = params.Q_z_d @ c_val
            # Shipment Gain (n_z, n_x)
            term_shipment = L_b * pi_z
            
            # Calculate Gross Continuation Value
            c_val_gross_new = (term_macro + term_micro + term_shipment) / event_hazard
            c_val_new = np.maximum(-F + c_val_gross_new, 0) # endogenous dissolution
            
            # Update Total Match Value (Flow + Continuation)
            pi_z_new = current_payoff + c_val_new
            
            # Check Convergence
            error = np.max(np.abs(pi_z - pi_z_new))
            
            # Update
            pi_z = pi_z_new
            c_val = c_val_new
        
        if iteration == params.max_iter:
            print(f"Warning: Match profit solver did not converge for phi index {j}")

        # Store results
        continuation_value[:, j, :] = np.maximum(c_val, 0)
        expected_match_profit[j, :] = np.maximum(params.erg_pz @ pi_z, 0)

    return expected_match_profit, continuation_value


# ==============================================================================
# Home Value Function Solver
# ==============================================================================

def solve_policy_home(policy, params):
    """
    Solve value function in Home Market (formula 10)

    Args:
        policy: Policy object
        params: ModelConfig object
    """
    print("  > Solving Home Policy ...")
    
    # Discretization
    dim1 = params.dim1        # Theta1 grid size
    net_max = params.net_size # Maximum network size
    phi_size = params.phi_size
    x_size = params.x_size
    
    # Restore
    V_h = np.zeros((dim1, net_max + 1, 2*phi_size+1, 2*x_size+1)) # Value Function
    L_h = np.zeros((dim1, net_max + 1, 2*phi_size+1, 2*x_size+1)) # Search Intensity

    # monitor
    total_steps = (2*phi_size+1) * dim1 * (net_max + 1)

    with tqdm(total=total_steps, desc="Home HJB Progress") as pbar:
    
        # Outer Loop: Productivity Phi
        for i_phi in range(len(params.Phi)): # given Phi
            pi_h_vec = policy.pi_h[i_phi, :]
            
            # Inner Loop Layer I: Popularity Theta (Home)
            for k in range(dim1):
                theta_val = params.theta1[k]
                
                V_next = np.zeros(2*x_size+1) # initial guess
                
                # Inner Loop Layer II: matched pairs A
                for m in range(net_max, -1, -1):
                    net_size_val = float(m + 1)
                    
                    if m == net_max:
                        V_succ_ref = V_next # Boundary
                        is_boundary = True
                    else:
                        V_succ_ref = V_h[k, m+1, i_phi, :]
                        is_boundary = False
                    
                    V_curr, s_curr = _solve_hjb_layer_no_learning(
                        params, pi_h_vec, theta_val, net_size_val, V_succ_ref, is_boundary, params.cs_h
                    )
                    
                    # restore the data
                    V_h[k, m, i_phi, :] = V_curr
                    L_h[k, m, i_phi, :] = s_curr

                    pbar.update(1)

    return V_h, L_h



def _solve_hjb_layer_no_learning(params, payoff_flow, theta, net_size, V_next, is_boundary,cost_scale):
    """
    HJB solver with implicit method
    """
    # restriction and restore
    x_size = 2*params.x_size+1
    V_guess = np.zeros(x_size) if not is_boundary else V_next.copy()
    s_star = np.zeros(x_size)
    #print("V_guess shape:",V_guess.shape,V_next.shape)
    
    for iter_k in range(params.max_iter):
        
        val_success = V_guess if is_boundary else V_next
        val_fail = V_guess # keep current circumstance
        
        # optimal search intensity (max s)
        s_star = get_optimal_search_intensity(
            expected_theta=theta, # Home market theta is known
            net_size=net_size,
            payoff_success=payoff_flow,
            val_success=val_success,
            val_fail=val_fail,
            val_current=V_guess,
            kappa0=cost_scale, kappa1=params.kappa1, gamma=params.gam
        )
        
        cost = get_search_cost(s_star, net_size, cost_scale, params.kappa1, params.gam)
        
        # Construct Linear System
        data, rows, cols = [], [], []
        rhs = np.zeros(x_size)
        
        for x in range(x_size): # given macro X
            macro_out = abs(params.Q_h[x, x])
            s = s_star[x]
            
            # Diagonal: rho + s * theta + macro_out
            diag_val = params.r + (s * theta) + macro_out
            
            # Boundary case specific: success leads to V_curr
            if is_boundary:
                diag_val -= s * theta
            
            data.append(diag_val); rows.append(x); cols.append(x)
            
            # Off-diagonal Q (Macro jumps)
            start, end = params.Q_h.indptr[x], params.Q_h.indptr[x+1]
            for i in range(start, end):
                x_prime = params.Q_h.indices[i]
                if x_prime != x: # Not diagonal
                    q_val = params.Q_h.data[i]
                    data.append(-q_val); rows.append(x); cols.append(x_prime)
            
            # RHS
            # Term: s * theta * (Payoff + [V_next if not boundary]) - Cost
            if is_boundary:
                rhs[x] = s * theta * payoff_flow[x] - cost[x]
            else:
                rhs[x] = s * theta * (payoff_flow[x] + V_next[x]) - cost[x]
                
        # Solve Linear System
        A_mat = sp.coo_matrix((data, (rows, cols)), shape=(x_size, x_size)).tocsr()
        V_new = spsolve(A_mat, rhs)
        
        if np.max(np.abs(V_new - V_guess)) < params.v_tolerance:
            return V_new, s_star
        V_guess = V_new

    raise RuntimeError(
        f"Home HJB Failed to Converge! \n"
        f"  > Max Iterations Reached: {params.max_iter} \n"
        f"  > Last Error: {np.max(np.abs(V_new - V_guess)):.2e} (Tolerance: {params.v_tolerance}) \n"
        f"  > Debug Info: Theta={theta:.3f}, NetworkSize={net_size:.1f}"
    )
        
    return V_guess, s_star



# ==============================================================================
# Foreign Value Function Solver
# ==============================================================================

def solve_policy_foreign(policy, params):
    """
    Solve HJB for Foreign Market (5D Rigorous Version).
    
    Dimensions: [n, a, m, phi, x]
    - n: Trials (Belief)
    - a: Successes (Belief)
    - m: Network Size (Physical State)
    """
    print("  > Solving Foreign Policy (5D Full Storage)...")

    n_max = params.n_size         
    net_max = params.net_size     
    phi_size = params.phi_size
    x_size = params.x_size
    
    # restore
    V_f = np.zeros((n_max + 1, n_max + 1, net_max + 1, 2*phi_size+1, 2*x_size+1))
    L_f = np.zeros((n_max + 1, n_max + 1, net_max + 1, 2*phi_size+1, 2*x_size+1))
    
    # Lower Triangle Only
    total_steps = (2*phi_size+1) * ((n_max + 1) * (n_max + 2) // 2)

    with tqdm(total=total_steps, desc="Foreign HJB Process") as pbar:
        for i_phi in range(params.Phi.shape[0]):
            pi_f_vec = policy.pi_f[i_phi, :]
            #if i_phi%5==0:
            #    print(pi_f_vec)

            for a in range(n_max+1): # if n = n_max -> no learning
                theta_fixed = policy.post_probs[n_max, a]

                V_bdy_guess = np.zeros(2*x_size+1)
                V_curr, s_curr = _solve_hjb_layer_no_learning(
                    params, pi_f_vec, theta_fixed, float(net_max + 1), 
                    V_next=V_bdy_guess, is_boundary=True, cost_scale=params.cs_f
                    )
                V_f[n_max, a, net_max, i_phi, :] = V_curr # restore
                L_f[n_max, a, net_max, i_phi, :] = s_curr

                V_next_net = V_curr
                
                for m in range(net_max-1,-1,-1): # backward algorithm for visibility
                    V_curr, s_curr = _solve_hjb_layer_no_learning(
                        params, pi_f_vec, theta_fixed, float(m + 1), 
                        V_next=V_next_net, is_boundary=False, cost_scale=params.cs_f
                        )
                    V_f[n_max, a, m, i_phi, :] = V_curr
                    L_f[n_max, a, m, i_phi, :] = s_curr
                    V_next_net = V_curr # Update for next m

                pbar.update(1)

            for n in range(n_max-1,-1,-1): # if n != n_max -> learning
                for a in range(n+1):
                    m = a # since learning exists
                    theta_hat = policy.post_probs[n, a] # dynamic belief
                    V_succ_known = V_f[n+1, a+1, m+1, i_phi, :] # known from previous period
                    V_fail_known = V_f[n+1, a, m, i_phi, :]
                    V_curr, s_curr = _solve_hjb_layer_foreign_learning(
                        params, pi_f_vec, theta_hat, float(m+1), 
                        V_succ_known, V_fail_known
                        )
                    V_f[n, a, m, i_phi, :] = V_curr # restore
                    L_f[n, a, m, i_phi, :] = s_curr

                    pbar.update(1)

    return V_f, L_f


def _solve_hjb_layer_foreign_learning(params, payoff_flow, theta, net_size, V_succ_known, V_fail_known):
    """
    HJB Solver for Active Learning Phase.
    
    Equation:
    (r + s)V = -Cost + s*theta*(pi + V_succ) + s*(1-theta)*V_fail + Q*V
    
    Key Difference from Home:
    V_succ (n+1, a+1) and V_fail (n+1, a) are BOTH constant known vectors from the future.
    There is NO self-reference on the RHS for search terms.
    """
    # initial guess
    x_size = 2*params.x_size+1
    V_guess = np.zeros(x_size)
    s_star = np.zeros(x_size)
    

    for iter_k in range(params.max_iter):
        
        s_star = get_optimal_search_intensity(
            expected_theta=theta,
            net_size=net_size,
            payoff_success=payoff_flow,
            val_success=V_succ_known, # given, no need for iteration
            val_fail=V_fail_known, # given, no need for iteration
            val_current=V_guess,
            kappa0=params.cs_f, kappa1=params.kappa1, gamma=params.gam
        )
        
        cost = get_search_cost(s_star, net_size, params.cs_f, params.kappa1, params.gam)
        
        
        data, rows, cols = [], [], []
        rhs = np.zeros(x_size)
        
        for x in range(x_size):
            macro_out = abs(params.Q_f[x, x])
            s = s_star[x]
            
            # LHS Diagonal: r + s + |Q|
            diag_val = params.r + s + macro_out
            data.append(diag_val); rows.append(x); cols.append(x)
            
            # LHS Off-diagonal: -Q
            start, end = params.Q_f.indptr[x], params.Q_f.indptr[x+1]
            for i in range(start, end):
                x_prime = params.Q_f.indices[i]
                if x_prime != x:
                    data.append(-params.Q_f.data[i]); rows.append(x); cols.append(x_prime)
            
            # RHS: -Cost + s*theta*(pi + V_succ) + s*(1-theta)*V_fail
            term_success = s * theta * (payoff_flow[x] + V_succ_known[x])
            term_fail = s * (1 - theta) * V_fail_known[x]
            rhs[x] = term_success + term_fail - cost[x]
            
        # Solve Linear System
        #print("x size:",x_size)
        A_mat = sp.coo_matrix((data, (rows, cols)), shape=(x_size, x_size)).tocsr()
        V_new = spsolve(A_mat, rhs)
        
        if np.max(np.abs(V_new - V_guess)) < params.v_tolerance:
            return V_new, s_star
        
        V_guess = V_new

    raise RuntimeError(f"HJB Solver Failed to Converge! \n"
                       f"Last Error: {np.max(np.abs(V_new - V_guess)):.4e} (Tol: {params.v_tolerance}) \n"
                       f"State Info: Theta={theta:.2f}, Net={net_size}")
        
    return V_guess, s_star



