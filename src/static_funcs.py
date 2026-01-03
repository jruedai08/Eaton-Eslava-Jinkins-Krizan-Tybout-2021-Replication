import numpy as np

def make_foreign_success_posteriors(params):
    """
    Bayesian posterior belief (formula 9)
    
    Returns:
        post_probs (np.array): (n_size+1, n_size+1) 矩阵
        Rows (j): Trials (n = j)
        Cols (k): Successes (a = k)
        Value: E[theta | n, a] = (alpha + a) / (alpha + beta + n)
    """
    N = params.n_size
    n_vec = np.arange(N + 1)
    a_vec = np.arange(N + 1)
    
    alpha = params.af
    beta = params.bf
    opt = params.optimism
    
    numerator = alpha + opt + a_vec[None, :]
    denominator = alpha + opt + beta + n_vec[:, None]
    post_probs = numerator / denominator # shape (len(n_vec),len(a_vec))
    
    # Keep lower triangle (a<n)
    mask = np.tril(np.ones((N + 1, N + 1)))
    post_probs = post_probs * mask
    
    return post_probs




def get_optimal_search_intensity(expected_theta, net_size, payoff_success, val_success, val_fail, val_current, kappa0, kappa1, gamma):
    """
    Solves for optimal s* by inverting the FOC condition. (formula 11)
    Formula: s* = ( (MB * Network / kappa0) + 1 )^(1/(k-1)) - 1 
    
    Args:
        expected_theta (a): Probability of success (Belief)
        net_size: Network term (1 + cumulative successes)
        payoff_success (pi): Immediate flow profit from a new match (or capitalized value)
        val_success (V_succ): Continuation value if search yields a match
        val_fail (V_fail): Continuation value if search yields a failure
        val_current (V_orig): Current continuation value
        kappa0, kappa1, gamma: Cost parameters
    """
    # Marginal Benefit of a match
    '''
    print(
        expected_theta.shape,
        payoff_success.shape,
        val_success.shape,
        val_fail.shape,
        val_current.shape
    ) # check shapes
    '''
    mb = expected_theta * (payoff_success + val_success) + (1 - expected_theta) * val_fail - val_current
    # Visibility effect
    network_factor = (1 + np.log(1+net_size))**gamma
    
    # FOC
    term_inside = np.maximum((network_factor * mb / kappa0) + 1, 0) # max for safety
    exponent = 1.0 / (kappa1 - 1.0)
    s_star = np.maximum(term_inside**exponent - 1,0) # ensure positive
    
    return s_star

def get_search_cost(s, net_size, kappa0, kappa1, gamma):
    """
    Calculates the cost of search intensity s given network size a^m. (formula 12)
    
    Args:
        s: Search intensity (float or array)
        net_size: Network effect term a^m
        kappa0: Cost scalar
        kappa1: Convexity parameter
        gamma: Visibility parameter
        
    Returns:
        Cost value
    """
    denominator = kappa1 * (1 + np.log(net_size + 1))**gamma
    numerator = kappa0 * ((1 + s)**kappa1 - (1 + kappa1 * s))
    return numerator / denominator