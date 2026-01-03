import numpy as np
import time

from src.config import ModelConfig
from src.objects import Policy
from src.static_funcs import (
    make_foreign_success_posteriors,
    get_optimal_search_intensity,
    get_search_cost
)
from src.solver_routines import (
    solve_expected_match_profit,
    solve_policy_home,
    solve_policy_foreign
)
from src.policy import make_exporter_transition_probabilities

# ==========================================
# Orchestrators
# ==========================================

def generate_policy_and_value_functions(params):
    """
    Main function
    """
    print("--- Starting Solver ---")
    start_time = time.time()

    # solving policy
    policy = solve_policy_main(params)

    # Discretization of state-space
    policy = make_exporter_transition_probabilities(params, policy)

    # (Optional) check_FOC(mm, policy) 
    
    elapsed = time.time() - start_time
    print(f"--- Solver Finished in {elapsed:.2f} seconds ---")
    return policy


def solve_policy_main(params):
    """
    
    """
    # empty class
    policy = Policy()

    print("Step 1: Computing Expected Match Profits...")
    policy.pi_f, policy.c_val_f = solve_expected_match_profit(
        params, params.scale_f, params.L_bF, params.X_f, params.Q_f, params.Q_f_d, params.F_f
        )
    policy.pi_h, policy.c_val_h = solve_expected_match_profit(
        params, params.scale_h, params.L_bH, params.X_h, params.Q_h, params.Q_h_d, params.F_h
        )

    print("Step 2: Computing Bayesian Beliefs...")
    policy.post_probs = make_foreign_success_posteriors(params)

    print("Step 3: Solving Value Function...")
    policy.value_f, policy.lambda_f = solve_policy_foreign(policy, params)
    policy.value_h, policy.lambda_h = solve_policy_home(policy, params)

    return policy

