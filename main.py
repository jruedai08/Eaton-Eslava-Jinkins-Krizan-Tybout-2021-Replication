import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from src.config import ModelConfig
from src.objects import Policy
from src.solver_routines import (
    solve_expected_match_profit, 
    solve_policy_home, 
    solve_policy_foreign
)
from src.policy import make_exporter_transition_probabilities
from src.solver import solve_policy_main


sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

def run_all():
    """
    Run the full solver and return the model parameters and policy object.
    """
    # 1. Load model configuration
    params = ModelConfig()
    
    # 2. Generate policy and value functions
    policy = solve_policy_main(params)
    
    return params, policy

# ============================================================
#               Visualization Functions
# ============================================================

def plot_value_function(params, policy):
    """Value Function"""
    phi_idx = params.phi_size
    macro_idx = params.x_size
    m_network = 0 # no network advantage when entering the market

    # Value Function: [n, a, m, phi, x]
    # Belief State (n, a)
    V_slice = policy.value_f[:, :, m_network, phi_idx, macro_idx]
    
    # 1. always fails (a=0)
    # 2. always succeeds (a=n)
    # 3. succeeds half of them (a=n/2)
    n_range = np.arange(params.n_size + 1)
    v_fail = V_slice[n_range, 0]
    v_succ = [V_slice[n, n] for n in n_range]
    
    plt.figure()
    plt.plot(n_range, v_succ, 'o-', label='All Successes (a=n)', color='#2ca02c')
    plt.plot(n_range, v_fail, 's-', label='All Failures (a=0)', color='#d62728')
    
    plt.title(f"Value Function $V(n, a)$ \n(Productivity $\phi$={params.Phi[phi_idx]:.2f}, Macro $X$=Avg)")
    plt.xlabel("Number of Trials (n)")
    plt.ylabel("Value of the Firm")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('figure_value_function.png')
    plt.show()

def plot_value_function(params, policy,phi_idx,macro_idx,m_network,elev=30,azim=-60):
    '''Value Function 3D Plot'''

    V_matrix = policy.value_f[:,:, m_network, phi_idx, macro_idx]

    fig = plt.figure(figsize=(12,10))
    ax  = fig.add_subplot(111, projection='3d')

    # grid
    n_dim = V_matrix.shape[0] # n
    a_dim = V_matrix.shape[1] # a
    X, Y = np.meshgrid(np.arange(n_dim), np.arange(a_dim))
    Z = V_matrix.astype(float)
    Z[Y > X] = np.nan # mask upper triangle

    # plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', 
                               edgecolor='k', linewidth=0.1, alpha=0.9,
                               vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    ax.set_title(f"3D Value Landscape\nPhi={phi_idx}, Macro={macro_idx}, Net={m_network}")
    ax.set_xlabel('Trials (n)')
    ax.set_ylabel('Successes (a)')
    ax.set_zlabel('Firm Value')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Value')
    plt.tight_layout()
    plt.show()




def plot_search_policy(params, policy,phi_idx,macro_idx,m_network):
    """Search Intensity Heatmap"""    
    # Lambda: [n, a, m, phi, x]
    L_slice = policy.lambda_f[:, :, m_network, phi_idx, macro_idx]
    
    # Mask upper triangle
    mask = np.zeros_like(L_slice)
    mask[np.triu_indices_from(mask, k=1)] = True

    plt.figure(figsize=(18, 12))
    sns.heatmap(L_slice, mask=mask, cmap="viridis", annot=True, fmt=".2f", 
                cbar_kws={'label': 'Search Intensity $\lambda$'})
    plt.title("Optimal Search Intensity $\lambda(n, a)$")
    plt.ylabel("Number of Trials (n)")
    plt.xlabel("Number of Successes (a)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figure_search_policy.png')
    plt.show()

def plot_posterior_beliefs(params, policy):
    """Posterior Belief Heatmap"""
    post_probs = policy.post_probs
    
    P_plot = post_probs
    
    mask = np.zeros_like(P_plot)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    plt.figure()
    sns.heatmap(P_plot, mask=mask, cmap="Blues", annot=True, fmt=".2f",
                cbar_kws={'label': 'Expected Product Appeal $E[\\theta]$'})
    plt.title("Evolution of Beliefs (Posterior Probability)")
    plt.ylabel("Number of Trials (n)")
    plt.xlabel("Number of Successes (a)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('figure_beliefs.png')
    plt.show()