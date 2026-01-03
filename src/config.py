import numpy as np
import scipy.stats as stats
from scipy.stats import beta as beta_dist
from src.grids import make_q, make_erg

class ModelConfig:
    """
    Central configuration for the Eaton et al. (2025) replication.
    """
    def __init__(self):
        # ==========================================
        # Profit Function (formula 5)
        # Estimate Profit Scaler Pi & Stimulate X & Y with O-U Process
        # ==========================================
        self.eta            = 5.0 # elasticity of demand
        # Estimation from Table 6
        self.scale_h        = -19.635 # profit scale
        self.scale_f        = self.scale_h + 1
        self.sig_p          = 2.384 # std of log productivity (normal dist)
        self.D_z            = 0.044
        # Discretization of state-space
        self.z_size         = 7     # Number of discretized demand shock states (2*n+1) 
        self.x_size         = 7     # Number of different discretized macro shocks; same for home and foreign (2*n+1)
        self.phi_size       = 8     # number of different discretized profit shocks (2*n+1)
        # Exogenous Markov Process (O-U Process)
        ## Estimation from Table 3
        gam_h = 1.0 - 0.875
        sig_h = 0.0469
        gam_f = 1.0 - 0.639
        sig_f = 0.1101
        ## Macro State X in Home market
        self.L_h = gam_h * self.x_size
        self.D_h = sig_h * (self.L_h**(-0.5))
        self.Q_h, self.X_h = make_q(self.L_h, self.D_h, self.x_size)
        ## Macro State X in Foreign market
        self.L_f = gam_f * self.x_size
        self.D_f = sig_f * (self.L_f**(-0.5))
        self.Q_f, self.X_f = make_q(self.L_f, self.D_f, self.x_size)
        ## jump hazard for idiosyncratic match shocks Y
        self.pd_per_yr      = 12.0
        self.L_z = 4.0 / self.pd_per_yr # Constrains
        self.Q_z, self.Z = make_q(self.L_z, self.D_z, self.z_size)
        self.erg_pz = make_erg(self.L_z, self.D_z, self.Z)
        ## Diagonal Zeroing (For Profit Calculation)
        self.Q_z_d = self.Q_z.copy()
        self.Q_z_d.setdiag(0)
        self.Q_h_d = self.Q_h.copy()
        self.Q_h_d.setdiag(0)
        self.Q_f_d = self.Q_f.copy()
        self.Q_f_d.setdiag(0)
        # Productivity Distribution (Phi) - LogNormal Distribution
        phi_steps = np.linspace(-3, 3, 2 * self.phi_size + 1)
        erg_pp_raw = stats.norm.pdf(phi_steps)
        self.erg_pp = erg_pp_raw / np.sum(erg_pp_raw) # Discretization
        self.Phi = phi_steps * self.sig_p

        # ==========================================
        # Profit and Expected Profit (formula 6)
        # Discount Rate for time, exogenous dissolution, shipment arrival
        # (Note: jump to new marketwide state & buyer-specific shock is in previous section)
        # ==========================================
        self.r              = 0.13 / self.pd_per_yr # rate of time preference
        self.delta          = 0.326 / self.pd_per_yr # Exogenous match separation rate delta
        # Estimation from Table 6
        self.F_h            = np.exp(-3.874) # Fixed cost F
        self.F_f            = np.exp(-3.874)
        self.L_bF           = 1.014 # shipment order arrival hazard lambda_b
        # Shipment Arrival Processes (lambda_b)
        self.L_bH = 2 * self.L_bF # constraint from (Alessandria, Kaboski, and Midrigan, AER, 2010)
        ## Foreign Market
        self.max_shipsF = 3 * int(round(self.L_bF)) # maximum within-period shipments
        k_vec_f = np.arange(1, self.max_shipsF + 1)
        self.shipment_cdf_f = stats.poisson.cdf(k_vec_f, self.L_bF)
        ## Home Market
        self.max_shipsH = 3 * int(round(self.L_bH))
        k_vec_h = np.arange(1, self.max_shipsH + 1)
        self.shipment_cdf_h = stats.poisson.cdf(k_vec_h, self.L_bH)

        # ==========================================
        # Belief (formula 9)
        # Theta distribution of the market
        # ==========================================
        # Estimation from Table 3
        self.ah             = 0.032 # learning
        self.bh             = 0.192 # learning
        self.optimism       = 0
        # Discretization of state-space
        self.n_size         = 20    # Maximum number of informative signals per firm 
        self.dim1           = 7     # Number of possible theta1 values (specific to home market);
        self.dim2           = 7     # Number pf possible theta2 values (specific to foreign market);
        self.theta_size     = 20    # ??
        # True Theta
        self.af             = self.ah
        self.bf             = self.bh
        self._build_theta_grids()

        # ==========================================
        # Search Costs (formula 12)
        # ==========================================
        # Estimation from Table 6
        self.gam            = 0.046 # visibility in searching cost
        self.cs_h           = np.exp(5.132) # cost scalar home
        self.cs_f           = np.exp(15.161) # cost scalar foreign
        # Discretization of state-space
        self.net_size       = 40
        # Constraint
        self.kappa1         = 2

        # ==========================================
        # Stimulation Constraints
        # Tolerance & Maximum iterations & Upper Bounds
        # ==========================================
        self.v_tolerance    = 1e-3 # value function tolerance
        self.max_iter       = 50000 # value function maximum iteration
        self.pi_tolerance   = 1e-8 # Profit function tolerance
        self.T_profit       = 50 # Profit function iteration
        self.tot_yrs        = 50 # Stimulation iteration
        self.periods        = int(self.tot_yrs * self.pd_per_yr) # switch to month
        self.n_firms        = 50000 # potential exporting firms
        self.burn_years     = 10 # "burn-in" years
        self.burn_months    = int(self.burn_years * self.pd_per_yr)
        # Upper bounds
        self.max_match_f    = 50 # Upper bound on number of matches
        self.max_match_h    = 70
        self.max_match_month= 1e7 
        self.max_home_clients= 500
        self.abort_time     = 5000

        # ==========================================
        # Population Initialization (Firm Types)
        # Index & Population about different types of firms
        # ==========================================
        # population (firm types)
        n_phi = len(self.Phi)
        n_theta = len(self.theta2)
        self.N_pt = n_phi * n_theta
        # index
        phi_idxs = np.arange(n_phi)
        col_1 = np.repeat(phi_idxs, n_theta)
        theta_idxs = np.arange(n_theta)
        col_2 = np.tile(theta_idxs, n_phi)
        self.pt_type = np.column_stack((col_1, col_2))
        # number of different firms
        probs_phi = self.erg_pp[self.pt_type[:, 0]]
        probs_theta = self.th2_pdf[self.pt_type[:, 1]]
        joint_probs_flat = probs_phi * probs_theta
        self.sim_firm_num_by_prod_succ_type = np.round(joint_probs_flat * self.n_firms).astype(int)

        # ==========================================
        # Miscellaneous
        # ==========================================
        self.firm_death_haz = 0.08 / self.pd_per_yr # time preferece 

    def _build_theta_grids(self):
        """
        Discretization of theta distribution
        """
        
        # --- Generate grids ---
        # Home Market (theta1)
        step_h = 1.0 / self.dim1
        start_h = step_h * 0.5
        end_h = 1.0 - start_h
        self.theta1 = np.linspace(start_h, end_h, self.dim1)
        
        # Foreign Market (theta2)
        step_f = 1.0 / self.dim2
        start_f = step_f * 0.5
        end_f = 1.0 - start_f
        self.theta2 = np.linspace(start_f, end_f, self.dim2)
        
        # Numerical adjustment
        self.theta1[-1] -= 0.0001
        self.theta2[-1] -= 0.0001
        
        # --- Calculate CDF & PDF ---
        
        # Home CDF
        self.th1_cdf = beta_dist.cdf(self.theta1, self.ah, self.bh)
        self.th1_cdf[-1] = 1.0
        
        # Foreign CDF & PDF
        self.th2_cdf = beta_dist.cdf(self.theta2, self.af, self.bf)
        pdf_diff_f = np.diff(self.th2_cdf)
        self.th2_pdf = np.concatenate(([self.th2_cdf[0]], pdf_diff_f)) # calculate with difference method
        # print(f"Sum of Foreign Theta PDF: {np.sum(self.th2_pdf)}")


    def print_summary(self):
        """Helper to verify parameters match the paper"""
        print("=== Model Configuration ===")
        print(f"Time Step: Monthly (1/{int(self.pd_per_yr)} year)")
        print(f"Discount Rate (r): {self.r:.5f} (monthly)")
        print(f"Exog. Separation (delta): {self.delta:.5f} (monthly)")
        print("-" * 30)
        print(f"Macro Process (Foreign):")
        print(f"  Target AR(1) rho: {self.ar1_persistence_f}")
        print(f"  Target RMSE: {self.rmse_x_f}")
        print(f"  Derived Lambda (annual): {self.lambda_x_f:.4f}")
        print(f"  Derived Step Size (Delta): {self.step_x_f:.4f}")
        print(f"  Grid Points: {self.n_x} (Range: {self.get_macro_grid()[0]:.2f} to {self.get_macro_grid()[-1]:.2f})")
        print("-" * 30)
        print(f"Structural Parameters:")
        print(f"  Profit Scalar (Pi): {self.Pi_f:.2e} (ln: {self.ln_Pi_f})")
        print(f"  Fixed Cost (F): {self.F:.4f} (ln: {self.ln_F})")
        print(f"  Search Cost Scalar (k0): {self.kappa0_f:.2e}")
        print(f"  Visibility (gamma): {self.gamma}")
        print(f"  Priors: Alpha={self.alpha}, Beta={self.beta}")
        print("=========================")

if __name__ == "__main__":
    # Test run
    conf = ModelConfig()
    #conf.print_summary()