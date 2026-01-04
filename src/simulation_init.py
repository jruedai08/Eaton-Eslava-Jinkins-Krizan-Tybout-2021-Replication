'''
Code still in development ...
'''

import numpy as np
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class IterIn:
    """
    Simulation State Container (Inputs/Intermediates)
    """
    pt_ndx: int # index firms by type (productivity & theta)
    macro_state_f: np.ndarray 
    year_lag: int = 1
    
    # Time variables
    t: int = 0
    year: int = 1
    season: int = 1
    
    # Firm-level time series (Firms x Periods)
    cur_cli_cnt: np.ndarray = field(default=None) # client that active currently
    cur_duds: np.ndarray = field(default=None) # dud meeting currently
    add_cli_cnt: np.ndarray = field(default=None) # gross addition to client
    actv_cli_cnt: np.ndarray = field(default=None) # active client
    cum_meets: np.ndarray = field(default=None) # cumulative number of meetings
    cum_succ: np.ndarray = field(default=None) # cumulative number of successes
    new_firm: np.ndarray = field(default=None) 
    exit_flag: np.ndarray = field(default=None)
    exit_firm: np.ndarray = field(default=None)
    exog_deaths: np.ndarray = field(default=None) # exogenous match death
    micro_state: np.ndarray = field(default=None) # index for success/meeting

    # Z-state breakdown (Firms x Z_size)
    lag_cli_zst: np.ndarray = field(default=None) # lagged clidents
    new_cli_zst: np.ndarray = field(default=None) # new client
    die_cli_zst: np.ndarray = field(default=None) # client death counts
    surviv_zst: np.ndarray = field(default=None) # surviving client
    trans_zst: np.ndarray = field(default=None) 
    
    # Age vectors
    flrlag: np.ndarray = field(default=None)
    cumage: np.ndarray = field(default=None)
    
    # Dynamic logic helpers
    # initialize seas_tran as list with factory
    seas_tran: List[Any] = field(default_factory=list)
    seas_Zcut: np.ndarray = field(default=None)
    
    mat_cont_2yr: np.ndarray = field(default_factory=lambda: np.empty((0, 14)))
    mkt_exit: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    mat_yr_sales_lag: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))
    firm_yr_sales_lag: np.ndarray = field(default=None) # contain (firm ID, sales, shipment, firm age)
    
    # Transition Count Matrix (Z+1, Z+1, Firms)
    trans_count: np.ndarray = field(default=None)
    
    # Client flags
    keep_cli: np.ndarray = field(default=None) # clients existing in period 1
    keep_cli_lag: np.ndarray = field(default=None)
    
    # Scalars
    N_match: int = 0
    Zcut_eoy: float = 0.0
    Zcut_eoy_lag: float = 0.0
    mic_type: int = 0
    pmat_cum_t: np.ndarray = field(default=None)  # Current transition matrix

    def initialize(self, params):
        
        n_firms = int(params.sim_firm_num_by_prod_succ_type[self.pt_ndx]) # number of the specific type of firm
        n_periods = int(params.periods) # number of total periods in months
        n_z = int(params.Z.shape[0]) # number of demand states
        
        # Lists
        self.seas_tran = [None] * params.pd_per_yr
        self.seas_Zcut = np.zeros(params.pd_per_yr)
        
        # Time Series
        self.cur_cli_cnt = np.zeros((n_firms, n_periods))
        self.cur_duds = np.zeros((n_firms, n_periods))
        self.add_cli_cnt = np.zeros((n_firms, n_periods))
        self.actv_cli_cnt = np.zeros((n_firms, n_periods))
        self.cum_meets = np.zeros((n_firms, n_periods))
        self.cum_succ = np.zeros((n_firms, n_periods))
        self.new_firm = np.zeros((n_firms, n_periods))
        self.exit_flag = np.zeros((n_firms, n_periods))
        self.exit_firm = np.zeros((n_firms, n_periods))
        self.exog_deaths = np.zeros((n_firms, n_periods))
        self.micro_state = np.ones((n_firms, n_periods), dtype=int)
        
        # Z Breakdowns
        self.lag_cli_zst = np.zeros((n_firms, n_z))
        self.new_cli_zst = np.zeros((n_firms, n_z))
        self.die_cli_zst = np.zeros((n_firms, n_z))
        self.surviv_zst = np.zeros((n_firms, n_z))
        self.trans_zst = np.zeros((n_firms, n_z))
        
        # Age
        self.flrlag = np.ones(n_firms, dtype=int)
        self.cumage = np.zeros(n_firms, dtype=int)
        
        # Trans Count
        self.trans_count = np.zeros((n_z + 1, n_z + 1, n_firms))
        
        # Client Flags
        self.keep_cli = np.ones(n_z)
        self.keep_cli_lag = np.ones(n_z)
        self.keep_cli[0:5] = 0
        
        self.firm_yr_sales_lag = np.zeros((n_firms, 4))
        
        return self


@dataclass
class IterOut:
    """
    Simulation Results Container
    """
    # Transaction Data Storage
    transF: List[List[Any]] = field(default=None)
    match_hist: np.ndarray = field(default=None)
    match_histD: np.ndarray = field(default=None)
    
    dud_matches: np.ndarray = field(default_factory=lambda: np.empty((0, 9)))
    mat_yr_sales: np.ndarray = field(default_factory=lambda: np.empty((0, 9)))
    mat_yr_sales_adj: np.ndarray = field(default_factory=lambda: np.empty((0, 9)))
    firm_f_yr_sales: np.ndarray = field(default_factory=lambda: np.empty((0, 6)))
    time_gaps: np.ndarray = field(default_factory=lambda: np.empty((0, 7)))
    
    # Moments
    moms_xx: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    moms_xy: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    ysum: float = 0.0
    nobs: int = 0
    ship_obs: int = 0
    ln_ships: float = 0.0
    duds: int = 0
    
    # Firm Moments
    fmoms_xx: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))
    fmoms_xy: np.ndarray = field(default_factory=lambda: np.zeros((4, 1)))
    fnobs: int = 0
    
    # Exit Moments
    exit_xx: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    exit_xy: np.ndarray = field(default_factory=lambda: np.zeros((1, 6)))
    sum_succ_rate: float = 0.0
    exit_obs: int = 0
    sum_exits: int = 0
    
    # Additional containers
    stackF: np.ndarray = field(default=None)
    iterF_check: dict = field(default_factory=dict)
    exptr_count: int = 0

    def initialize(self, params):
        self.transF = [[None for _ in range(6)] for _ in range(params.N_pt)]
        self.match_hist = np.zeros(params.max_match)
        self.match_histD = np.zeros(params.max_match)
        return self


def simulate_foreign_inner_initialize(params, pt_ndx, macro_state_f):
    iter_in = IterIn(pt_ndx=pt_ndx, macro_state_f=macro_state_f)
    iter_in.initialize(params)
    
    iter_out = IterOut()
    iter_out.initialize(params)
    
    return iter_in, iter_out