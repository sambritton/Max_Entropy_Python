import logging
from optlang.glpk_interface import Variable, Model, Constraint, Objective
from numpy import array, zeros, ones, sign, diag, log, isnan, nonzero, \
    vstack, hstack, sqrt, eye, exp, nan

from equilibrator_api import settings
from equilibrator_api.bounds import Bounds

class MDFResult(object):

    def __init__(self, model, mdf,
                 concentrations, dG0_r_cov_eigen,
                 reaction_prices, compound_prices,
                 min_total_dG=None, max_total_dG=None):
        """Initialize.

        Args:
            model: PathwayModel generating these results.
            mdf: value of the result [in kJ/mol]
            concentrations: metabolite concentrations at optimum [in M]
            dG0_r_cov_eigen: dGr0' covariance eigenvalues at optimum [in kJ/mol]
            reaction_prices: shadow prices for reactions.
            compound_prices: shadow prices for compounds.
        """
        self.model = model
        self.mdf = mdf
        self.concentrations = concentrations
        self.dG0_r_cov_eigen = dG0_r_cov_eigen
        self.reaction_prices = reaction_prices
        self.compound_prices = compound_prices

        self.dG_r_prime = model.CalculateReactionEnergiesUsingConcentrations(
            concentrations)
        self.dG_r_prime_raw = self.dG_r_prime + model.dG0_r_std @ dG0_r_cov_eigen

        # adjust dG to flux directions
        self.dG_r_prime_adj = model.I_dir @ self.dG_r_prime_raw

        # May be set after initialization. Optional.
        self.max_total_dG = max_total_dG
        self.min_total_dG = min_total_dG


class PathwayThermoModel(object):
    """Container for doing pathway-level thermodynamic analysis."""

    DEFAULT_FORMATION_LB = -1e6
    DEFAULT_FORMATION_UB = 1e6
    DEFAULT_REACTION_LB = -1e3
    DEFAULT_REACTION_UB = 0.0
    DEFAULT_C_RANGE = (1e-6, 0.1)
    DEFAULT_PHYSIOLOGICAL_CONC = 1e-3

    def __init__(self, S, fluxes, dG0_r_prime, cids,
                 dG0_r_std=None, concentration_bounds=None):
        """Create a pathway object.

        Args:
            S: Stoichiometric matrix of the pathway.
                Reactions are on the rows, compounds on the columns.
            fluxes: the list of relative fluxes through each of the reactions.
                By default, all fluxes are 1.
            dG0_r_prime: the change in Gibbs energy for the reactions
                in standard conditions, corrected for pH, ionic strength, etc.
                Should be a column vector in np.matrix format.
            dG0_r_std: (optional) the square root of the covariance matrix
                corresponding to the uncertainty in the dG0_r values.
            concentration_bounds: a bounds.Bounds object expressing bounds on
                the metabolite concentrations.
        """
        self.S = S
        self.Nc, self.Nr = S.shape

        self.dG0_r_prime = dG0_r_prime
        if dG0_r_std is None:
            self.dG0_r_std = zeros((self.Nr, self.Nr))
        else:
            self.dG0_r_std = dG0_r_std

        # Make sure dG0_r' is the right size
        assert self.dG0_r_prime.shape[0] == self.Nr
        assert self.dG0_r_std.shape[0] == self.Nr
        assert self.dG0_r_std.shape[1] == self.Nr

        self.fluxes = fluxes
        if self.fluxes is None:
            self.fluxes = ones((1, self.Nr))
        self.fluxes = array(self.fluxes, ndmin=2)
        assert self.fluxes.shape[1] == self.Nr, 'Fluxes required for all reactions'

        _signs = list(map(sign, self.fluxes.flat))
        self.I_dir = array(diag(_signs), ndmin=2)
        self.Nr_active = int(sum(self.fluxes.T != 0))

        self.cids = cids
        self.concentration_bounds = concentration_bounds

        if self.concentration_bounds is None:
            lb, ub = self.DEFAULT_C_RANGE
            self.concentration_bounds = Bounds(default_lb=lb, default_ub=ub)

        # Currently unused bounds on reaction dGs.
        self.r_bounds = None

    def CalculateReactionEnergiesUsingConcentrations(self, concentrations):
        log_conc = log(concentrations)
        if isnan(self.dG0_r_prime).any():
            dG_r_prime = self.dG0_r_prime.copy()
            for r in range(self.Nr):
                reactants = list(self.S[:, r].nonzero()[0].flat)
                dG_r_prime[0, r] += settings.RT * \
                    log_conc[reactants, 0].T @ self.S[reactants, r]
            return dG_r_prime
        else:
            return self.dG0_r_prime + settings.RT * self.S.T @ log_conc

    def GetPhysiologicalConcentrations(self, bounds=None):
        conc = ones((self.Nc, 1)) * self.DEFAULT_PHYSIOLOGICAL_CONC
        if bounds:
            for i, bound in enumerate(bounds):
                lb, ub = bound
                if lb is not None and ub is not None:
                    if not (lb < conc[i, 0] < ub):
                        conc[i, 0] = sqrt(lb * ub)

        return conc

    def _MakeLnConcentratonBounds(self):
        """Make bounds on logarithmic concentrations.

        Returns:
            A two-tuple (lower bounds, upper bounds).
        """
        bounds = self.concentration_bounds.GetLnBounds(self.cids)
        return bounds

    def _MakeDrivingForceConstraints(self, ln_conc_lb, ln_conc_ub):
        """Generates the A matrix and b & c vectors that can be used in a
        standard form linear problem:
                max          c'x
                subject to   Ax <= b

        x is the vector of (y | log-conc | B)
        where y dG'0 are the reaction Gibbs energy variables, log-conc
        are the natural log of the concentrations of metabolites, and
        B is the max-min driving force variable which is being maximized
        by the LP
        """
        inds = nonzero(diag(self.I_dir))[0].tolist()

        # driving force
        A11 = self.I_dir[inds] @ self.dG0_r_std
        A12 = settings.RT * self.I_dir[inds] @ self.S.T
        A13 = ones((len(inds), 1))

        # covariance var ub and lb
        A21 = eye(self.Nr)
        A22 = zeros((self.Nr, self.Nc))
        A23 = zeros((self.Nr, 1))

        # log conc ub and lb
        A31 = zeros((self.Nc, self.Nr))
        A32 = eye(self.Nc)
        A33 = zeros((self.Nc, 1))

        # upper bound values
        b1 = -self.I_dir[inds] @ self.dG0_r_prime
        b2 = ones((self.Nr, 1))

        A = vstack([hstack([ A11,  A12,  A13]),   # driving force
                    hstack([ A21,  A22,  A23]),   # covariance var ub
                    hstack([-A21,  A22,  A23]),   # covariance var lb
                    hstack([ A31,  A32,  A33]),   # log conc ub
                    hstack([ A31, -A32,  A33])]) # log conc lb

        b = vstack([b1, b2, b2, ln_conc_ub, -ln_conc_lb])

        c = zeros((A.shape[1], 1))
        c[-1, 0] = 1.0

        # change the constaints such that reaction that have an explicit
        # r_bound will not be constrained by B, but will be constained by
        # their specific bounds. Note that we need to divide the bound
        # by R*T since the variables in the LP are not in kJ/mol but in units
        # of R*T.
        if self.r_bounds:
            for i, r_ub in enumerate(self.r_bounds):
                if r_ub is not None:
                    A[i, -1] = 0.0
                    b[i, 0] += r_ub

        return A, b, c

    def _GetPrimalVariablesAndConstants(self):
        # Define and apply the constraints on the concentrations
        ln_conc_lb, ln_conc_ub = self._MakeLnConcentratonBounds()

        # Create the driving force variable and add the relevant constraints
        A, b, c = self._MakeDrivingForceConstraints(ln_conc_lb, ln_conc_ub)

        # the dG'0 covariance eigenvariables
        y = [Variable("y%d" % i) for i in range(self.Nr)]

        # ln-concentration variables
        l = [Variable("l%d" % i) for i in range(self.Nc)]

        return A, b, c, y, l

    def _GetDualVariablesAndConstants(self):
        # Define and apply the constraints on the concentrations
        ln_conc_lb, ln_conc_ub = self._MakeLnConcentratonBounds()

        # Create the driving force variable and add the relevant constraints
        A, b, c = self._MakeDrivingForceConstraints(ln_conc_lb, ln_conc_ub)

        w = [Variable('w%d' % i, lb=0) for i in range(self.Nr_active)]
        g = [Variable('g%d' % i, lb=0) for i in range(2*self.Nr)]
        z = [Variable('z%d' % i, lb=0) for i in range(self.Nc)]
        u = [Variable('u%d' % i, lb=0) for i in range(self.Nc)]

        return A, b, c, w, g, z, u

    def _GetTotalEnergyProblem(self, min_driving_force=0.0, direction='min'):

        A, b, _c, y, l = self._GetPrimalVariablesAndConstants()
        x = y + l + [min_driving_force]
        lp = Model(name='MDF')

        constraints = []
        for j in range(A.shape[0]):
            row = [A[j, i] * x[i] for i in range(A.shape[1])]
            constraints.append(Constraint(sum(row), ub=b[j, 0], name='row_%d' % j))

        total_g0 = float(self.fluxes @ self.dG0_r_prime)
        total_reaction = self.S @ self.fluxes.T
        row = [total_reaction[i, 0] * x[i] for i in range(self.Nc)]
        total_g = total_g0 + sum(row)
        
        lp.add(constraints)
        lp.objective = Objective(total_g, direction=direction)

        return lp

    def _MakeMDFProblem(self):
        """Create a CVXOPT problem for finding the Maximal Thermodynamic
        Driving Force (MDF).

        Does not set the objective function... leaves that to the caller.

        Returns:
            the linear problem object, and the three types of variables as arrays
        """
        A, b, c, y, l = self._GetPrimalVariablesAndConstants()
        B = Variable('mdf')
        x = y + l + [B]
        lp = Model(name="MDF_PRIMAL")

        cnstr_names = ["driving_force_%02d" % j for j in range(self.Nr_active)] + \
                      ["covariance_var_ub_%02d" % j for j in range(self.Nr)] + \
                      ["covariance_var_lb_%02d" % j for j in range(self.Nr)] + \
                      ["log_conc_ub_%02d" % j for j in range(self.Nc)] + \
                      ["log_conc_lb_%02d" % j for j in range(self.Nc)]

        constraints = []
        for j in range(A.shape[0]):
            row = [A[j, i] * x[i] for i in range(A.shape[1])]
            constraints.append(Constraint(sum(row), ub=b[j, 0],
                                          name=cnstr_names[j]))

        lp.add(constraints)

        row = [c[i, 0] * x[i] for i in range(c.shape[0])]
        lp.objective = Objective(sum(row), direction='max')

        return lp, y, l, B

    def _MakeMDFProblemDual(self):
        """Create a CVXOPT problem for finding the Maximal Thermodynamic
        Driving Force (MDF).

        Does not set the objective function... leaves that to the caller.

        Returns:
            the linear problem object, and the four types of variables as arrays
        """
        A, b, c, w, g, z, u = self._GetDualVariablesAndConstants()
        x = w + g + z + u
        lp = Model(name="MDF_DUAL")

        cnstr_names = ["y_%02d" % j for j in range(self.Nr)] + \
                      ["l_%02d" % j for j in range(self.Nc)] + \
                      ["MDF"]
        
        constraints = []
        for i in range(A.shape[1]):
            row = [A[j, i] * x[j] for j in range(A.shape[0])]
            constraints.append(Constraint(sum(row), lb=c[i, 0], ub=c[i, 0],
                                          name=cnstr_names[i]))
        
        lp.add(constraints)

        row = [b[i, 0] * x[i] for i in range(A.shape[0])]
        lp.objective = Objective(sum(row), direction='min')

        return lp, w, g, z, u

    def FindMDF(self, calculate_totals=True):
        """Find the MDF (Optimized Bottleneck Energetics).

        Args:
            c_range: a tuple (min, max) for concentrations (in M).
            bounds: a list of (lower bound, upper bound) tuples for compound
                concentrations.

        Returns:
            A 2 (optimal dGfs, optimal concentrations, optimal mdf).
        """
        
        def get_primal_array(l):
            return array([v.primal for v in l], ndmin=2).T
        
        lp_primal, y, l, B = self._MakeMDFProblem()

        if lp_primal.optimize() != 'optimal':
            logging.warning('LP status %s', lp_primal.status)
            raise Exception("Cannot solve MDF primal optimization problem")

        y = get_primal_array(y)
        l = get_primal_array(l)
        mdf = lp_primal.variables['mdf'].primal
        conc = exp(l)
        #dG0_r_prime = self.dG0_r_prime + self.dG0_r_std @ y

        lp_dual, w, g, z, u = self._MakeMDFProblemDual()
        if lp_dual.optimize() != 'optimal':
            raise Exception("cannot solve MDF dual")

        primal_obj = lp_primal.objective.value
        dual_obj = lp_dual.objective.value
        if abs(primal_obj - dual_obj) > 1e-3:
            raise Exception("Primal != Dual (%.5f != %.5f)"
            % (primal_obj, dual_obj))

        w = get_primal_array(w)
        z = get_primal_array(z)
        u = get_primal_array(u)
        reaction_prices = w
        compound_prices = z-u

        ret = MDFResult(self, mdf, conc, y, reaction_prices, compound_prices)

        if calculate_totals:
            res = {}
            for direction in ['min', 'max']:
                # find the maximum and minimum total Gibbs energy of the pathway,
                # under the constraint that the driving force of each reaction is >= MDF
                lp_total = self._GetTotalEnergyProblem(mdf - 1e-2,
                                                       direction=direction)
                lp_total.optimize()
                if lp_total.status != 'optimal':
                    logging.warning("cannot solve %d total ΔG problem" % direction)
                    res[direction] = nan
                else:
                    res[direction] = lp_total.objective.value

            ret.min_total_dG = res['min']
            ret.max_total_dG = res['max']

        return ret

    @property
    def mdf_result(self):
        ret = self.FindMDF()
        return ret
