"""SHEPHERD MDP Solver for optimal drug cycling.

Implements the continuous-time Fokker-Planck discretization of the Wright-Fisher
evolutionary model in u-coordinates (stick-breaking parameterization of the
frequency simplex).  The resulting finite MDP is solved exactly via value
iteration using pymdptoolbox.

The u-coordinate system maps the (M-1)-simplex of genotype frequencies x to
unconstrained coordinates u ∈ [0,1]^{M-1} via stick-breaking:

    x_0 = Π_{j=1}^{M-1} (1 - u_j)           (wildtype / residual)
    x_k = u_k · Π_{j=1}^{k-1} (1 - u_j)     for k = 1, ..., M-1

The Fokker-Planck equation in u-space has:
    - Drift:     A_k(u) = A_k^{mut}(u) + A_k^{sel}(u, action)
    - Diffusion: D_k(u) = u_k(1 - u_k) / (2N · S_k)

These are discretized on an L^d grid to form a CTMC generator Ω, then
exponentiated W = exp(Ω · dt) to get a row-stochastic MDP transition matrix.

Reference
---------
Nichol et al., "Steering Evolution with Sequential Therapy to Prevent the
Emergence of Bacterial Antibiotic Resistance" (PLoS Comput. Biol., 2015).
"""

import mdptoolbox.mdp as mdp
import numpy as np
from scipy.sparse import dok_matrix, diags

try:
    from scipy.sparse.linalg import expm as sparse_expm
except ImportError:
    from scipy.linalg import expm as _dense_expm

    def sparse_expm(A):
        """Fallback: convert sparse → dense, compute expm, return result."""
        return _dense_expm(A.toarray())


class ShepherdMDP:
    """Builds and solves the SHEPHERD MDP for optimal drug cycling.

    Parameters
    ----------
    mutation_matrix : ndarray, shape (M, M)
        Column-stochastic per-generation mutation matrix U.
        Internally converted to the rate matrix Mmat = U − I.
    drug_landscapes : ndarray, shape (num_drugs, M)
        Fitness vector for each drug.
    pop_size : int
        Effective population size N (enters the diffusion coefficient).
    L : int
        Number of grid points per axis in u-space.
    dt : float
        Time between drug switches, in generations.
    discount : float
        MDP discount factor γ for value iteration.
    """

    def __init__(self, mutation_matrix, drug_landscapes, pop_size=10000,
                 L=3, dt=500, discount=0.99):
        self.M = mutation_matrix.shape[0]       # number of genotypes
        self.d = self.M - 1                     # dimension of u-space
        self.L = L
        self.a = 1.0 / L                       # grid cell width
        self.dt = dt
        self.discount = discount
        self.pop_size = pop_size

        # Infinitesimal mutation generator: Mmat = U − I
        self.Mmat = mutation_matrix - np.eye(self.M)

        # Fitness landscapes
        self.drug_landscapes = np.asarray(drug_landscapes, dtype=float)
        self.num_drugs = len(drug_landscapes)
        self.actions = list(range(self.num_drugs))

        # u-space grid
        self.N_states = L ** self.d
        self.centers = self.a * (0.5 + np.arange(L))
        self.states = self._build_states()
        self.strides = [L ** r for r in range(self.d)]

        # Populated by solve()
        self.policy = None
        self.value = None

    # ------------------------------------------------------------------
    # Coordinate transformations
    # ------------------------------------------------------------------

    @staticmethod
    def x_from_u(u):
        """Stick-breaking transform: u (len M−1) → x (len M)."""
        u = np.asarray(u, float)
        M = u.size + 1
        x = np.zeros(M, float)
        S = 1.0
        for k in range(1, M):
            uk = u[k - 1]
            x[k] = uk * S
            S *= (1.0 - uk)
        x[0] = S
        return x

    @staticmethod
    def S_prefix(u):
        """Prefix products S_k = Π_{j<k}(1 − u_j).

        Returns array of length d, 0-indexed:
        S[0] = 1, S[1] = (1 − u_0), S[2] = (1 − u_0)(1 − u_1), ...
        """
        u = np.asarray(u, float)
        d = u.size
        S = np.ones(d, float)
        prod = 1.0
        for i in range(d):
            S[i] = prod
            prod *= (1.0 - u[i])
        return S

    @staticmethod
    def pack_from_x(x):
        """Inverse stick-breaking: x (len M) → u (len M−1)."""
        x = np.asarray(x, float)
        M = x.size
        u = np.zeros(M - 1, float)
        cum = 0.0
        for i in range(1, M):
            denom = 1.0 - cum
            u[i - 1] = 0.0 if denom <= 0 else x[i] / denom
            cum += x[i]
        return u

    # ------------------------------------------------------------------
    # Drift and diffusion in u-space
    # ------------------------------------------------------------------

    def get_f(self, action):
        """Fitness vector for a drug (length M)."""
        return self.drug_landscapes[action]

    def get_s_vec(self, action):
        """Selection coefficients: s_k = f_{k+1} − f_0 (length d).

        These are fitness differences relative to the wildtype (genotype 0),
        which is the natural parameterization for the stick-breaking u-coordinates.
        """
        f = self.drug_landscapes[action]
        return f[1:] - f[0]

    def drift_u_mut(self, u):
        """Mutation drift A^{mut}(u) from Mmat · x(u), transformed to u-space."""
        u = np.asarray(u, float)
        d = u.size
        x = self.x_from_u(u)
        Ax = self.Mmat @ x                         # length M
        # Prefix sums of Ax over indices 1..M-1
        pref = np.concatenate([[0.0], np.cumsum(Ax[1:])])   # length d+1
        S = self.S_prefix(u)
        A = np.zeros(d, float)
        for k in range(d):          # u-index k ↔ x-index k+1
            Akx = Ax[k + 1]
            sum_prev = pref[k]      # Σ_{i=1}^{k} Ax[i]
            A[k] = (Akx + u[k] * sum_prev) / S[k]
        return A

    def drift_u_sel(self, u, action):
        """Selection drift A^{sel}(u) using compact tail-sum form."""
        u = np.asarray(u, float)
        d = u.size
        s = self.get_s_vec(action)      # length d, s[0] ↔ genotype 1
        x = self.x_from_u(u)
        S = self.S_prefix(u)
        # Tail sums: T_k = Σ_{j=k}^{d-1} x_{j+1} · s_j
        tail = np.zeros(d + 1, float)
        for k in range(d - 1, -1, -1):
            tail[k] = tail[k + 1] + x[k + 1] * s[k]
        A = np.zeros(d, float)
        for k in range(d):
            A[k] = (u[k] / S[k]) * (S[k] * s[k] - tail[k])
        return A

    def drift_u(self, u, action):
        """Total drift = mutation + selection."""
        return self.drift_u_mut(u) + self.drift_u_sel(u, action)

    def diffusion_u(self, u):
        """Diagonal diffusion D_k(u) = u_k(1 − u_k) / (2N · S_k)."""
        u = np.asarray(u, float)
        S = self.S_prefix(u)
        return u * (1.0 - u) / (2.0 * self.pop_size * S)

    # ------------------------------------------------------------------
    # L^d grid in u-space
    # ------------------------------------------------------------------

    def _build_states(self):
        """Enumerate L^d grid centers in lexicographic order."""
        states = []
        for n in range(self.N_states):
            idx = self.index_to_multi(n)
            u = self.centers[idx]
            states.append(u)
        return states

    def index_to_multi(self, n):
        """Flat index → multi-index (length d)."""
        idx = np.empty(self.d, int)
        for r in range(self.d):
            idx[r] = (n // (self.L ** r)) % self.L
        return idx

    def multi_to_index(self, idx):
        """Multi-index → flat index."""
        n = 0
        for r, ir in enumerate(idx):
            n += (self.L ** r) * ir
        return n

    # ------------------------------------------------------------------
    # Transition rate matrix Ω (Fokker-Planck finite-difference)
    # ------------------------------------------------------------------

    def build_transition_rate_matrix(self, action):
        """Build the infinitesimal CTMC generator Ω for a given drug action.

        Uses central-difference discretization on the L^d u-grid:
            forward  rate: D_r/a² + A_r/(2a)
            backward rate: D_r/a² − A_r/(2a)
        Diagonal entries enforce row-sum = 0.
        """
        Omega = dok_matrix((self.N_states, self.N_states), dtype=np.float64)

        for n in range(self.N_states):
            u = self.states[n]
            A = self.drift_u(u, action)
            D = self.diffusion_u(u)
            idx = self.index_to_multi(n)

            for r in range(self.d):
                # Forward neighbor along axis r
                if idx[r] < self.L - 1:
                    m = n + self.strides[r]
                    rate = D[r] / self.a ** 2 + A[r] / (2 * self.a)
                    Omega[n, m] = rate
                # Backward neighbor along axis r
                if idx[r] > 0:
                    m = n - self.strides[r]
                    rate = D[r] / self.a ** 2 - A[r] / (2 * self.a)
                    Omega[n, m] = rate

        # Diagonal: rows sum to zero
        for n in range(self.N_states):
            row_sum = Omega[n, :].sum()
            Omega[n, n] = -row_sum

        return Omega.tocsr()

    def build_transition_matrix(self, Omega):
        """Compute W = exp(Ω · dt), clipped and row-normalized.

        Parameters
        ----------
        Omega : sparse CSR matrix
            Infinitesimal generator (from build_transition_rate_matrix).

        Returns
        -------
        W : sparse CSR matrix
            Row-stochastic transition probability matrix.
        """
        W = sparse_expm(Omega * self.dt)
        W = W.tocsr()

        # Clip tiny negative entries from expm numerical noise
        if W.nnz:
            data = W.data
            data[data < 0] = 0.0
            W.data = data

        # Row-normalize
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        Dinv = diags(1.0 / row_sums)
        W = Dinv @ W
        return W.tocsr()

    # ------------------------------------------------------------------
    # Reward: −mean fitness (lower fitness = better drug efficacy)
    # ------------------------------------------------------------------

    def compute_reward(self, u_state, action):
        """Immediate reward for a single state–action pair."""
        x = self.x_from_u(u_state)
        return -x.dot(self.get_f(action))

    # ------------------------------------------------------------------
    # Build full MDP (P, R) and solve
    # ------------------------------------------------------------------

    def build_W_and_R(self):
        """Build transition matrices P and reward matrix R for all drug actions.

        Returns
        -------
        P : list of sparse CSR matrices, length num_drugs
            P[a] is the (N_states × N_states) transition matrix for drug a.
        R : ndarray, shape (N_states, num_drugs)
            R[s, a] = −x(u_s) · f_a  (negative mean fitness).
        """
        # Vectorize reward computation
        S_arr = np.array(self.states)                           # (N_states, d)
        X = np.vstack([self.x_from_u(u) for u in S_arr])       # (N_states, M)

        P = []
        R = np.zeros((self.N_states, len(self.actions)), dtype=float)

        for a in self.actions:
            # Build sparse generator Ω(a)
            Omega = self.build_transition_rate_matrix(a)
            # Sparse expm + normalize
            W = self.build_transition_matrix(Omega)
            P.append(W)

            # Reward: −mean fitness under action a
            f = self.get_f(a)
            R[:, a] = -X.dot(f)

        return P, R

    def solve(self, max_iter=1000):
        """Build and solve the MDP via value iteration.

        Returns
        -------
        policy : ndarray, shape (N_states,)
            Optimal drug index for each discretized state.
        value : ndarray, shape (N_states,)
            Value function.
        """
        P, R = self.build_W_and_R()
        vi = mdp.ValueIteration(P, R, discount=self.discount, max_iter=max_iter)
        vi.run()
        self.policy = np.array(vi.policy)
        self.value = np.array(vi.V)
        return self.policy, self.value

    # ------------------------------------------------------------------
    # Policy query
    # ------------------------------------------------------------------

    def get_action(self, x):
        """Look up the optimal action for a frequency vector x.

        Maps x → u → nearest grid cell → policy lookup.

        Parameters
        ----------
        x : ndarray, shape (M,)
            Genotype frequency vector on the simplex.

        Returns
        -------
        action : int
            Optimal drug index from the solved MDP.
        """
        if self.policy is None:
            raise RuntimeError("Call solve() before get_action().")
        u = self.pack_from_x(x)
        # Snap to nearest grid cell
        idx = np.clip(np.round(u / self.a - 0.5).astype(int), 0, self.L - 1)
        n = self.multi_to_index(idx)
        return int(self.policy[n])

    def get_state_index(self, x):
        """Map a frequency vector x to its nearest grid-cell flat index."""
        u = self.pack_from_x(x)
        idx = np.clip(np.round(u / self.a - 0.5).astype(int), 0, self.L - 1)
        return self.multi_to_index(idx)

    # ------------------------------------------------------------------
    # Factory from WrightFisherEnv
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env, L=3, discount=0.99):
        """Create a ShepherdMDP from a WrightFisherEnv instance.

        Pulls the mutation matrix, fitness landscapes, population size,
        and switch interval directly from the environment.

        Parameters
        ----------
        env : WrightFisherEnv
            An initialized Wright-Fisher environment.
        L : int
            Grid resolution per axis (default 3).
        discount : float
            MDP discount factor (default 0.99).

        Returns
        -------
        ShepherdMDP
            Ready to call ``.solve()``.
        """
        return cls(
            mutation_matrix=env.mutation_matrix,
            drug_landscapes=env.drug_landscapes,
            pop_size=env.pop_size,
            L=L,
            dt=env.switch_interval,
            discount=discount,
        )
