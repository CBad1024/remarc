import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
import functools
from tianshou.env import DummyVectorEnv
from ..core.landscapes import Landscape


class WrightFisherEnv(gym.Env):
    """Wright-Fisher evolutionary environment following the SHEPHERD formulation.

    Population state is a frequency vector x ∈ ℝ^M on the (M-1)-simplex,
    where M = 2^seq_length genotypes.

    Per-generation update (Selection → Mutation → optional Drift):
        1. Selection:  x_sel_i = f_i * x_i / f̄(x)
        2. Mutation:   x_mut   = U @ x_sel      (U is column-stochastic)
        3. Drift:      if stochastic, sample Multinomial(pop_size, x_mut)
                       else x_{t+1} = x_mut     (Fokker-Planck / deterministic)

    Parameters
    ----------
    pop_size : int
        Population size N.  Only affects dynamics in stochastic mode.
    seq_length : int
        Number of bits per genotype.  M = 2^seq_length genotypes.
    mutation_rate : float
        Per-site per-generation mutation rate.  Total per-genotype rate
        μ = mutation_rate * seq_length.  Each Hamming-1 neighbor receives
        probability μ / seq_length = mutation_rate.
    gen_per_step : int
        Number of generations simulated per RL step.
    total_generations : int
        Total generations per episode.
    num_drugs : int
        Number of available drugs (actions).
    random_start : bool
        If True, start each episode at a random genotype.
    landscape_list : list[Landscape] | None
        Pre-built landscape objects.  If None, random landscapes are generated.
    reward_scale : float
        Multiplier for the reward signal.
    stochastic : bool
        If True, apply multinomial sampling (genetic drift) each generation.
        If False, use deterministic frequency update (Fokker-Planck limit).
    """

    def __init__(self, pop_size=10000, seq_length=4, mutation_rate=1e-4,
                 gen_per_step=500, total_generations=1000, num_drugs=10,
                 random_start=False, landscape_list=None, reward_scale=1.0,
                 stochastic=True):
        super(WrightFisherEnv, self).__init__()
        self.pop_size = pop_size
        self.seq_length = seq_length
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.switch_interval = gen_per_step
        self.total_generations = total_generations
        self.stochastic = stochastic
        self.fit_trajectory = []

        # --- Genotype bookkeeping ---
        self.num_genotypes = 2 ** self.seq_length
        self.genotypes = [''.join(seq) for seq in itertools.product("01", repeat=self.seq_length)]

        # --- Landscapes ---
        if landscape_list is not None:
            self.landscape_list = landscape_list
            self.num_drugs = len(landscape_list)
        else:
            self.num_drugs = num_drugs
            self.landscape_list = [Landscape(self.seq_length, sigma=0.5) for _ in range(self.num_drugs)]

        self.drug_landscapes = np.array([land.ls for land in self.landscape_list])  # (num_drugs, M)
        self.concentrations = [0.1]
        self.num_concs = 1

        # --- Spaces ---
        self.action_space = spaces.Discrete(self.num_drugs)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_genotypes,), dtype=np.float32)

        # --- Pre-compute mutation matrix U (column-stochastic, Hamming-1) ---
        self.mutation_matrix = self._build_mutation_matrix()

        # --- State ---
        self.freqs = np.zeros(self.num_genotypes)
        self.current_drug = 0
        self.current_conc = 0
        self.generation = 0
        self.reward_scale = reward_scale

        self.reset()

    # ------------------------------------------------------------------
    # Mutation matrix construction
    # ------------------------------------------------------------------

    def _build_mutation_matrix(self):
        """Build the column-stochastic mutation matrix U.

        For N-bit genotypes with Hamming-1 connectivity:
        - Each genotype i has exactly seq_length neighbors (one per bit flip)
        - Each neighbor j gets U[j, i] = mutation_rate  (= μ / seq_length)
        - Diagonal: U[i, i] = 1 - seq_length * mutation_rate  (= 1 - μ)

        This makes each column sum to 1.
        """
        M = self.num_genotypes
        N = self.seq_length
        mu_per_neighbor = self.mutation_rate  # μ/N = mutation_rate (per-site rate)

        U = np.zeros((M, M))
        for i in range(M):
            for bit in range(N):
                j = i ^ (1 << bit)  # Hamming-1 neighbor
                U[j, i] = mu_per_neighbor
            U[i, i] = 1.0 - N * mu_per_neighbor

        return U

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if len(self.fit_trajectory) == 0:
            self.prev_avg_fitness = 1.0  # Sensible default for raw Chen values
        else:
            self.prev_avg_fitness = np.mean(self.fit_trajectory)
        self.fit_trajectory = []
        self.prev_step_fitness = None  # Track step-to-step changes
        super().reset(seed=seed)

        self.freqs = np.zeros(self.num_genotypes)
        if self.random_start:
            idx = np.random.randint(self.num_genotypes)
            self.freqs[idx] = 1.0
        else:
            self.freqs[0] = 1.0  # All mass on wildtype (00...0)

        self.generation = 0
        self.current_drug = 0
        self.current_conc = 0
        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        return self.freqs.astype(np.float32)

    def get_obs(self):
        """Public alias for _get_obs(); returns current genotype frequency vector."""
        return self._get_obs()

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def avg_fitness(self):
        """Mean fitness under the current drug: f̄ = Σ f_i x_i."""
        fitness_vec = self.drug_landscapes[self.current_drug]
        return np.dot(fitness_vec, self.freqs)

    def get_fitness_map(self):
        """Return dict mapping genotype strings to fitness values (legacy compat)."""
        fitness = {geno: self.drug_landscapes[self.current_drug, i]
                   for i, geno in enumerate(self.genotypes)}
        return fitness

    def get_fitness(self, raw=False):
        """Mean fitness of current population under current drug."""
        return np.dot(self.drug_landscapes[self.current_drug], self.freqs)

    # ------------------------------------------------------------------
    # Evolution: Selection → Mutation → Drift
    # ------------------------------------------------------------------

    def time_step(self, fitness_vec):
        """One generation of Wright-Fisher evolution (SHEPHERD formulation).

        1. Selection:  x_sel_i = f_i · x_i / f̄(x)
        2. Mutation:   x_mut   = U @ x_sel
        3. Drift:      Multinomial(N, x_mut) / N   (if stochastic)
                        or x_{t+1} = x_mut           (if deterministic)
        """
        # --- 1. Selection ---
        f_bar = np.dot(fitness_vec, self.freqs)
        if f_bar > 1e-12:
            x_sel = (fitness_vec * self.freqs) / f_bar
        else:
            x_sel = self.freqs.copy()

        # --- 2. Mutation ---
        x_mut = self.mutation_matrix @ x_sel

        # Clamp tiny negatives from floating-point and re-normalize
        np.maximum(x_mut, 0.0, out=x_mut)
        total = x_mut.sum()
        if total > 1e-12:
            x_mut /= total
        else:
            # Degenerate case: reset to uniform
            x_mut[:] = 1.0 / self.num_genotypes

        # --- 3. Genetic drift (optional) ---
        if self.stochastic:
            counts = np.random.multinomial(self.pop_size, x_mut)
            self.freqs = counts.astype(np.float64) / self.pop_size
        else:
            self.freqs = x_mut

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        self.current_drug = action
        self.current_conc = 0

        if self.current_drug >= self.num_drugs:
            raise ValueError(f"Current Drug {self.current_drug} is out of bounds for num_drugs {self.num_drugs}")

        fitness_vec = self.drug_landscapes[self.current_drug]

        for _ in range(self.switch_interval):
            self.time_step(fitness_vec)
            self.generation += 1
            if self.generation >= self.total_generations:
                break

        obs = self._get_obs()
        avg_fit = self.avg_fitness()
        self.fit_trajectory.append(avg_fit)

        # --- Reward: lower fitness = better drug efficacy ---
        reward = (1 - avg_fit) * self.reward_scale

        # Delta bonus: reward the agent for *reducing* fitness vs previous step.
        if self.prev_step_fitness is not None:
            delta = self.prev_step_fitness - avg_fit  # positive when fitness drops
            reward += delta * self.reward_scale * 0.5
        self.prev_step_fitness = avg_fit

        terminated = self.generation >= self.total_generations
        truncated = False  # Gymnasium requires this explicitly
        info = {'avg_fitness': avg_fit}

        if terminated:
            # Terminal bonus: reward sustained low fitness over the episode
            final_avg_fit = np.mean(self.fit_trajectory[-20:])
            reward += (1 - final_avg_fit) * self.reward_scale * 10

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Transition matrix (analytical)
    # ------------------------------------------------------------------

    def get_transition_matrix(self, drug_index, conc_index=0):
        """
        Computes the expected one-generation transition matrix T.

        T maps a frequency vector x to the expected post-selection+mutation
        frequencies: x' = T(x) is NOT linear in x (because of the f̄ denominator),
        but this method returns the mutation matrix weighted by fitness, which
        gives the correct transition when applied column-wise to a pure state.

        T[i, j] = U[i, j] * f[j] / Σ_k U[k, j] * f[k]
        """
        M = self.num_genotypes
        fitness = self.drug_landscapes[drug_index]
        fitness = np.maximum(fitness, 0)

        # Weight mutation matrix columns by fitness
        T = self.mutation_matrix * fitness[np.newaxis, :]  # broadcast: T[i,j] = U[i,j] * f[j]

        # Normalize each column
        col_sums = T.sum(axis=0)
        for j in range(M):
            if col_sums[j] > 1e-9:
                T[:, j] /= col_sums[j]
            else:
                T[:, j] = self.mutation_matrix[:, j] / self.mutation_matrix[:, j].sum()

        return T

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def getEnv(cls, n_train, n_test, landscape_list=None, num_drugs=10,
               gen_per_step=25, seq_length=4, random_start=False,
               episode_steps=20, reward_scale=1.0, stochastic=True):
        total_generations = gen_per_step * episode_steps
        fn_train = functools.partial(
            _make_env_train, landscape_list, num_drugs, gen_per_step,
            seq_length, random_start, total_generations, reward_scale, stochastic)
        fn_test = functools.partial(
            _make_env_test, landscape_list, num_drugs, gen_per_step,
            seq_length, total_generations, reward_scale, stochastic)
        train_envs = DummyVectorEnv([fn_train for _ in range(n_train)])
        test_envs = DummyVectorEnv([fn_test for _ in range(n_test)])
        return train_envs, test_envs




class ThreeGenotypeEnv(WrightFisherEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_genotypes = 3
    
    def _build_mutation_matrix(self):
        mu = self.mutation_rate
        return np.array([
            [1 - mu, mu, mu],
            [mu, 1 - mu, mu],
            [mu, mu, 1 - mu],
        ])



# --------------------------------------------------------------------------
# Factory helpers (module-level for pickling compatibility)
# --------------------------------------------------------------------------

def _make_env_train(landscape_list, num_drugs, gen_per_step, seq_length,
                    random_start, total_generations, reward_scale, stochastic):
    return WrightFisherEnv(
        landscape_list=landscape_list, num_drugs=num_drugs,
        gen_per_step=gen_per_step, seq_length=seq_length,
        random_start=random_start, total_generations=total_generations,
        reward_scale=reward_scale, stochastic=stochastic)


def _make_env_test(landscape_list, num_drugs, gen_per_step, seq_length,
                   total_generations, reward_scale, stochastic):
    return WrightFisherEnv(
        landscape_list=landscape_list, num_drugs=num_drugs,
        gen_per_step=gen_per_step, seq_length=seq_length,
        random_start=False, total_generations=total_generations,
        reward_scale=reward_scale, stochastic=stochastic)




