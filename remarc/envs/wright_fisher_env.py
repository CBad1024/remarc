import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
from tianshou.env import DummyVectorEnv
from ..core.landscapes import Landscape

class WrightFisherEnv(gym.Env):

    def __init__(self, pop_size=10000, seq_length=4, mutation_rate=1e-4, gen_per_step=500, total_generations=1000, num_drugs = 10, random_start=False, landscape_list=None, reward_scale=1.0):
        super(WrightFisherEnv, self).__init__()
        self.pop_size = pop_size
        self.seq_length = seq_length
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.switch_interval = gen_per_step # Use gen_per_step as the interval between actions
        self.total_generations = total_generations
        self.genotypes = [''.join(seq) for seq in itertools.product("01", repeat=self.seq_length)]
        self.fit_trajectory = []
    
        if landscape_list is not None:
            self.landscape_list = landscape_list
            self.num_drugs = len(landscape_list)
        else:
            self.num_drugs = num_drugs
            self.landscape_list = [Landscape(self.seq_length, sigma=0.5) for _ in range(self.num_drugs)]

        self.drug_landscapes = np.array([land.ls for land in self.landscape_list])  # (drug, genotype)
        self.concentrations = [0.1]
        self.num_concs = 1

        # Action space: (drug, concentration) pairs (concentration is always 0.1, so just self.num_drugs)
        self.action_space = spaces.Discrete(self.num_drugs)
    
        # Observation space: genotype frequencies
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.genotypes),), dtype=np.float32)

        # State initialization
        self.pop = {}
        self.current_drug = 0
        self.current_conc = 0
        self.generation = 0
        
        self.reward_scale = reward_scale
        
        self.reset()

    def reset(self, seed=None, options=None):
        if len(self.fit_trajectory) == 0:
            self.prev_avg_fitness = 3.5
        else:
            self.prev_avg_fitness = np.mean(self.fit_trajectory)
        self.fit_trajectory = []
        super().reset(seed=seed)
        if self.random_start:
            # Start at a random genotype
            idx = np.random.randint(len(self.genotypes))
            self.pop = {self.genotypes[idx]: self.pop_size}
        else:
            self.pop = {'0' * self.seq_length: self.pop_size}
        self.generation = 0
        self.current_drug = 0
        self.current_conc = 0 
        obs = self._get_obs()
        return obs, {}

    def avg_fitness(self):
        fitness = self.get_fitness_map()
        return sum((self.pop.get(g, 0) / self.pop_size) * fitness[g] for g in self.genotypes)

    def get_fitness_map(self):
        fitness = {geno: self.drug_landscapes[self.current_drug, i] for i, geno in enumerate(self.genotypes)}
        return fitness

    def step(self, action):
        self.current_drug = action
        self.current_conc = 0

        if self.current_drug >= self.num_drugs:
           raise ValueError(f"Current Drug {self.current_drug} is out of bounds for num_drugs {self.num_drugs}")
        
        fitness = {geno: self.drug_landscapes[self.current_drug, i] for i, geno in enumerate(self.genotypes)}

        for _ in range(self.switch_interval):
            self.time_step(fitness)
            self.generation += 1
            if self.generation >= self.total_generations:
                break

        obs = self._get_obs()
        avg_fit = self.avg_fitness()
        self.fit_trajectory.append(avg_fit)

        # Reward structure quadratically prioritizes small fitnesses
        reward = ((1-avg_fit) * self.reward_scale)

        terminated = self.generation >= self.total_generations
        truncated = False  # Gymnasium requires this explicitly
        info = {'avg_fitness': avg_fit}

        if terminated:
            # terminal reward structure
            final_avg_fit = np.mean(self.fit_trajectory[-20:])
            reward += (((1-final_avg_fit)*50) * np.abs((1-final_avg_fit)*50)) * (self.reward_scale / 10.0)
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        freqs = np.array([self.pop.get(geno, 0) for geno in self.genotypes]) / self.pop_size
        return freqs.astype(np.float32)

    def get_obs(self):
        """Public alias for _get_obs(); returns current genotype frequency vector."""
        return self._get_obs()

    def time_step(self, fitness):
        self.mutation_step()
        self.offspring_step(fitness)

    def mutation_step(self):
        mutation_count = np.random.poisson(self.mutation_rate * self.pop_size * self.seq_length)
        for _ in range(mutation_count):
            haplotype = self.get_random_haplotype()
            if self.pop[haplotype] > 1:
                self.pop[haplotype] -= 1
                mutant = self.get_mutant(haplotype)
                self.pop[mutant] = self.pop.get(mutant, 0) + 1

    def get_random_haplotype(self):
        haplotypes, frequencies = zip(*self.pop.items())
        frequencies = np.array(frequencies) / self.pop_size
        return np.random.choice(haplotypes, p=frequencies)

    def get_mutant(self, haplotype):
        site = np.random.randint(0, self.seq_length)
        new_base = '1' if haplotype[site] == '0' else '0'
        return haplotype[:site] + new_base + haplotype[site + 1:]

    def offspring_step(self, fitness):
        haplotypes = list(self.pop.keys())
        frequencies = np.array([self.pop[h] / self.pop_size for h in haplotypes])
        fit_values = np.array([fitness[h] for h in haplotypes])
        
        # Ensure non-negative fitness for weights
        fit_values = np.maximum(fit_values, 0)
        np.nan_to_num(fit_values, copy=False, nan=0.0)
        
        weights = frequencies * fit_values
        total_weight = weights.sum()
        
        if total_weight > 1e-9:
            weights /= total_weight
        else:
            # Fallback to frequencies (neutral drift) if fitnesses are all 0/invalid
            weights = frequencies
            weights /= weights.sum()

        counts = np.random.multinomial(self.pop_size, weights)
        self.pop.clear()
        for haplotype, count in zip(haplotypes, counts):
            if count > 0:
                self.pop[haplotype] = count

    @classmethod
    def getEnv(cls, n_train, n_test, landscape_list = None, num_drugs = 10, gen_per_step=25, seq_length=4, random_start=False, episode_steps=20, reward_scale=1.0):
        total_generations = gen_per_step * episode_steps
        import functools
        fn_train = functools.partial(_make_env_train, landscape_list, num_drugs, gen_per_step, seq_length, random_start, total_generations, reward_scale)
        fn_test = functools.partial(_make_env_test, landscape_list, num_drugs, gen_per_step, seq_length, total_generations, reward_scale)
        train_envs = DummyVectorEnv([fn_train for _ in range(n_train)])
        test_envs = DummyVectorEnv([fn_test for _ in range(n_test)])
        return train_envs, test_envs

    def get_fitness(self, raw=False):
        frequencies = np.array(list(self.pop.values())) / self.pop_size
        haplotypes = list(self.pop.keys())
        state_vector = np.zeros(2**self.seq_length)
        hap_inds = [int(hap, 2) for hap in haplotypes]
        for i, hap in enumerate(hap_inds):
            state_vector[hap] = frequencies[i]

        fitness_vec = self.drug_landscapes[self.current_drug]
        mean_fitness = np.dot(state_vector, fitness_vec)

        if raw:
            # Get g_min, g_max from the active landscape object
            ls = self.landscape_list[self.current_drug]
            if hasattr(ls, "g_min") and ls.g_min is not None:
                return mean_fitness * (ls.g_max - ls.g_min) + ls.g_min
        return mean_fitness

    def get_transition_matrix(self, drug_index, conc_index=0):
        """
        Computes the expected transition matrix T where T[i, j] is the 
        expected frequency of genotype i in the next generation if the 
        current population consists entirely of genotype j.
        
        T[i, j] = (m[i, j] * w[i]) / sum_k (m[k, j] * w[k])
        where m is the mutation matrix and w is the fitness vector.
        """
        num_genotypes = 2**self.seq_length
        fitness = self.drug_landscapes[drug_index]
        
        # Ensure non-negative fitness
        fitness = np.maximum(fitness, 0)
        
        # 1. Construct Mutation Matrix M
        # M[i, j] is prob of mutating from j to i
        M = np.zeros((num_genotypes, num_genotypes))
        for j in range(num_genotypes):
            # Probability of no mutation
            M[j, j] = 1 - (self.mutation_rate * self.seq_length)
            # Probability of single mutations
            for bit in range(self.seq_length):
                i = j ^ (1 << bit)
                M[i, j] = self.mutation_rate
        
        # 2. Coupled with Selection
        # T[i, j] = M[i, j] * fitness[i] / normalized_by_column
        T = np.zeros((num_genotypes, num_genotypes))
        for j in range(num_genotypes):
            col_weights = M[:, j] * fitness
            total_w = np.sum(col_weights)
            if total_w > 1e-9:
                T[:, j] = col_weights / total_w
            else:
                # Neutral drift fallback
                T[:, j] = M[:, j] / np.sum(M[:, j])
                
        return T


def _make_env_train(landscape_list, num_drugs, gen_per_step, seq_length, random_start, total_generations, reward_scale):
    return WrightFisherEnv(landscape_list=landscape_list, num_drugs=num_drugs, gen_per_step=gen_per_step, seq_length=seq_length, random_start=random_start, total_generations=total_generations, reward_scale=reward_scale)


def _make_env_test(landscape_list, num_drugs, gen_per_step, seq_length, total_generations, reward_scale):
    return WrightFisherEnv(landscape_list=landscape_list, num_drugs=num_drugs, gen_per_step=gen_per_step, seq_length=seq_length, random_start=False, total_generations=total_generations, reward_scale=reward_scale)
