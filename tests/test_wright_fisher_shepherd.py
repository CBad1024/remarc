"""Tests for SHEPHERD-formulation Wright-Fisher environment.

Validates:
- Mutation matrix properties (column-stochastic, Hamming-1 connectivity)
- Selection step correctness
- Deterministic vs. stochastic mode behavior
- 4-genotype (N=2) and 8-genotype (N=3) systems
"""

import numpy as np
import pytest
from remarc.envs.wright_fisher_env import WrightFisherEnv
from remarc.core.landscapes import Landscape


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def env_4state():
    """4-genotype (N=2) deterministic environment."""
    landscapes = [
        Landscape(N=2, sigma=0.0, ls=np.array([1.0, 1.1, 0.9, 1.05])),
        Landscape(N=2, sigma=0.0, ls=np.array([0.95, 1.0, 1.1, 0.9])),
    ]
    return WrightFisherEnv(
        seq_length=2, landscape_list=landscapes,
        mutation_rate=1e-3, pop_size=10000,
        gen_per_step=10, total_generations=100,
        stochastic=False
    )


@pytest.fixture
def env_8state():
    """8-genotype (N=3) deterministic environment."""
    ls_vals = np.array([0.993, 0.998, 1.009, 1.003, 1.007, 1.001, 0.992, 0.997])
    landscapes = [
        Landscape(N=3, sigma=0.0, ls=ls_vals),
        Landscape(N=3, sigma=0.0, ls=ls_vals[::-1]),
    ]
    return WrightFisherEnv(
        seq_length=3, landscape_list=landscapes,
        mutation_rate=1e-3, pop_size=10000,
        gen_per_step=10, total_generations=100,
        stochastic=False
    )


@pytest.fixture
def env_4state_stochastic():
    """4-genotype stochastic environment."""
    landscapes = [
        Landscape(N=2, sigma=0.0, ls=np.array([1.0, 1.1, 0.9, 1.05])),
    ]
    return WrightFisherEnv(
        seq_length=2, landscape_list=landscapes,
        mutation_rate=1e-3, pop_size=10000,
        gen_per_step=10, total_generations=100,
        stochastic=True
    )


# ── Mutation Matrix Tests ────────────────────────────────────────────────

class TestMutationMatrix:
    """Tests for the pre-computed mutation matrix U."""

    def test_column_stochastic_4state(self, env_4state):
        """Each column of U must sum to 1."""
        U = env_4state.mutation_matrix
        col_sums = U.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-12)

    def test_column_stochastic_8state(self, env_8state):
        """Each column of U must sum to 1."""
        U = env_8state.mutation_matrix
        col_sums = U.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-12)

    def test_hamming1_connectivity_4state(self, env_4state):
        """Only Hamming-1 neighbors should have nonzero off-diagonal entries."""
        U = env_4state.mutation_matrix
        M = env_4state.num_genotypes
        N = env_4state.seq_length
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                hamming = bin(i ^ j).count('1')
                if hamming == 1:
                    assert U[i, j] > 0, f"U[{i},{j}] should be nonzero (Hamming-1 neighbor)"
                else:
                    assert U[i, j] == 0, f"U[{i},{j}] should be zero (Hamming distance {hamming})"

    def test_hamming1_connectivity_8state(self, env_8state):
        """Only Hamming-1 neighbors should have nonzero off-diagonal entries."""
        U = env_8state.mutation_matrix
        M = env_8state.num_genotypes
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                hamming = bin(i ^ j).count('1')
                if hamming == 1:
                    assert U[i, j] > 0
                else:
                    assert U[i, j] == 0

    def test_off_diagonal_values(self, env_4state):
        """Each Hamming-1 neighbor should get μ/N = mutation_rate."""
        U = env_4state.mutation_matrix
        mu_per_neighbor = env_4state.mutation_rate
        M = env_4state.num_genotypes
        N = env_4state.seq_length
        for i in range(M):
            for bit in range(N):
                j = i ^ (1 << bit)
                np.testing.assert_allclose(U[j, i], mu_per_neighbor, atol=1e-15)

    def test_diagonal_values(self, env_4state):
        """Diagonal should be 1 - μ = 1 - seq_length * mutation_rate."""
        U = env_4state.mutation_matrix
        expected_diag = 1.0 - env_4state.seq_length * env_4state.mutation_rate
        for i in range(env_4state.num_genotypes):
            np.testing.assert_allclose(U[i, i], expected_diag, atol=1e-15)

    def test_non_negative(self, env_4state):
        """All entries of U must be non-negative."""
        assert np.all(env_4state.mutation_matrix >= 0)

    def test_non_negative_8state(self, env_8state):
        """All entries of U must be non-negative."""
        assert np.all(env_8state.mutation_matrix >= 0)


# ── Selection Step Tests ─────────────────────────────────────────────────

class TestSelection:
    """Tests for the selection step: x_sel = f * x / f_bar."""

    def test_selection_preserves_simplex(self, env_4state):
        """Post-selection frequencies must sum to 1."""
        env_4state.freqs = np.array([0.25, 0.25, 0.25, 0.25])
        fitness_vec = env_4state.drug_landscapes[0]
        f_bar = np.dot(fitness_vec, env_4state.freqs)
        x_sel = (fitness_vec * env_4state.freqs) / f_bar
        np.testing.assert_allclose(x_sel.sum(), 1.0, atol=1e-12)

    def test_selection_increases_fit_genotypes(self, env_4state):
        """Genotypes with above-average fitness should increase in frequency."""
        env_4state.freqs = np.array([0.25, 0.25, 0.25, 0.25])
        fitness_vec = env_4state.drug_landscapes[0]
        f_bar = np.dot(fitness_vec, env_4state.freqs)
        x_sel = (fitness_vec * env_4state.freqs) / f_bar

        for i in range(len(fitness_vec)):
            if fitness_vec[i] > f_bar:
                assert x_sel[i] > env_4state.freqs[i]
            elif fitness_vec[i] < f_bar:
                assert x_sel[i] < env_4state.freqs[i]


# ── Deterministic Mode Tests ─────────────────────────────────────────────

class TestDeterministicMode:
    """Tests for the Fokker-Planck (deterministic) mode."""

    def test_deterministic_is_reproducible(self, env_4state):
        """Two runs with the same initial state should give identical results."""
        env_4state.reset()
        init_freqs = env_4state.freqs.copy()

        env_4state.step(0)
        result1 = env_4state.freqs.copy()

        env_4state.reset()
        env_4state.freqs = init_freqs.copy()
        env_4state.step(0)
        result2 = env_4state.freqs.copy()

        np.testing.assert_array_equal(result1, result2)

    def test_frequencies_stay_on_simplex(self, env_4state):
        """Frequencies must sum to 1 after multiple steps."""
        env_4state.reset()
        for _ in range(5):
            env_4state.step(0)
            np.testing.assert_allclose(env_4state.freqs.sum(), 1.0, atol=1e-10)
            assert np.all(env_4state.freqs >= 0)

    def test_frequencies_stay_on_simplex_8state(self, env_8state):
        """8-state: frequencies must sum to 1 after multiple steps."""
        env_8state.reset()
        for _ in range(5):
            env_8state.step(0)
            np.testing.assert_allclose(env_8state.freqs.sum(), 1.0, atol=1e-10)
            assert np.all(env_8state.freqs >= 0)


# ── Stochastic Mode Tests ───────────────────────────────────────────────

class TestStochasticMode:
    """Tests for the stochastic Wright-Fisher mode."""

    def test_frequencies_sum_to_one(self, env_4state_stochastic):
        """Frequencies must still sum to 1 after stochastic step."""
        env_4state_stochastic.reset()
        env_4state_stochastic.step(0)
        np.testing.assert_allclose(env_4state_stochastic.freqs.sum(), 1.0, atol=1e-10)

    def test_frequencies_are_multiples_of_inv_N(self, env_4state_stochastic):
        """In stochastic mode, freq * pop_size should be integer counts."""
        env_4state_stochastic.reset()
        env_4state_stochastic.step(0)
        counts = env_4state_stochastic.freqs * env_4state_stochastic.pop_size
        np.testing.assert_allclose(counts, np.round(counts), atol=1e-10)

    def test_counts_sum_to_pop_size(self, env_4state_stochastic):
        """Integer counts must sum to pop_size."""
        env_4state_stochastic.reset()
        env_4state_stochastic.step(0)
        counts = env_4state_stochastic.freqs * env_4state_stochastic.pop_size
        np.testing.assert_allclose(counts.sum(), env_4state_stochastic.pop_size, atol=1e-10)

    def test_stochastic_introduces_variance(self, env_4state_stochastic):
        """Multiple runs from the same state should NOT all be identical."""
        results = []
        for _ in range(10):
            env_4state_stochastic.reset()
            env_4state_stochastic.freqs = np.array([0.25, 0.25, 0.25, 0.25])
            env_4state_stochastic.step(0)
            results.append(env_4state_stochastic.freqs.copy())

        # Check that not all results are identical (stochasticity should produce variation)
        all_same = all(np.array_equal(results[0], r) for r in results[1:])
        assert not all_same, "Stochastic mode should produce different results across runs"


# ── Gymnasium Interface Tests ────────────────────────────────────────────

class TestGymnasiumInterface:
    """Tests for Gymnasium compatibility."""

    def test_reset_returns_tuple(self, env_4state):
        """reset() must return (obs, info) tuple."""
        result = env_4state.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert obs.shape == (4,)
        assert isinstance(info, dict)

    def test_step_returns_5tuple(self, env_4state):
        """step() must return (obs, reward, terminated, truncated, info)."""
        env_4state.reset()
        result = env_4state.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (4,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_matches_freqs(self, env_4state):
        """get_obs() should return the frequency vector."""
        env_4state.reset()
        obs = env_4state.get_obs()
        np.testing.assert_array_equal(obs, env_4state.freqs.astype(np.float32))

    def test_action_space_4state(self, env_4state):
        assert env_4state.action_space.n == 2

    def test_observation_space_4state(self, env_4state):
        assert env_4state.observation_space.shape == (4,)

    def test_action_space_8state(self, env_8state):
        assert env_8state.action_space.n == 2

    def test_observation_space_8state(self, env_8state):
        assert env_8state.observation_space.shape == (8,)


# ── Transition Matrix Tests ──────────────────────────────────────────────

class TestTransitionMatrix:
    """Tests for the analytical transition matrix."""

    def test_column_stochastic(self, env_4state):
        """Transition matrix columns must sum to 1."""
        T = env_4state.get_transition_matrix(drug_index=0)
        col_sums = T.sum(axis=0)
        np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_non_negative(self, env_4state):
        """All entries must be non-negative."""
        T = env_4state.get_transition_matrix(drug_index=0)
        assert np.all(T >= 0)


# ── End-to-End Integration Tests ─────────────────────────────────────────

class TestIntegration:
    """End-to-end tests simulating full episodes."""

    def test_full_episode_4state_deterministic(self):
        """Run a complete episode in 4-state deterministic mode."""
        landscapes = [
            Landscape(N=2, sigma=0.0, ls=np.array([1.0, 1.1, 0.9, 1.05])),
            Landscape(N=2, sigma=0.0, ls=np.array([0.95, 1.0, 1.1, 0.9])),
        ]
        env = WrightFisherEnv(
            seq_length=2, landscape_list=landscapes,
            mutation_rate=1e-3, pop_size=10000,
            gen_per_step=50, total_generations=200,
            stochastic=False
        )
        obs, _ = env.reset()
        assert obs.shape == (4,)

        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            assert obs.shape == (4,)
            np.testing.assert_allclose(obs.sum(), 1.0, atol=1e-6)

        assert steps > 0

    def test_full_episode_8state_deterministic(self):
        """Run a complete episode in 8-state deterministic mode."""
        ls_vals = np.array([0.993, 0.998, 1.009, 1.003, 1.007, 1.001, 0.992, 0.997])
        landscapes = [
            Landscape(N=3, sigma=0.0, ls=ls_vals),
            Landscape(N=3, sigma=0.0, ls=ls_vals[::-1]),
        ]
        env = WrightFisherEnv(
            seq_length=3, landscape_list=landscapes,
            mutation_rate=1e-3, pop_size=10000,
            gen_per_step=50, total_generations=200,
            stochastic=False
        )
        obs, _ = env.reset()
        assert obs.shape == (8,)

        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            np.testing.assert_allclose(obs.sum(), 1.0, atol=1e-6)

        assert steps > 0

    def test_getEnv_factory(self):
        """Verify getEnv factory produces valid vectorized environments."""
        landscapes = [
            Landscape(N=2, sigma=0.0, ls=np.array([1.0, 1.1, 0.9, 1.05])),
            Landscape(N=2, sigma=0.0, ls=np.array([0.95, 1.0, 1.1, 0.9])),
        ]
        train_envs, test_envs = WrightFisherEnv.get_env(
            2, 1, landscape_list=landscapes, seq_length=2,
            gen_per_step=10, episode_steps=5, stochastic=False
        )
        assert train_envs is not None
        assert test_envs is not None
