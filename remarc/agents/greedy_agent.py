import numpy as np

class GreedyAgent:
    """
    A baseline greedy agent that evaluates all available drug actions
    and picks the one that minimizes the immediate mean fitness of the population.
    """
    def __init__(self, drug_landscapes):
        """
        Args:
            drug_landscapes: numpy array of shape (num_drugs, num_genotypes)
                             representing the fitness of each genotype under each drug.
        """
        self.drug_landscapes = np.asarray(drug_landscapes)
        
    def get_action(self, state_array):
        """
        Args:
            state_array: numpy array of genotype frequencies summing to 1.0.
        Returns:
            The index of the drug that minimizes the mean fitness.
        """
        # Calculate mean fitness for each drug: dot product of frequencies and fitness
        mean_fitnesses = self.drug_landscapes.dot(state_array)
        
        # We want to minimize fitness (maximize drug efficacy / death rate)
        best_drug = np.argmin(mean_fitnesses)
        return int(best_drug)
