import numpy as np

class CriticalResourceRouter:
    """
    Constrained dynamic resource routing engine.
    Allocates limited national medical reserves (ICU beds, ventilators, antivirals)
    prioritizing nodes with high forecasted infection acceleration and high population risk.
    """
    def __init__(self, num_nodes, total_budget=6.0):
        self.num_nodes = num_nodes
        self.total_budget = total_budget  # Total intervention units available
        
    def optimize_dispatch(self, predicted_surges, populations):
        """
        Calculates optimal intervention vector:
        dispatch_i proportional to predicted_surge_i * log(population_i)
        constrained to sum(dispatch) <= total_budget and max per node <= 1.0.
        """
        weights = predicted_surges.flatten() * np.log10(populations / 1e4 + 1.0)
        total_w = np.sum(weights)
        
        if total_w < 1e-6:
            return np.zeros(self.num_nodes)
            
        raw_alloc = (weights / total_w) * self.total_budget
        intervention = np.clip(raw_alloc, 0.0, 1.0)
        return intervention
