import numpy as np
import networkx as nx

class MetapopulationNetwork:
    """
    30-Node Scale-Free Metapopulation Network with Transit Mobility
    and Coupled SEIRS Contagion Dynamics.
    """
    def __init__(self, num_nodes=30, seed=42):
        np.random.seed(seed)
        self.num_nodes = num_nodes
        
        # Scale-free Barabási-Albert network for realistic metropolitan hub topologies
        self.graph = nx.barabasi_albert_graph(num_nodes, 2, seed=seed)
        self.adj_matrix = nx.to_numpy_array(self.graph)
        
        # Node populations (50,000 to 2,500,000 residents per hub)
        self.populations = np.random.uniform(5e4, 2.5e6, size=num_nodes)
        
        # Dynamic SEIRS states per node: S, E, I, R (proportions of population)
        self.S = np.ones(num_nodes)
        self.E = np.zeros(num_nodes)
        self.I = np.zeros(num_nodes)
        self.R = np.zeros(num_nodes)
        
        # Seed infection in the two highest-degree hub nodes
        hub_indices = np.argsort(-np.sum(self.adj_matrix, axis=1))[:2]
        for hub in hub_indices:
            self.I[hub] = 0.04
            self.S[hub] = 0.96
            
        # Epidemiological parameters
        self.beta_base = 0.38   # Transmission rate
        self.sigma = 0.20       # Incubation rate (1/incubation period ~ 5 days)
        self.gamma = 0.10       # Recovery rate (1/infectious period ~ 10 days)
        self.xi = 0.015         # Immunity waning rate
        self.mobility_rate = 0.025  # Inter-city commuter flux rate
        
    def step(self, intervention_vector=None):
        """
        Advances one day of contagion dynamics using Euler integration
        over coupled metapopulation differential equations.
        """
        if intervention_vector is None:
            intervention_vector = np.zeros(self.num_nodes)
            
        # Effective beta after targeted supply/antiviral intervention [0 to 1]
        effective_beta = self.beta_base * (1.0 - 0.65 * np.clip(intervention_vector, 0.0, 1.0))
        
        # Cross-node infection pressure from commuter influx
        deg = np.sum(self.adj_matrix, axis=1) + 1e-6
        commuter_flux = (self.adj_matrix @ self.I) / deg
        force_of_infection = effective_beta * (self.I + self.mobility_rate * commuter_flux)
        
        # SEIRS differentials
        new_exposed = force_of_infection * self.S
        new_infected = self.sigma * self.E
        new_recovered = self.gamma * self.I
        waned_immune = self.xi * self.R
        
        self.S = np.clip(self.S - new_exposed + waned_immune, 0.0, 1.0)
        self.E = np.clip(self.E + new_exposed - new_infected, 0.0, 1.0)
        self.I = np.clip(self.I + new_infected - new_recovered, 0.0, 1.0)
        self.R = np.clip(self.R + new_recovered - waned_immune, 0.0, 1.0)
        
        return self.I.copy()
