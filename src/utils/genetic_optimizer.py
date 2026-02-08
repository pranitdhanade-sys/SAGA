"""
Genetic Algorithm Optimizer for Sentiment Analysis
Uses GA to optimize model parameters and feature selection
"""

import random
import logging
from typing import List, Tuple, Callable, Dict
import numpy as np
from deap import base, creator, tools, algorithms
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneticOptimizer:
    """
    Genetic Algorithm for optimizing sentiment analysis model parameters.
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 20,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        seed: int = 42
    ):
        """
        Initialize Genetic Optimizer.
        
        Args:
            population_size: Size of GA population
            generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            seed: Random seed
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        random.seed(seed)
        np.random.seed(seed)

    def optimize_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fitness_function: Callable,
        feature_bounds: List[Tuple[float, float]],
    ) -> Dict:
        """
        Optimize feature weights using genetic algorithm.
        
        Args:
            X: Feature matrix
            y: Target labels
            fitness_function: Function to evaluate fitness
            feature_bounds: List of (min, max) bounds for each feature
            
        Returns:
            Best solution and its fitness
        """
        # Define fitness and individual
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Attribute generators
        for i, (min_val, max_val) in enumerate(feature_bounds):
            toolbox.register(
                f"feature_{i}",
                random.uniform,
                min_val,
                max_val
            )
        
        # Individual and population
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [toolbox.register(f"feature_{i}", random.uniform, *bound)
             for i, bound in enumerate(feature_bounds)],
            n=1
        )
        
        # Create population with proper attributes
        def create_individual():
            ind = creator.Individual([random.uniform(b[0], b[1]) for b in feature_bounds])
            return ind
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Genetic operators
        toolbox.register(
            "evaluate",
            lambda ind: (fitness_function(np.array(ind), X, y),)
        )
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", self._mutate_bounds, bounds=feature_bounds, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population
        pop = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Main GA loop
        for gen in range(self.generations):
            # Select next generation
            offspring = toolbox.select(pop, len(pop))
            offspring = [copy.deepcopy(ind) for ind in offspring]
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
            
            # Log best fitness
            best_ind = tools.selBest(pop, 1)[0]
            logger.info(f"Generation {gen}: Best fitness = {best_ind.fitness.values[0]:.4f}")
        
        # Return best solution
        best_ind = tools.selBest(pop, 1)[0]
        return {
            'best_features': list(best_ind),
            'best_fitness': best_ind.fitness.values[0],
            'population': pop
        }

    def _mutate_bounds(self, individual, bounds, indpb=0.3):
        """
        Mutate individual within bounds.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                min_val, max_val = bounds[i]
                individual[i] = random.uniform(min_val, max_val)
        return (individual,)

    def optimize_model_parameters(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        fitness_function: Callable,
        *args,
        **kwargs
    ) -> Dict:
        """
        Optimize model hyperparameters using GA.
        
        Args:
            parameter_ranges: Dict of parameter names to (min, max) ranges
            fitness_function: Function to evaluate fitness
            *args, **kwargs: Additional arguments for fitness function
            
        Returns:
            Best parameters and fitness
        """
        param_names = list(parameter_ranges.keys())
        bounds = [parameter_ranges[name] for name in param_names]
        
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        def create_individual():
            return creator.Individual([random.uniform(b[0], b[1]) for b in bounds])
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register(
            "evaluate",
            lambda ind: (fitness_function(dict(zip(param_names, ind)), *args, **kwargs),)
        )
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", self._mutate_bounds, bounds=bounds, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run algorithm
        pop = toolbox.population(n=self.population_size)
        pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            verbose=True
        )
        
        # Return best solution
        best_ind = tools.selBest(pop, 1)[0]
        best_params = dict(zip(param_names, best_ind))
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best fitness: {best_ind.fitness.values[0]:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_fitness': best_ind.fitness.values[0],
            'population': pop
        }
