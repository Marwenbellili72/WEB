# optimizer/optimizer.py
import pygad
import numpy as np
import matplotlib.pyplot as plt

class CTScannerOptimization:
    def __init__(self, lower_bounds, upper_bounds):
        # Paramètres de configuration
        self.num_generations = 50
        self.num_parents_mating = 4
        self.sol_per_pop = 50
        self.num_genes = 3  # mA, time, rpm

        # Bornes des variables
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Créer une population initiale aléatoire
        self.initial_population = np.random.uniform(
            low=self.lower_bounds, 
            high=self.upper_bounds, 
            size=(self.sol_per_pop, self.num_genes)
        )

    def fitness_function(self, ga_instance, solution, solution_idx):
        # Ajouter le paramètre ga_instance pour correspondre à la nouvelle API
        mA, time, rpm = solution
        
        # Calcul de la dose de rayonnement
        dose = mA * time / rpm
        
        # Calcul de la qualité de l'image 
        quality = (mA * time) / np.log(rpm + 1)
        
        # Normalisation et calcul de la fitness
        normalized_dose = (dose - np.min(self.lower_bounds[0] * self.lower_bounds[1] / self.upper_bounds[2])) / \
                  (np.max(self.upper_bounds[0] * self.upper_bounds[1] / self.upper_bounds[2]) - 
                   np.min(self.lower_bounds[0] * self.lower_bounds[1] / self.upper_bounds[2]))

        normalized_quality = (quality - np.min((self.lower_bounds[0] * self.lower_bounds[1]) / np.log(self.upper_bounds[2] + 1))) / \
                     (np.max((self.upper_bounds[0] * self.upper_bounds[1]) / np.log(self.lower_bounds[2] + 1)) - 
                      np.min((self.lower_bounds[0] * self.lower_bounds[1]) / np.log(self.upper_bounds[2] + 1)))
        
        # Fitness combinée : minimiser la dose, maximiser la qualité
        combined_fitness = 1 / (normalized_dose + 1 - normalized_quality)
        
        return combined_fitness

    def run_optimization(self):
        # Initialiser l'algorithme génétique
        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=self.fitness_function,
            initial_population=self.initial_population,
            num_genes=self.num_genes,
            sol_per_pop=self.sol_per_pop,
            init_range_low=self.lower_bounds,
            init_range_high=self.upper_bounds,
            mutation_percent_genes=10,
            mutation_type="random",
            mutation_by_replacement=True,
            random_mutation_min_val=self.lower_bounds,
            random_mutation_max_val=self.upper_bounds
        )

        # Exécuter l'algorithme génétique
        ga_instance.run()

        # Récupérer la meilleure solution
        best_solution, best_fitness, _ = ga_instance.best_solution()
        
        # Analyser et afficher la meilleure solution
        mA, time, rpm = best_solution
        dose = mA * time / rpm
        quality = (mA * time) / np.log(rpm + 1)
        
        result = {
            "mA": mA,
            "time": time,
            "rpm": rpm,
            "dose": dose,
            "quality": quality,
            "fitness": best_fitness
        }

        return result
