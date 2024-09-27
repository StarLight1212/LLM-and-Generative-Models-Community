import numpy as np
import random
from typing import List


class GeneticAlgorithmNQueensAllSolutions:
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.1, generations: int = 10000,
                 max_no_improvement_generations: int = 1000):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.max_no_improvement_generations = max_no_improvement_generations
        self.population = self.initialize_population()
        self.solutions = []  # 用于存储找到的所有解
        self.no_improvement_counter = 0  # 用于记录无进展的世代数

    def initialize_population(self) -> List[List[int]]:
        """Initialize the population with random sequences."""
        return [random.sample(range(1, 9), 8) for _ in range(self.population_size)]

    def calculate_fitness(self, individual: List[int]) -> int:
        """Calculate the fitness of an individual based on the number of attacking queens."""
        a = np.zeros((9, 9))
        attacks = 0

        for i in range(1, 9):
            a[individual[i - 1]][i] = 1

        for i in range(1, 9):
            for k in range(1, i):
                if a[individual[i - 1]][k] == 1:
                    attacks += 1

            t1 = t2 = individual[i - 1]
            for j in range(i - 1, 0, -1):
                if t1 != 1:
                    t1 -= 1
                    if a[t1][j] == 1:
                        attacks += 1
                if t2 != 8:
                    t2 += 1
                    if a[t2][j] == 1:
                        attacks += 1

            t1 = t2 = individual[i - 1]
            for j in range(i + 1, 9):
                if t1 != 1:
                    t1 -= 1
                    if a[t1][j] == 1:
                        attacks += 1
                if t2 != 8:
                    t2 += 1
                    if a[t2][j] == 1:
                        attacks += 1

        return attacks // 2

    def selection(self, population: List[List[int]], fitness_scores: List[int]) -> List[List[int]]:
        """Perform selection using a roulette wheel selection."""
        total_fitness = sum(fitness_scores)
        probabilities = [1 - (f / total_fitness) for f in fitness_scores]
        selected_population = random.choices(population, weights=probabilities, k=self.population_size)
        return selected_population

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform crossover between two parents."""
        crossover_point = random.randint(1, 7)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        print(child)
        return child

    def mutate(self, individual: List[int]) -> List[int]:
        """Perform mutation on an individual."""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(8), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def repair(self, individual: List[int]) -> List[int]:
        """Repair individuals with duplicate entries to ensure validity."""
        missing_values = list(set(range(1, 9)) - set(individual))
        seen = set()
        for i in range(8):
            if individual[i] in seen:
                individual[i] = missing_values.pop()
            else:
                seen.add(individual[i])
        return individual

    def evolve_population(self) -> List[List[int]]:
        """Evolve the population through selection, crossover, and mutation."""
        fitness_scores = [self.calculate_fitness(ind) for ind in self.population]
        # Selection
        selected_population = self.selection(self.population, fitness_scores)

        # Crossover
        next_generation = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            next_generation.append(self.mutate(self.repair(child1)))
            next_generation.append(self.mutate(self.repair(child2)))

        return next_generation

    def is_unique_solution(self, individual: List[int]) -> bool:
        """Check if a solution is unique by comparing it to all found solutions."""
        return individual not in self.solutions

    def store_solution(self, individual: List[int]):
        """Store a solution if it is unique and reset the no improvement counter."""
        if self.is_unique_solution(individual):
            self.solutions.append(individual)
            self.no_improvement_counter = 0  # Reset the counter if a new solution is found
            print(f"New solution found: {individual}. Total solutions: {len(self.solutions)}")

    def run(self) -> List[List[int]]:
        """Run the genetic algorithm to find all solutions, terminating if no progress is made for a while."""
        for generation in range(self.generations):
            self.population = self.evolve_population()
            fitness_scores = [self.calculate_fitness(ind) for ind in self.population]

            new_solutions_found = False
            for i, fitness in enumerate(fitness_scores):
                if fitness == 0:
                    self.store_solution(self.population[i])
                    new_solutions_found = True

            if not new_solutions_found:
                self.no_improvement_counter += 1  # Increase counter if no new solution is found
            else:
                self.no_improvement_counter = 0  # Reset the counter if a new solution is found

            # If no new solution is found for too long, stop the algorithm
            if self.no_improvement_counter >= self.max_no_improvement_generations:
                print(f"No new solutions found for {self.no_improvement_counter} generations. Stopping early.")
                break

            # Every 100 generations, report progress
            if generation % 100 == 0:
                print(f"Generation {generation}: Solutions found so far: {len(self.solutions)}")

        print(f"Completed {generation} generations. Total solutions found: {len(self.solutions)}")
        return self.solutions


# Example usage
ga_solver = GeneticAlgorithmNQueensAllSolutions(population_size=200, mutation_rate=0.2, generations=10000,
                                                max_no_improvement_generations=1000)
all_solutions = ga_solver.run()

print(f"All unique solutions ({len(all_solutions)}) are found.")
