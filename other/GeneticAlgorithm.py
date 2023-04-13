import random
import copy

class CHCGeneticAlgorithm:
    def __init__(self, pop_size, chromosome_size, max_generations, crossover_prob, mutation_prob, num_parents):
        self.pop_size = pop_size
        self.chromosome_size = chromosome_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_parents = num_parents
        self.population = []
        
    def initialize_population(self):
        for i in range(self.pop_size):
            chromosome = [random.randint(0, 1) for j in range(self.chromosome_size)]
            self.population.append(chromosome)
    
    def evaluate_fitness(self, chromosome):
        fitness = 0
        # fitness function implementation here
        return fitness
    
    def tournament_selection(self, population, tournament_size):
        tournament = [random.choice(population) for i in range(tournament_size)]
        winner = max(tournament, key=self.evaluate_fitness)
        return winner
    
    def mixing(self, parent1, parent2):
        child1 = []
        child2 = []
        for i in range(self.chromosome_size):
            if parent1[i] == parent2[i]:
                child1.append(parent1[i])
                child2.append(parent1[i])
            else:
                child1.append(random.choice([parent1[i], parent2[i]]))
                child2.append(random.choice([parent1[i], parent2[i]]))
        return child1, child2
    
    def mutation(self, chromosome):
        for i in range(self.chromosome_size):
            if random.random() < self.mutation_prob:
                chromosome[i] = 1 - chromosome[i]
        return chromosome
    
    def run(self):
        self.initialize_population()
        
        for generation in range(self.max_generations):
            parents = [self.tournament_selection(self.population, self.num_parents) for i in range(self.pop_size)]
            offspring = []
            
            for i in range(0, self.pop_size, 2):
                parent1 = parents[i]
                parent2 = parents[i+1]
                
                if random.random() < self.crossover_prob:
                    child1, child2 = self.mixing(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                offspring.append(child1)
                offspring.append(child2)
            
            self.population = copy.deepcopy(offspring)
            
            print("Generation:", generation+1, "Best Fitness:", max([self.evaluate_fitness(chromosome) for chromosome in self.population]))
