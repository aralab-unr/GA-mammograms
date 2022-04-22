from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournament2Selection, BinaryTournamentSelection
from jmetal.problem import ZDT1
from jmetal.problem.singleobjective.unconstrained import Sphere
from mammogram import Mammogram
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.evaluator import MapEvaluator
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.evaluator import SparkEvaluator
from dask.distributed import Client
from distributed import LocalCluster
from genetic_algorithm import GeneticAlgorithm, DistributedGeneticAlgorithm
import os

# remove log files
# tracks how many times GA fitness function has been invoked
if os.path.exists("logs_fitness_function_invoked.txt"):
    os.remove("logs_fitness_function_invoked.txt")

# logs general logging comments
if os.path.exists("logs_common.txt"):
    os.remove("logs_common.txt")

# logs reward for each run of fitness function
if os.path.exists("reward.txt"):
    os.remove("reward.txt")

if __name__ == "__main__":
    problem = Mammogram(6)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=50,
        selection=BinaryTournamentSelection(),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=4000),
        population_evaluator=SparkEvaluator(processes=6)
    )

    # # setup Dask client
    # client = Client(LocalCluster(n_workers=24))
    #
    # ncores = sum(client.ncores().values())
    # print(f'{ncores} cores available')
    #
    # # creates the algorithm
    # max_evaluations = 25000
    #
    # algorithm = DistributedGeneticAlgorithm(
    #     problem=problem,
    #     population_size=100,
    #     mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    #     crossover=SBXCrossover(probability=1.0, distribution_index=20),
    #     termination_criterion=StoppingByEvaluations(max=4000),
    #     number_of_cores=ncores,
    #     client=client
    # )

    algorithm.run()
    result = algorithm.get_result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.get_name()))
    print("Solution: {}".format(result.variables))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))

    # Save results to file
    print_function_values_to_file(result, "FUN." + algorithm.label)
    print_variables_to_file(result, "VAR." + algorithm.label)

    # front = get_non_dominated_solutions([algorithm.get_result()])
    # plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
    # plot_front.plot(front, label='GA-E2E', filename='GA-E2E', format='png')