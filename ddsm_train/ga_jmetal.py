from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournament2Selection, BinaryTournamentSelection
from jmetal.problem import ZDT1
from jmetal.problem.singleobjective.unconstrained import Sphere
from mammogram import Mammogram
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, \
    print_variables_to_file

from jmetal.algorithm.singleobjective import GeneticAlgorithm

if __name__ == "__main__":
    problem = Mammogram(6)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        selection=BinaryTournamentSelection(),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=4000)
    )

    algorithm.run()
    result = algorithm.get_result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.get_name()))
    print("Solution: {}".format(result.variables))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))

    # front = get_non_dominated_solutions([algorithm.get_result()])
    # plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
    # plot_front.plot(front, label='GA-E2E', filename='GA-E2E', format='png')