import functools
from abc import ABC, abstractmethod
from multiprocessing.pool import Pool, ThreadPool
from typing import Generic, List, TypeVar

try:
    import dask
except ImportError:
    pass

try:
    from pyspark import SparkConf, SparkContext
except ImportError:
    pass

from jmetal.core.problem import Problem

S = TypeVar("S")


class Evaluator(Generic[S], ABC):
    @abstractmethod
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        pass

    @staticmethod
    def evaluate_solution(solution: S, problem: Problem) -> None:
        problem.evaluate(solution)


class SequentialEvaluator(Evaluator[S]):
    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        for solution in solution_list:
            Evaluator.evaluate_solution(solution, problem)

        return solution_list


class MapEvaluator(Evaluator[S]):
    def __init__(self, processes: int = None):
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        self.pool.map(lambda solution: Evaluator.evaluate_solution(solution, problem), solution_list)

        return solution_list


class MultiprocessEvaluator(Evaluator[S]):
    def __init__(self, processes: int = None):
        super().__init__()
        self.pool = Pool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)


class SparkEvaluator(Evaluator[S]):
    def __init__(self, processes: int = 8):
        self.spark_conf = SparkConf()\
            .setAppName("jmetalpy") \
            .set("spark.task.resource.gpu.amount", "1") \
            .set("spark.default.parallelism", processes) \
            .set("spark.acls.enable", "false") \
            .set("spark.modify.acls", "adarshsehgal") \
            .set("spark.executor.memory", "8g") \
            .set("spark.executor.resource.gpu.discoveryScript", "/home/adarshsehgal/workspace/GA-mammograms/ddsm_train/getGpusResources.sh") \
            .set("spark.driver.resource.gpu.discoveryScript", "/home/adarshsehgal/workspace/GA-mammograms/ddsm_train/getGpusResources.sh") \
            .setMaster("spark://192.168.0.152:7077")

            # .set("spark.task.cpus", "12") \
            # .set("spark.driver.allowMultipleContexts", "true") \
            # .set("spark.driver.maxResultSize", "0") \
            #  \
            #     .set("spark.submit.deployMode", "client") \
            #     .set("spark.driver.cores", "4") \
            #     .set("spark.driver.supervise", "true") \
            #     .set("spark.task.cpus", "2") \
        self.spark_context = SparkContext(conf=self.spark_conf)

        #adding this to avoid no module found error in pickle
        self.spark_context.addPyFile('/home/adarshsehgal/workspace/GA-mammograms/ddsm_train/jMetalPy.zip')

        self.spark_context.setCheckpointDir("spark_checkpoint_location")
        logger = self.spark_context._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        # spark_context = SparkContext(conf=self.spark_conf)
        # spark_context.setCheckpointDir("spark_checkpoint_location")
        # logger = spark_context._jvm.org.apache.log4j
        # logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
        solutions_to_evaluate = self.spark_context.parallelize(solution_list)

        return solutions_to_evaluate.map(lambda s: problem.evaluate(s)).collect()


def evaluate_solution(solution, problem):
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    def __init__(self, scheduler="processes"):
        self.scheduler = scheduler

    def evaluate(self, solution_list: List[S], problem: Problem) -> List[S]:
        with dask.config.set(scheduler=self.scheduler):
            return list(
                dask.compute(
                    *[dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list]
                )
            )
