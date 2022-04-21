import math
import random

from jmetal.core.problem import BinaryProblem, FloatProblem
from jmetal.core.solution import BinarySolution, FloatSolution
import image_clf_train

class Mammogram(FloatProblem):
    def __init__(self, number_of_variables: int = 6):
        super(Mammogram, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [0 for _ in range(number_of_variables)]
        self.upper_bound = [1 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        TRAIN_DIR = "Inbreast/train"
        VAL_DIR = "Inbreast/val"
        TEST_DIR = "Inbreast/test"
        BEST_MODEL = "ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"

        total = image_clf_train.run(
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            test_dir=TEST_DIR,
            resume_from=BEST_MODEL,
            img_size=[1152, 896],
            rescale_factor=0.003891,
            featurewise_mean=44.33,
            patch_net='resnet50',
            block_type='resnet',
            batch_size=2, #tweak this parameter for better performance
            all_layer_epochs=4, #tweak this parameter for better performance
            load_val_ram=False,
            load_train_ram=False,
            weight_decay=float(solution.variables[0]),
            weight_decay2=solution.variables[1],
            init_lr=solution.variables[2],
            all_layer_multiplier=solution.variables[3],
            pos_cls_weight=solution.variables[4],
            neg_cls_weight=solution.variables[5],
            lr_patience=10,
            es_patience=25,
            augmentation=True,
            nb_epoch = 0
        )
        print(total)

        solution.objectives[0] = total

        return solution

    def get_name(self) -> str:
        return "E2E-Mammogram"

class Sphere(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Sphere, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for x in solution.variables:
            total += x * x

        solution.objectives[0] = total

        return solution

    def get_name(self) -> str:
        return "Sphere"


class SubsetSum(BinaryProblem):
    def __init__(self, C: int, W: list):
        """The goal is to find a subset S of W whose elements sum is closest to (without exceeding) C.

        :param C: Large integer.
        :param W: Set of non-negative integers."""
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = W

        self.number_of_bits = len(self.W)
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MAXIMIZE]
        self.obj_labels = ["Sum"]

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_sum = 0.0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_sum += self.W[index]

        if total_sum > self.C:
            total_sum = self.C - total_sum * 0.1

            if total_sum < 0.0:
                total_sum = 0.0

        solution.objectives[0] = -1.0 * total_sum

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(
            number_of_variables=self.number_of_variables, number_of_objectives=self.number_of_objectives
        )
        new_solution.variables[0] = [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def get_name(self) -> str:
        return "Subset Sum"
