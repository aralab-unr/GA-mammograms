#!/usr/bin/env python3.5
from mchgenalg import GeneticAlgorithm
import mchgenalg
import numpy as np
import os
import time

start_time = time.time()

timesEvaluated = 0
bestauc = -1

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

# First, define function that will be used to evaluate the fitness
def fitness_function(genome):
    global timesEvaluated
    timesEvaluated += 1
    start_time = time.time()
    with open('logs_fitness_function_invoked.txt', 'a') as output:
        output.write(str(timesEvaluated) + "\n")
    print("Fitness function invoked " + str(timesEvaluated) + " times")

    # setting parameter values using genome
    pos_cls_weight = decode_function(genome[0:10])
    if pos_cls_weight >= 1:
        pos_cls_weight = 0.999

    neg_cls_weight = decode_function(genome[11:21])
    if neg_cls_weight >= 1:
        neg_cls_weight = 0.999

    weight_decay = decode_function(genome[22:33])
    if weight_decay >= 1:
        weight_decay = 0.999

    weight_decay2 = decode_function(genome[34:44])
    if weight_decay2 >= 1:
        weight_decay2 = 0.999

    init_lr = decode_function(genome[45:55])
    if init_lr >= 1:
        init_lr = 0.999  # 1

    all_layer_multiplier = decode_function(genome[56:66])
    if all_layer_multiplier >= 1:
        all_layer_multiplier = 0.999  # 1

    epochs_default = 4  # 50

    with open('logs_common.txt', 'a') as output:
        output.write("======Setting Parameters value=========" + "\n")
        output.write("weight_decay = " + str(weight_decay))
        output.write(" || weight_decay2 = " + str(weight_decay2))
        output.write(" || init_lr = " + str(init_lr))
        output.write(" || all_layer_multiplier = " + str(all_layer_multiplier))
        output.write(" || pos_cls_weight = " + str(pos_cls_weight))
        output.write(" || neg_cls_weight = " + str(neg_cls_weight) + "\n")

    # query = "export NUM_CPU_CORES=6 \ " \
    query = "python image_clf_train.py \
	--no-patch-model-state \
	--resume-from 'CBIS-DDSM/Combined_full_ROI/inbreast_vgg16_[512-512-1024]x2_hybrid.h5' \
    --img-size 1152 896 \
    --no-img-scale \
    --rescale-factor 0.003891 \
	--featurewise-center \
    --featurewise-mean 44.33 \
    --no-equalize-hist \
    --patch-net resnet50 \
    --block-type resnet \
    --top-depths 512 512 \
    --top-repetitions 2 2 \
    --bottleneck-enlarge-factor 2 \
    --no-add-heatmap \
    --avg-pool-size 7 7 \
    --add-conv \
    --no-add-shortcut \
    --hm-strides 1 1 \
    --hm-pool-size 5 5 \
    --fc-init-units 64 \
    --fc-layers 2 \
    --batch-size 2 \
    --train-bs-multiplier 0.5 \
	--no-augmentation \
	--class-list neg pos \
	--nb-epoch 0 \
    --all-layer-epochs " + str(epochs_default) + "\
    --no-load-val-ram \
    --no-load-train-ram \
    --optimizer adam \
    --weight-decay " + str(weight_decay) + "\
    --hidden-dropout 0.0 \
    --weight-decay2 " + str(weight_decay2) + "\
    --hidden-dropout2 0.0 \
    --init-learningrate " + str(init_lr) + "\
    --all-layer-multiplier " + str(all_layer_multiplier) + "\
	--lr-patience 2 \
	--es-patience 10 \
	--auto-batch-balance \
    --pos-cls-weight " + str(pos_cls_weight) + "\
	--neg-cls-weight " + str(neg_cls_weight) + "\
	--best-model 'INbreast/train_dat_mod/final_hybrid_model.h5' \
	--final-model 'NOSAVE' \
	'INbreast/train_dat_mod/train' 'INbreast/train_dat_mod/val' 'INbreast/train_dat_mod/test'"

    print(query)
    # calling training to calculate number of epochs required to reach close to maximum success rate
    os.system(query)

    if os.path.exists("reward.txt"):
        file = open('reward.txt', 'r')

        # after reading auc value, reset the reward file to avoid incorrect auc being saved
        if os.path.exists("reward.txt"):
            os.remove("reward.txt")

        auc = float(file.read())
    # auc was not calculated for some reason and reward file doesn't exist
    else:
        auc = 0

    if auc == None:
        auc = 0

    ##tracking time to execute one run
    programExecutionTime = time.time() - start_time  # seconds
    programExecutionTime = programExecutionTime / (60)  # minutes
    with open('logs_common.txt', 'a') as output:
        output.write("AUC calculated " + str(auc) + "\n")
        output.write("======Run " + str(timesEvaluated) + " took " + str(
            programExecutionTime) + " minutes to complete=========" + "\n")

    global bestauc
    if bestauc == -1:
        bestauc = auc
    if auc >= bestauc:
            bestauc = auc
            with open('BestParameters.txt', 'a') as output:
                output.write("AUC : " + str(bestauc) + "\n")
                output.write("weight_decay = " + str(weight_decay) + "\n")
                output.write("weight_decay2 = " + str(weight_decay2) + "\n")
                output.write("init_lr = " + str(init_lr) + "\n")
                output.write("all_layer_multiplier = " + str(all_layer_multiplier) + "\n")
                output.write("pos_cls_weight = " + str(pos_cls_weight) + "\n")
                output.write("neg_cls_weight = " + str(neg_cls_weight) + "\n")
                output.write("\n")
                output.write("=================================================")
                output.write("\n")

    print("Best auc so far : " + str(bestauc))

    return auc


def decode_function(genome_partial):
    prod = 0
    for i, e in reversed(list(enumerate(genome_partial))):
        if e == False:
            prod += 0
        else:
            prod += 2 ** abs(i - len(genome_partial) + 1)
    return float(prod)/1000


# Configure the algorithm:
population_size = 50  # 30
genome_length = 66
ga = GeneticAlgorithm(fitness_function)
ga.generate_binary_population(size=population_size, genome_length=genome_length)

# How many pairs of individuals should be picked to mate
ga.number_of_pairs = 5

# Selective pressure from interval [1.0, 2.0]
# the lower value, the less will the fitness play role
ga.selective_pressure = 1.5
ga.mutation_rate = 0.1

# If two parents have the same genotype, ignore them and generate TWO random parents
# This helps preventing premature convergence
ga.allow_random_parent = True  # default True
# Use single point crossover instead of uniform crossover
ga.single_point_cross_over = False  # default False

# Run 100 iteration of the algorithm
# You can call the method several times and adjust some parameters
# (e.g. number_of_pairs, selective_pressure, mutation_rate,
# allow_random_parent, single_point_cross_over)
ga.run(50)  # 30 default 1000
best_genome, best_fitness = ga.get_best_genome()

print("BEST CHROMOSOME IS")
print(best_genome)
print("It's decoded value is")
print("pos_cls_weight = " + str(decode_function(best_genome[0:10])))
print("neg_cls_weight = " + str(decode_function(best_genome[11:22])))
print("weight_decay = " + str(decode_function(best_genome[23:33])))
print("weight_decay2 = " + str(decode_function(best_genome[34:44])))
print("init_lr = " + str(decode_function(best_genome[45:55])))
print("all_layer_multiplier = " + str(decode_function(best_genome[56:66])))

# If you want, you can have a look at the population:
population = ga.population

# and the fitness of each element:
fitness_vector = ga.get_fitness_vector()

# time tracking
programExecutionTime = time.time() - start_time  # seconds
programExecutionTime = programExecutionTime / (60 * 60)  # hours
with open('logs_common.txt', 'a') as output:
    output.write("======Total program execution time is " + str(programExecutionTime) + " hours=========" + "\n")
print("--- %s seconds ---" % (time.time() - start_time))