import os
import numpy as np
import matplotlib.pyplot as plt

aucArray=[]
timesFitnessFuncEvaluated=[]
population_size = 50
generation_best = []
generation_average = []
generations_index_list = []

if os.path.exists("logs_common-2.txt"):
    with open('logs_common-2.txt', 'r') as f:
        for line in f:
            if 'AUC calculated ' in line:
                aucArray.append(float(line[15:].replace('\n','')))
                with open('logs_rewards.txt', 'a') as output:
                    output.write(str(float(line[15:].replace('\n',''))) + "\n")

for i in range(0, len(aucArray)):
    timesFitnessFuncEvaluated.append(i)

generations = len(aucArray)/population_size

for i in range(0, int(generations)):
    tempArray = aucArray[i*population_size:((i+1)*population_size)]
    tempArray = np.array(tempArray)
    generation_best.append(np.max(tempArray))
    generation_average.append(np.average(tempArray))
    generations_index_list.append(i+1)

generation_best = np.array(generation_best)
generation_average = np.array(generation_average)



plot1 = plt.figure(1)
plt.title("AUC over GA generations")
plt.plot(generations_index_list, generation_best, label='Best fitness')
plt.plot(generations_index_list, generation_average, label='Average fitness')
plt.xlabel("Generations")
plt.ylabel("Fitness value")
plt.legend()

plt.show()

