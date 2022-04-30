import matplotlib.pyplot as plt
import numpy as np

best_aucs = []
avg_aucs = []
generations = []
generation = 0

try:
  fp = open('generation_stats.txt', "r")
except IOError:
  print("File not found")
  exit()

for line in fp:
  best_aucs.append(float(line.split(',')[0]))
  avg_aucs.append(float(line.split(',')[1].replace('\n', '')))
  generation += 1
  generations.append(generation)

fp.close()

best_aucs = np.array(best_aucs)
avg_aucs = np.array(avg_aucs)
generations = np.array(generations)

plot1 = plt.figure(1)
plt.title("AUC over GA generations")
plt.plot(generations, best_aucs, label='Best fitness')
plt.plot(generations, avg_aucs, label='Average fitness')
plt.xlabel("Generations")
plt.ylabel("Fitness value")
plt.legend()

plt.show()

