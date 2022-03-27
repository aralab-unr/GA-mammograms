import os
import numpy as np
import matplotlib.pyplot as plt

aucArray=[]
timesFitnessFuncEvaluated=[]

if os.path.exists("logs_common.txt"):
    with open('logs_common.txt', 'r') as f:
        for line in f:
            if 'AUC calculated ' in line:
                aucArray.append(float(line[15:].replace('\n','')))

for i in range(0,len(aucArray)):
    timesFitnessFuncEvaluated.append(i)

x = np.array(timesFitnessFuncEvaluated)
y = np.array(aucArray)

plt.title("AUC vs. Times Fitness function evaluated")
plt.plot(x, y)
plt.xlabel('Times Fitness function evaluated')
plt.ylabel('AUC')

plt.savefig('AUC vs. Times Fitness function evaluated')

