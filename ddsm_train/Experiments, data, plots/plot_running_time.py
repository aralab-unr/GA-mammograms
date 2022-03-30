import os

runTimeArray=[]
timesFitnessFuncEvaluated=[]

if os.path.exists("../logs_common.txt"):
    with open('../logs_common.txt', 'r') as f:
        for line in f:
            if ' took ' in line:
                runTimeArray.append(float(line[17:27].replace('k','')))
                with open('logs_running_time.txt', 'a') as output:
                    output.write(str(float(line[17:27].replace('k',''))) + "\n")

for i in range(0,len(runTimeArray)):
    timesFitnessFuncEvaluated.append(i)