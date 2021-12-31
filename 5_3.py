import numpy as np
import importlib
import matplotlib.pyplot as plt
from discretize import discretize
from split import split
naive_bayes = importlib.import_module("nbc")

fraction=[0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
#discretize back to 5 bins
discretize(5,False,'dating.csv','dating-binned.csv')
split('dating-binned.csv','trainingSet.csv','testSet.csv')

training_scores = []
testing_scores = []

for f in fraction:
    print("Fraction:", f)
    #calculate accuracy for each fraction of training
    training_accuracy,testing_accuracy =naive_bayes.nbc(f)
    training_scores.append(training_accuracy)
    testing_scores.append(testing_accuracy)

space=np.arange(len(fraction))
plt.plot(training_scores,label='Training')
plt.plot(testing_scores,label='Testing')
plt.xticks(space,fraction)
plt.legend()
plt.title('Variance of accuracies based on fraction size of training')
plt.show()
