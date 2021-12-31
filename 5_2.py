import pandas as pd
import importlib
import matplotlib.pyplot as plt
import numpy as np
from discretize import discretize
from split import split

naive_bayes = importlib.import_module("nbc")


bins = [2,5,10,50,100,200]

training_scores = []
testing_scores = []

for bin in bins:
    print('Bin size:',bin)
    discretize(bin,False,'dating.csv','dating-binned.csv')
    split('dating-binned.csv','trainingSet.csv','testSet.csv')
    training_accuracy,testing_accuracy = naive_bayes.nbc(1)
    training_scores.append(training_accuracy)
    testing_scores.append(testing_accuracy)

space=np.arange(len(bins))
plt.plot(training_scores,label='Training')
plt.plot(testing_scores,label='Testing')
plt.xticks(space,bins)
plt.legend()
plt.title('Accuracy based on Bin Size')
plt.show()
