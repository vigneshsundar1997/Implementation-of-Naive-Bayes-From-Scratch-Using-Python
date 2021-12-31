import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sys import argv

input_file = argv[1]
data = pd.read_csv(input_file)

rating_of_partner_from_participant = ['attractive_partner', 'sincere_partner', 'intelligence_parter', 'funny_partner', 'ambition_partner', 'shared_interests_partner']

for column in rating_of_partner_from_participant:
    uniqueValuesCount = data[column].value_counts()
    uniqueKeys=uniqueValuesCount.keys()
    successValues=[]
    for index in range(0,len(uniqueValuesCount)):
        successValue = data[(data[column] == uniqueKeys[index]) & (data['decision'] == 1) ]
        successValues.append(len(successValue)/uniqueValuesCount.values[index])
    plt.scatter(uniqueKeys,successValues)
    plt.title(column)
    plt.xlabel('Values')
    plt.ylabel('Success Rate')
    plt.show()