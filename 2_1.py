import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

input_file = argv[1]
data = pd.read_csv(input_file)



data_male = data[data['gender']==1]
data_female = data[data['gender']==0]

preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']

mean_preference_scores_of_maleParticipant = []
mean_preference_scores_of_femaleParticipant = []

for column in preference_scores_of_participant:
    mean_preference_scores_of_maleParticipant.append(data_male[column].mean())
    mean_preference_scores_of_femaleParticipant.append(data_female[column].mean())

space=np.arange(len(preference_scores_of_participant))
plt.bar(space+0.00, mean_preference_scores_of_femaleParticipant, color = 'b', width = 0.25,label='female')
plt.bar(space+0.25, mean_preference_scores_of_maleParticipant, color = 'r', width = 0.25,label ='male')
plt.xticks(space,preference_scores_of_participant)
plt.xlabel('Attributes')
plt.ylabel('Mean')
plt.legend()
plt.show()