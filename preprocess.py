import pandas as pd
from sys import argv

input_file = argv[1]
output_file = argv[2]

data = pd.read_csv(input_file)
numberOfCellsStripped=0
columnsToStrip = ["race", "race_o", "field"]
encodedDictionary = {}
temp_data = data[columnsToStrip].copy()

#1 starts here
for stripColumn in columnsToStrip:
    data[stripColumn] = data[stripColumn].str.strip("'")
    numberOfCellsStripped+=temp_data[stripColumn].isin(data[stripColumn]).value_counts()[False]

print('Quotes removed from' , numberOfCellsStripped , 'cells.')


#2 starts here
temp_data = data[['field']].copy()
data['field'] = data['field'].str.lower()
numberOfCellsStandardized = temp_data['field'].isin(data['field']).value_counts()[False]
print('Standardized', numberOfCellsStandardized ,'cells to lower case.')

#3 starts here
encodedColumns ={'gender': 'male', 'race': 'European/Caucasian-American', 'race_o': 'Latino/Hispanic American','field':'law'}
for column in encodedColumns:
    data[column], encodedMapping = data[column].factorize(sort=True)
    encodedDictionary[column]=encodedMapping
    print('Value assigned for', encodedColumns[column] ,'in column', column + ':', str(encodedMapping.get_loc(encodedColumns[column])) + '.')

#4 starts here
preference_scores_of_participant = ['attractive_important','sincere_important','intelligence_important','funny_important','ambition_important','shared_interests_important']
preference_scores_of_partner = ['pref_o_attractive','pref_o_sincere', 'pref_o_intelligence','pref_o_funny','pref_o_ambitious','pref_o_shared_interests']

temp_data['total'] = 0

for column in preference_scores_of_participant:
    temp_data['total'] += data[column] 

for column in preference_scores_of_participant:
    data[column] = data[column] / temp_data['total']
    print('Mean of', column + ':' , str(data[column].mean().round(2)) + '.')

temp_data['total'] = 0

for column in preference_scores_of_partner:
    temp_data['total'] += data[column] 

for column in preference_scores_of_partner:
    data[column] = data[column] / temp_data['total']
    print('Mean of', column + ':' , str(data[column].mean().round(2)) + '.')

data.to_csv(output_file,index=False)