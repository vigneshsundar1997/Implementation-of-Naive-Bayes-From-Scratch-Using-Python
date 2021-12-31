import pandas as pd
from sys import argv

def discretize(bins,isPrint,input_file,output_file):
    data = pd.read_csv(input_file)

    excludedColumns = ['gender','race','race_o','samerace','field','decision']

    rangeDictionary = { 'age' : [18,58],
                        'age_o' : [18,58],
                        'importance_same_race' : [0,10],
                        'importance_same_religion' : [0,10],
                        'pref_o_attractive' : [0,1],
                        'pref_o_sincere' : [0,1],
                        'pref_o_intelligence' : [0,1],
                        'pref_o_funny' : [0,1],
                        'pref_o_ambitious' : [0,1],
                        'pref_o_shared_interests' : [0,1],
                        'attractive_important' : [0,1],
                        'sincere_important' : [0,1],
                        'intelligence_important' : [0,1],
                        'funny_important' : [0,1],
                        'ambition_important' : [0,1],
                        'shared_interests_important' : [0,1],
                        'attractive' : [0,10],
                        'sincere' : [0,10],
                        'intelligence' : [0,10],
                        'funny' : [0,10],
                        'ambition' : [0,10],
                        'attractive_partner':[0,10],
                        'sincere_partner':[0,10],
                        'intelligence_parter':[0,10],
                        'funny_partner':[0,10],
                        'ambition_partner':[0,10],
                        'shared_interests_partner':[0,10],
                        'sports':[0,10],
                        'tvsports': [0,10],
                        'exercise': [0,10],
                        'dining':   [0,10],
                        'museums':  [0,10],
                        'art':      [0,10],
                        'hiking':   [0,10],
                        'gaming': [0,10],
                        'clubbing': [0,10],
                        'reading': [0,10],
                        'tv': [0,10],
                        'theater': [0,10],
                        'concerts': [0,10],
                        'music': [0,10],
                        'shopping': [0,10],
                        'yoga': [0,10],
                        'interests_correlate': [-1,1],
                        'expected_happy_with_sd_people': [0,10],
                        'like': [0,10],
                        'movies': [0,10],
                        }

    for column in data.columns:
        if column not in excludedColumns:
            #convert any values greater than the maximum value of the attribute to the maximum value
            data[column] = data[column].apply(lambda x : rangeDictionary[column][1] if x > rangeDictionary[column][1] else x)
            min_value = rangeDictionary[column][0]
            max_value = rangeDictionary[column][1]
            bin_range = (max_value - min_value)/bins
            binValues = [min_value + (i*bin_range) for i in range(0,bins+1)]
            data[column]=pd.cut(data[column],binValues,include_lowest=True,labels=list(range(0,bins)))
            valuesCount = data[column].value_counts().sort_index().values
            if isPrint:
                print(column + ':' , valuesCount)
    
    data.to_csv(output_file,index=False)

if __name__ == "__main__":
    input_file = argv[1]
    output_file = argv[2]
    discretize(5,True,input_file,output_file)