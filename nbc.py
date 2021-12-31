import pandas as pd
import numpy as np


def split_features_outcome(data):
    features = data.drop([data.columns[-1]],axis=1)
    decision = data[data.columns[-1]]
    return features,decision

#return the prior probability
#occurence of a particular decision/total number of samples
def get_prior_probability(decision):
    prior_probability = {}
    size=decision.shape[0]
    for outcome in np.unique(decision):
        count = sum(decision == outcome)
        prior_probability.update({outcome: count/size})
    return prior_probability

#initialize all the attributes values to zero
def initialize_likelihood(features,decision,outcomes):
    featureList = list(features.columns)
    likelihoods = {}

    for feature in featureList:
        likelihoods[feature] = {}
        for feature_value in np.unique(features[feature]):
            for outcome in outcomes:
                likelihoods[feature].update({str(feature_value)+'_'+str(outcome):0})
    return likelihoods

def get_likelihood(features,decision,likelihoods,outcomes,combined_set):
    featureList = features.columns
    for feature in featureList:
        unique_count = combined_set[feature].unique()
        for outcome in outcomes:
            count = sum(decision == outcome)
            #find the conditional probability of each attribute given each outcome
            feature_likelihood = features[feature][decision[decision == outcome].index.values.tolist()].value_counts().to_dict()
            
            #perform the laplace smoothing with smoothing parameter as 1
            for feature_val, feature_count in feature_likelihood.items():
                likelihoods[feature].update({(str(feature_val)+'_'+str(outcome)) : (feature_count+1)/(count + len(unique_count))})
            
            #apply laplace smoothing for values not found in training set
            featureMap = likelihoods[feature]
            for feat_val,feat_count in featureMap.items():
                if feat_count == 0:
                    likelihoods[feature].update({feat_val : (1)/(count + len(unique_count))})
    return likelihoods

def predict(features,decision,prior_probability,likelihoods):
    prediction_outcomes = []

    featureList = features.columns
    rows = np.array(features)

    for row in rows:
        bayes_outcome = {}
        for outcome in np.unique(decision):
            prior = prior_probability[outcome]
            likelihood = 1

            for feature_name, feature_value in zip(featureList,row):
                likelihood *= likelihoods[feature_name][str(feature_value)+'_'+str(outcome)]

            #bayes formula. The prediction probability is removed as it is a constant.
            # P(Y=y|X) = P(X|y) * P(y) . P(X) is removed as it constant
            posterior_probability = likelihood * prior

            bayes_outcome[outcome] = posterior_probability

        label = max(bayes_outcome, key = lambda x: bayes_outcome[x])

        prediction_outcomes.append(label)
    return np.array(prediction_outcomes)

def train_model(features,decision,likelihoods,outcomes,combined_set):
    #calculate the prior probability of the decisions
    prior_probability = get_prior_probability(decision)
    #calculate the conditional probability of each of the attributes unique values given different outcomes
    likelihoods=get_likelihood(features,decision,likelihoods,outcomes,combined_set)
    return prior_probability,likelihoods

def get_accuracy(actual,predcited):
    #round off to two decimals
    return round(float(sum(actual==predcited))/float(len(actual)),2)

def nbc(t_frac):
    data_train=pd.read_csv('trainingSet.csv')
    data_test=pd.read_csv('testSet.csv')
    #Combine the training and testing sets in order to find all the unique values of an attribute
    combined_set = pd.concat([data_train,data_test])
    #split the training data
    train = data_train.sample(frac=t_frac,random_state=47)

    #split the features and decision
    features,decision = split_features_outcome(combined_set)
    outcomes = np.unique(decision)
    #Initialize likelihoods for all the unique values of all the attributes to zero
    likelihoods = initialize_likelihood(features,decision,outcomes)

    #split the features and decision of training
    features,decision = split_features_outcome(train)
    #train the model
    prior_probability,likelihoods = train_model(features,decision,likelihoods,outcomes,combined_set)
    #test the model after training
    prediction_outcomes = predict(features,decision,prior_probability,likelihoods)

    #get the accuracy
    training_accuracy = get_accuracy(decision,prediction_outcomes)
    print('Training Accuracy:', training_accuracy)

    #test the model
    features,decision = split_features_outcome(data_test)
    prediction_outcomes = predict(features,decision,prior_probability,likelihoods)
    testing_accuracy = get_accuracy(np.array(decision),prediction_outcomes)
    print('Testing Accuracy:',testing_accuracy)

    return training_accuracy,testing_accuracy

if __name__ == "__main__":
    nbc(1)