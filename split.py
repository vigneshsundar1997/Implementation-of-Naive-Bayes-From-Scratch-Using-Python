import pandas as pd
from sys import argv

def split(input_file,training_file,test_file):
    data = pd.read_csv(input_file)
    test=data.sample(frac=0.2,random_state=47)
    train=data.drop(test.index)

    train.to_csv(training_file,index=False)
    test.to_csv(test_file,index=False)

if __name__ == "__main__":
    input_file = argv[1]
    training_file = argv[2]
    test_file=argv[3]
    split(input_file,training_file,test_file)