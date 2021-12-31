# Implementation-of-Naive-Bayes-From-Scratch-Using-Python
Implementation of Naive Bayes classifier using python on the Dating dataset

Name : Vignesh Somasundaram
PUID : 0033886171

The project folder contains 8 files: 
1. preprocess.py
2. 2_1.py
3. 2_2.py
4. discretize.py
5. split.py
6. 5_1.py
7. 5_2.py
8. 5_3.py

1. preprocess.py

This script contains the preprocessing steps like removing the quotes, converting to lower case, categorization and normalization.

Execution : python3 preprocess.py input_file output_file

eg: dating-full.csv dating.csv

2. 2_1.py

This file does visualization of the attributes and provides bar graph. It does not output to any file.

Execution : python3 preprocess.py input_file

eg: python3 2_1.py dating.csv

3. 2_2.py

This script does visualization of the attributes and provides scatter plot. It does not output to any file.

Execution : python3 2_2.py input_file

eg: python3 2_3.py dating.csv

4. discretize.py

This script does the binning of the attributes. It takes in input_file and output_file as arguments. 

Execution : python3 discretize.py input_file output_file

eg: python3 2_3.py dating.csv dating-binned.csv

5. split.py

This script does the binning of the attributes. It takes in input_file and output_file as arguments. 

Execution : python3 discretize.py input_file output_file

eg: python3 2_3.py dating.csv dating-binned.csv

6. nbc.py

This script does the training of the model and calculates the training and testing. It uses trainingSet.csv and testSet.csv

Execution : python3 5_1.py

7. 5_2.py

This script split the trainingSet according to various bin sizes and does the training and testing.

It also contains a plot showing the variance of accuracies based on bin sizes.

Execution : python3 5_2.py

8. 5_3.py

This script split the trainingSet according to various fraction sizes of the training set and does the training and testing.

It also contains a plot showing the variance of accuracies based on fraction of training set and how testing set performs on it.

Execution : python3 5_3.py
