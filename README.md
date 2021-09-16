# COMP20008-EDP-Assignment-2
COMP20008 Elements of Data Processing 2020 Semester 1 - Assignment 2

## Introduction
For this project, you will perform a data linkage on two real-world datasets (Part 1) and explore different classification algorithms (Part 2).

The project is to be coded in Python 3. Eight (8) relevant data sets are provided for this project: 

* datasets for Part 1: Data Linkage 
  * amazon.csv
  * google.csv 
  * amazon google truth.csv 
  * amazon small.csv 
  * google small.csv 
  * amazon google truth small.csv
* datasets for Part 2: Classification: 
  * life.csv
  * world.csv

## Part 1 - Data Linkage
Amazon and Google both have product databases. They would like to link the same products in order to perform joint market research.

### Na¨ıve data linkage without blocking
For this part, data linkage without blocking is performed on two smaller data sets: amazon small.csv and google small.csv.

**Task - 1A:** Using `amazon small.csv` and `google small.csv`, implement the linkage between the two data sets.
Your code for this question is to be contained in a single Python file called `task1a.py` and produce a single csv file `task1a.csv` containing the following two column headings:

 `idAmazon, idGoogleBase`

Each row in the datafile must contain a pair of matched products. For example, if your algorithm only matched product `b000hcz8ey` from the Amazon dataset with product http://www.google.com/base/feeds/snippets/11114112522015930440 from the Google dataset your output task1a.csv would be as follows:

```idAmazon, idGoogleBase b000hcz8ey, http://www.google.com/base/feeds/snippets/11114112522015930440```

The performance is evaluated in terms of recall and precision and the marks in this section will be awarded based on the two measures of your algorithm.

```
recall = tp/(tp + fn)
precision = tp/(tp + fp)
```

where `tp` (true-positive) is the number of true positive pairs, `fp` the number of false positive pairs, `tn` the number of true negatives, and `fn` the number of false negative pairs.

### Blocking for efficient data linkage
Blocking is a method to reduce the computational cost for record linkage.

**Task - B:** Implement a blocking method for the linkage of the `amazon.csv` and `google.csv` data sets.

Your code is be contained in a single Python file called `task1b.py` and must produce two csv files `amazon blocks.csv` and `google blocks.csv`, each containing the following two column headings:

`block_key, product_id`

The product id field corresponds to the `idAmazon` and `idGoogleBase` of the `amazon.csv` and `google.csv` files respectively. Each row in the output files matches a product to a block. For example, if your algorithm placed product `b000hcz8ey` from the Amazon dataset in blocks with block keys x & y, your `amazon blocks.csv` would be as follows:
```
block_key, product_id x, 
b000hcz8ey y, b000hcz8ey
```
A block is uniquely identified by the block key. The same block key in the two block-files (`amazon blocks.csv` and `google blocks.csv`) indicates that the corresponding products cooccur in the same block. For example, if your algorithm placed the amazon product `b000hcz8ey` in block x and placed google product http://www.google.com/base/feeds/snippets/11114112522015930440 in block x, your `amazon blocks.csv` and `google blocks.csv` would be as follows respectively:
```
amazon_blocks.csv: 
  block_key, product_id 
  x, b000hcz8ey
google_blocks.csv: 
  block_key, product_id 
  x, http://www.google.com/base/feeds/snippets/11114112522015930440
```
The two products co-occur in the same block x.
To measure the quality of blocking, we assume that when comparing a pair of records, the pair are always 100% similar and are a match. A pair of records are categorised as follows:

* a record-pair is a true positive if the pair are found in the ground truth set and also the pair co-occur in the same block.
* a record-pair is a false positive if the pair co-occur in some block but are not found in the ground truth set.
* a record-pair is a false negative if the pair do not co-occur in any block but are found in the ground truth set.
* a record-pair is a true negative if the pair do not co-occur in any block and are also not found in the ground truth set.
Then, the quality of blocking can be evaluated using the following two measures:
```
PC (pair completeness) = tp/(tp + fn) 
RR (reduction ratio) = 1 − (tp + fp)/n
```
where n is the total number of all possible record pairs from the two data sets.
`(n = fp + fn + tp + tn)`

### Report on the Data Linkage project
**Task - 1C:** Write a one-page report describing your algorithms and implementations of tasks 1a and 1b. You should discuss:
* How your product comparison works, including your choice of similarity functions and final scoring function and threshold.
* An evaluation of the overall performance of your product comparison and what opportunities exist to improve it.
* How your blocking implementation works.
* An evaluation of the overall performance of your blocking method, how the method relates to the performance measures and what opportunities exist to improve it.

Your report for this task should be contained in a single file called `task1c.pdf` or `task1c.docx`.

## Part 2 - Classification
Each year, the World Bank publishes the World Development Indicators which provide high quality and international comparable statistics about global development and the fight against poverty [1]. As data scientists, we wish to understand how the information can be used to predict average lifespan in different countries. To this end, we have provided the `world.csv` file, which contains some of the World Development Indicators for each country and the `life.csv` file containing information about the average lifespan for each country (based on data from the World Health Organization) [2]. Each data file also contains a country name, country code and year as identifiers for each record. These may be used to link the two datasets but should not be considered features.

### Comparing Classification Algorithms
**Task - 2A:** Compare the performance of the following 3 classification algorithms: k-NN (k=5 and k=10) and Decision tree (with a maximum depth of 4) on the provided data. You may use sklearn’s KNeighborsClassifier and DecisionTreeClassifier functions for this task. For the k-NN classifier, all parameters other than k should be kept at their defaults. For the Decision tree classifier, all parameters other than the maximum depth should be kept at their defaults. Use each classification algorithm to predict the class feature **life expectancy at birth(years)** of the data (**Low**, **Medium** and **High** life expectancy) using the remaining features.

For each of the algorithms, fit a model with the following processing steps:
* Split the dataset into a training set comprising 2/3 of the data and a test set comprising the remaining 1/3 using the train test split function with a random state of 100.
* Perform the same imputation and scaling to the training set: 
  * For each feature, perform median imputation to impute missing values. 
  * Scale each feature by removing the mean and scaling to unit variance.
* Train the classifiers using the training set 
* Test the classifiers by applying them to the test set.
Your code must produce a CSV file called `task2a.csv` describing the median used for imputation for each feature, as well as the mean and variance used for scaling, all rounded to three decimal places. The CSV file must have one row corresponding to each feature. The first three lines of the output should be as follows (where x is a number calculated by your program):
```
feature, median, mean, variance 
Access to electricity, rural (% of rural population) [EG.ELC.ACCS.RU.ZS], x, x, x 
Adjusted savings: particulate emission damage (% of GNI) [NY.ADJ.DPEM.GN.ZS], x, x, x
```
Your code must print the classification accuracy of each classifier to standard output. Your output should look as follows (where the % symbol is replaced by the accuracy of each algorithm, rounded to 3 decimal places):
```
Accuracy of decision tree: % 
Accuracy of k-nn (k=5): % 
Accuracy of k-nn (k=10): %
```
Your code for this question should be contained in a single Python file called `task2a.py`.

### Feature Engineering and Selection
**Task - 2B:** This task will focus on k-NN with k=5 (from here on referred to as 5-NN). In order to achieve higher prediction accuracy for 5-NN, one can investigate the use of feature engineering and selection to predict the class feature of the data. Feature generation involves the creation of additional features. Two possible methods are:
* Interaction term pairs. Given a pair of features f1 and f2, create a new feature f12 = f1 × f2. All possible pairs can be considered.
* Clustering labels: apply k-means clustering to the data in `world` and then use the resulting cluster labels as the values for a new feature fclusterlabel. You will need to decide how many clusters to use. At test time, a label for a testing instance can be created by assigning it to its nearest cluster.
Given a set of N features (the original features plus generated features), feature selection involves selecting a smaller set of n features (n < N).

An alternative method of performing feature engineering & selection is to use Principal Component Analysis (PCA). The first n principal components can be used as features.

Your task in this question is to evaluate how the above methods for feature engineering and selection affect the prediction accuracy compared to using 5-NN on a subset of the original features in `world`. You should:
* Implement feature engineering using interaction term pairs and clustering labels. This should produce a dataset with 211 features (20 original features, 190 features generated by interaction term pairs and 1 feature generated by clustering). You should (in some principled manner) select 4 features from this dataset and perform 5-NN classification.
* Implement feature engineering and selection via PCA by taking the first four principal components. You should use only these four features to perform 5-NN classification.
* Take the first four features (columns D-G, if the dataset is opened in Excel) from the original dataset as a sample of the original 20 features. Perform 5-NN classification.
Your output for this question should include:
* Any text, numbers, or other numerical data you reference in your report, printed to standard output
* Any graphs or charts as `png` files with the prefix `task2b` (e.g. `task2bgraph1.png`, `task2bgraph2.png`)
* The classification accuracy for the test set for of the three methods in the following format, as the last three lines printed to standard output (where the % symbol is replaced by the accuracy of 5-NN using each feature set, rounded to 3 decimal places):
```
Accuracy of feature engineering: % 
Accuracy of PCA: %
Accuracy of first four features: %
```

### Report
**Task - 2C:** Write a 1-2 page report describing your implementations. You should discuss: 
* Which algorithm (decision trees or k-nn) in Task-2A performed better on this dataset? For k-nn, which value of k performed better? Explain your experiments and the results.
* A description of the precise steps you took to perform the analysis in Task-2B.
* The method you used to select the number of clusters for the clustering label feature generation and a justification for the number of clusters you selected.
* The method you used to select four features from the generated dataset of 211 features for your analysis and a justification for this method.
* Which of the three methods investigated in Task-2B produced the best results for classification using 5-NN and why this was the case.
* What other techniques you could implement to improve classification accuracy with this data.
* How reliable you consider the classification model to be.
Your report for this task should be contained in a single file called `task2c.pdf` or `task2c.docx`.

### References
[1] The World Bank, “World development indicators,” 2016, data retrieved from World Development Indicators, https://databank.worldbank.org/source/world-development-indicators.

[2] World Health Organization, “Life expectancy and healthy life expectancy data by country,” 2016, data retrieved from Global Health Observatory data repository, https://apps.who.int/gho/data/node.main.688?lang=en.
