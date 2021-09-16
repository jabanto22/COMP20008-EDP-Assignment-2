import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

def data_classification():
    # load in the data
    world = pd.read_csv('world.csv',encoding = 'ISO-8859-1')
    life = pd.read_csv('life.csv',encoding = 'ISO-8859-1')
    life.columns = ['Country Name','Country Code','Time','Life expectancy at birth (years)']
    
    # replace .. with Nan values for later imputation
    for i in world.columns:
        for j in range(len(world)):
            if world[i][j] == '..':
                world[i][j] = np.nan
    
    life_world_linked = life.merge(world,how='left',on=['Country Name','Country Code']).iloc[:264,:]
    
    # get the features
    data=life_world_linked.iloc[:,5:].astype(float)
    
    # get just the class labels
    classlabel=life_world_linked.iloc[:,3]

    # randomly select 66% of the instances to be training and the rest to be testing
    X_train, X_test, y_train, y_test = train_test_split(data, classlabel, train_size=0.66, test_size=0.34, random_state=100)
    
    # impute missing values using the median
    imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    
    # normalise the data to have 0 mean and unit variance
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    
    # get the median, mean, and variance used to impute and scale each feature on the training set
    feature = pd.DataFrame(data.columns)
    feature.columns = ['feature']
    median = pd.DataFrame(imputer.statistics_.round(3))
    median.columns = ['median']
    mean = pd.DataFrame(scaler.mean_.round(3))
    mean.columns = ['mean']
    variance = pd.DataFrame(scaler.var_.round(3))
    variance.columns = ['variance']
    statistics = pd.concat([feature,median,mean,variance],axis=1)
    open('task2a.csv','w').write(statistics.to_csv(index=False))

    # k-NN classification with k=5
    k5_nn = neighbors.KNeighborsClassifier(n_neighbors=5)
    k5_nn.fit(X_train, y_train)
    k5_y_pred=k5_nn.predict(X_test)
    
    # k-NN classification with k=10
    k10_nn = neighbors.KNeighborsClassifier(n_neighbors=10)
    k10_nn.fit(X_train, y_train)
    k10_y_pred=k10_nn.predict(X_test)
    
    # Decision Tree with maximum depth=4
    dt = DecisionTreeClassifier(max_depth=4)
    dt.fit(X_train, y_train)
    dt_y_pred=dt.predict(X_test)
    
    # print accuracy scores of the three classification algorithms
    print('Accuracy of decision tree:','{0:.3%}'.format(accuracy_score(y_test, dt_y_pred)))
    print('Accuracy of k-nn (k=5):','{0:.3%}'.format(accuracy_score(y_test, k5_y_pred)))
    print('Accuracy of k-nn (k=10):','{0:.3%}'.format(accuracy_score(y_test, k10_y_pred)))
    
    
# Test function 
def test():
    
    data_classification()
    
if __name__ == "__main__":
    test()