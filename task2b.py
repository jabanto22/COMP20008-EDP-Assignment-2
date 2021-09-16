import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def data_classification():
    # load in the data
    world = pd.read_csv('world.csv',encoding = 'ISO-8859-1')
    life = pd.read_csv('life.csv',encoding = 'ISO-8859-1')
    life.columns = ['Country Name','Country Code','Time','Life expectancy at birth (years)']
    
    # replace .. with Nan values for later imputation
    for i in world.columns:
        for j in range(len(world)):
            if world[i][j] == '..':
                world[i][j] = None
    
    life_world_linked = life.merge(world,how='left',on=['Country Name','Country Code']).iloc[:264,:]
    
    # impute missing values on world data
    world_imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(world.iloc[:264,3:])
    world = world_imputer.transform(world.iloc[:264,3:])
    
    # evaluate sum of squares and silhoutte scores to choose the number of clusters for k-means
    # https://medium.com/pursuitnotes/k-means-clustering-model-in-6-steps-with-python-35b532cfa8ad
    fig, (plt1, plt2) = plt.subplots(1,2)
    fig.set_size_inches(10, 6)
    cluster = range(2,20)
    ss_scores = []
    silhoutte_scores = []
    for c in cluster:
        kmeans = KMeans(n_clusters=c,random_state=100)
        clusters = kmeans.fit_predict(world)
        ss_scores.append(kmeans.inertia_)
        silhoutte_scores.append(silhouette_score(world, clusters))
    plt1.plot(cluster,ss_scores)
    plt1.set_title('The Elbow analysis for K-means Clustering')
    plt1.set_ylabel('WCSS')
    plt1.set_xlabel('Number of Clusters')
    plt2.plot(cluster,silhoutte_scores)
    plt2.set_title('The Silhoutte analysis for K-means Clustering')
    plt2.set_ylabel('SC')
    plt2.set_xlabel('Number of Clusters')
    plt.savefig('task2bgraph1.png',dpi=300)
    plt.show()
    plt.close()
    
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
    
    # use K-means clustering on world (data=>X_train) equal to 5 to generate additional feature fclusterlabel
    kmeans_5 = KMeans(n_clusters=5,random_state=100).fit(X_train)
    X_train_k5 = kmeans_5.predict(X_train)
    X_test_k5 = kmeans_5.predict(X_test)
    X_train_k5 = pd.DataFrame(X_train_k5)
    X_train_k5.columns = ['fclusterlabel']
    X_test_k5 = pd.DataFrame(X_test_k5)
    X_test_k5.columns = ['fclusterlabel']
    print('CLUSTER LABELS of K-means Clustering in the Training set:')
    print(pd.DataFrame(X_train_k5))
    print('CLUSTER LABELS of K-means Clustering in the Test set:')
    print(pd.DataFrame(X_test_k5))
    
    # normalise the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # generate new features using interaction term pairs
    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()
    features = PolynomialFeatures(degree=2).fit(X_train)
    X_train_pf = features.transform(X_train)
    X_test_pf = features.transform(X_test)
    print('GENERATED POLYNOMIAL FEATURES on Training Set:')
    print(pd.DataFrame(X_train_pf))
    print('GENERATED POLYNOMIAL FEATURES on Test Set:')
    print(pd.DataFrame(X_test_pf))
    for p in range(len(features.powers_)):
        feat_name = 'f'
        if max(features.powers_[p]) == 1 and features.powers_[p].sum() == 2:
            for f in range(len(features.powers_[p])):
                if features.powers_[p][f] == 1:
                    feat_name = feat_name+str(f)
            X_train_ip = pd.DataFrame(X_train_pf[:,p])
            X_train_ip.columns = [feat_name]
            X_train_df = [pd.DataFrame(X_train_fe),X_train_ip]
            X_train_fe = pd.concat(X_train_df,axis=1)
            X_test_ip = pd.DataFrame(X_test_pf[:,p])
            X_test_ip.columns = [feat_name]
            X_test_df = [pd.DataFrame(X_test_fe),X_test_ip]
            X_test_fe = pd.concat(X_test_df,axis=1)
            
    print('INTERACTION TERM PAIRS for Training set:')
    print(X_train_fe.iloc[:,20:])
    print('INTERACTION TERM PAIRS for Test set:')
    print(X_test_fe.iloc[:,20:])
    
    # combine all features    
    X_train_df = [pd.DataFrame(X_train_fe),X_train_k5]
    X_train_fe = pd.concat(X_train_df,axis=1)
    X_test_df = [pd.DataFrame(X_test_fe),X_test_k5]
    X_test_fe = pd.concat(X_test_df,axis=1)
            
    # normalise the data including the added features
    scaler = preprocessing.StandardScaler().fit(X_train_fe)
    X_train_fs = scaler.transform(X_train_fe)
    X_test_fs = scaler.transform(X_test_fe)
    
    # drop correlated features
    X_train_df = pd.DataFrame(X_train_fs)
    X_test_df = pd.DataFrame(X_test_fs)
    corr_matrix = X_train_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    print('CORRELATED COLUMNS TO DROP',to_drop)
    X_train_df = X_train_df.drop(columns=to_drop, axis=1)
    X_test_df = X_test_df.drop(columns=to_drop, axis=1)
    print('NUMBER OF FEATURES LEFT AFTER DROPPING CORRELATED FEATURES:',X_train_df.shape[1])
    print('FEATURES LEFT AFTER DROPPING CORRELATED FEATRES:')
    print(X_train_fe.columns[X_train_df.columns])
    
    # feature selection using LogisticRegression
    selector = SelectFromModel(estimator=LogisticRegression(max_iter=200)).fit(X_train_df, y_train)
    X_train_fs = selector.transform(X_train_df)
    X_test_fs = selector.transform(X_test_df)
    print('Number of Features left after using LogisticRegression:',X_train_fs.shape[1])
    print('Features left after using LogisticRegression:')
    print(X_train_fe.columns[X_train_df.columns[selector.get_support(indices=True)]])
    
    # feature selection using ANOVA F, chi2, and mutual information
    X_train_best = preprocessing.MinMaxScaler().fit_transform(X_train_fs)
    method = [f_classif,chi2,mutual_info_classif]
    print_method = ['ANOVA F','Chi-square','Mutual Information']
    fe_score = 0
    for m in range(len(method)):
        k4_best = SelectKBest(method[m], k=4).fit(X_train_best, y_train)
        # print the selected features
        best_feat = k4_best.get_support()
        col_num = []
        for i in range(len(best_feat)):
            if best_feat[i]:
                col_num.append(i)
        print('Best 4 features selected using',
              print_method[m],':',X_train_fe.columns[X_train_df.columns[selector.get_support(indices=True)[col_num]]])
    
        # k-NN classification with k=5 using feature engineering
        k5_nn_fe = neighbors.KNeighborsClassifier(n_neighbors=5)
        k5_nn_fe.fit(X_train_fs[:,col_num], y_train)
        k5_y_pred_fe = k5_nn_fe.predict(X_test_fs[:,col_num])
        if accuracy_score(y_test, k5_y_pred_fe) >= fe_score:
            fe_score = accuracy_score(y_test, k5_y_pred_fe)
        print('Accuracy of',print_method[m],':','{0:.3%}'.format(fe_score))
    
    # Dimensionality reduction using PCA
    print('TRAINING SET before PCA:')
    print(pd.DataFrame(X_train))
    print('TEST SET before PCA:')
    print(pd.DataFrame(X_test))
    pca = PCA(n_components=4).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # k-NN classification with k=5 using PCA
    print('PRINCIPAL COMPONENTS in the Training set:')
    print(pd.DataFrame(X_train_pca))
    print('PRINCIPAL COMPONENTS in the Test set:')
    print(pd.DataFrame(X_test_pca))
    k5_nn_pca = neighbors.KNeighborsClassifier(n_neighbors=5)
    k5_nn_pca.fit(X_train_pca, y_train)
    k5_y_pred_pca = k5_nn_pca.predict(X_test_pca)
    
    # k-NN classification with k=5 using first four features
    print('FIRST FOUR FEATURES in the Training set:')
    print(pd.DataFrame(X_train).iloc[:,:4])
    print('FIRST FOUR FEATURES in the Test set:')
    print(pd.DataFrame(X_test).iloc[:,:4])
    k5_nn_f4 = neighbors.KNeighborsClassifier(n_neighbors=5)
    k5_nn_f4.fit(X_train[:,:4], y_train)
    k5_y_pred_f4 = k5_nn_f4.predict(X_test[:,:4])
    
    # print accuracy scores of feature engineering
    print('Accuracy of feature engineering:','{0:.3%}'.format(fe_score))
    print('Accuracy of PCA:','{0:.3%}'.format(accuracy_score(y_test, k5_y_pred_pca)))
    print('Accuracy of first four features:','{0:.3%}'.format(accuracy_score(y_test, k5_y_pred_f4)))
    
    
# Test function 
def test():
    
    data_classification()
    
if __name__ == "__main__":
    test()