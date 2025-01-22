import pandas as pd
import numpy as np
import math
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn import model_selection
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, cross_val_score, RandomizedSearchCV 
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.pipeline import Pipeline
import pickle
import scipy.stats as stats
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Load raw data
drugML = pd.read_csv('Tox21curveCombinedidentifier.csv', low_memory=False, index_col = 0).drop(['target_gene','cas','smiles','parent_smiles','MappingID','inchikey'],axis=1)

# Perform feture selection
def feature_selection(df, nonzero_thrd = 0.0, cor_thrd = 0.95):
    '''
    remove the zero variance and highly correlated features
    
    df: train features
    
    '''
    selector = VarianceThreshold(nonzero_thrd)
    selector.fit(df)
    nonzero_df = df[df.columns[selector.get_support(indices=True)]]
    
    #remove high correlated features
    ## Create correlation matrix
    corr_matrix = nonzero_df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > cor_thrd)]
    
    return nonzero_df.drop(nonzero_df[to_drop], axis=1)

# filtered_drugML = feature_selection(drugML)
filtered_drugML = drugML

# Normalize the tox curve
def feature_norm_fit(train_df , scaler = MinMaxScaler()):
    '''
    train_df: training data
    
    scaler: return the scaler which will be used for test set.
    '''
    array =  train_df.values
    df_norm = pd.DataFrame(scaler.fit_transform(array), columns=train_df.columns, index=train_df.index)
    return df_norm, scaler

norm_train_features, scaler_features = feature_norm_fit(filtered_drugML)
print(norm_train_features.head(5))

#load gene_enrichment result and create target gene dataframe
df_enrichment = pd.read_csv('enrichment_analysis_results_7_20.csv')
df_enrichment = df_enrichment[df_enrichment['significant'] == True]
target_gene = df_enrichment.target_gene.unique()

gene = pd.read_csv('Tox21curveCombinedidentifier.csv', low_memory=False, index_col = 0).target_gene
gene = gene.tolist()

# create target_gene list as Y in the model
value_list = [[0 for i in range(len(target_gene))] for i in range(len(gene))]
for j in range(len(target_gene)):
    for i in range(len(gene)):
        if target_gene[j] in gene[i]:
            value_list[i][j] = 1
        else:
            value_list[i][j] = 0
df_gene = pd.DataFrame(value_list, columns = target_gene, dtype = int) 
print(df_gene.head(5))

# create classifier
svc_clf = SVC()
knn_clf = KNeighborsClassifier()
xgb_clf = XGBClassifier(random_state =123, n_jobs=-1)
rf_clf =  RandomForestClassifier(random_state =123, n_jobs=-1)

# model selection 
def model_selection(model, params_grid, X, y, 
                    scoring = None, cv=5, n_jobs=6, GridSearch = True, n_iter=20, refit = True):
    '''
    return the refitted model on the whole train data
    '''
    if GridSearch == True:
        model_train = GridSearchCV(model, params_grid, cv=cv, n_jobs=n_jobs, scoring = scoring,refit = refit)
    else:
        model_train = RandomizedSearchCV(model, param_distributions = params_grid, 
                                         n_iter = n_iter,cv=cv, n_jobs=n_jobs, scoring=scoring, refit = refit)
    
    model_train.fit(X, y)
    print("Best parameters set found on development set:", model_train.best_params_ )
    print("Best score:", model_train.best_score_ )
    
    print("Grid scores on development set:")
    print()
    means = model_train.cv_results_['mean_test_score']
    stds = model_train.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, model_train.cv_results_['params']):
       print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return model_train

# params for single gene svc second round selected from the first round
svc_parameters = {
        'kernel': ['rbf'],
        'gamma':[1,1e-2,0.05,1e-1,0.5,1e-3],
        'C':[0.001, 0.01, 0.1, 1, 10, 15, 20, 50, 100]
    }

N = 143
Singlegene_svc = []
y_pro_array = []

for k in range(N):
    # choose the most popular 15 gene related to different compounds
    sum_col = df_gene.sum()
    df_gene_single = df_gene[sum_col.sort_values(ascending=False).index[k]]
    
    # split train and test 
    x_train, x_test, y_train, y_test = train_test_split(norm_train_features, df_gene_single, test_size=0.2, random_state = 42)
    
    # print('start svc model selection for target gene', df_gene_single.name)
    # Singlegene_svc.append(model_selection(svc_clf, svc_parameters, x_train, y_train, scoring = 'roc_auc', cv=5, GridSearch = True, n_jobs=-1))
    
    # print('start svc prediction on test set')
    clf = SVC(kernel='rbf', gamma=0.5, C=20, probability=True).fit(x_train, y_train)
    y_pro = np.array(clf.predict_proba(norm_train_features))[:,1]
    y_pro_array.append(y_pro)

print(y_pro_array)

df_pred = pd.DataFrame(y_pro_array)
df_pred = df_pred.T

df_true = df_gene[sum_col.sort_values(ascending=False).index[:N]]
col = df_true.columns.values
df_pred.columns=col
print(df_pred.head(5))

df_difference = df_pred.sub(df_true)
df_difference.index=norm_train_features.index
print(df_difference.head(5))

interest_targets = df_difference.gt(0.7).apply(lambda x: x.index[x].tolist(), axis=1)
df_interest_targets = pd.DataFrame(interest_targets,columns=['interest_gene_targets'])
print(df_interest_targets.head(5))

df_interest_gene_svc = df_interest_targets[df_interest_targets['interest_gene_targets'].astype(str) != '[]']
print(df_interest_gene_svc.head(5))

df_interest_gene_svc.to_csv('InterestGeneForCompoundsSVC.csv')
df_difference.to_csv('ProbabilityDifferencebetweenPredandTureSVC')