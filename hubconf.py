
import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
import sklearn.cluster as skl_cluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.linear_model import LogisticRegression

# You can import whatever standard packages are required
#
# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  
  # write your code here
  # Refer to sklearn data sets
  return X,y

def get_data_circles(n_points=100):
 
  X,y = make_circles(n_samples=n_points,noise=0.05)
  # write your code here
  # Refer to sklearn data sets
 
  # write your code ...
  return X,y

def get_data_mnist():
  from sklearn.datasets import load_digits
  X,y= load_digits()
  
  
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  Km = skl_cluster.KMeans(n_clusters=k)
  Km.fit(X)
  
 
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  # this is the KMeans object
  # write your code ...
  return Km

def assign_kmeans(km=None,X=None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  h = homogeneity_score(ypred_1,ypred_2)
  c = completeness_score(ypred_1,ypred_2)
  v = v_measure_score(ypred_1,ypred_2)
  
  #h,c,v=sklearn.metrics.homogeneity_completeness_v_measure(ypred1, ypred2)
  # refer to sklearn documentation for homogeneity, completeness and vscore
  #h,c,v = 0,0,0 # you need to write your code to find proper values
  return h,c,v
###### PART 2 ######
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Fetching MNIST Dataset
mnist = fetch_openml('mnist_784', version=1)

# Get the data and target


def get_data_mnist():
  mnist = fetch_openml('mnist_784', version=1)

# Get the data and target
  X, y = mnist["data"], mnist["target"]

  
  #print(X.shape)
  return X,y
def build_lr_model(X=None, y=None):
  
  lr_model = LogisticRegression().fit(X, y)
  return lr_model

def build_rf_model(X=None, y=None):
  
  rf_model = RandomForestClassifier().fit(X,y)
  
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  return rf_model

def get_metrics(model1=None,X=None,y=None):
  
  y_pred = model1.predict(X)
  acc = accuracy_score(y, y_pred,average='weighted')
 
  prec = precision_score(y, y_pred,average='weighted')
  
  # recall: tp / (tp + fn)
  rec = recall_score(y, y_pred,average='weighted')
  
  # f1: 2 tp / (2 tp + fp + fn)
  f1 = f1_score(y, y_pred,average='weighted')
  auc=0
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  #acc, prec, rec, f1, auc = 0,0,0,0,0
  # write your code here...
  return acc, prec, rec, f1, auc

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid={"penalty":["l1","l2"]}
  

  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid= {
    
    'max_depth': [1,10,None],
    'n_estimators':[1,10,100],
    'criterion': ['gini', 'entropy']
    
}
  

  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  top1_scores = []
  le = len(X)
  if X.ndim > 2:
      X= X.reshape((le, -1))
      
  for i in metrics:
      grid_search_cv = GridSearchCV(model1,param_grid,scoring = i,cv=cv)
      grid_search_cv.fit(X,y)
      top1_scores.append(grid_search_cv.best_estimator_.get_params())
  
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
 
  
  return top1_scores
