# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 18:37:51 2021

@author: asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

questions = ['Q2A', 'Q4A', 'Q7A', 'Q9A', 'Q15A', 'Q19A', 'Q20A', 'Q23A', 'Q25A', 'Q28A', 'Q30A', 'Q36A', 'Q40A', 'Q41A']
demographic = ['education', 'urban', 'gender', 'age', 'religion', 'orientation', 'race', 'married']
categorical = ['major', "country"]

data = pd.read_csv("Anxiety.csv")
demo = pd.read_csv("Demog.csv")
# Load data
dataframe = pd.DataFrame(data)
dataframedemo = pd.DataFrame(demo)
cat = dataframe['Anxiety Level']
dataframe.drop('Anxiety Level', inplace=True, axis=1)
#desciptive feature

X=dataframe.copy()
#target
y=cat.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                               test_size=0.1,
                                               random_state=11)
accur={}
result=[]
def logistic():
    pipeline = Pipeline(steps = [
                                 ['classifier', LogisticRegression(multi_class='multinomial',random_state=11,
                                                                   max_iter=100)]])
    
    lg=LogisticRegression(multi_class='multinomial',random_state=11,max_iter=1000)
                                                                   
# =============================================================================
#     KFold(n_splits = 2, shuffle = True, random_state = 100)
# =============================================================================
    #used stratifiedKfold because data is inbalanced.
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
    for train, test in folds.split(X_train,y_train):
    
        print("Train data",train,"Test data",test)
       
    param_grid = {'classifier__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    grid_search1 = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               cv=folds,
                               n_jobs=-1)
    
    grid_search1.fit(X_train, y_train)
    y_pred=grid_search1.predict(X_test)
    print(grid_search1.best_score_)
    print(grid_search1.best_params_)
    cv_score = grid_search1.best_score_
    test_score = grid_search1.score(X_test, y_test)
    cv_results = cross_val_score(grid_search1, X_train, y_train, cv=folds, scoring='accuracy')
    result.append(cv_results)
    accur['Multinomial Logistic Regression'] = accuracy_score(y_test, grid_search1.predict(X_test))
    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')
    print(classification_report(y_test, y_pred))
    
    matrix = confusion_matrix( y_test,y_pred)
    # Create pandas dataframe
    df = pd.DataFrame(matrix, index=['Extremely Severe','Moderate','Severe'], columns=['Extremely Severe','Moderate','Severe'])
    # Create heatmap
    sns.heatmap(df,annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix Multinomial Logistic Regression"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()
# =============================================================================
#     #LEARNING CURVE 
#     train_sizes, train_scores, test_scores = learning_curve(grid_search1 , X, 
#                                                             y, cv=folds, scoring='accuracy', 
#                                                             n_jobs=-1, 
#                                                             train_sizes=np.linspace( 0.01, 1.0, 50))
#     # Create means and standard deviations of training set scores
#     train_mean = np.mean(train_scores, axis=1)
#     train_std = np.std(train_scores, axis=1)
#     # Create means and standard deviations of test set scores
#     test_mean = np.mean(test_scores, axis=1)
#     test_std = np.std(test_scores, axis=1)
#     # Draw lines
#     plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
#     plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")
#     # Draw bands
#     plt.fill_between(train_sizes, train_mean - train_std,
#     train_mean + train_std, color="#DDDDDD")
#     plt.fill_between(train_sizes, test_mean - test_std,
#     test_mean + test_std, color="#DDDDDD")
#     # Create plot
#     plt.title("Learning Curve")
#     plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"),
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.show()
# =============================================================================

   
logistic()
def multiNB():
    modelNB = MultinomialNB(alpha=0.0001, fit_prior = False)
    params = {}
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
    for train, test in folds.split(X_train,y_train):
    
        print("Train data",train,"Test data",test)
    grid_search1 = GridSearchCV(estimator=modelNB,
                               param_grid=params,
                               cv=folds,
                               n_jobs=-1)
    
    grid_search1.fit(X_train, y_train)
    y_pred=grid_search1.predict(X_test)
    print(grid_search1.best_score_)
    print(grid_search1.best_params_)
    cv_score = grid_search1.best_score_
    test_score = grid_search1.score(X_test, y_test)
    cv_results = cross_val_score(grid_search1, X_train, y_train, cv=folds, scoring='accuracy')
    result.append(cv_results)
    accur['Multinomial Naive Bayes'] = accuracy_score(y_test, grid_search1.predict(X_test))
    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')
    print(classification_report(y_test, y_pred))
    
    plot_confusion_matrix(grid_search1, X_test, y_test)
    f1_score(y_test,y_pred, average = 'micro')
    matrix = confusion_matrix( y_test,y_pred)
    # Create pandas dataframe
    df = pd.DataFrame(matrix, index=['Extremely Severe','Moderate','Severe'], columns=['Extremely Severe','Moderate','Severe'])
    # Create heatmap
    sns.heatmap(df,annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix Multinomial Naive Bayes"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()
        
# =============================================================================
#     
# multiNB()   
# =============================================================================
def randomforest():  
    
    modelRF = RandomForestClassifier(random_state=0) 
    
    params = {}
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
    for train, test in folds.split(X_train,y_train):
    
        print("Train data",train,"Test data",test)
    grid_search1 = GridSearchCV(estimator=modelRF,
                               param_grid=params,
                               cv=folds,
                               n_jobs=-1)
    
    grid_search1.fit(X_train, y_train)
    y_pred=grid_search1.predict(X_test)
    print(grid_search1.best_score_)
    print(grid_search1.best_params_)
    cv_score = grid_search1.best_score_
    test_score = grid_search1.score(X_test, y_test)
    cv_results = cross_val_score(grid_search1, X_train, y_train, cv=folds, scoring='accuracy')
    result.append(cv_results)
    accur['Random Forest'] = accuracy_score(y_test, grid_search1.predict(X_test))
    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')
    print(classification_report(y_test, y_pred))
    
    plot_confusion_matrix(grid_search1, X_test, y_test)
    f1_score(y_test,y_pred, average = 'micro')
    matrix = confusion_matrix( y_test,y_pred)
    # Create pandas dataframe
    df = pd.DataFrame(matrix, index=['Extremely Severe','Moderate','Severe'], columns=['Extremely Severe','Moderate','Severe'])
    # Create heatmap
    sns.heatmap(df,annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix Random Forest"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()  
    
# =============================================================================
# randomforest()
# =============================================================================

def KNN():
  
    modelKNN = KNeighborsClassifier(n_neighbors=3)
    params = {}
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
    for train, test in folds.split(X_train,y_train):
        print("Train data",train,"Test data",test)
        
    grid_search1 = GridSearchCV(estimator=modelKNN,
                               param_grid=params,
                               cv=folds,
                               n_jobs=-1)
    
    grid_search1.fit(X_train, y_train)
    y_pred=grid_search1.predict(X_test)
    print(grid_search1.best_score_)
    print(grid_search1.best_params_)
    cv_score = grid_search1.best_score_
    test_score = grid_search1.score(X_test, y_test)
    cv_results = cross_val_score(grid_search1, X_train, y_train, cv=folds, scoring='accuracy')
    result.append(cv_results)
    accur['KNN'] = accuracy_score(y_test, grid_search1.predict(X_test))
    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')
    print(classification_report(y_test, y_pred))
    
    plot_confusion_matrix(grid_search1, X_test, y_test)
    f1_score(y_test,y_pred, average = 'micro')
    matrix = confusion_matrix( y_test,y_pred)
    # Create pandas dataframe
    df = pd.DataFrame(matrix, index=['Extremely Severe','Moderate','Severe'], columns=['Extremely Severe','Moderate','Severe'])
    # Create heatmap
    sns.heatmap(df,annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix KNN"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()  
    
    
# =============================================================================
# KNN()
# =============================================================================

def decisiontree():
    modelDT = DecisionTreeClassifier()
    params = {}
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
    for train, test in folds.split(X_train,y_train):
        print("Train data",train,"Test data",test)
        
    grid_search1 = GridSearchCV(estimator=modelDT,
                               param_grid=params,
                               cv=folds,
                               n_jobs=-1)
    
    grid_search1.fit(X_train, y_train)
    y_pred=grid_search1.predict(X_test)
    print(grid_search1.best_score_)
    print(grid_search1.best_params_)
    cv_score = grid_search1.best_score_
    test_score = grid_search1.score(X_test, y_test)
    cv_results = cross_val_score(grid_search1, X_train, y_train, cv=folds, scoring='accuracy')
    result.append(cv_results)
    accur['Decision Tree']= accuracy_score(y_test, grid_search1.predict(X_test))
    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')
    print(classification_report(y_test, y_pred))
    
    plot_confusion_matrix(grid_search1, X_test, y_test)
    f1_score(y_test,y_pred, average = 'micro')
    matrix = confusion_matrix( y_test,y_pred)
    # Create pandas dataframe
    df = pd.DataFrame(matrix, index=['Extremely Severe','Moderate','Severe'], columns=['Extremely Severe','Moderate','Severe'])
    # Create heatmap
    sns.heatmap(df,annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix Decision Tree"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()  
    

decisiontree()
def comparemodel():
    colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(16,5))
    plt.yticks(np.arange(0,100,10))
    plt.ylabel("Accuracy %")
    plt.xlabel("Algorithms")
    sns.barplot(x=list(accur.keys()), y=list(accur.values()), palette=colors)
    plt.show()

    # Compare Algorithms
    plt.boxplot(result, labels=accur.keys())
    plt.title('Algorithm Comparison')
    plt.show()
comparemodel()

concatt = pd.concat([dataframedemo, dataframe], axis=1, join='inner')
concatt.drop("Unnamed: 0",axis=1, inplace=True )

# Correlation and heatmap
plt.figure(figsize=(16,16),dpi=80)
corr = concatt[demographic + ["Scores"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, robust=True, center=0, square=True, linewidths=.5)
plt.title('Correlation', fontsize=15)
plt.show()







