#%%
#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
#%%
#loading the dataset
os.chdir("E:\Data science\Projects")
df=pd.read_csv('creditcard.csv')
#%%
#Explore the dataset
print(df.columns)
print(df.shape)
print(df.describe())
df.isnull().sum()

#%%
#take fraction of dataset
df=df.sample(frac=0.1, random_state=1)
print(df.shape)

#%%
#plot Histogram of each parameter`
df.hist(figsize=(20,20))
plt.show()

#%%
#Determine the number of fraud cases in dataset
fraud=df[df['Class']==1]
valid=df[df['Class']==0]

outlier_fraction=(float(len(fraud))/len(valid))
print(outlier_fraction)
print('Fraud cases : {}'.format(len(fraud)))
print('Valid cases : {}'.format(len(valid)))

#%%
print("Amount Details of Fradulent transaction") 
fraud.Amount.describe() 
#%%
print("Amount details of valid transaction") 
valid.Amount.describe() 


#%%
#Correlation matrix
corr_mat=df.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(corr_mat, vmax=.8, square=True)
plt.show()

#%%
#Get all the columns from the dataframe
columns=df.columns.tolist()
#Filter the columns to remove. data we do not want
columns=[c for c in columns if c not in ["Class"]]
#store the variable we will predicting on
target="Class"
x=df[columns]
y=df[target]
#shape of x and y
print(x.shape)
print(y.shape)

#%%
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state=1
#define the outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(x),
                        contamination=outlier_fraction,
                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                            contamination=outlier_fraction)
}

#%%
#fit the model
n_outliers=len(fraud)
for i, (clf_name, clf) in enumerate(classifiers.items()):
    #fit the model and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred=clf.fit_predict(x)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores=clf.decision_function(x)
        y_pred=clf.predict(x)
    #reshape the predicton value 0 for valid, 1 for fraud
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors=(y_pred != y).sum()
    #return classification matrics
    print('{} : {}'.format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))

#%%
# printing the confusion matrix 
from sklearn.metrics import confusion_matrix
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS, 
			yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 
# %%
