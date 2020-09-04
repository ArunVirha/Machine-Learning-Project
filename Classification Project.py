#!/usr/bin/env python
# coding: utf-8

# # Supervised Machine Learning  Project on classification

# In this notebook we are to implement the classification algorithms in the available dataset. We import a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[541]:



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# **About dataset**
# This dataset is about past loans. The Loan_train.csv data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# Field	Description
# 
# **Loan_status**-	Whether a loan is paid off on in collection
# 
# **Principal**	-Basic principal loan amount at the
# 
# **Terms**	-Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule
# 
# **Effective_date**-	When the loan got originated and took effects
# 
# **Due_date**	-Since it’s one-time payoff schedule, each loan has one single due date
# 
# **Age**	-Age of applicant
# 
# **Education**	-Education of applicant
# 
# **Gender**	-The gender of applicant
# 
# Lets import the dataset

# In[542]:


df = pd.read_csv("C:/Users/hp/Desktop/loan_train.csv")


# 
# 

# In[543]:


df.head(10)


# In[544]:


df.shape


# #### Convert to date time object
# 

# In[545]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# #### Data visualization and pre-processing

# Let’s see how many of each class is in our data set

# In[546]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection.
# Lets plot some columns to underestand data better

# In[547]:


import seaborn as sns


# In[548]:


binsP = np.linspace(df.Principal.min(), df.Principal.max(), 10)   #creating a sequence if equally spaced numbers within defined range 
binsP


# In[549]:


g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=binsP, ec="k")
g.axes[-1].legend()
plt.show()


# In[550]:


binsA = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=binsA, ec="k")

g.axes[-1].legend()
plt.show()


# #### Pre-processing: Feature selection/extraction

# Lets look at the day of the week people get the loan

# In[551]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
binsW = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=binsW, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less than day 4.

# In[552]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# #### Convert Categorical features to numerical values
# 
# Firstly **Gender**

# In[553]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 
# converting male to 0 and female to 1:

# In[554]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# For **education**

# In[555]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### converting categorical varables to binary variables and append them to  Data Frame

# In[556]:


X = df[['Principal','terms','age','Gender','weekend']]
X = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
X.drop(['Master or Above'], axis = 1,inplace=True)
X.head()


# In[557]:


X[0:5]


# In[558]:


y = df['loan_status'].values
y[0:5]


# #### Normalizing Data

# Data Standardization give data zero mean and unit variance.

# In[560]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# ### Classification

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model You should use the following algorithm:
# 
# 1.K Nearest Neighbor(KNN)
# 
# 2.Decision Tree
# 
# 3.Support Vector Machine
# 
# 4.Logistic Regression

# ### 1. K Nearest Neighbor(KNN)

# We should find the best k to build the model with the best accuracy.

# In[561]:


X[0:5]


# #### Classifier implementing the k-nearest neighbors vote

# In[562]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
x_train, x_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=4)
print('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)


# #### Training Model
#  with k=4

# In[563]:


model= KNeighborsClassifier(n_neighbors=4)
model.fit(x_train, y_train)


# #### Predictions

# In[564]:


y_hat=model.predict(x_test)
y_hat[0:5]


# #### Accuracy evaluation

# 1. In multilabel classification, **accuracy classification score** is a function that computes subset accuracy. This function is equal to the jaccard_similarity_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.

# In[565]:


from sklearn import metrics
print('Train Accuracy:', metrics.accuracy_score(y_train, model.predict(x_train)))
print('Test Accuracy:', metrics.accuracy_score(y_test, y_hat))


# 2. **Jaccard Index**

# In[566]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, y_hat)


# 3. **F-1 Score**
# 
# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.

# In[567]:


from sklearn.metrics import f1_score
f1_score(y_test, y_hat, average='weighted')


# Here, we see that out of sample accuracy is around 68% by three metrics.

# ### 2. Decision Tree

# Adding one more variable,'education' in dataset.
# But, decision tree model does not handle categorical data, hence we will convert these variables into nu,erical values.

# In[569]:


# Covering 'gender' and 'education'
import random
df['education'] = random.choices(["High School or Below", "Bechalor", "college", "Master or Above"], k = 346)
df["education"] = df["education"].map({"High School or Below":2, "Bechalor":4, "college":6, "Master or Above":8})

df=df[['loan_status','Principal','terms','age','education','Gender']]

df.head()


# In[570]:


XD = df[['Principal', 'terms', 'age', 'education', 'Gender']].values
XD[0:5]


# In[571]:


yd= df['loan_status']
yd[0:5]


# #### Training Model

# In[572]:


from sklearn.tree import DecisionTreeClassifier
x_train, x_test,y_train,  y_test = train_test_split(XD, yd, test_size=0.2, random_state=3)
tree=DecisionTreeClassifier(criterion='entropy', max_depth=4)
tree.fit(x_train, y_train)


# #### Predictions

# In[573]:


T_predict= tree.predict(x_test)
print(T_predict[0:10])
print(y_test[0:10])


# Here,we notice that out of 6 randam observations, 4 values are predicted incorrectly i.e., **60%** accuracy.
# Further, we will check the accuracy for complete model.

# In[574]:


from sklearn.metrics import jaccard_similarity_score, f1_score


# **1. Acuracy Classification Score**

# In[575]:


print('Train Accuracy:', metrics.accuracy_score( y_train, tree.predict(x_train)))
print('Test Accuracy:', metrics.accuracy_score( y_test, tree.predict(x_test)))


# **2. Jaccard Index**

# In[576]:


print('Train Accuracy:', jaccard_similarity_score( y_train, tree.predict(x_train)))
print('Test Accuracy:', jaccard_similarity_score( y_test, tree.predict(x_test)))


# **3. f1_score**

# In[577]:


print('Train Accuracy:', f1_score( y_train, tree.predict(x_train), average='weighted'))
print('Test Accuracy:', f1_score( y_test, tree.predict(x_test), average='weighted'))


# Here, we notice that f1_score has considerable difference as compared with score given other two metrics.

# ### 3. Support Vector Machine

# SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.

# In[578]:


import pylab as pl
import scipy.optimize as opt


# In[579]:


df.head()


# All variables are numerical except loan_status. We will convert it into numerical form.

# In[580]:


loan_status= random.choices(['PAIDOFF','COLLECTION'], k=346)
df['loan_status']= df['loan_status'].map({'PAIDOFF':5, 'COLLECTION': 10})    # assigning PAIDOFF = 5, COLLECTION= 10
df.head()


# #### Visualizing AGE vs Gender

# In[586]:


ax = df[df['loan_status'] == 5][0:350].plot(kind='scatter', x='age', y='Gender', color='Red', label='PAIDOFF');
df[df['loan_status'] == 10][0:350].plot(kind='scatter', x='age', y='Gender', color='Green', label='COLLECTION', ax=ax);
plt.show()


# #### Data pre-processing and selection
# 

# In[587]:


# Data types
df.dtypes


# In[588]:


X_svm = df.drop(['loan_status'], axis=1)


# In[589]:


X_svm[0:5]


# We want the model to predict the value of loan_status (that is, PAIDOFF (=5) or COLLECTION (=10)). As this field can have one of only two possible values, we need to change its measurement level to reflect this.

# In[590]:


df['loan_status'] = df['loan_status'].astype('int')
y_svm = df['loan_status']
y_svm[0:5]


# #### Train/Test dataset

# In[591]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size= 0.2, random_state= 4)
print('Train data:', x_train.shape, y_train.shape)
print('Test data:', x_test.shape, y_test.shape)


# #### Trainnig the model

# The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:
# 
# 1.Linear
# 
# 2.Polynomial
# 
# 3.Radial basis function (RBF)
# 
# 4.Sigmoid
# 

# Let's just use the default, RBF (Radial Basis Function) for this model.

# In[592]:


from sklearn import svm
SVM= svm.SVC(kernel='rbf')


# In[593]:


SVM.fit(x_train, y_train)


# #### Predictions

# In[594]:


pred= SVM.predict(x_test)
pred[0:5]


# Recall that we have labelled PAIDOFF as '5' and COLLECTION as '10'.
# 
# Further, we will **EVALUATE** the efficiency of the model by following metrics:
# 
# 1. confusion_matrix
# 
# 2. Jaccard Index
# 
# 3. f1_score

# #### 1. confusion_matrix

# In[595]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[596]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
    
    print('Confudion Matrix without normalization')    
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks= np.arange(len(classes))
    plt.xticks(tick_marks,  classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt= '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j], fmt), horizontalalignment='center', color='white' if cm[i,j] > thresh else 'black')
        
        plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
        
    


# In[597]:


# Compute confusion matrix
confus_matrix= confusion_matrix(y_test, pred, labels=[5,10])
np.set_printoptions(precision=4)

print(classification_report(y_test, pred))
plt.figure()
plot_confusion_matrix(confus_matrix, classes=['PAIDOFF(5)', 'COLLECTION(10)'], normalize= False, title= 'Confusion_Matrix')


# #### 2 Jaccard Index.

# In[598]:


print('Train Accuracy:', jaccard_similarity_score(y_train, SVM.predict(x_train)))
print('Test Accuracy:', jaccard_similarity_score(y_test, SVM.predict(x_test)))


# #### 3. f1_score

# In[599]:


print('Train Accuracy:', f1_score(y_train, SVM.predict(x_train), average='weighted'))
print('Test Accuracy:', f1_score(y_test, SVM.predict(x_test), average='weighted'))


# Here, we see that test accuracy is higher than train accuracy.

# ### 4. Logistic Regression

# In[600]:


import scipy.optimize as opt


# In[601]:


df.head()


# In[602]:


X_logr= df.drop('loan_status', axis=1)
X_logr.head()


# In[603]:


y_logr= df['loan_status']
y_logr[0:5]


# Normalizing the datast

# In[604]:


X_logr= preprocessing.StandardScaler().fit(X_logr).transform(X_logr)
X_logr[0:5]


# #### Train and Test split

# In[605]:


x_train, x_test, y_train, y_test = train_test_split(X_logr, y_logr, test_size= 0.2, random_state=4)
print('train set:', x_train.shape, y_train.shape)
print('test set:', x_test.shape, y_test.shape)


# #### Modeling Logistic Regression

# Lets build our model using **LogisticRegression** from Scikit-learn package. This function implements logistic regression and can use different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. Every optimizer has its own importance.
# 
# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem in machine learning models. **C** parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify stronger regularization. Now lets fit our model with train set:

# In[606]:


from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(C=0.01, solver='liblinear').fit(x_train, y_train)


# #### Predictions

# In[607]:


y_train_hat= LR.predict(x_train)
y_train_hat[0:5]


# In[ ]:





# Here, we can also use **predict_proba** which estimates the probabilities of class 1, i.e., P(Y=1|X) as well as  probabilities of class 2 i.e., 1-P(Y=1|X)= P(Y=0|X)

# In[608]:


y_train_hat_prob= LR.predict_proba(x_train) 
y_test_hat_prob= LR.predict_proba(x_test)


# In[609]:


y_test_hat[0:10]


# #### Evaluation
#  1.  Jaccard Index
#  2.  f1_score
#  3.  Log Loss

# #### 1.Jaccard Index

# In[610]:


print('Train Accuracy:', jaccard_similarity_score(y_train, y_train_hat))
print('Test Accuracy:', jaccard_similarity_score(y_test, LR.predict(x_test)))


# #### 2.f1_score

# In[611]:


print('Train Accuracy:', f1_score(y_train, y_train_hat, average='weighted'))
print('Test Accuracy:', f1_score(y_test, LR.predict(x_test), average='weighted'))


# #### 3. Log Loss
# In logistic regression, the output can be the probability. This probability is a value between 0 and 1. Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.The goal of our machine learning models is to minimize this value.

# In[612]:


from sklearn.metrics import log_loss
print('Train Accuracy:', log_loss(y_train, y_train_hat_prob))
print('Test Accuracy:', log_loss(y_test, y_test_hat_prob))


# ### Thus, we have final report based on Classification Models

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | 0.68    | 0.68     | NA      |
# | Decision Tree      | 0.67    | 0.53     | NA      |
# | SVM                | 0.78    | 0.69     | NA      |
# | LogisticRegression | 0.78    | 0.69     | 0.60    |

# ### Conclusion:

# From the above report, we interpret that SVM and Logistic Regression models fit with higher accuracy and hence we should select any one model out of these two. 
# Further, we also see that the probability of log_loss is higher than 0.5. Hence we are most likely to choose the **SVM model** for best efficiency.

# 
# 
# 
# 
# Author: **Arun Virha**

# In[ ]:




