#!/usr/bin/env python
# coding: utf-8

# There are 150 observations of the iris flower in this dataset. All of the observations belong to three species of flowers. There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed.
# 
# This dataset is available for download at https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv. 

# Imported the needed libaries to do the analysis.

# In[20]:


import sys
import scipy
import numpy
import matplotlib
import pandas 
import sklearn


# Verified version of each library. 

# In[21]:


print("sys version:", sys.version_info)
print("scipy version:", scipy.__version__)
print("numpy version:", numpy.__version__)
print("matplotlib:", matplotlib.__version__)
print("pandas version:", pandas.__version__)
print("sklearn version:", sklearn.__version__)


# In[24]:


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Read in the dataset and set the name of the columns. 

# In[25]:


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = read_csv(url, names=names)


# This gives us an overview of the shape of the data set. This dataset has 5 columns and 150 rows. 

# In[28]:


iris.shape


# The head method allows us to see a sample of the first 20 rows. 

# In[29]:


iris.head(20)


# The describe methods give us basis statistics on the dataset. 

# In[30]:


iris.describe()


# The group by method filters the dataset by class. The size method request the size of the class size. In this situation the Iris-setosa, Iris-versicolor and Iris-virginica class size are all 50. 

# In[31]:


iris.groupby('class').size()


# This creates a box and whisker plot. This shows a graphical distribution of the dataset. 

# In[36]:


iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# This creats a histogram for each variable 

# In[37]:


iris.hist()
plt.show()


# Scatter plot matrices are used in exploratory data analysis to identify relationships between pairs of variables. A grouping of data points in a diagonal manner suggests a relationship: from the bottom-left to the top-right indicates a positive correlation, while from the top-left to the bottom-right indicates a negative correlation.

# In[39]:


scatter_matrix(iris)
plt.show()


# Create a validation dataset. 
# 
# We're going to build a model that predict Iris flower type based on the previous data.

# In[40]:


array = iris.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


# Build models
# 
# We're going identify which algorithm would be a good fit for this dataset. 

# In[43]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVB', SVC(gamma='auto')))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# The Support Vector Machines (SVM) has the largest estimated accuracy score at 98.3%. 

# In[44]:


plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()


# We're using the Support Vector Machines to train the model.   

# In[47]:


model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# In[49]:


print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In summary, we used the Iris dataset to train the Support Vector Machines algorithm. In essence we would be able to predict the type of Iris flower if we had the flower's sepal-length, sepal-width, petal-length and petal-width.
