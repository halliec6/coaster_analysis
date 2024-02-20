#the goal of this file is to display the decision tree 
#using this example https://mljar.com/blog/visualize-decision-tree/https://mljar.com/blog/visualize-decision-tree/
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
import pandas as pd
import seaborn as sns
import imblearn
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
import graphviz
import numpy as np
from dtreeviz.trees import dtreeviz 


data = pd.read_csv("clean_total.csv")

#cols we're analyzing
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"]]

array = dataframe.values
X = array[:,0:4]
y = array[:,4]
print("Y before: ", Counter(y))

#new info
oversample = SVMSMOTE()
X, y = oversample.fit_resample(X,y)
print("Y after: ", Counter(y))

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)


clf = DecisionTreeClassifier()
model = clf.fit(X_train, Y_train)

#this is bomb.com, make sure to edit the max depth for better visual
# fig = plt.figure(figsize=(100,100))
# _ = tree.plot_tree(clf,
#                 #    max_depth = 2,
#                    feature_names = dataframe.columns[:-1],
#                    class_names = np.unique(y).astype(str),
#                    filled = True)
# fig.savefig("decision_tree.png")

viz = dtreeviz(clf, X, y,
                target_name="target",
                feature_names=dataframe.columns[:-1],
                class_names=list(np.unique(y).astype(str)))

viz.save("decision_tree.svg")
