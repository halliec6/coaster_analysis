#the goal of this file is to display the decision tree 
#using this example https://mljar.com/blog/visualize-decision-tree/https://mljar.com/blog/visualize-decision-tree/

#explore additional visuals for trees

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
import numpy as np


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


clf = DecisionTreeClassifier(max_depth=4)
model = clf.fit(X_train, Y_train)

#this is bomb.com, make sure to edit the max depth for better visual
fig = plt.figure(figsize=(100,100))
_ = tree.plot_tree(clf,
                #    max_depth = 2,
                   feature_names = dataframe.columns[:-1],
                   class_names = np.unique(y).astype(str),
                   filled = True)
fig.savefig("decision_tree.png")




