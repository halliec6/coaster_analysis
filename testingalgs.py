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
import pandas as pd
import seaborn as sns

import pandas as pd
import seaborn as sns

# Read the data
data = pd.read_csv("data.csv")

#cols we're analyzing
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

# Drop rows with missing values
dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
dataframe.dropna(inplace=True)

# Create box and whisker plots for each column separately
dataframe.plot(kind='box', subplots=True, layout=(3,2), figsize=(10, 10), sharex=False, sharey=False)
plt.show()
#dataframe.hist()
# plt.show()
#scatter_matrix(dataframe)
# plt.show()
# Split-out validation dataset
# Split-out validation dataset
array = dataframe.values
X = array[:,0:4]
y = array[:,4]
print("\nFirst 20 items of X")
print(X[:20])
print("\nFirst 20 items of y")
print(y[:20])
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    