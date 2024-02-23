#I am working to balance the data in this file
#this once again compares the algorithms but when SMOTE is used, we decided to then use carts based on results

from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
# Read the data
data = pd.read_csv("clean_total.csv")

#cols we're analyzing
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"]]
# yes_count = dataframe['Rank'].value_counts().get('Yes', 0)
# no_count = dataframe['Rank'].value_counts().get('No', 0)

# print("yes: ", yes_count, "no: ", no_count)

array = dataframe.values
X = array[:,0:4]
y = array[:,4]
print("Y before: ", Counter(y))

#new info
oversample = SVMSMOTE()
X, y = oversample.fit_resample(X,y)
print("Y after: ", Counter(y))

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
# Spot Check Algorithms
# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(gamma='auto')))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold,
#      scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#     scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
#     print('\n%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
#     print('Mean ROC AUC: %.3f' % mean(scores))

# # Compare Algorithms
# plt.boxplot(results, labels=names)
# plt.title('Algorithm Comparison')
# plt.show()