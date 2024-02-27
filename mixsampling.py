#meet in the middle of the two datasets, oversample yes and undersample no to meet in the middle
#visualize confusion matrix w a heat map

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SVMSMOTE
from numpy import mean
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
# Read the data
data = pd.read_csv("clean_total.csv")

#cols we're analyzing
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"]]


array = dataframe.values
X = array[:,0:4]
y = array[:,4]
print("Y before: ", Counter(y))

#new info
# Define the pipeline
over_sampler = SVMSMOTE()
under_sampler = RandomUnderSampler()
pipeline = Pipeline(steps=[('o', over_sampler), ('u', under_sampler)])

# Apply mix sampling
X, y = pipeline.fit_resample(X, y)
print("Y after: ", Counter(y))

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

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Mixsampling: Algorithm Comparison')
plt.show()


# using decision tree for heatmap
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
# Calculate confusion matrix
conf_matrix = confusion_matrix(Y_validation, predictions)
print("confusion matrix \n", conf_matrix)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Mix Sampling Confusion Matrix')
plt.show()
# summarize performance
print('Mean ROC AUC: %.3f' % mean(scores))
print("accuracy_score\n", accuracy_score(Y_validation, predictions))
print("confustion matrix \n",confusion_matrix(Y_validation, predictions)) #this is what we want to make pretty
print("classification_report\n",classification_report(Y_validation, predictions))