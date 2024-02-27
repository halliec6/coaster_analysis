#follow oversampling, instead of using smote, undersample the no's so they balance the yes values
#make a heatmap of the confusion matrix

#maybe run comparisons from balancing.py which is what makes the box plots for all of the different algorithms]
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
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

# Select columns for analysis
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"]]

# Extract features and target variable
X = dataframe.drop("Rank", axis=1)
y = dataframe["Rank"]

print("Y before: ", Counter(y))

# Splitting data into train and validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Undersampling
# Undersampling the training set
undersample = RandomUnderSampler(sampling_strategy='auto')
X_train_under, Y_train_under = undersample.fit_resample(X_train, Y_train)
print("Y after undersampling: ", Counter(Y_train_under))

# Undersampling the validation set
print("Y_validation before: ", Counter(Y_validation))
undersample_val = RandomUnderSampler(sampling_strategy='auto')
X_validation_under, Y_validation_under = undersample_val.fit_resample(X_validation, Y_validation)
print("Class distribution of Y_validation_under:", Counter(Y_validation_under))

# Model training
# model = DecisionTreeClassifier()
# model.fit(X_train_under, Y_train_under)

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
    cv_results = cross_val_score(model, X_train_under, Y_train_under, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Undersampling: Algorithm Comparison')
plt.show()

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train_under, Y_train_under)
# Predictions on undersampled validation set
predictions = model.predict(X_validation_under)

# Confusion matrix
conf_matrix = confusion_matrix(Y_validation_under, predictions)

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Undersampling Confusion Matrix')
plt.show()

# Evaluation
print("accuracy_score\n", accuracy_score(Y_validation_under, predictions))
print("confusion matrix \n", conf_matrix)  # this is what we want to make pretty
print("classification_report\n", classification_report(Y_validation_under, predictions))

