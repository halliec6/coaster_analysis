#the goal of this file is to work through the process of cleaning the data.csv file
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
import warnings
warnings.filterwarnings("ignore", message="The least populated class in y has only")

df = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None)

#first dropped unnecessary cols
#based on too many NaN values/not significant to our analysis
to_drop = [
    'Δ Elevation (ft)',
    'Airtime Points',
    'Crossings',
    'Bank Angle (°)',
    'Drop',
    'Designer',
    'G-Force',
    'Vertical Angle (°)',
    'Uphill Length (ft)',
    'Downhill Length (ft)',

]
df.drop(to_drop, inplace = True, axis = 1)
# print(df.head(10))


#next step is removing rows with a null value for height, speed or length
df.dropna(subset=["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)", "Design"], inplace=True)
#print(df.head(10))
print(df.shape)

#with drop we go down from 2k values to 400... need to consider if we want to include this or not
# print("Number of NaN values in Drop(ft) column:",nan_count)
# print(df.shape)

dataframe = df[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)", "Design"]]

# dataframe.plot(kind='box', subplots=True, layout=(3,2), figsize=(10, 10), sharex=False, sharey=False)
# plt.show()
array = dataframe.values
X = array[:,0:5]
y = array[:,5]


X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
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
plt.title('Algorithm Comparison')
plt.show()    