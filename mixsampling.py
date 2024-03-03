#PURPOSE: Applying both oversampling and undersampling on the dataset, visualizes confusion matrix w a heat map

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from collections import Counter
from numpy import mean

def read_data(file_path):
    return pd.read_csv(file_path)

def select_columns(data, columns):
    return data[columns]

def apply_mix_sampling(X, y):
    over_sampler = SVMSMOTE()
    under_sampler = RandomUnderSampler()
    pipeline = Pipeline(steps=[('o', over_sampler), ('u', under_sampler)])
    return pipeline.fit_resample(X, y)

def spot_check_models(models, X_train, Y_train, cv=10):
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    return results, names

def plot_algorithm_comparison(results, names, title):
    plt.boxplot(results, labels=names)
    plt.title(title)
    plt.show()


def train_model_and_evaluate(model, X_train, Y_train, X_validation, Y_validation):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    
    conf_matrix = confusion_matrix(Y_validation, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Mix Sampling Confusion Matrix')
    plt.show()
    
    print("accuracy_score\n", accuracy_score(Y_validation, predictions))
    print("confusion matrix \n", conf_matrix)  
    print("classification_report\n", classification_report(Y_validation, predictions))

def main():
    # Read the data
    data = read_data("clean_total.csv")

    # Select columns for analysis
    dataframe = select_columns(data, ["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"])

    array = dataframe.values
    X = array[:,0:4]
    y = array[:,4]
    print("Y before: ", Counter(y))

    # Apply mix sampling
    X, y = apply_mix_sampling(X, y)
    print("Y after: ", Counter(y))

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

    # Spot Check Algorithms
    models = [
        ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', SVC(gamma='auto'))
    ]
    results, names = spot_check_models(models, X_train, Y_train)
    plot_algorithm_comparison(results, names, 'Mixsampling: Algorithm Comparison')

    # Using decision tree for heatmap
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)

    # Evaluate predictions
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    conf_matrix = confusion_matrix(Y_validation, predictions)

    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Mix Sampling Confusion Matrix')
    plt.show()

    # Summarize performance
    print('Mean ROC AUC: %.3f' % mean(scores))
    print("accuracy_score\n", accuracy_score(Y_validation, predictions))
    print("confusion matrix \n", conf_matrix)
    print("classification_report\n", classification_report(Y_validation, predictions))

if __name__ == "__main__":
    main()
