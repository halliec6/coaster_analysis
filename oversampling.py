#I am working to balance the data in this file
#first I am trying this using SMOTE, then I am looking at the confusion matricies of the most accurate algorithms

#TO DO: confusion matrix as a visual
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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

def oversample_data(X, y):
    oversample = SVMSMOTE()
    return oversample.fit_resample(X, y)

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
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Oversampling Confusion Matrix')
    plt.show()
    
    print("accuracy_score\n", accuracy_score(Y_validation, predictions))
    print("confusion matrix \n", conf_matrix)  
    print("classification_report\n", classification_report(Y_validation, predictions))

def main():
    # Read the data
    data = read_data("clean_total.csv")

    # Columns we're analyzing
    dataframe = select_columns(data, ["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"])

    array = dataframe.values
    X = array[:,0:4]
    y = array[:,4]
    print("Y before: ", Counter(y))

    # Oversampling
    X, y = oversample_data(X, y)
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
    plot_algorithm_comparison(results, names, 'Oversampling Algorithm Comparison')

    # Create the decision tree classifier
    model = DecisionTreeClassifier()

    # Evaluate model using cross-validation
    cv_scores = cross_val_score(model, X_train, Y_train, cv=10)  # You can adjust the number of folds as needed
    print("Cross-validation scores:", cv_scores)
    print("Mean cross-validation score:", mean(cv_scores))

    # Fit the model
    model.fit(X_train, Y_train)

    # Make predictions
    predictions = model.predict(X_validation)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(Y_validation, predictions)
    print("Confusion matrix:\n", conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Oversampling Confusion Matrix')
    plt.show()

    # Print performance metrics
    print('Mean ROC AUC:', mean(cv_scores))
    print("Accuracy score:", accuracy_score(Y_validation, predictions))
    print("Confusion matrix:\n", confusion_matrix(Y_validation, predictions))
    print("Classification report:\n", classification_report(Y_validation, predictions))

if __name__ == "__main__":
    main()
