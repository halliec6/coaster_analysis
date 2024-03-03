#PURPOSE: Undersampling the dataset, visualizes confusion matrix w a heat map
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from collections import Counter

def read_data(file_path):
    return pd.read_csv(file_path)

def select_columns(data, columns):
    return data[columns]

def split_data(X, y, test_size=0.2, random_state=1):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)

def undersample_data(X_train, Y_train, X_validation, Y_validation):
    undersample = RandomUnderSampler(sampling_strategy='auto')
    X_train_under, Y_train_under = undersample.fit_resample(X_train, Y_train)
    print("Y after undersampling: ", Counter(Y_train_under))

    undersample_val = RandomUnderSampler(sampling_strategy='auto')
    X_validation_under, Y_validation_under = undersample_val.fit_resample(X_validation, Y_validation)
    print("Class distribution of Y_validation_under:", Counter(Y_validation_under))

    return X_train_under, Y_train_under, X_validation_under, Y_validation_under

def spot_check_models(models, X_train_under, Y_train_under, cv=10):
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train_under, Y_train_under, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    return results, names

def plot_algorithm_comparison(results, names):
    plt.boxplot(results, labels=names)
    plt.title('Undersampling: Algorithm Comparison')
    plt.show()

def train_model_and_evaluate(model, X_train_under, Y_train_under, X_validation_under, Y_validation_under):
    model.fit(X_train_under, Y_train_under)
    predictions = model.predict(X_validation_under)
    
    conf_matrix = confusion_matrix(Y_validation_under, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Predicted Yes', 'Predicted No'], yticklabels=['Actual Yes', 'Actual No'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Undersampling Confusion Matrix')
    plt.show()
    
    print("accuracy_score\n", accuracy_score(Y_validation_under, predictions))
    print("confusion matrix \n", conf_matrix)  
    print("classification_report\n", classification_report(Y_validation_under, predictions))

def main():
    # Read the data
    data = read_data("clean_total.csv")

    # Select columns for analysis
    dataframe = select_columns(data, ["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"])

    # Extract features and target variable
    X = dataframe.drop("Rank", axis=1)
    y = dataframe["Rank"]

    print("Y before: ", Counter(y))

    # Splitting data into train and validation sets
    X_train, X_validation, Y_train, Y_validation = split_data(X, y)

    # Undersampling
    X_train_under, Y_train_under, X_validation_under, Y_validation_under = undersample_data(X_train, Y_train, X_validation, Y_validation)

    # Model training and evaluation
    models = [
        ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', SVC(gamma='auto'))
    ]
    results, names = spot_check_models(models, X_train_under, Y_train_under)
    plot_algorithm_comparison(results, names)
    chosen_model = LogisticRegression(solver='liblinear', multi_class='ovr')  # You can change this to any model you want to evaluate further
    train_model_and_evaluate(chosen_model, X_train_under, Y_train_under, X_validation_under, Y_validation_under)

if __name__ == "__main__":
    main()

