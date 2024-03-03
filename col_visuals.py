#PURPOSE: looking at the diferent factors we are analyzing in our project 
    # creates boxplots, histograms, and scatter plots
#REFERENCE: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("clean_total.csv")

#cols we're analyzing
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

# box and whisker plots
dataframe.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False)
plt.tight_layout()
plt.savefig("clean_boxplots.png")

# histograms
dataframe.hist()
plt.tight_layout()
plt.savefig("clean_histograms.png")

# scatter plot matrix
scatter_matrix(dataframe)
plt.tight_layout()
plt.savefig("clean_scatterplots.png")