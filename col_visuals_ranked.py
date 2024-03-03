#PURPOSE: making histograms for features of ranked data and histograms for unranked data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("clean_total.csv")

#analyzing yes ranked 
yes_rank_data = data[data["Rank"] == "Yes"]

dataframe = yes_rank_data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

dataframe.hist()
plt.tight_layout()
plt.title('Ranked Roller Coaster Attributes')

plt.savefig("yes_ranked_histograms.png")

#analyzing not ranked
no_rank_data = data[data["Rank"] == "No"]

dataframe = no_rank_data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

dataframe.hist()
plt.tight_layout()
plt.savefig("not_ranked_histograms.png")