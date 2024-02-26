#made correlation matrices for wood and steel to exxplore the
#assosciations between the different factors

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

#wood data
wood_data = pd.read_csv("wooden_main.csv")

dataframe = wood_data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

matrix = dataframe.corr()

sns.heatmap(matrix, cmap="Blues", annot=True)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="Blues", annot=True)
plt.savefig("wood_heatmap.png") 


#steel data
steel_data = pd.read_csv("steel_main.csv")
dataframe = steel_data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

matrix = dataframe.corr()

sns.heatmap(matrix, cmap="Greens", annot=True)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="Greens", annot=True)
plt.savefig("steel_heatmap.png") 

#big data file
data = pd.read_csv("data.csv")
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

matrix = dataframe.corr()

sns.heatmap(matrix, cmap="Reds", annot=True)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="Reds", annot=True)
plt.savefig("data_heatmap.png") 
