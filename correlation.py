#PURPOSE: Creates a correlation matrix to analyze the realtionships between the factors we are studying
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

#clean data file
data = pd.read_csv("clean_total.csv")
dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]
matrix = dataframe.corr()
sns.heatmap(matrix, cmap="Reds", annot=True)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="Reds", annot=True)
plt.title('Correlation Matrix for Total Dataset')

plt.savefig("clean_total_heatmap.png") 