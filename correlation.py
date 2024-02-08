import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

#wood data
wood_data = pd.read_csv("wooden_main.csv")

wood_data['Length (ft)'] = wood_data['Length (ft)'].str.replace(',', '')  
wood_data['Length (ft)'] = wood_data['Length (ft)'].str.split('.').str[0]  
wood_data['Length (ft)'] = wood_data['Length (ft)'].fillna(0)
wood_data['Length (ft)'] = wood_data['Length (ft)'].astype(int)

dataframe = wood_data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

matrix = dataframe.corr()

sns.heatmap(matrix, cmap="Blues", annot=True)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="Blues", annot=True)
plt.savefig("wood_heatmap.png") 


#steel data
steel_data = pd.read_csv("steel_main.csv")

steel_data['Length (ft)'] = steel_data['Length (ft)'].str.replace(',', '')  
steel_data['Length (ft)'] = steel_data['Length (ft)'].str.split('.').str[0]  
steel_data['Length (ft)'] = steel_data['Length (ft)'].fillna(0)
steel_data['Length (ft)'] = steel_data['Length (ft)'].astype(int)

dataframe = steel_data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

matrix = dataframe.corr()

sns.heatmap(matrix, cmap="Greens", annot=True)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="Greens", annot=True)
plt.savefig("steel_heatmap.png") 


#big data file
data = pd.read_csv("data.csv")

data['Length (ft)'] = data['Length (ft)'].str.replace(',', '')  
data['Length (ft)'] = data['Length (ft)'].str.split('.').str[0]  
data['Length (ft)'] = data['Length (ft)'].fillna(0)
data['Length (ft)'] = data['Length (ft)'].astype(int)

dataframe = data[["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Drop (ft)"]]

matrix = dataframe.corr()

sns.heatmap(matrix, cmap="Reds", annot=True)
plt.figure(figsize=(10, 8))
sns.heatmap(matrix, cmap="Reds", annot=True)
plt.savefig("data_heatmap.png") 
