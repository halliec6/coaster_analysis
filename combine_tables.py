#PURPOSE: File combines rank and data.csv set to make one total_ranks dataset
import pandas as pd 

#combining all of the ranked data with the big data.csv, col stores yes or no for if a coaster is ranked
total = pd.read_csv("data.csv")
print(total.shape)
allranks = pd.read_csv("allranks.csv")
allranks = allranks.drop_duplicates(subset = ["Name", "Park"])
total_ranks = pd.merge( total, allranks, on=["Name", "Park"],how = "left")
total_ranks['Rank'] = total_ranks['Rank'].fillna('No').apply(lambda x: 'Yes' if x != 'No' else 'No')

#now going to clean the data, get rid of silly cols, then NaN values
pd.set_option('display.max_columns', None)
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
    'Location', 
    'Supplier',
    'Year',
    'Awarded',
    'Duration', 
    'Opened'
]

#total_ranks.to_csv("total_ranks.csv", index = False)
total_ranks.drop(to_drop, inplace = True, axis = 1)
total_ranks.dropna(subset=["Length (ft)", "Speed (mph)", "Height (ft)", "Inversions", "Rank"], inplace=True)
total_ranks.to_csv("clean_total.csv", index = False)
print(total_ranks.head)