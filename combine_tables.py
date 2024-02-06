import pandas as pd 

steel = pd.read_csv("steel_ranks.csv")
wooden = pd.read_csv("wooden_ranks.csv")
total = pd.read_csv("data.csv")

#joining our master dataset (data.csv) with our ranking data
steel_main = pd.merge(steel, total, on=["Name", "Park"])
steel_main.to_csv("steel_main.csv", index = False)

wooden_main = pd.merge(wooden, total, on=["Name", "Park"])
wooden_main.to_csv("wooden_main.csv", index = False)