import pandas as pd

path =r"C:\Users\HARIKRISHNAN\.cache\kagglehub\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3"
train = pd.read_csv(f"{path}/exoTrain.csv")
print(f" {train.info}")
