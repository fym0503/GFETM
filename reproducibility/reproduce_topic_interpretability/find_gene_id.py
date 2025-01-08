import numpy as np
import pandas as pd

mapping_dict={}
with open("data/name_to_id.csv") as f:
    for line in f:
        data = line.rstrip().split("\t")
        mapping_dict[data[0]] = data[1]
data = pd.read_csv("data/peak_by_topic_panel.csv")
for i in range(data.shape[0]):
    data['Unnamed: 0'][i] = mapping_dict[data['Unnamed: 0'][i]]
data = data.set_index("Unnamed: 0")
data.to_csv("data/peak_by_topic_panel_transform.csv")
