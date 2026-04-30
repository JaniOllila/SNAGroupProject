import numpy as np
import pandas as pd
import os 
import networkx as nx

THRESHOLD = 25

G = nx.read_gexf("graph.gexf")

total_timesteps = 0
total_gust_over_threshold = 0
directory = "Wind datasets"

# Loop for counting total of times that max gust speed goes over threshold in any datasets and also counting the total of timesteps
for file in os.listdir(directory):
    file_path = f"{directory}/{file}"
    df = pd.read_csv(file_path, usecols=["Time [Local time]", "Maximum gust speed [m/s]"])
    gust_speed = list(df["Maximum gust speed [m/s]"])
    time_steps = list(df["Time [Local time]"])
    total_timesteps += len(time_steps)
    total_gust_over_threshold += len([x for x in gust_speed if x > THRESHOLD])

# Probability for turbine failure based on too big of a gust
overall_failure_probability = total_gust_over_threshold / total_timesteps

risk_scores = pd.read_csv("combined.csv", usecols=["location", "risk_score_model"])
#Adding failure probability for each turbine
risk_scores.insert(2, "failure_probability", risk_scores["risk_score_model"] 
                   * overall_failure_probability)


edgelist = nx.to_pandas_edgelist(G)
print(edgelist)
#print(risk_scores)
propagation_matrix = np.zeros((11,11))
for i in range(11):
    for j in range(11):
        if j==i:
            continue
        
        

#print(propagation_matrix)
