"""
Module for task 12, 13, 14, 15. 
Includes most functions and impelentation of probabilities and simulation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import scipy
import supplementary as supp

#-----------------Task 12-----------------------------------
plt.close("all")

CUT_IN_THRESHOLD = 3
CUT_OUT_THRESHOLD = 25
DIRECTORY = "Wind datasets"

G = nx.read_gexf("graph.gexf")

gust_speeds = []

# Loop for counting total of times that max gust speed goes over threshold
# in any datasets and also counting the total of timesteps
for file in os.listdir(DIRECTORY):
    file_path = f"{DIRECTORY}/{file}"
    df = pd.read_csv(file_path, usecols=["Time [Local time]", "Maximum gust speed [m/s]"])
    for speed in df["Maximum gust speed [m/s]"]:
        gust_speeds.append(speed)

# Probability for turbine failure based on too big of a gust
k, loc, scale = scipy.stats.weibull_min.fit(gust_speeds, floc=0)
overall_failure_probability = scipy.stats.weibull_min.cdf(CUT_IN_THRESHOLD, k, loc=0,
                                scale=scale) + (1 - scipy.stats.weibull_min.cdf(
                                CUT_OUT_THRESHOLD, k, loc=0, scale=scale))
#print(overall_failure_probability)

risk_scores = pd.read_csv("combined.csv", usecols=["location", "risk_score_model"])
#Adding failure probability for each turbine
risk_scores.insert(2, "failure_probability", risk_scores["risk_score_model"]
                   * overall_failure_probability)

list_of_edges = nx.to_pandas_edgelist(G)

#Creating propagation matrix where row is source of failure
#and column is the target. Values in cells are probabilities for failure propagation.
propagation_matrix = np.zeros((12,12))

edge_lookup = {}
for _, row in list_of_edges.iterrows():
    edge_lookup[(row["source"], row["target"])] = row["weight"]
    edge_lookup[(row["target"], row["source"])] = row["weight"]

for row in range(12):
    for column in range(12):
        if column==row:
            continue

        source = risk_scores["location"][row]
        target = risk_scores["location"][column]
        weight = np.abs(edge_lookup.get((source, target), 0))
        propagation_matrix[row,column] = weight * risk_scores["risk_score_model"][column]


supp.display_heatmap(propagation_matrix, risk_scores, "Propagation Matrix [i → j]",
                 save=True, save_path="plots_and_fiqures/propagation_matrix.png")
#-----------------Task 12-----------------------------------
adj_list = {}
node_ids = {}

for index, key in enumerate(risk_scores["location"]):
    adj_list[key] = []
    node_ids[key] = index

for source, target in edge_lookup.keys():
    adj_list[source].append(target)



#-----------------Task 13-----------------------------------


results = supp.simulate_propagation(risk_scores, propagation_matrix, adj_list, node_ids)

sizes = [len(s) for s in results]
#print(f"Mean failures per simulation: {np.mean(sizes):.2f}")
#print(f"P(zero failures): {np.mean([s == 0 for s in sizes]):.3f}")
#print(f"P(propagation > 3): {np.mean([s > 3 for s in sizes]):.3f}")

# Per-turbine failure rate
all_locations = list(risk_scores["location"])
failure_rates = {}

for loc in all_locations:
    rate = np.mean([loc in scenario for scenario in results])
    failure_rates[loc] = rate
    #print(f"{loc}: {rate*100:.1f}% of simulations")


heatmap_for_propagation = supp.transition_heatmap(results, risk_scores)

supp.display_heatmap(heatmap_for_propagation, risk_scores,
                "Heatmap for propagation simulation [row → column]",
                save=True, save_path="plots_and_fiqures/simulation_propagation.png")
RANDOM_INDEX = int(np.random.randint(10000))

supp.plot_propagation(results[RANDOM_INDEX], risk_scores, G, node_ids)

#supp.plot_failure_rates(risk_scores, failure_rates)

#-----------------Task 14-----------------------------------
betweenness_centrality = nx.betweenness_centrality(G)
#print(betweenness_centrality)

#-----------------Task 14-----------------------------------

#-----------------Task 15-----------------------------------


initiators = supp.get_initiators(results)
amplifiers = supp.get_amplifiers(results, risk_scores)
print("Top initiators:", initiators)
print("Top amplifiers:", amplifiers)


summary_of_strategies = supp.compare_strategies(risk_scores, propagation_matrix, adj_list, 
                                                node_ids, initiators, amplifiers,)


supp.plot_comparison(summary_of_strategies)

#-----------------Task 15-----------------------------------