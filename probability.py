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
rng = np.random.default_rng(50)

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
# The reason it is approximated this way is that we have no data of previous
# failures and this is the only plausible way of approximating failure that I came up with. 
k, weibull_loc, scale = scipy.stats.weibull_min.fit(gust_speeds, floc=0)
overall_failure_probability = 1 - scipy.stats.weibull_min.cdf(CUT_OUT_THRESHOLD,
                                                              k, loc=0, scale=scale)

print(f"Overall failure probability: {overall_failure_probability}")

risk_scores = pd.read_csv("combined.csv", usecols=["location", "risk_score_model"])
#Adding failure probability for each turbine
risk_scores.insert(2, "failure_probability", risk_scores["risk_score_model"]
                   * overall_failure_probability)

list_of_edges = nx.to_pandas_edgelist(G)

# Creating a dictionary where key is a tuple of turbines and value is weight between those turbines
# The tuple as a key is created both ways
edge_lookup = {}
for _, row in list_of_edges.iterrows():
    edge_lookup[(row["source"], row["target"])] = row["weight"]
    edge_lookup[(row["target"], row["source"])] = row["weight"]

# Creating propagation matrix
propagation_matrix = supp.create_propagation_matrix(edge_lookup, risk_scores)

# Displaying propagation matrix
supp.display_heatmap(propagation_matrix, risk_scores, "Propagation Matrix [row → column]",
                 save=True, save_path="plots_and_fiqures/propagation_matrix.png", display=True)
#-----------------Task 12-----------------------------------

# Creating dictionary for adjacency list and node ids
adj_list = {}
node_ids = {}

# Enumerating through locations in the network
for index, key in enumerate(risk_scores["location"]):
    adj_list[key] = [] # Adding a list to each value of adjacency list
    node_ids[key] = index # Indexing each location of the network for easier usage

for source, target in edge_lookup.keys():
    adj_list[source].append(target) # If there is a edge between location and another node, add it to adjacency list



#-----------------Task 13-----------------------------------

# Doing Monte Carlo simulation (10000 iterations)
results = supp.simulate_propagation(risk_scores, propagation_matrix, adj_list, node_ids)

# Printing statistics of the simulation
sizes = [len(s) for s in results]
print(f"Mean failures per simulation: {np.mean(sizes):.2f}")
print(f"P(propagation > 3): {np.mean([s > 3 for s in sizes]):.3f}")

# Making a list out of all locations
all_locations = list(risk_scores["location"])
# Creating a dictionary for failure rates, where key is a turbine location and value rate for failure in the simulation
failure_rates = {}

# Iterating through all the locations
for loc in all_locations:
    rate = np.mean([loc in scenario for scenario in results])
    failure_rates[loc] = rate

# Creating a heatmap for back to back failure sequence rates of the simulation
heatmap_for_propagation = supp.transition_heatmap(results, risk_scores)

# Displaying results of the simulation with a heatmap
supp.display_heatmap(heatmap_for_propagation, risk_scores,
                "Heatmap for propagation simulation [row → column]",
                save=True, save_path="plots_and_fiqures/simulation_propagation.png", display=True)

# Finding out what is the most common failure sequence in the network
most_common_propagation = max(results, key=results.count)

# Plotting the propagation
supp.plot_propagation(most_common_propagation, risk_scores, G, node_ids,
            "plots_and_fiqures/most_common_propagation.png", "Most Common Propagation Result")

# Plotting failure rates
supp.plot_failure_rates(risk_scores, failure_rates, "plots_and_fiqures/failure_rates.png")

#-----------------Task 14-----------------------------------
# Getting top nodes based on different categories
top_betweenness_nodes = supp.get_top_betweenness_cetrality(G)
high_risk_nodes = supp.get_top_risk_nodes(risk_scores)
critical_nodes = supp.get_critical_nodes(G, risk_scores)
# Printing the found nodes
print("High betweenness centrality nodes: ", top_betweenness_nodes)
print("High risk nodes: ", high_risk_nodes)
print("Critical nodes: ", critical_nodes)

#-----------------Task 14-----------------------------------

#-----------------Task 15-----------------------------------

# Creating the summary of the three different strategies
summary_of_strategies = supp.compare_strategies(risk_scores, propagation_matrix, adj_list,
                        node_ids, high_risk_nodes, critical_nodes, edge_lookup)

# Plotting the effect of the three different strategies
supp.plot_comparison(summary_of_strategies, "plots_and_fiqures/intervention_improvements.png")

#-----------------Task 15-----------------------------------
