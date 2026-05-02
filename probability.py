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
from main import pos as position

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
overall_failure_probability = scipy.stats.weibull_min.cdf(CUT_IN_THRESHOLD, k, loc=0, scale=scale) + (1 - 
                            scipy.stats.weibull_min.cdf(CUT_OUT_THRESHOLD, k, loc=0, scale=scale))
#print(overall_failure_probability)

risk_scores = pd.read_csv("combined.csv", usecols=["location", "risk_score_model"])
#Adding failure probability for each turbine
risk_scores.insert(2, "failure_probability", risk_scores["risk_score_model"] 
                   * overall_failure_probability)


edgelist = nx.to_pandas_edgelist(G)

#Creating propagation matrix where row is source of failure
#and column is the target. Values in cells are probabilities for failure propagation. 
propagation_matrix = np.zeros((12,12))

edge_lookup = {}
for _, row in edgelist.iterrows():
    edge_lookup[(row["source"], row["target"])] = row["weight"]
    edge_lookup[(row["target"], row["source"])] = row["weight"]

for i in range(12):
    for j in range(12):
        if j==i:
            continue
        
        source = risk_scores["location"][i]
        target = risk_scores["location"][j]
        weight = np.abs(edge_lookup.get((source, target), 0))
        propagation_matrix[i,j] = weight * risk_scores["risk_score_model"][j]

def display_heatmap(matrix, risk_score, title, save=False, save_path=None):
    """
    Function for displayn 12 by 12 heatmap

    param: 
    matrix: np.matrix
    riks_score: pd.dataframe
    title: string
    save: boolean
    save_path: string
    """
    _, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd")

    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels(risk_score["location"], rotation=45, ha="right")
    ax.set_yticklabels(risk_score["location"])

    for i in range(12):
        for j in range(12):
            ax.text(j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=7)

    ax.set_title(title)
    plt.tight_layout()
    if save and save_path is not None:
        plt.savefig(save_path)

    plt.show()

display_heatmap(propagation_matrix, risk_scores, "Propagation Matrix [i → j]",
                 save=True, save_path="plots_and_fiqures/propagation_matrix.png")
#-----------------Task 12-----------------------------------
adj_list = {}
node_id = {}

for i, key in enumerate(risk_scores["location"]): 
    adj_list[key] = []
    node_id[key] = i

for source, target in edge_lookup.keys():
    adj_list[source].append(target)



#-----------------Task 13-----------------------------------

def one_simulation(queue_propagation, risk_score, failed, adjacency_list, propagation_matrix, node_id, 
                   queue_independent=None, primary=True):
    if not primary: 
        test = queue_independent.pop(0)
        if test in failed:
            return
        
        fail = np.random.rand() < risk_score["failure_probability"][node_id[test]]
        if not fail:
            return
          
        failed.append(test)
        for node in adjacency_list[test]:
            if node not in failed:
                queue_propagation.append((test, node))
        return


    failure_source, test = queue_propagation.pop(0)
    fail = np.random.rand() < propagation_matrix[node_id[failure_source], node_id[test]]
    if not fail:
        return

    failed.append(test)
    for node in adjacency_list[test]:
        if node not in failed:
            queue_propagation.append((test, node))

    

def simulate_propagation(risk_score, propagation_matrix, adjacency_list, n_simulations=10000):
    sets = []
    for i in range(n_simulations):
        failed = []
        queue_propagation = []
        sorted_risks = risk_score.sort_values("risk_score_model", ascending=False)
        queue_independent = list(sorted_risks["location"])

        while len(queue_independent) > 0 or len(queue_propagation) > 0:
            if len(queue_propagation) > 0:
                one_simulation(queue_propagation, risk_score, failed, adjacency_list, propagation_matrix, node_id)
                continue

            one_simulation(queue_propagation, risk_score, failed, adjacency_list, propagation_matrix, node_id, queue_independent, primary=False)
        
        sets.append(failed)

    return sets

results = simulate_propagation(risk_scores, propagation_matrix, adj_list)

sizes = [len(s) for s in results]
print(f"Mean failures per simulation: {np.mean(sizes):.2f}")
print(f"P(zero failures): {np.mean([s == 0 for s in sizes]):.3f}")
print(f"P(propagation > 3): {np.mean([s > 3 for s in sizes]):.3f}")

# Per-turbine failure rate
all_locations = list(risk_scores["location"])
failure_rates = {}

for loc in all_locations:
    rate = np.mean([loc in scenario for scenario in results])
    failure_rates[loc] = rate
    #print(f"{loc}: {rate*100:.1f}% of simulations")

def plot_propagation(failed, risk_score, edgelist, title="Propagation Simulation"):
    graph = nx.from_pandas_edgelist(edgelist, "source", "target", edge_attr="weight")
    
    pos = position
    
    node_colors = ["red" if node in failed else "steelblue" for node in graph.nodes()]
    node_sizes  = [risk_score["risk_score_model"][node_id[node]] * 1000 for node in graph.nodes()]
    edge_weights = [graph[u][v]["weight"] * 3 for u, v in G.edges()]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")
    nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.5)
    
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_failure_rates(results, risk_score):
    locations = risk_score["location"].values
    failure_rates = [
        np.mean([loc in s for s in results]) for loc in locations
    ]
    
    # Sort by failure rate
    sorted_idx = np.argsort(failure_rates)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(locations)), 
                  [failure_rates[i] for i in sorted_idx],
                  color="steelblue")
    
    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels([locations[i] for i in sorted_idx], rotation=45, ha="right")
    ax.set_ylabel("Failure Rate")
    ax.set_title("Per-Turbine Failure Rate Across Simulations")
    ax.legend()
    plt.tight_layout()
    plt.show()

def transition_heatmap(result_list, risk_score): 
    heatmap = np.zeros((12, 12))
    for i in range(12):
        preceding = risk_score["location"][i]
        encounters = []
        for chase in result_list:
            if preceding in chase:
                encounters.append(chase)
        for j in range(12):
            if j==i:
                continue
            subsequent = risk_score["location"][j]
            count = count_subsequent_indexes(encounters, preceding, subsequent)
            heatmap[i, j] = count/len(encounters)
    return heatmap

def count_subsequent_indexes(check_list, preceding, subsequent):
    count = 0
    for scenario in check_list:
        index = scenario.index(preceding)
        try:
            if scenario[index + 1] == subsequent:
                count += 1
        except (IndexError, ValueError):
            continue
    
    return count


heatmap_for_propagation = transition_heatmap(results, risk_scores)

display_heatmap(heatmap_for_propagation, risk_scores,
                "Heatmap for propagation simulation [row → column]",
                save=True, save_path="plots_and_fiqures/simulation_propagation.png")
random_index = int(np.random.randint(10000))

#plot_propagation(results[random_index], risk_scores, edgelist)

#plot_failure_rates(results, risk_scores)

#-----------------Task 14-----------------------------------
betweenness_centrality = nx.betweenness_centrality(G)
print(betweenness_centrality)
