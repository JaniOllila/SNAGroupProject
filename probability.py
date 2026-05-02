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

def display_heatmap(matrix, risk_score, title, save=False, save_path=None):
    """
    Function for displayn 12 by 12 heatmap

    Parameters: 
        matrix (np.matrix): Matrix to be displayed
        riks_score (pd.dataframe): Df which contains turbines and risk scores
        title (string): Title of figure
        save (boolean): Save picture or no
        save_path (string): Path to save picture
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
node_ids = {}

for index, key in enumerate(risk_scores["location"]):
    adj_list[key] = []
    node_ids[key] = index

for source, target in edge_lookup.keys():
    adj_list[source].append(target)



#-----------------Task 13-----------------------------------

def one_simulation(queue_propagation, risk_score, failed, adjacency_list, prop_matrix,
                   node_id, queue_independent=None, propagation=True):
    """
    Function for on simulation. One simulation tries if turbine fails based on probability. 
    If catches failure in a turbine, add that turbine to failed list and add adjacent nodes
    to propagation queue. 

    Parameters: 
        queue_propagation (list): Queue for propagation failure testing
        risk_score (pd.dataframe): Df which contains turbines and risk scores
        failed (list): To keep track of already failed turbines
        adjacency_list (dict): Adjacent nodes for each node
        prop_matrix (np.matrix): Matrix which contains propagation probabilities
        node_id (dict): Contains node ID for every turbine based on turbine location
        queue_independent (list): Queue for individual fail testing
        propagation (boolean): Is simulation for propagation or individual testing

    """
    if not propagation:
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
    fail = np.random.rand() < prop_matrix[node_id[failure_source], node_id[test]]
    if not fail:
        return

    failed.append(test)
    for node in adjacency_list[test]:
        if node not in failed:
            queue_propagation.append((test, node))



def simulate_propagation(risk_score, prop_matrix, adjacency_list, node_id, n_simulations=10000):
    """
    Function for simulating how failures propagate based on individual probability for failure and
    probability for failure propagation between adjacent nodes. If one turbine fails, 
    simulation puts adjacent nodes in queue to test failure propagation in those turbines.

    Parameters:  
        risk_score (pd.dataframe): Df which contains turbines and risk scores
        prop_matrix (np.matrix): Matrix which contains propagation probabilities
        adjacency_list (dict): Adjacent nodes for each node
        node_id (dict): Contains node ID for every turbine based on turbine location
        n_simulations (int): Number of simulations

    """
    sets = []
    for _ in range(n_simulations):
        failed = []
        queue_propagation = []
        sorted_risks = risk_score.sort_values("risk_score_model", ascending=False)
        queue_independent = list(sorted_risks["location"])

        while len(queue_independent) > 0 or len(queue_propagation) > 0:
            if len(queue_propagation) > 0:
                one_simulation(queue_propagation, risk_score, failed,
                               adjacency_list, prop_matrix, node_id)
                continue

            one_simulation(queue_propagation, risk_score, failed, adjacency_list,
                           prop_matrix, node_id, queue_independent, propagation=False)

        sets.append(failed)

    return sets

results = simulate_propagation(risk_scores, propagation_matrix, adj_list, node_ids)

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

def plot_propagation(failed, risk_score, edgelist, node_id, title="Propagation Simulation"):
    """
    Function for plotting propagation. Displays failed nodes in red and
    others in blue. Also nodes that have higher risk score are displayed bigger.

    Parameters: 
        failed (list): To keep track of already failed turbines
        risk_score (pd.dataframe): Df which contains turbines and risk scores
        edgelist (pd.edgelist): Edgelist of graph of turbines
        node_id (dict): Contains node ID for every turbine based on turbine location
        title (string): Title of figure

    """
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

def plot_failure_rates(risk_score, failure_rate):
    """
    Function for plotting simulated probability of failure for each turbines

    Parameters: 
        risk_score (pd.dataframe): Df which contains turbines and risk scores
        failure_rate (dict): Dictionary of failure rates 

    """
    locations = risk_score["location"].values

    # Sort by failure rate
    sorted_idx = np.argsort(failure_rate)[::-1]

    _, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(locations)),
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
    """
    Function for creating heatmap of probabiliets of each turbine failure following a certain
    turbine failure. 

    Parameters: 
        result_list (list): List of simulation results
        risk_score (pd.dataframe): Df which contains turbines and risk scores

    """
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

def count_subsequent_indexes(result_list, preceding, subsequent):
    """
    Counts how many scenarios have a certain turbine failure following a certain turbine failrue

    Parameters: 
        result_list (list): List of simulation results
        preceding (string): Location of preceding turbine failure
        subsequent (string): Location of subsequent turbine failure

    """
    count = 0
    for scenario in result_list:
        index_of_pre = scenario.index(preceding)
        try:
            if scenario[index_of_pre + 1] == subsequent:
                count += 1
        except (IndexError, ValueError):
            continue

    return count


heatmap_for_propagation = transition_heatmap(results, risk_scores)

#display_heatmap(heatmap_for_propagation, risk_scores,
#                "Heatmap for propagation simulation [row → column]",
#                save=True, save_path="plots_and_fiqures/simulation_propagation.png")
RANDOM_INDEX = int(np.random.randint(10000))

#plot_propagation(results[RANDOM_INDEX], risk_scores, list_of_edges, node_ids)

#plot_failure_rates(results, risk_scores)

#-----------------Task 14-----------------------------------
betweenness_centrality = nx.betweenness_centrality(G)
#print(betweenness_centrality)

#-----------------Task 14-----------------------------------

#-----------------Task 15-----------------------------------
def apply_intervention(risk_score, prop_matrix, strategy, node_id, targets=None, factor=0.5):
    risk_copy = risk_score.copy()
    prop_matrix_copy = prop_matrix.copy()
    
    if strategy == "monitor":
        # Prioritizing monitoring (affects failure probability of target nodes)
        for target in targets:
            idx = node_id[target]
            risk_copy["failure_probability"][idx] *= factor

    elif strategy == "reinforce_edges":
        # Reinforce connections around target nodes (affects propagation weight)
        for target in targets:
            idx = node_id[target]
            prop_matrix_copy[idx, :] *= factor
            prop_matrix_copy[:, idx] *= factor

    elif strategy == "isolate":
        # Fully isolate a node (take off, no failures)
        for target in targets:
            idx = node_id[target]
            prop_matrix_copy[idx, :] = 0
            prop_matrix_copy[:, idx] = 0
            risk_copy["failure_probability"][idx] = 0


    return risk_copy, prop_matrix_copy


def get_initiators(result_list, top_n=3):
    first_failures = [s[0] for s in result_list if len(s) > 0]
    unique, counts = np.unique(first_failures, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1]
    return unique[sorted_idx][:top_n]


def get_amplifiers(result_list, risk_score):
    locations = risk_score["location"].values
    rates = {loc: np.mean([loc in s for s in result_list]) for loc in locations}
    return sorted(rates, key=rates.get, reverse=True)[:3]

initiators = get_initiators(results)
amplifiers = get_amplifiers(results, risk_scores)
print("Top initiators:", initiators)
print("Top amplifiers:", amplifiers)

def compare_strategies(risk_score, prop_matrix, adjacency_list, node_id, n_simulations=10000):
    strategies = {
        "baseline": (risk_score, prop_matrix),
        "monitor_initiators": apply_intervention(
            risk_score, prop_matrix, "monitor", node_ids, targets=initiators, factor=0.5),
        "reinforce_amplifiers": apply_intervention(
            risk_score, prop_matrix, "reinforce_edges", node_ids, targets=amplifiers, factor=0.5),
        "isolate_worst": apply_intervention(
            risk_score, prop_matrix, "isolate", node_ids, targets=initiators[:1]),
    }
    
    summary = {}
    for name, (rs, pm) in strategies.items():
        res = simulate_propagation(rs, pm, adjacency_list, node_id, n_simulations)
        sizes = [len(s) for s in res]
        summary[name] = {
            "mean_failures": np.mean(sizes),
            "p_zero_failures": np.mean([s == 0 for s in sizes]),
            "p_many_failures": np.mean([s > 3 for s in sizes]),
            "results": res
        }
    
    return summary

summary_of_strategies = compare_strategies(risk_scores, propagation_matrix, adj_list, node_ids)

def plot_comparison(summary):
    strategies = [s for s in summary if s != "baseline"]
    metrics = ["mean_failures", "p_zero_failures", "p_many_failures"]
    labels = ["Mean Failures", "P(Zero Failures)", "P(Failures > 3)"]
    
    _, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for ax, metric, label in zip(axes, metrics, labels):
        baseline_value = summary["baseline"][metric]
        values = [summary[s][metric] for s in strategies]
        improvements = [(baseline_value - value) / baseline_value * 100 for value in values]
        
        colors = ["green" if i > 0 else "red" for i in improvements]
        ax.bar(strategies, improvements, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label}\n(% improvement vs baseline)")
        ax.set_xticklabels(strategies, rotation=25, ha="right")
        ax.set_ylabel("% improvement")
    
    plt.suptitle("Comparison of strategies (improvements based on baseline)", fontsize=14)
    plt.tight_layout()
    plt.show()

plot_comparison(summary_of_strategies)

#-----------------Task 15-----------------------------------