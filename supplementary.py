"""
This is a Module for functions in task 12, 13, 14, 15
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from main import pos

def create_propagation_matrix(edge_lookup, risk_score):
    """
    Function for creating propagation matrix where row is source of failure
    and column is the target. Values in cells are probabilities for failure propagation.

    Parameters

    """
    size = risk_score["location"].size
    prop_matrix = np.zeros((size,size))

    for row in range(size):
        for column in range(size):
            if column==row:
                continue

            source = risk_score["location"][row]
            target = risk_score["location"][column]
            weight = np.abs(edge_lookup.get((source, target), 0))
            prop_matrix[row,column] = weight * risk_score["risk_score_model"][column]
    
    return prop_matrix

def display_heatmap(matrix, risk_score, title, save=False, save_path=None, display=False):
    """
    Function for displaying 12 by 12 heatmap

    Parameters: 
        matrix (np.matrix): Matrix to be displayed
        riks_score (pd.dataframe): Df which contains turbines and risk scores
        title (string): Title of figure
        save (boolean): Save picture or no
        save_path (string): Path to save picture
        display (boolean): wheter or not to diplay heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 6))
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
    
    if display:
        plt.show()

def get_random_choice(risk_score):
    sum_of_risk_scores = sum(risk_score["risk_score_model"])
    probabilities = [loc / sum_of_risk_scores for loc in risk_score["risk_score_model"]]
    return np.random.choice(list(risk_score["location"]), p=probabilities)


def one_simulation(queue_propagation, risk_score, failed, adjacency_list, prop_matrix,
                   node_id, queue_independent=None, propagation=True):
    """
    Function for one simulation. One simulation tries if turbine fails based on probability. 
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
            
            #Making sure that there is at least one failure
            if len(queue_independent) == 0 and len(failed) == 0:
                random_choice = get_random_choice(risk_score)
                failed.append(random_choice)
                for node in adjacency_list[random_choice]:
                    if node not in failed:
                        queue_propagation.append((random_choice, node))

        sets.append(failed)

    return sets

def plot_propagation(failed, risk_score, G, node_id, title="Propagation Simulation"):
    """
    Function for plotting propagation. Displays failed nodes in red and
    others in blue. Also nodes that have higher risk score are displayed bigger.

    Parameters: 
        failed (list): To keep track of already failed turbines
        risk_score (pd.dataframe): Dataframe which contains turbines and risk scores
        edgelist (pd.edgelist): Edgelist of graph of turbines
        node_id (dict): Contains node ID for every turbine based on turbine location
        title (string): Title of figure

    """
    P = G.copy()

    node_colors = ["red" if node in failed else "steelblue" for node in P.nodes()]
    node_sizes  = [risk_score["risk_score_model"][node_id[node]] * 1000 for node in P.nodes()]
    edge_weights = [P[u][v]["weight"] * 3 for u, v in G.edges()]

    nx.draw_networkx_nodes(P, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(P, pos, font_size=8, font_color="black")
    nx.draw_networkx_edges(P, pos, width=edge_weights, alpha=0.5)

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
    plt.clf()
    locations = risk_score["location"].values

    # Sort by failure rate
    sorted_idx = np.argsort(failure_rate)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(locations)),
                  [failure_rate[i] for i in sorted_idx],
                  color="steelblue")

    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels([locations[i] for i in sorted_idx], rotation=45, ha="right")
    ax.set_ylabel("Failure Rate")
    ax.set_title("Per-Turbine Failure Rate Across Simulations")
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

def apply_intervention(risk_score, prop_matrix, strategy, node_id, edge_lookup, targets=None, factor=0.5):
    """
    Function for applying one of three tactics to turbine-network.
    Prioritizing monitoring affects failure probability of target nodes.
    Reinforcing connections around target nodes affects propagation weight.
    Fully isolating a node takes it off which means no failures on that node.

    Parameters: 
        risk_score (pd.dataframe): Df which contains turbines and risk scores
        prop_matrix (np.matrix): Matrix which contains propagation probabilities
        strategy (string): Tactic name (monitor, reinforce edges, isolate)
        node_id (dict): Contains node ID for every turbine based on turbine location
        targets (list): List of target nodes to apply intervention to
        factor (double): How much decreasing score (smaller value affects more)
    """
    risk_copy = risk_score.copy()
    prop_matrix_copy = prop_matrix.copy()

    if strategy == "monitor":
        for target in targets:
            risk_copy.loc[risk_copy["location"] == target, "risk_score_model"] *= factor

        prop_matrix_copy = create_propagation_matrix(edge_lookup, risk_copy)
        display_heatmap(prop_matrix_copy, risk_score, "MOI", display=True)


    elif strategy == "reinforce_edges":
        for target in targets:
            idx = node_id[target]
            prop_matrix_copy[idx, :] *= factor
            prop_matrix_copy[:, idx] *= factor

    elif strategy == "isolate":
        for target in targets:
            idx = node_id[target]
            prop_matrix_copy[idx, :] = 0
            prop_matrix_copy[:, idx] = 0
            risk_copy.loc[risk_copy["location"] == target, "failure_probability"] = 0


    return risk_copy, prop_matrix_copy


def get_top_betweenness_cetrality(G, top_n=3):
    betweenness_centrality = nx.betweenness_centrality(G)
    return sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:top_n]

def get_top_risk_nodes(risk_score, top_n=3):
    risk_copy = risk_score.sort_values(by=["risk_score_model"], ascending=False)
    sorted_locations = list(risk_copy["location"])
    return sorted_locations[:top_n]

def get_critical_nodes(G, risk_score):
    bet_cent = nx.betweenness_centrality(G, weight="weight")
    df_centrality = pd.DataFrame({
        "location": list(bet_cent.keys()),
        "betweenness_centrality": list(bet_cent.values())
    })

    #df_ranked = df_centrality.sort_values(by="betweenness_centrality", ascending=False)

    df_analysis = df_centrality.merge(risk_score[["location","risk_score_model"]],on="location")

    df_analysis.sort_values(by=["risk_score_model", "betweenness_centrality"],ascending=False)

    #These nodes have high risk score and also important node in network this might change if edge weights are recalc or changed 
    critical = df_analysis[
        (df_analysis["risk_score_model"] > df_analysis["risk_score_model"].mean()) & (df_analysis["betweenness_centrality"] > df_analysis["betweenness_centrality"].mean())
    ]
    return list(critical["location"])[:3]


def compare_strategies(risk_score, prop_matrix, adjacency_list, node_id, risk_nodes, critical_nodes, edge_lookup, n_simulations=10000):
    strategies = {
        "baseline": (risk_score, prop_matrix),
        "monitor_high_risk_nodes": apply_intervention(
            risk_score, prop_matrix, "monitor", node_id, edge_lookup, targets=risk_nodes, factor=0.5),
        "reinforce_critical_nodes": apply_intervention(
            risk_score, prop_matrix, "reinforce_edges", node_id, edge_lookup, targets=critical_nodes, factor=0.5),
        "isolate_worst": apply_intervention(
            risk_score, prop_matrix, "isolate", node_id, edge_lookup, targets=risk_nodes[:1]),
    }
    summary = {}
    for name, (rs, pm) in strategies.items():
        res = simulate_propagation(rs, pm, adjacency_list, node_id, n_simulations)
        sizes = [len(s) for s in res]
        summary[name] = {
            "mean_failures": np.mean(sizes),
            "p_zero_propagations": np.mean([s != 1 for s in sizes]),
            "p_many_failures": np.mean([s > 4 for s in sizes]),
            "results": res
        }

    return summary


def plot_comparison(summary, save_path):
    strategies = [s for s in summary if s != "baseline"]
    metrics = ["mean_failures", "p_zero_propagations", "p_many_failures"]
    labels = ["Mean Failures", "P(Zero Propagations)", "P(Failures > 4)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

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
    plt.savefig(save_path)
    plt.show()

